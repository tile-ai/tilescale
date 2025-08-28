#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <torch/types.h>
#include <vector>

namespace py = pybind11;

class EPException : public std::exception {
private:
  std::string message = {};

public:
  explicit EPException(const char *name, const char *file, const int line,
                       const std::string &error) {
    message = std::string("Failed: ") + name + " error " + file + ":" +
              std::to_string(line) + " '" + error + "'";
  }

  const char *what() const noexcept override { return message.c_str(); }
};

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond)                                                   \
  do {                                                                         \
    if (not(cond)) {                                                           \
      throw EPException("Assertion", __FILE__, __LINE__, #cond);               \
    }                                                                          \
  } while (0)
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e));    \
    }                                                                          \
  } while (0)
#endif

torch::Tensor create_tensor(const std::vector<int64_t> &shape,
                            c10::ScalarType dtype) {
  auto current_device = c10::cuda::current_device();
  auto option_gpu =
      at::TensorOptions(at::kCUDA).dtype(dtype).device_index(current_device);
  auto size = torch::elementSize(dtype) *
              std::accumulate(shape.begin(), shape.end(), (size_t)1,
                              std::multiplies<>());
  CUDA_CHECK(cudaDeviceSynchronize());
  void *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return at::from_blob(
      ptr, shape, [=](void *ptr) { CUDA_CHECK(cudaFree(ptr)); }, option_gpu);
}

pybind11::bytearray create_ipc_handle(void *ptr) {
  cudaIpcMemHandle_t *handle =
      (cudaIpcMemHandle_t *)malloc(sizeof(cudaIpcMemHandle_t));
  cudaIpcGetMemHandle(handle, ptr);
  pybind11::bytearray result = {handle->reserved, CUDA_IPC_HANDLE_SIZE};
  return result;
}

void sync_ipc_handles(
    int rank, const std::vector<int> &device_ids, void **buffer_ptrs_gpu,
    const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles,
    const std::optional<pybind11::bytearray> &root_unique_id_opt) {
  // TODO: Support inter-node case
  int rdma_rank = 0;
  int num_nvl_ranks = device_ids.size();
  cudaIpcMemHandle_t ipc_handles[device_ids.size()];
  // buffer_ptrs is a pointer on host and points to the address on device
  void *buffer_ptrs[device_ids.size()];
  // EP_HOST_ASSERT(num_ranks == device_ids.size());
  EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
  for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
    EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
    auto handle_str = std::string(all_gathered_handles[offset + i].value());
    EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
    if (offset + i != rank) {
      std::memcpy(ipc_handles[i].reserved, handle_str.c_str(),
                  CUDA_IPC_HANDLE_SIZE);
      CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i],
                                      cudaIpcMemLazyEnablePeerAccess));
    }
  }

  // Copy all buffer and barrier signal pointers to GPU
  CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs,
                        sizeof(void *) * device_ids.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
}

PYBIND11_MODULE(ipc_ext, m) {
  m.def(
      "_create_tensor",
      [](const std::vector<int64_t> &shape, const py::object &dtype) {
        return create_tensor(shape,
                             torch::python::detail::py_object_to_dtype(dtype));
      },
      py::arg("shape"), py::arg("dtype"));

  m.def(
      "_create_ipc_handle",
      [](uintptr_t ptr_value) {
        void *ptr = reinterpret_cast<void *>(ptr_value);
        return create_ipc_handle(ptr);
      },
      py::arg("ptr_value"));

  m.def(
      "_sync_ipc_handles",
      [](int rank, const std::vector<int> &device_ids,
         uintptr_t buffer_ptrs_gpu_addr,
         const std::vector<std::optional<py::bytearray>> &all_gathered_handles,
         const std::optional<py::bytearray> &root_unique_id_opt) {
        void **buffer_ptrs_gpu =
            reinterpret_cast<void **>(buffer_ptrs_gpu_addr);
        sync_ipc_handles(rank, device_ids, buffer_ptrs_gpu,
                         all_gathered_handles, root_unique_id_opt);
      },
      py::arg("rank"), py::arg("device_ids"), py::arg("buffer_ptrs_gpu_addr"),
      py::arg("all_gathered_handles"), py::arg("root_unique_id_opt"));
}