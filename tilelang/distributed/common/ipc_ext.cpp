#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <torch/types.h>
#include <tuple>
#include <vector>

namespace py = pybind11;

class EPException: public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char *what() const noexcept override { return message.c_str(); }
};


#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif


#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd) \
do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    } \
} while (0)
#endif

pybind11::bytearray create_ipc_handle(void* ptr) {
    cudaIpcMemHandle_t handle;
    cudaIpcGetMemHandle(&handle, ptr);
    return {handle.reserved, CUDA_IPC_HANDLE_SIZE};
}

void sync_ipc_handles(
    int rank,
    const std::vector<int> &device_ids,
    void** buffer_ptrs_gpu,
    const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles,
    const std::optional<pybind11::bytearray>& root_unique_id_opt
) {
    // TODO: Support inter-node case
    int rdma_rank = 0;
    int num_nvl_ranks = device_ids.size();
    cudaIpcMemHandle_t ipc_handles[device_ids.size()];
    // buffer_ptrs is a pointer on host and points to the address on device
    void* buffer_ptrs[device_ids.size()];
    // EP_HOST_ASSERT(num_ranks == device_ids.size());
    EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
    for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++ i) {
        EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
        auto handle_str = std::string(all_gathered_handles[offset + i].value());
        EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
        if (offset + i != rank) {
            std::memcpy(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
            CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i], cudaIpcMemLazyEnablePeerAccess));
        } 
    }

    // Copy all buffer and barrier signal pointers to GPU
    CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * device_ids.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

PYBIND11_MODULE(ipc_ext, m) {
    m.def("create_ipc_handle", [](uintptr_t ptr_value) {
        void* ptr = reinterpret_cast<void*>(ptr_value);
        return create_ipc_handle(ptr);
    }, py::arg("ptr_value"));

    m.def("sync_ipc_handles", [](
        int rank,
        const std::vector<int>& device_ids,
        uintptr_t buffer_ptrs_gpu_addr,
        const std::vector<std::optional<py::bytearray>>& all_gathered_handles,
        const std::optional<py::bytearray>& root_unique_id_opt
    ) {
        void** buffer_ptrs_gpu = reinterpret_cast<void**>(buffer_ptrs_gpu_addr);
        sync_ipc_handles(rank, device_ids, buffer_ptrs_gpu, all_gathered_handles, root_unique_id_opt);
    }, 
    py::arg("rank"), 
    py::arg("device_ids"),
    py::arg("buffer_ptrs_gpu_addr"),
    py::arg("all_gathered_handles"),
    py::arg("root_unique_id_opt")
    ); 
}