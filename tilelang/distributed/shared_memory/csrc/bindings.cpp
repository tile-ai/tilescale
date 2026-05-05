#include "ops.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "TileScale shared memory allocator (IPC + VMM)";

  // Tensor utilities
  m.def("tensor_from_ptr", &tensor_from_ptr, py::arg("ptr"), py::arg("shape"),
        py::arg("dtype") = std::string("float32"), py::arg("device") = 0,
        py::arg("take_ownership") = false);

  m.def(
      "_create_tensor",
      [](const std::vector<int64_t> &shape, const py::object &dtype) {
        return create_tensor(shape,
                             torch::python::detail::py_object_to_dtype(dtype));
      },
      py::arg("shape"), py::arg("dtype"));

  m.def("create_host_device_tensor", &create_host_device_tensor,
        "Create host/device shared pinned-mapped tensor (shape + dtype)");

  // IPC API
  m.def(
      "_create_ipc_handle",
      [](uintptr_t ptr_value) {
        return create_ipc_handle(reinterpret_cast<void *>(ptr_value));
      },
      py::arg("ptr_value"));

  m.def(
      "_sync_ipc_handles",
      [](int rank, const std::vector<int> &device_ids,
         uintptr_t buffer_ptrs_gpu_addr,
         const std::vector<std::optional<py::bytearray>> &all_gathered_handles,
         const std::optional<py::bytearray> &root_unique_id_opt) {
        sync_ipc_handles(rank, device_ids,
                         reinterpret_cast<void **>(buffer_ptrs_gpu_addr),
                         all_gathered_handles, root_unique_id_opt);
      },
      py::arg("rank"), py::arg("device_ids"), py::arg("buffer_ptrs_gpu_addr"),
      py::arg("all_gathered_handles"), py::arg("root_unique_id_opt"));

  // VMM API
  m.def("_supports_vmm_fabric", &supports_vmm_fabric,
        "Check if all GPUs support CUDA VMM fabric handles");

  m.def(
      "_vmm_malloc",
      [](size_t size) -> uintptr_t {
        return reinterpret_cast<uintptr_t>(vmm_malloc(size));
      },
      py::arg("size"));

  m.def(
      "_vmm_free",
      [](uintptr_t ptr_value) {
        vmm_free(reinterpret_cast<void *>(ptr_value));
      },
      py::arg("ptr_value"));

  m.def(
      "_create_vmm_handle",
      [](uintptr_t ptr_value) {
        return create_vmm_handle(reinterpret_cast<void *>(ptr_value));
      },
      py::arg("ptr_value"));

  m.def(
      "_open_vmm_handle",
      [](const py::bytearray &handle_bytes) -> uintptr_t {
        return reinterpret_cast<uintptr_t>(open_vmm_handle(handle_bytes));
      },
      py::arg("handle_bytes"));

  m.def(
      "_close_vmm_handle",
      [](uintptr_t ptr_value) {
        close_vmm_handle(reinterpret_cast<void *>(ptr_value));
      },
      py::arg("ptr_value"));

  m.def(
      "_sync_vmm_handles",
      [](int rank, const std::vector<int> &device_ids,
         uintptr_t buffer_ptrs_gpu_addr,
         const std::vector<std::optional<py::bytearray>> &all_gathered_handles) {
        sync_vmm_handles(rank, device_ids,
                         reinterpret_cast<void **>(buffer_ptrs_gpu_addr),
                         all_gathered_handles);
      },
      py::arg("rank"), py::arg("device_ids"), py::arg("buffer_ptrs_gpu_addr"),
      py::arg("all_gathered_handles"));
}
