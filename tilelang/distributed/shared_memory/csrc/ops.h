#pragma once
#include <cstdint>
#include <optional>
#include <pybind11/pytypes.h>
#include <string>
#include <torch/types.h>
#include <vector>

// Tensor utilities
torch::Tensor tensor_from_ptr(uint64_t ptr_val, std::vector<int64_t> shape,
                              const std::string &dtype = "float32",
                              int64_t device = 0, bool take_ownership = false);

torch::Tensor create_tensor(const std::vector<int64_t> &shape,
                            c10::ScalarType dtype);

std::pair<torch::Tensor, torch::Tensor>
create_host_device_tensor(const std::vector<int64_t> &shape,
                          c10::ScalarType dtype);

// IPC operations
pybind11::bytearray create_ipc_handle(void *ptr);

void sync_ipc_handles(
    int rank, const std::vector<int> &device_ids, void **buffer_ptrs_gpu,
    const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles,
    const std::optional<pybind11::bytearray> &root_unique_id_opt);

// VMM operations
void *vmm_malloc(size_t size_raw);
void vmm_free(void *ptr);
pybind11::bytearray create_vmm_handle(void *ptr);
void *open_vmm_handle(const pybind11::bytearray &handle_bytes);
void close_vmm_handle(void *ptr);
bool supports_vmm_fabric();
void sync_vmm_handles(
    int rank, const std::vector<int> &device_ids, void **buffer_ptrs_gpu,
    const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles);

// Multicast support detection
bool supports_multicast();
