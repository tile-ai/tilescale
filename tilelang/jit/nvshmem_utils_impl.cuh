#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <vector>

inline void tensor_deleter(void* void_ptr) {
    nvshmem_free(void_ptr);
}

extern "C" {
    void* nvshmem_create_tensor_impl(
        const std::vector<int64_t>& shape,
        c10::ScalarType dtype,
        int device_index,
        at::Tensor& result
    );

    void nvshmem_init_impl() noexcept;
} 