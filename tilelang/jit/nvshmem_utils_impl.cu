#include "nvshmem_utils_impl.cuh"

extern "C" void* nvshmem_create_tensor_impl(
    const std::vector<int64_t>& shape,
    c10::ScalarType dtype,
    int device_index,
    at::Tensor& result
) {
    int64_t element_size = c10::elementSize(dtype);
    int64_t num_elements = 1;
    for (auto dim : shape) num_elements *= dim;
    int64_t size = element_size * num_elements;

    if (size == 0) throw std::runtime_error("Tensor size cannot be zero");
    
    void* ptr = nvshmem_malloc(size);
    if (!ptr) throw std::runtime_error("NVSHMEM malloc failed");
    
    cudaMemset(ptr, 0, size);
    
    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::kCUDA, device_index);
    
    result = torch::from_blob(
        ptr,
        shape,
        [](void* p) { tensor_deleter(p); },
        options
    );
    
    return ptr;
}

extern "C" void nvshmem_init_impl() noexcept {
    try {
        nvshmem_init();
    } catch (const std::exception& e) {
        LOG(ERROR) << "NVSHMEM init failed: " << e.what();
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nvshmem_init", &nvshmem_init_impl, "Initialize NVSHMEM");
    m.def("nvshmem_create_tensor", 
        [](const std::vector<int64_t>& shape, torch::ScalarType dtype, int device) {
            at::Tensor result;
            void* ptr = nvshmem_create_tensor_impl(shape, dtype, device, result);
            return result;
        }, 
        py::arg("shape"), 
        py::arg("dtype"), 
        py::arg("device") = 0
    );
} 
