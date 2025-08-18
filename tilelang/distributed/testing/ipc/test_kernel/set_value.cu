#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdint>

__global__ void set_value_kernel_float(float* data, int64_t N, float value) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < N) data[idx] = value;
}

void set_value_cuda(uint64_t ptr_uint, int64_t N, double value) {
  if (N <= 0) return;

  void* raw = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr_uint));
  float* data_ptr = reinterpret_cast<float*>(raw);

  const int threads = 256;
  const int blocks = static_cast<int>((N + threads - 1) / threads);

  set_value_kernel_float<<<blocks, threads>>>(data_ptr, N, static_cast<float>(value));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("set_value_cuda", &set_value_cuda,
        "Set value to each element of a CUDA float32 buffer given device pointer and N");
}
