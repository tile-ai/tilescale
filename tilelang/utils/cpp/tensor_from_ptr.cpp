#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <limits>
#include <stdexcept>
#include <torch/extension.h>
#include <vector>

static int64_t safe_mul_int64(int64_t a, int64_t b) {
  if (a == 0 || b == 0)
    return 0;
  int64_t maxv = std::numeric_limits<int64_t>::max();
  if (a > maxv / b) {
    throw std::overflow_error("integer overflow in multiplication");
  }
  return a * b;
}

static at::ScalarType dtype_from_string(const std::string &s) {
  if (s == "float32" || s == "float")
    return at::kFloat;
  if (s == "float16" || s == "half")
    return at::kHalf;
  if (s == "bfloat16" || s == "bfloat")
    return at::kBFloat16;
  if (s == "float64" || s == "double")
    return at::kDouble;
  if (s == "uint32")
    return at::kUInt32;
  if (s == "uint64")
    return at::kUInt64;
  if (s == "int32" || s == "int")
    return at::kInt;
  if (s == "int64" || s == "long")
    return at::kLong;
  if (s == "uint8" || s == "byte")
    return at::kByte;
  if (s == "int8")
    return at::kChar;
  if (s == "bool")
    return at::kBool;
  throw std::runtime_error("Unsupported dtype string: " + s);
}

torch::Tensor tensor_from_ptr(uint64_t ptr_val, std::vector<int64_t> shape,
                              const std::string &dtype = "float32",
                              int64_t device = 0, bool take_ownership = false) {
  if (ptr_val == 0) {
    throw std::runtime_error("Received null pointer (0).");
  }
  void *data_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(ptr_val));

  at::ScalarType st = dtype_from_string(dtype);
  auto options = torch::TensorOptions().dtype(st).device(
      torch::kCUDA, static_cast<int>(device));

  // compute number of elements with overflow check
  int64_t nelems = 1;
  for (auto d : shape) {
    if (d < 0)
      throw std::runtime_error("Negative dimension in shape");
    nelems = safe_mul_int64(nelems, d);
  }

  // deleter
  std::function<void(void *)> deleter;
  if (take_ownership) {
    uint64_t saved_ptr = ptr_val;
    deleter = [saved_ptr](void * /*unused*/) {
      void *p = reinterpret_cast<void *>(static_cast<uintptr_t>(saved_ptr));
      cudaError_t cerr = cudaFree(p);
      if (cerr != cudaSuccess) {
        std::fprintf(stderr,
                     "tensor_from_ptr: cudaFree failed in deleter: %s\n",
                     cudaGetErrorString(cerr));
      }
    };
  } else {
    deleter = [](void *) { /* no-op */ };
  }

  torch::Tensor result;
  if (nelems == 0) {
    result = torch::empty(shape, options);
  } else {
    // at::from_blob with deleter + options (zero-copy)
    // Note: from_blob expects a pointer to memory of correct device when device
    // is CUDA.
    result = at::from_blob(data_ptr, shape, deleter, options);
  }

  return result;
}

PYBIND11_MODULE(alloc_cuda, m) {
  m.doc() = "Utility to wrap a CUDA device pointer (uintptr_t) into a "
            "torch.Tensor (zero-copy)";
  m.def("tensor_from_ptr", &tensor_from_ptr, py::arg("ptr"), py::arg("shape"),
        py::arg("dtype") = std::string("float32"), py::arg("device") = 0,
        py::arg("take_ownership") = false,
        "Create a torch.Tensor from an external CUDA device pointer "
        "(uintptr_t).\n\n"
        "ptr: integer device pointer (uintptr_t)\n"
        "shape: list/tuple of dims\n"
        "dtype: string, e.g. 'float32','float16','int32'\n"
        "device: int device ordinal (process-visible)\n"
        "take_ownership: if true the tensor's deleter will call cudaFree on "
        "ptr when freed");
}
