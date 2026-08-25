/*!
 * \file shared_memory/shared_memory.cc
 * \brief VMM/IPC/multicast shared memory ops registered via TVM FFI.
 *
 * All functions are registered under the "tl.shared_memory.*" namespace
 * and accessed from Python via tvm_ffi.get_global_func().
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/logging.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#include "cuda/stubs/dynlib.h"
#include "support/check.h"

using namespace tvm;
using namespace tvm::ffi;

// ---------- helpers ----------

#define SM_CUDA_CHECK(cmd)                                                     \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      LOG_FATAL << "CUDA error " << __FILE__ << ":" << __LINE__ << " '"        \
                << cudaGetErrorString(e) << "'";                               \
    }                                                                          \
  } while (0)

#define SM_CU_CHECK(cmd)                                                       \
  do {                                                                         \
    CUresult e = (cmd);                                                        \
    if (e != CUDA_SUCCESS) {                                                   \
      const char *err_str = nullptr;                                           \
      cuGetErrorString(e, &err_str);                                           \
      LOG_FATAL << "CU error " << __FILE__ << ":" << __LINE__ << " '"          \
                << (err_str ? err_str : "unknown") << "'";                     \
    }                                                                          \
  } while (0)

namespace {

void *load_libcuda() {
#if defined(_WIN32) && !defined(__CYGWIN__)
  constexpr const char *kLibCudaPaths[] = {"nvcuda.dll"};
#else
  constexpr const char *kLibCudaPaths[] = {"libcuda.so.1", "libcuda.so"};
#endif
  for (const char *path : kLibCudaPaths) {
    if (void *handle = tvm::tl::stubs::dynlib_open(path)) {
      return handle;
    }
  }
  return nullptr;
}

template <typename T> T load_optional_symbol(void *handle, const char *name) {
  if (handle == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(tvm::tl::stubs::dynlib_sym(handle, name));
}

template <typename T> T require_driver_symbol(T symbol, const char *name) {
  if (symbol == nullptr) {
    LOG_FATAL << "CUDA driver symbol '" << name
              << "' is unavailable; the requested shared-memory operation is "
                 "not supported by this driver";
  }
  return symbol;
}

struct SharedMemoryDriverAPI {
  decltype(&cuMemSetAccess) cuMemSetAccess_;
  decltype(&cuMemGetAllocationGranularity) cuMemGetAllocationGranularity_;
  decltype(&cuMemCreate) cuMemCreate_;
  decltype(&cuMemAddressReserve) cuMemAddressReserve_;
  decltype(&cuMemMap) cuMemMap_;
  decltype(&cuMemRetainAllocationHandle) cuMemRetainAllocationHandle_;
  decltype(&cuMemGetAddressRange_v2) cuMemGetAddressRange_v2_;
  decltype(&cuMemUnmap) cuMemUnmap_;
  decltype(&cuMemAddressFree) cuMemAddressFree_;
  decltype(&cuMemRelease) cuMemRelease_;
  decltype(&cuMemExportToShareableHandle) cuMemExportToShareableHandle_;
  decltype(&cuMemImportFromShareableHandle) cuMemImportFromShareableHandle_;
  decltype(&cuMulticastGetGranularity) cuMulticastGetGranularity_;
  decltype(&cuMulticastCreate) cuMulticastCreate_;
  decltype(&cuMulticastAddDevice) cuMulticastAddDevice_;
  decltype(&cuMulticastBindMem) cuMulticastBindMem_;

  bool HasVMM() const {
    return cuMemSetAccess_ != nullptr &&
           cuMemGetAllocationGranularity_ != nullptr &&
           cuMemCreate_ != nullptr && cuMemAddressReserve_ != nullptr &&
           cuMemMap_ != nullptr && cuMemRetainAllocationHandle_ != nullptr &&
           cuMemGetAddressRange_v2_ != nullptr && cuMemUnmap_ != nullptr &&
           cuMemAddressFree_ != nullptr && cuMemRelease_ != nullptr &&
           cuMemExportToShareableHandle_ != nullptr &&
           cuMemImportFromShareableHandle_ != nullptr;
  }

  bool HasMulticast() const {
    return HasVMM() && cuMulticastGetGranularity_ != nullptr &&
           cuMulticastCreate_ != nullptr && cuMulticastAddDevice_ != nullptr &&
           cuMulticastBindMem_ != nullptr;
  }

  static SharedMemoryDriverAPI *Get() {
    static SharedMemoryDriverAPI api = [] {
      void *handle = load_libcuda();
      SharedMemoryDriverAPI api{};
#define LOAD_OPTIONAL(name)                                                    \
  api.name##_ = load_optional_symbol<decltype(api.name##_)>(handle, #name)
      LOAD_OPTIONAL(cuMemSetAccess);
      LOAD_OPTIONAL(cuMemGetAllocationGranularity);
      LOAD_OPTIONAL(cuMemCreate);
      LOAD_OPTIONAL(cuMemAddressReserve);
      LOAD_OPTIONAL(cuMemMap);
      LOAD_OPTIONAL(cuMemRetainAllocationHandle);
      LOAD_OPTIONAL(cuMemGetAddressRange_v2);
      LOAD_OPTIONAL(cuMemUnmap);
      LOAD_OPTIONAL(cuMemAddressFree);
      LOAD_OPTIONAL(cuMemRelease);
      LOAD_OPTIONAL(cuMemExportToShareableHandle);
      LOAD_OPTIONAL(cuMemImportFromShareableHandle);
      LOAD_OPTIONAL(cuMulticastGetGranularity);
      LOAD_OPTIONAL(cuMulticastCreate);
      LOAD_OPTIONAL(cuMulticastAddDevice);
      LOAD_OPTIONAL(cuMulticastBindMem);
#undef LOAD_OPTIONAL
      return api;
    }();
    return &api;
  }
};

#define SHARED_MEMORY_DRIVER_SYMBOL(name)                                      \
  require_driver_symbol(SharedMemoryDriverAPI::Get()->name##_, #name)

#define cuMemSetAccess SHARED_MEMORY_DRIVER_SYMBOL(cuMemSetAccess)
#define cuMemGetAllocationGranularity                                          \
  SHARED_MEMORY_DRIVER_SYMBOL(cuMemGetAllocationGranularity)
#define cuMemCreate SHARED_MEMORY_DRIVER_SYMBOL(cuMemCreate)
#define cuMemAddressReserve SHARED_MEMORY_DRIVER_SYMBOL(cuMemAddressReserve)
#define cuMemMap SHARED_MEMORY_DRIVER_SYMBOL(cuMemMap)
#define cuMemRetainAllocationHandle                                            \
  SHARED_MEMORY_DRIVER_SYMBOL(cuMemRetainAllocationHandle)
#define cuMemGetAddressRange_v2                                                \
  SHARED_MEMORY_DRIVER_SYMBOL(cuMemGetAddressRange_v2)
#define cuMemUnmap SHARED_MEMORY_DRIVER_SYMBOL(cuMemUnmap)
#define cuMemAddressFree SHARED_MEMORY_DRIVER_SYMBOL(cuMemAddressFree)
#define cuMemRelease SHARED_MEMORY_DRIVER_SYMBOL(cuMemRelease)
#define cuMemExportToShareableHandle                                           \
  SHARED_MEMORY_DRIVER_SYMBOL(cuMemExportToShareableHandle)
#define cuMemImportFromShareableHandle                                         \
  SHARED_MEMORY_DRIVER_SYMBOL(cuMemImportFromShareableHandle)
#define cuMulticastGetGranularity                                              \
  SHARED_MEMORY_DRIVER_SYMBOL(cuMulticastGetGranularity)
#define cuMulticastCreate SHARED_MEMORY_DRIVER_SYMBOL(cuMulticastCreate)
#define cuMulticastAddDevice SHARED_MEMORY_DRIVER_SYMBOL(cuMulticastAddDevice)
#define cuMulticastBindMem SHARED_MEMORY_DRIVER_SYMBOL(cuMulticastBindMem)

class ScopedGenericAllocationHandle {
public:
  explicit ScopedGenericAllocationHandle(CUmemGenericAllocationHandle handle)
      : handle_(handle) {}

  ScopedGenericAllocationHandle(const ScopedGenericAllocationHandle &) = delete;
  ScopedGenericAllocationHandle &
  operator=(const ScopedGenericAllocationHandle &) = delete;

  ~ScopedGenericAllocationHandle() noexcept {
    if (active_) {
      auto release = SharedMemoryDriverAPI::Get()->cuMemRelease_;
      if (release != nullptr) {
        release(handle_);
      }
    }
  }

  void Disarm() noexcept { active_ = false; }

private:
  CUmemGenericAllocationHandle handle_;
  bool active_{true};
};

class ScopedVirtualAddress {
public:
  ScopedVirtualAddress(CUdeviceptr ptr, size_t size, bool mapped = false)
      : ptr_(ptr), size_(size), mapped_(mapped) {}

  ScopedVirtualAddress(const ScopedVirtualAddress &) = delete;
  ScopedVirtualAddress &operator=(const ScopedVirtualAddress &) = delete;

  ~ScopedVirtualAddress() noexcept {
    if (!active_) {
      return;
    }
    auto *api = SharedMemoryDriverAPI::Get();
    if (mapped_ && api->cuMemUnmap_ != nullptr) {
      api->cuMemUnmap_(ptr_, size_);
    }
    if (api->cuMemAddressFree_ != nullptr) {
      api->cuMemAddressFree_(ptr_, size_);
    }
  }

  void MarkMapped() noexcept { mapped_ = true; }
  void MarkUnmapped() noexcept { mapped_ = false; }
  void Disarm() noexcept { active_ = false; }

private:
  CUdeviceptr ptr_;
  size_t size_;
  bool mapped_;
  bool active_{true};
};

void cleanup_vmm_mapping_noexcept(void *ptr) noexcept {
  if (ptr == nullptr) {
    return;
  }
  auto *api = SharedMemoryDriverAPI::Get();
  size_t size = 0;
  if (api->cuMemGetAddressRange_v2_ == nullptr ||
      api->cuMemGetAddressRange_v2_(nullptr, &size, (CUdeviceptr)ptr) !=
          CUDA_SUCCESS ||
      size == 0) {
    return;
  }
  ScopedVirtualAddress mapping((CUdeviceptr)ptr, size, true);
}

class ScopedPeerMappings {
public:
  ScopedPeerMappings(std::vector<void *> &ptrs, bool use_vmm)
      : ptrs_(ptrs), use_vmm_(use_vmm) {}

  ScopedPeerMappings(const ScopedPeerMappings &) = delete;
  ScopedPeerMappings &operator=(const ScopedPeerMappings &) = delete;

  ~ScopedPeerMappings() noexcept {
    if (!active_) {
      return;
    }
    for (void *ptr : ptrs_) {
      if (ptr == nullptr) {
        continue;
      }
      if (use_vmm_) {
        cleanup_vmm_mapping_noexcept(ptr);
      } else {
        cudaIpcCloseMemHandle(ptr);
      }
    }
  }

  void Disarm() noexcept { active_ = false; }

private:
  std::vector<void *> &ptrs_;
  bool use_vmm_;
  bool active_{true};
};

class ScopedIpcMapping {
public:
  explicit ScopedIpcMapping(void *ptr) : ptr_(ptr) {}

  ScopedIpcMapping(const ScopedIpcMapping &) = delete;
  ScopedIpcMapping &operator=(const ScopedIpcMapping &) = delete;

  ~ScopedIpcMapping() noexcept {
    if (active_ && ptr_ != nullptr) {
      cudaIpcCloseMemHandle(ptr_);
    }
  }

  void Disarm() noexcept { active_ = false; }

private:
  void *ptr_;
  bool active_{true};
};

size_t checked_positive_size(int64_t value, const char *argument_name) {
  ICHECK_GT(value, 0) << argument_name << " must be > 0";
  const uint64_t unsigned_value = static_cast<uint64_t>(value);
  ICHECK_LE(unsigned_value,
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
      << argument_name << " exceeds the platform size limit";
  return static_cast<size_t>(unsigned_value);
}

size_t checked_serialized_size(size_t value, const char *argument_name) {
  ICHECK_GT(value, 0U) << argument_name << " must be > 0";
  ICHECK_LE(static_cast<uintmax_t>(value),
            static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()))
      << argument_name << " exceeds the public int64 size limit";
  return value;
}

uintptr_t checked_address(int64_t value, const char *argument_name) {
  ICHECK_GT(value, 0) << argument_name << " must be a non-zero address";
  const uint64_t unsigned_value = static_cast<uint64_t>(value);
  ICHECK_LE(unsigned_value,
            static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max()))
      << argument_name << " exceeds the platform address limit";
  const uintptr_t address = static_cast<uintptr_t>(unsigned_value);
  ICHECK_EQ(address % alignof(void *), 0U)
      << argument_name << " must be pointer-aligned";
  return address;
}

CUmemGenericAllocationHandle checked_handle(int64_t value,
                                            const char *argument_name) {
  ICHECK_GT(value, 0) << argument_name << " must be a non-zero handle";
  return static_cast<CUmemGenericAllocationHandle>(value);
}

int checked_device_count(int64_t value, const char *argument_name) {
  ICHECK_GT(value, 0) << argument_name << " must be > 0";
  ICHECK_LE(static_cast<uint64_t>(value),
            static_cast<uint64_t>(std::numeric_limits<int>::max()))
      << argument_name << " exceeds the CUDA multicast count limit";

  int visible_device_count = 0;
  SM_CUDA_CHECK(cudaGetDeviceCount(&visible_device_count));
  ICHECK_GT(visible_device_count, 0) << "no visible CUDA devices";
  ICHECK_LE(value, static_cast<int64_t>(visible_device_count))
      << argument_name << " exceeds the visible CUDA device count ("
      << visible_device_count << ")";
  return static_cast<int>(value);
}

int checked_device_id(int64_t value, const char *argument_name) {
  ICHECK_GE(value, 0) << argument_name << " must be >= 0";
  ICHECK_LE(value, static_cast<int64_t>(std::numeric_limits<int>::max()))
      << argument_name << " exceeds the CUDA device index limit";

  int visible_device_count = 0;
  SM_CUDA_CHECK(cudaGetDeviceCount(&visible_device_count));
  ICHECK_GT(visible_device_count, 0) << "no visible CUDA devices";
  ICHECK_LT(value, static_cast<int64_t>(visible_device_count))
      << argument_name << " is outside the visible CUDA device range [0, "
      << visible_device_count << ")";
  return static_cast<int>(value);
}

size_t checked_rank_count(int64_t rank, int64_t num_ranks) {
  const size_t count = checked_positive_size(num_ranks, "num_ranks");
  ICHECK_GE(rank, 0) << "rank must be >= 0";
  ICHECK_LT(static_cast<uint64_t>(rank), static_cast<uint64_t>(count))
      << "rank must be smaller than num_ranks";
  return count;
}

size_t checked_multiply(size_t lhs, size_t rhs, const char *description) {
  if (lhs != 0) {
    ICHECK_LE(rhs, std::numeric_limits<size_t>::max() / lhs)
        << description << " overflows size_t";
  }
  return lhs * rhs;
}

size_t checked_align_to_granularity(size_t size_raw, size_t granularity,
                                    const char *argument_name) {
  ICHECK_GT(size_raw, 0U) << argument_name << " must be > 0";
  ICHECK_GT(granularity, 0U) << "CUDA reported zero allocation granularity";

  const size_t remainder = size_raw % granularity;
  const size_t padding = remainder == 0 ? 0 : granularity - remainder;
  ICHECK_LE(size_raw, std::numeric_limits<size_t>::max() - padding)
      << argument_name << " alignment overflows size_t";
  const size_t aligned_size = size_raw + padding;
  ICHECK_LE(static_cast<uintmax_t>(aligned_size),
            static_cast<uintmax_t>(std::numeric_limits<int64_t>::max()))
      << argument_name << " aligned value exceeds the public int64 size limit";
  return aligned_size;
}

void check_exact_bytes(const ffi::Bytes &bytes, size_t expected_size,
                       const char *argument_name) {
  ICHECK_EQ(bytes.size(), expected_size)
      << argument_name << " must contain exactly " << expected_size
      << " bytes, got " << bytes.size();
}

int64_t checked_output_address(CUdeviceptr address, const char *description) {
  ICHECK_NE(address, 0U) << description << " returned a null address";
  ICHECK_LE(address,
            static_cast<CUdeviceptr>(std::numeric_limits<int64_t>::max()))
      << description << " returned an address outside the public int64 range";
  ICHECK_EQ(address % alignof(void *), 0U)
      << description << " returned an address that is not pointer-aligned";
  return static_cast<int64_t>(address);
}

int64_t checked_output_handle(CUmemGenericAllocationHandle handle,
                              const char *description) {
  ICHECK_NE(handle, 0U) << description << " returned a null handle";
  ICHECK_LE(handle, static_cast<CUmemGenericAllocationHandle>(
                        std::numeric_limits<int64_t>::max()))
      << description << " returned a handle outside the public int64 range";
  return static_cast<int64_t>(handle);
}

} // namespace

static void cu_mem_set_access_all(void *ptr, size_t size) {
  int device_count = 0;
  SM_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  ICHECK_GT(device_count, 0) << "no visible CUDA devices";

  std::vector<CUmemAccessDesc> access_desc(device_count);
  for (int idx = 0; idx < device_count; ++idx) {
    access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc[idx].location.id = idx;
    access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }

  SM_CU_CHECK(
      cuMemSetAccess((CUdeviceptr)ptr, size, access_desc.data(), device_count));
}

static void cu_mem_set_access_devices(void *ptr, size_t size, int num_devices) {
  ICHECK_GT(num_devices, 0) << "num_devices must be > 0";
  std::vector<CUmemAccessDesc> access_desc(static_cast<size_t>(num_devices));
  for (int idx = 0; idx < num_devices; ++idx) {
    access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc[idx].location.id = idx;
    access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }

  SM_CU_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, access_desc.data(),
                             static_cast<size_t>(num_devices)));
}

static bool can_create_pinned_allocation(CUdevice device,
                                         CUmemAllocationHandleType handle_type) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = handle_type;
  prop.location.id = device;

  size_t granularity = 0;
  CUresult result = cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (result != CUDA_SUCCESS || granularity == 0) {
    return false;
  }

  CUmemGenericAllocationHandle handle;
  result = cuMemCreate(&handle, granularity, &prop, 0);
  if (result != CUDA_SUCCESS) {
    return false;
  }

  return cuMemRelease(handle) == CUDA_SUCCESS;
}

static bool can_create_fabric_allocation(CUdevice device) {
  return can_create_pinned_allocation(device, CU_MEM_HANDLE_TYPE_FABRIC);
}

static bool can_create_multicast_object(int device_count,
                                        CUmemAllocationHandleType handle_type) {
  CUmulticastObjectProp prop = {};
  prop.numDevices = static_cast<unsigned int>(device_count);
  prop.handleTypes = handle_type;

  size_t granularity = 0;
  CUresult result = cuMulticastGetGranularity(
      &granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED);
  if (result != CUDA_SUCCESS || granularity == 0) {
    return false;
  }

  prop.size = granularity;

  CUmemGenericAllocationHandle mc_handle;
  result = cuMulticastCreate(&mc_handle, &prop);
  if (result != CUDA_SUCCESS) {
    return false;
  }

  bool ok = false;
  if (handle_type == CU_MEM_HANDLE_TYPE_FABRIC) {
    CUmemFabricHandle fabric_handle;
    result = cuMemExportToShareableHandle(&fabric_handle, mc_handle,
                                          CU_MEM_HANDLE_TYPE_FABRIC, 0);
    if (result == CUDA_SUCCESS) {
      CUmemGenericAllocationHandle imported_handle;
      result = cuMemImportFromShareableHandle(&imported_handle, &fabric_handle,
                                              CU_MEM_HANDLE_TYPE_FABRIC);
      if (result == CUDA_SUCCESS) {
        ok = cuMemRelease(imported_handle) == CUDA_SUCCESS;
      }
    }
  }
#if !defined(_WIN32)
  else if (handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    int fd = -1;
    result = cuMemExportToShareableHandle(
        &fd, mc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
    if (result == CUDA_SUCCESS) {
      CUmemGenericAllocationHandle imported_handle;
      result = cuMemImportFromShareableHandle(
          &imported_handle, reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
      if (result == CUDA_SUCCESS) {
        ok = cuMemRelease(imported_handle) == CUDA_SUCCESS;
      }
      close(fd);
    }
  }
#endif

  return cuMemRelease(mc_handle) == CUDA_SUCCESS && ok;
}

// ---------- VMM malloc/free ----------

static int64_t vmm_malloc_with_type(int64_t size_raw,
                                    CUmemAllocationHandleType handle_type) {
  const size_t requested_size = checked_positive_size(size_raw, "size");

  CUdevice device;
  SM_CU_CHECK(cuCtxGetDevice(&device));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = handle_type;
  prop.location.id = device;

  size_t granularity = 0;
  SM_CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                            CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  const size_t size =
      checked_align_to_granularity(requested_size, granularity, "size");

  CUmemGenericAllocationHandle handle;
  SM_CU_CHECK(cuMemCreate(&handle, size, &prop, 0));
  ScopedGenericAllocationHandle handle_guard(handle);

  CUdeviceptr ptr = 0;
  SM_CU_CHECK(cuMemAddressReserve(&ptr, size, granularity, 0, 0));
  ScopedVirtualAddress address_guard(ptr, size);
  SM_CU_CHECK(cuMemMap(ptr, size, 0, handle, 0));
  address_guard.MarkMapped();
  cu_mem_set_access_all(reinterpret_cast<void *>((uintptr_t)ptr), size);
  SM_CU_CHECK(cuMemRelease(handle));
  handle_guard.Disarm();
  const int64_t result = checked_output_address(ptr, "vmm_malloc");
  address_guard.Disarm();

  return result;
}

static int64_t vmm_malloc_impl(int64_t size_raw) {
  return vmm_malloc_with_type(size_raw, CU_MEM_HANDLE_TYPE_FABRIC);
}

// Granularity-aligned size of the mapping backing `ptr` (needed by importers,
// since cuMemMap must cover the full allocation).
static int64_t vmm_alloc_size_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>(checked_address(ptr_val, "ptr"));
  size_t size = 0;
  SM_CU_CHECK(cuMemGetAddressRange_v2(NULL, &size, (CUdeviceptr)ptr));
  return static_cast<int64_t>(size);
}

// POSIX-FD flavor: works on Linux without IMEX channels.
static int64_t vmm_malloc_posix_impl(int64_t size_raw) {
  return vmm_malloc_with_type(size_raw,
                              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

static void vmm_free_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>(checked_address(ptr_val, "ptr"));
  CUmemGenericAllocationHandle handle;
  SM_CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
  ScopedGenericAllocationHandle handle_guard(handle);

  size_t size = 0;
  SM_CU_CHECK(cuMemGetAddressRange_v2(NULL, &size, (CUdeviceptr)ptr));
  ScopedVirtualAddress address_guard((CUdeviceptr)ptr, size, true);

  SM_CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  address_guard.MarkUnmapped();
  SM_CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  address_guard.Disarm();
  SM_CU_CHECK(cuMemRelease(handle));
  handle_guard.Disarm();
}

// ---------- handle export/import ----------

// Returns serialized handle as Bytes.
// Format: 8 bytes size + sizeof(CUmemFabricHandle) bytes fabric handle.
static ffi::Bytes create_vmm_handle_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>(checked_address(ptr_val, "ptr"));
  CUmemGenericAllocationHandle handle;
  SM_CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
  ScopedGenericAllocationHandle handle_guard(handle);

  size_t size = 0;
  SM_CU_CHECK(cuMemGetAddressRange_v2(NULL, &size, (CUdeviceptr)ptr));

  CUmemFabricHandle fabric_handle;
  SM_CU_CHECK(cuMemExportToShareableHandle(&fabric_handle, handle,
                                           CU_MEM_HANDLE_TYPE_FABRIC, 0));
  SM_CU_CHECK(cuMemRelease(handle));
  handle_guard.Disarm();

  std::string raw(sizeof(size_t) + sizeof(CUmemFabricHandle), '\0');
  std::memcpy(&raw[0], &size, sizeof(size_t));
  std::memcpy(&raw[sizeof(size_t)], &fabric_handle, sizeof(CUmemFabricHandle));
  return ffi::Bytes(raw);
}

static int64_t open_vmm_handle_impl(ffi::Bytes handle_bytes) {
  check_exact_bytes(handle_bytes, sizeof(size_t) + sizeof(CUmemFabricHandle),
                    "handle_bytes");
  const char *data = handle_bytes.data();

  size_t size = 0;
  std::memcpy(&size, data, sizeof(size_t));
  size = checked_serialized_size(size, "serialized allocation size");

  CUmemFabricHandle fabric_handle;
  std::memcpy(&fabric_handle, data + sizeof(size_t), sizeof(CUmemFabricHandle));

  CUmemGenericAllocationHandle alloc_handle;
  SM_CU_CHECK(cuMemImportFromShareableHandle(&alloc_handle, &fabric_handle,
                                             CU_MEM_HANDLE_TYPE_FABRIC));
  ScopedGenericAllocationHandle handle_guard(alloc_handle);

  CUdeviceptr ptr = 0;
  SM_CU_CHECK(cuMemAddressReserve(&ptr, size, 0, 0, 0));
  ScopedVirtualAddress address_guard(ptr, size);
  SM_CU_CHECK(cuMemMap(ptr, size, 0, alloc_handle, 0));
  address_guard.MarkMapped();
  cu_mem_set_access_all(reinterpret_cast<void *>((uintptr_t)ptr), size);
  SM_CU_CHECK(cuMemRelease(alloc_handle));
  handle_guard.Disarm();
  const int64_t result = checked_output_address(ptr, "open_vmm_handle");
  address_guard.Disarm();

  return result;
}

static void close_vmm_handle_impl(int64_t ptr_val) { vmm_free_impl(ptr_val); }

#if !defined(_WIN32)
// Export the allocation backing `ptr` as a POSIX file descriptor. The caller
// owns the returned fd (pass it to a peer process via SCM_RIGHTS, then close).
static int64_t vmm_export_posix_fd_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>(checked_address(ptr_val, "ptr"));
  CUmemGenericAllocationHandle handle;
  SM_CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
  ScopedGenericAllocationHandle handle_guard(handle);

  int fd = -1;
  SM_CU_CHECK(cuMemExportToShareableHandle(
      &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  SM_CU_CHECK(cuMemRelease(handle));
  handle_guard.Disarm();
  return static_cast<int64_t>(fd);
}

// Import a peer allocation from a POSIX fd received over SCM_RIGHTS and map
// it into this process. The caller still owns (and should close) the fd.
static int64_t vmm_import_posix_fd_impl(int64_t fd_val, int64_t size_raw) {
  const size_t size = checked_positive_size(size_raw, "size");
  const int fd = static_cast<int>(fd_val);
  if (fd < 0) {
    LOG_FATAL << "vmm_import_posix_fd: invalid fd " << fd_val;
  }

  CUmemGenericAllocationHandle alloc_handle;
  SM_CU_CHECK(cuMemImportFromShareableHandle(
      &alloc_handle, reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  ScopedGenericAllocationHandle handle_guard(alloc_handle);

  CUdeviceptr ptr = 0;
  SM_CU_CHECK(cuMemAddressReserve(&ptr, size, 0, 0, 0));
  ScopedVirtualAddress address_guard(ptr, size);
  SM_CU_CHECK(cuMemMap(ptr, size, 0, alloc_handle, 0));
  address_guard.MarkMapped();
  cu_mem_set_access_all(reinterpret_cast<void *>((uintptr_t)ptr), size);
  SM_CU_CHECK(cuMemRelease(alloc_handle));
  handle_guard.Disarm();
  const int64_t result = checked_output_address(ptr, "vmm_import_posix_fd");
  address_guard.Disarm();

  return result;
}
#endif

// ---------- IPC handle ----------

static ffi::Bytes create_ipc_handle_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>(checked_address(ptr_val, "ptr"));
  cudaIpcMemHandle_t handle{};
  SM_CUDA_CHECK(cudaIpcGetMemHandle(&handle, ptr));
  return ffi::Bytes(reinterpret_cast<const char *>(handle.reserved),
                    CUDA_IPC_HANDLE_SIZE);
}

static int64_t open_ipc_handle_impl(ffi::Bytes handle_bytes) {
  check_exact_bytes(handle_bytes, CUDA_IPC_HANDLE_SIZE, "handle_bytes");
  cudaIpcMemHandle_t handle{};
  std::memcpy(handle.reserved, handle_bytes.data(), CUDA_IPC_HANDLE_SIZE);

  void *ptr = nullptr;
  SM_CUDA_CHECK(
      cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
  ScopedIpcMapping mapping_guard(ptr);
  const int64_t result = checked_output_address(
      reinterpret_cast<CUdeviceptr>(ptr), "open_ipc_handle");
  mapping_guard.Disarm();
  return result;
}

static void close_ipc_handle_impl(int64_t ptr_val) {
  void *ptr = reinterpret_cast<void *>(checked_address(ptr_val, "ptr"));
  SM_CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
}

// ---------- support detection ----------

static bool supports_vmm_fabric_impl() {
  if (!SharedMemoryDriverAPI::Get()->HasVMM()) {
    return false;
  }

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0)
    return false;

  int driver_version = 0;
  CUresult cu_err = cuDriverGetVersion(&driver_version);
  if (cu_err != CUDA_SUCCESS || driver_version < 12040)
    return false;

  for (int i = 0; i < device_count; ++i) {
    CUdevice dev = static_cast<CUdevice>(i);
    int supported = 0;
    CUresult result = cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
    if (result != CUDA_SUCCESS) {
      return false;
    }
    if (!supported)
      return false;
    if (!can_create_fabric_allocation(dev))
      return false;
  }
  return true;
}

static bool supports_multicast_impl() {
  if (!SharedMemoryDriverAPI::Get()->HasMulticast()) {
    return false;
  }

  if (!supports_vmm_fabric_impl()) {
    return false;
  }

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0)
    return false;

  int driver_version = 0;
  CUresult cu_err = cuDriverGetVersion(&driver_version);
  if (cu_err != CUDA_SUCCESS || driver_version < 12040)
    return false;

  for (int i = 0; i < device_count; ++i) {
    CUdevice dev = static_cast<CUdevice>(i);
    int supported = 0;
    CUresult result = cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev);
    if (result != CUDA_SUCCESS) {
      return false;
    }
    if (!supported)
      return false;
  }
  return can_create_multicast_object(device_count, CU_MEM_HANDLE_TYPE_FABRIC);
}

// POSIX-FD flavor of the probes: usable on Linux single-node setups without
// IMEX channels (fabric handles require IMEX; POSIX fds only require peer
// processes on the same host exchanging fds over SCM_RIGHTS).
static bool supports_vmm_posix_impl() {
#if defined(_WIN32)
  return false;
#else
  if (!SharedMemoryDriverAPI::Get()->HasVMM()) {
    return false;
  }

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0)
    return false;

  for (int i = 0; i < device_count; ++i) {
    CUdevice dev = static_cast<CUdevice>(i);
    int supported = 0;
    CUresult result = cuDeviceGetAttribute(
        &supported,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, dev);
    if (result != CUDA_SUCCESS || !supported) {
      return false;
    }
    if (!can_create_pinned_allocation(dev,
                                      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR))
      return false;
  }
  return true;
#endif
}

static bool supports_multicast_posix_impl() {
#if defined(_WIN32)
  return false;
#else
  if (!SharedMemoryDriverAPI::Get()->HasMulticast()) {
    return false;
  }

  if (!supports_vmm_posix_impl()) {
    return false;
  }

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0)
    return false;

  for (int i = 0; i < device_count; ++i) {
    CUdevice dev = static_cast<CUdevice>(i);
    int supported = 0;
    CUresult result = cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev);
    if (result != CUDA_SUCCESS || !supported) {
      return false;
    }
  }
  return can_create_multicast_object(device_count,
                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
#endif
}

// ---------- Multicast (NVSwitch) ----------
// Multi-process multi-GPU with fabric handles (same as vmm_malloc).
// Each process manages one GPU. MC handle shared via fabric export/import.

// Create multicast object, returns handle as int64.
static int64_t mc_create_with_type(int64_t size_raw, int64_t num_devices,
                                   CUmemAllocationHandleType handle_type) {
  const size_t requested_size = checked_positive_size(size_raw, "size");
  const int device_count = checked_device_count(num_devices, "num_devices");

  CUmulticastObjectProp prop = {};
  prop.numDevices = static_cast<unsigned int>(device_count);
  prop.handleTypes = handle_type;

  size_t granularity = 0;
  SM_CU_CHECK(cuMulticastGetGranularity(&granularity, &prop,
                                        CU_MULTICAST_GRANULARITY_RECOMMENDED));

  const size_t size =
      checked_align_to_granularity(requested_size, granularity, "size");
  prop.size = size;

  CUmemGenericAllocationHandle mc_handle;
  SM_CU_CHECK(cuMulticastCreate(&mc_handle, &prop));
  ScopedGenericAllocationHandle handle_guard(mc_handle);
  const int64_t result = checked_output_handle(mc_handle, "mc_create");
  handle_guard.Disarm();

  return result;
}

static int64_t mc_create_impl(int64_t size_raw, int64_t num_devices) {
  return mc_create_with_type(size_raw, num_devices, CU_MEM_HANDLE_TYPE_FABRIC);
}

static int64_t mc_create_posix_impl(int64_t size_raw, int64_t num_devices) {
  return mc_create_with_type(size_raw, num_devices,
                             CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

// Export multicast handle as fabric handle bytes (for sharing across processes)
static ffi::Bytes mc_export_handle_impl(int64_t mc_handle_val) {
  CUmemGenericAllocationHandle mc_handle =
      checked_handle(mc_handle_val, "mc_handle");

  CUmemFabricHandle fabric_handle;
  SM_CU_CHECK(cuMemExportToShareableHandle(&fabric_handle, mc_handle,
                                           CU_MEM_HANDLE_TYPE_FABRIC, 0));

  return ffi::Bytes(reinterpret_cast<const char *>(&fabric_handle),
                    sizeof(CUmemFabricHandle));
}

// Import multicast handle from fabric handle bytes, returns handle as int64.
static int64_t mc_import_handle_impl(ffi::Bytes handle_bytes) {
  check_exact_bytes(handle_bytes, sizeof(CUmemFabricHandle), "handle_bytes");

  CUmemFabricHandle fabric_handle;
  std::memcpy(&fabric_handle, handle_bytes.data(), sizeof(CUmemFabricHandle));

  CUmemGenericAllocationHandle mc_handle;
  SM_CU_CHECK(cuMemImportFromShareableHandle(&mc_handle, &fabric_handle,
                                             CU_MEM_HANDLE_TYPE_FABRIC));
  ScopedGenericAllocationHandle handle_guard(mc_handle);
  const int64_t result = checked_output_handle(mc_handle, "mc_import_handle");
  handle_guard.Disarm();

  return result;
}

#if !defined(_WIN32)
// Export multicast handle as a POSIX fd (caller owns and closes the fd).
static int64_t mc_export_posix_fd_impl(int64_t mc_handle_val) {
  CUmemGenericAllocationHandle mc_handle =
      checked_handle(mc_handle_val, "mc_handle");

  int fd = -1;
  SM_CU_CHECK(cuMemExportToShareableHandle(
      &fd, mc_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  return static_cast<int64_t>(fd);
}

// Import multicast handle from a POSIX fd received over SCM_RIGHTS. The
// caller still owns (and should close) the fd.
static int64_t mc_import_posix_fd_impl(int64_t fd_val) {
  const int fd = static_cast<int>(fd_val);
  if (fd < 0) {
    LOG_FATAL << "mc_import_posix_fd: invalid fd " << fd_val;
  }

  CUmemGenericAllocationHandle mc_handle;
  SM_CU_CHECK(cuMemImportFromShareableHandle(
      &mc_handle, reinterpret_cast<void *>(static_cast<intptr_t>(fd)),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  ScopedGenericAllocationHandle handle_guard(mc_handle);
  const int64_t result = checked_output_handle(mc_handle, "mc_import_posix_fd");
  handle_guard.Disarm();

  return result;
}
#endif

// Add a device to the multicast object
static void mc_add_device_impl(int64_t mc_handle_val, int64_t device_id) {
  CUmemGenericAllocationHandle mc_handle =
      checked_handle(mc_handle_val, "mc_handle");
  CUdevice device =
      static_cast<CUdevice>(checked_device_id(device_id, "device_id"));
  SM_CU_CHECK(cuMulticastAddDevice(mc_handle, device));
}

// Bind a physical memory VA (from vmm_malloc) to the multicast object
static void mc_bind_mem_impl(int64_t mc_handle_val, int64_t ptr_val,
                             int64_t size_raw) {
  CUmemGenericAllocationHandle mc_handle =
      checked_handle(mc_handle_val, "mc_handle");
  void *ptr = reinterpret_cast<void *>(checked_address(ptr_val, "ptr"));
  const size_t size = checked_positive_size(size_raw, "size");

  // Retrieve the physical allocation handle from the mapped pointer
  CUmemGenericAllocationHandle phys_handle;
  SM_CU_CHECK(cuMemRetainAllocationHandle(&phys_handle, ptr));
  ScopedGenericAllocationHandle handle_guard(phys_handle);

  // Bind to multicast
  SM_CU_CHECK(cuMulticastBindMem(mc_handle, 0, phys_handle, 0, size, 0));

  // Release the temporary handle reference
  SM_CU_CHECK(cuMemRelease(phys_handle));
  handle_guard.Disarm();
}

static size_t mc_granularity(int device_count,
                             CUmemAllocationHandleType handle_type) {
  CUmulticastObjectProp prop = {};
  prop.numDevices = static_cast<unsigned int>(device_count);
  prop.handleTypes = handle_type;

  size_t granularity = 0;
  SM_CU_CHECK(cuMulticastGetGranularity(&granularity, &prop,
                                        CU_MULTICAST_GRANULARITY_RECOMMENDED));
  return granularity;
}

// Map multicast object to a VA, returns mc_ptr. Does NOT release handle.
static int64_t mc_map_with_type(int64_t mc_handle_val, int64_t size_raw,
                                int64_t num_devices,
                                CUmemAllocationHandleType handle_type) {
  CUmemGenericAllocationHandle mc_handle =
      checked_handle(mc_handle_val, "mc_handle");
  const size_t requested_size = checked_positive_size(size_raw, "size");
  const int device_count = checked_device_count(num_devices, "num_devices");

  const size_t granularity = mc_granularity(device_count, handle_type);
  const size_t size =
      checked_align_to_granularity(requested_size, granularity, "size");

  CUdeviceptr mc_ptr = 0;
  SM_CU_CHECK(cuMemAddressReserve(&mc_ptr, size, granularity, 0, 0));
  ScopedVirtualAddress address_guard(mc_ptr, size);
  SM_CU_CHECK(cuMemMap(mc_ptr, size, 0, mc_handle, 0));
  address_guard.MarkMapped();
  cu_mem_set_access_devices(reinterpret_cast<void *>((uintptr_t)mc_ptr), size,
                            device_count);
  const int64_t result = checked_output_address(mc_ptr, "mc_map");
  address_guard.Disarm();

  return result;
}

static int64_t mc_map_impl(int64_t mc_handle_val, int64_t size_raw,
                           int64_t num_devices) {
  return mc_map_with_type(mc_handle_val, size_raw, num_devices,
                          CU_MEM_HANDLE_TYPE_FABRIC);
}

static int64_t mc_map_posix_impl(int64_t mc_handle_val, int64_t size_raw,
                                 int64_t num_devices) {
  return mc_map_with_type(mc_handle_val, size_raw, num_devices,
                          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

// Release a multicast handle (call after map)
static void mc_release_handle_impl(int64_t mc_handle_val) {
  CUmemGenericAllocationHandle mc_handle =
      checked_handle(mc_handle_val, "mc_handle");
  ScopedGenericAllocationHandle handle_guard(mc_handle);
  SM_CU_CHECK(cuMemRelease(mc_handle));
  handle_guard.Disarm();
}

// Free multicast VA mapping
static void mc_unmap_with_type(int64_t mc_ptr_val, int64_t size_raw,
                               int64_t num_devices,
                               CUmemAllocationHandleType handle_type) {
  void *ptr = reinterpret_cast<void *>(checked_address(mc_ptr_val, "mc_ptr"));
  const size_t requested_size = checked_positive_size(size_raw, "size");
  const int device_count = checked_device_count(num_devices, "num_devices");

  const size_t granularity = mc_granularity(device_count, handle_type);
  const size_t size =
      checked_align_to_granularity(requested_size, granularity, "size");

  ScopedVirtualAddress address_guard((CUdeviceptr)ptr, size, true);
  SM_CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  address_guard.MarkUnmapped();
  SM_CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  address_guard.Disarm();
}

static void mc_unmap_impl(int64_t mc_ptr_val, int64_t size_raw,
                          int64_t num_devices) {
  mc_unmap_with_type(mc_ptr_val, size_raw, num_devices,
                     CU_MEM_HANDLE_TYPE_FABRIC);
}

static void mc_unmap_posix_impl(int64_t mc_ptr_val, int64_t size_raw,
                                int64_t num_devices) {
  mc_unmap_with_type(mc_ptr_val, size_raw, num_devices,
                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

// Get the aligned size for multicast
static int64_t mc_get_aligned_size_with_type(
    int64_t size_raw, int64_t num_devices,
    CUmemAllocationHandleType handle_type) {
  const size_t requested_size = checked_positive_size(size_raw, "size");
  const int device_count = checked_device_count(num_devices, "num_devices");

  const size_t granularity = mc_granularity(device_count, handle_type);
  return static_cast<int64_t>(
      checked_align_to_granularity(requested_size, granularity, "size"));
}

static int64_t mc_get_aligned_size_impl(int64_t size_raw, int64_t num_devices) {
  return mc_get_aligned_size_with_type(size_raw, num_devices,
                                       CU_MEM_HANDLE_TYPE_FABRIC);
}

static int64_t mc_get_aligned_size_posix_impl(int64_t size_raw,
                                              int64_t num_devices) {
  return mc_get_aligned_size_with_type(size_raw, num_devices,
                                       CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

// ---------- sync helpers ----------

// Synchronize VMM handles: open all peer handles and write pointers to GPU.
// peer_handles is a comma-separated list of hex-encoded handle bytes (or "SELF"
// for local rank). We pass individual handle open results back via buffer_ptrs.
// packed_handles: num_ranks concatenated raw handle bytes
static void sync_vmm_handles_impl(int64_t rank, int64_t num_ranks,
                                  int64_t buffer_ptrs_gpu_addr,
                                  ffi::Bytes packed_handles) {
  const size_t handle_size = sizeof(size_t) + sizeof(CUmemFabricHandle);
  const size_t rank_count = checked_rank_count(rank, num_ranks);
  const size_t rank_index = static_cast<size_t>(rank);
  const uintptr_t table_address =
      checked_address(buffer_ptrs_gpu_addr, "buffer_ptrs_gpu_addr");
  const size_t expected_bytes =
      checked_multiply(handle_size, rank_count, "packed VMM handle size");
  check_exact_bytes(packed_handles, expected_bytes, "packed_handles");

  std::vector<void *> buffer_ptrs(rank_count, nullptr);
  ScopedPeerMappings mappings_guard(buffer_ptrs, true);

  for (size_t i = 0; i < rank_count; ++i) {
    if (i != rank_index) {
      ffi::Bytes h(packed_handles.data() + i * handle_size, handle_size);
      buffer_ptrs[i] =
          reinterpret_cast<void *>((uintptr_t)open_vmm_handle_impl(h));
    }
  }

  void **gpu_ptr = reinterpret_cast<void **>(table_address);
  const size_t table_bytes = checked_multiply(
      sizeof(void *), buffer_ptrs.size(), "peer pointer table size");
  SM_CUDA_CHECK(cudaMemcpy(gpu_ptr, buffer_ptrs.data(), table_bytes,
                           cudaMemcpyHostToDevice));
  SM_CUDA_CHECK(cudaDeviceSynchronize());
  mappings_guard.Disarm();
}

static void sync_ipc_handles_impl(int64_t rank, int64_t num_ranks,
                                  int64_t buffer_ptrs_gpu_addr,
                                  ffi::Bytes packed_handles) {
  const size_t rank_count = checked_rank_count(rank, num_ranks);
  const size_t rank_index = static_cast<size_t>(rank);
  const uintptr_t table_address =
      checked_address(buffer_ptrs_gpu_addr, "buffer_ptrs_gpu_addr");
  const size_t expected_bytes = checked_multiply(
      CUDA_IPC_HANDLE_SIZE, rank_count, "packed IPC handle size");
  check_exact_bytes(packed_handles, expected_bytes, "packed_handles");

  std::vector<void *> buffer_ptrs(rank_count, nullptr);
  ScopedPeerMappings mappings_guard(buffer_ptrs, false);

  for (size_t i = 0; i < rank_count; ++i) {
    if (i != rank_index) {
      ffi::Bytes h(packed_handles.data() + i * CUDA_IPC_HANDLE_SIZE,
                   CUDA_IPC_HANDLE_SIZE);
      buffer_ptrs[i] =
          reinterpret_cast<void *>((uintptr_t)open_ipc_handle_impl(h));
    }
  }

  void **gpu_ptr = reinterpret_cast<void **>(table_address);
  const size_t table_bytes = checked_multiply(
      sizeof(void *), buffer_ptrs.size(), "peer pointer table size");
  SM_CUDA_CHECK(cudaMemcpy(gpu_ptr, buffer_ptrs.data(), table_bytes,
                           cudaMemcpyHostToDevice));
  SM_CUDA_CHECK(cudaDeviceSynchronize());
  mappings_guard.Disarm();
}

// ---------- Registration ----------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  // VMM
  refl::GlobalDef().def("tl.shared_memory.vmm_malloc", vmm_malloc_impl);
  refl::GlobalDef().def("tl.shared_memory.vmm_free", vmm_free_impl);
  refl::GlobalDef().def("tl.shared_memory.vmm_alloc_size", vmm_alloc_size_impl);
  refl::GlobalDef().def("tl.shared_memory.create_vmm_handle",
                        create_vmm_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.open_vmm_handle",
                        open_vmm_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.close_vmm_handle",
                        close_vmm_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.sync_vmm_handles",
                        sync_vmm_handles_impl);

  // IPC
  refl::GlobalDef().def("tl.shared_memory.create_ipc_handle",
                        create_ipc_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.open_ipc_handle",
                        open_ipc_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.close_ipc_handle",
                        close_ipc_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.sync_ipc_handles",
                        sync_ipc_handles_impl);

  // Support detection
#if !defined(_WIN32)
  // POSIX-FD flavors (Linux single-node, no IMEX required)
  refl::GlobalDef().def("tl.shared_memory.vmm_malloc_posix",
                        vmm_malloc_posix_impl);
  refl::GlobalDef().def("tl.shared_memory.vmm_export_posix_fd",
                        vmm_export_posix_fd_impl);
  refl::GlobalDef().def("tl.shared_memory.vmm_import_posix_fd",
                        vmm_import_posix_fd_impl);
  refl::GlobalDef().def("tl.shared_memory.supports_vmm_posix",
                        supports_vmm_posix_impl);
  refl::GlobalDef().def("tl.shared_memory.supports_multicast_posix",
                        supports_multicast_posix_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_create_posix",
                        mc_create_posix_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_export_posix_fd",
                        mc_export_posix_fd_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_import_posix_fd",
                        mc_import_posix_fd_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_map_posix", mc_map_posix_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_unmap_posix",
                        mc_unmap_posix_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_get_aligned_size_posix",
                        mc_get_aligned_size_posix_impl);
#endif

  refl::GlobalDef().def("tl.shared_memory.supports_vmm_fabric",
                        supports_vmm_fabric_impl);
  refl::GlobalDef().def("tl.shared_memory.supports_multicast",
                        supports_multicast_impl);

  // Multicast (NVSwitch)
  refl::GlobalDef().def("tl.shared_memory.mc_create", mc_create_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_export_handle",
                        mc_export_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_import_handle",
                        mc_import_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_add_device", mc_add_device_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_bind_mem", mc_bind_mem_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_map", mc_map_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_release_handle",
                        mc_release_handle_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_unmap", mc_unmap_impl);
  refl::GlobalDef().def("tl.shared_memory.mc_get_aligned_size",
                        mc_get_aligned_size_impl);
}
