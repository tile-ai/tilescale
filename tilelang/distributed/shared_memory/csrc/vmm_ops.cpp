#include <cuda.h>
#include <cuda_runtime.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <vector>

#include "exception.h"
#include "ops.h"

namespace py = pybind11;

// ---------- helpers ----------

static void cu_mem_set_access_all(void *ptr, size_t size) {
  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  std::vector<CUmemAccessDesc> access_desc(device_count);
  for (int idx = 0; idx < device_count; ++idx) {
    access_desc[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc[idx].location.id = idx;
    access_desc[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  }

  CU_CHECK(cuMemSetAccess((CUdeviceptr)ptr, size, access_desc.data(),
                           device_count));
}

static size_t align_to_granularity(size_t size_raw, size_t granularity) {
  size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
  if (size == 0)
    size = granularity;
  return size;
}

// ---------- VMM malloc/free ----------

void *vmm_malloc(size_t size_raw) {
  CUdevice device;
  CU_CHECK(cuCtxGetDevice(&device));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  prop.location.id = device;

  size_t granularity = 0;
  CU_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  size_t size = align_to_granularity(size_raw, granularity);

  CUmemGenericAllocationHandle handle;
  CU_CHECK(cuMemCreate(&handle, size, &prop, 0));

  void *ptr = nullptr;
  CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&ptr, size, granularity, 0, 0));
  CU_CHECK(cuMemMap((CUdeviceptr)ptr, size, 0, handle, 0));
  cu_mem_set_access_all(ptr, size);

  return ptr;
}

void vmm_free(void *ptr) {
  CUmemGenericAllocationHandle handle;
  CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

  size_t size = 0;
  CU_CHECK(cuMemGetAddressRange_v2(NULL, &size, (CUdeviceptr)ptr));

  CU_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  CU_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  CU_CHECK(cuMemRelease(handle));
}

// ---------- handle export/import ----------

py::bytearray create_vmm_handle(void *ptr) {
  CUmemGenericAllocationHandle handle;
  CU_CHECK(cuMemRetainAllocationHandle(&handle, ptr));

  size_t size = 0;
  CU_CHECK(cuMemGetAddressRange_v2(NULL, &size, (CUdeviceptr)ptr));

  CUmemFabricHandle fabric_handle;
  CU_CHECK(cuMemExportToShareableHandle(&fabric_handle, handle,
                                         CU_MEM_HANDLE_TYPE_FABRIC, 0));

  // Serialize: 8 bytes size + sizeof(CUmemFabricHandle)
  std::string buf(sizeof(size_t) + sizeof(CUmemFabricHandle), '\0');
  std::memcpy(&buf[0], &size, sizeof(size_t));
  std::memcpy(&buf[sizeof(size_t)], &fabric_handle, sizeof(CUmemFabricHandle));
  return py::bytearray(buf.data(), buf.size());
}

void *open_vmm_handle(const py::bytearray &handle_bytes) {
  std::string s = (std::string)handle_bytes;
  TS_HOST_ASSERT(s.size() == sizeof(size_t) + sizeof(CUmemFabricHandle));

  size_t size = 0;
  std::memcpy(&size, s.data(), sizeof(size_t));

  CUmemFabricHandle fabric_handle;
  std::memcpy(&fabric_handle, s.data() + sizeof(size_t),
              sizeof(CUmemFabricHandle));

  CUmemGenericAllocationHandle alloc_handle;
  CU_CHECK(cuMemImportFromShareableHandle(&alloc_handle, &fabric_handle,
                                           CU_MEM_HANDLE_TYPE_FABRIC));

  void *ptr = nullptr;
  CU_CHECK(cuMemAddressReserve((CUdeviceptr *)&ptr, size, 0, 0, 0));
  CU_CHECK(cuMemMap((CUdeviceptr)ptr, size, 0, alloc_handle, 0));
  cu_mem_set_access_all(ptr, size);

  return ptr;
}

void close_vmm_handle(void *ptr) {
  vmm_free(ptr);
}

// ---------- fabric support detection ----------

bool supports_vmm_fabric() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0)
    return false;

  int driver_version = 0;
  CUresult cu_err = cuDriverGetVersion(&driver_version);
  if (cu_err != CUDA_SUCCESS || driver_version < 12040)
    return false;

  for (int i = 0; i < device_count; ++i) {
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, i));
    int supported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev));
    if (!supported)
      return false;
  }
  return true;
}

// ---------- unified sync entry ----------

void sync_vmm_handles(
    int rank, const std::vector<int> &device_ids, void **buffer_ptrs_gpu,
    const std::vector<std::optional<py::bytearray>> &all_gathered_handles) {

  const int num = (int)device_ids.size();
  TS_HOST_ASSERT((size_t)num == all_gathered_handles.size());

  std::vector<void *> buffer_ptrs(num, nullptr);

  for (int i = 0; i < num; ++i) {
    TS_HOST_ASSERT(all_gathered_handles[i].has_value());
    if (i != rank) {
      buffer_ptrs[i] = open_vmm_handle(all_gathered_handles[i].value());
    }
  }

  CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs.data(),
                         sizeof(void *) * buffer_ptrs.size(),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------- Multicast support detection ----------

bool supports_multicast() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0)
    return false;

  int driver_version = 0;
  CUresult cu_err = cuDriverGetVersion(&driver_version);
  if (cu_err != CUDA_SUCCESS || driver_version < 12040)
    return false;

  for (int i = 0; i < device_count; ++i) {
    CUdevice dev;
    CU_CHECK(cuDeviceGet(&dev, i));
    int supported = 0;
    CU_CHECK(cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev));
    if (!supported)
      return false;
  }
  return true;
}
