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
            // barrier_signal_ptrs[i] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
        } 
        // else {
        //     EP_HOST_ASSERT(std::memcmp(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE) == 0);
        // }
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


// Buffer::Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode, bool explicitly_destroy):
//         rank(rank), num_ranks(num_ranks),
//         num_nvl_bytes(num_nvl_bytes), num_rdma_bytes(num_rdma_bytes),
//         low_latency_mode(low_latency_mode),
//         explicitly_destroy(explicitly_destroy),
//         comm_stream(at::cuda::getStreamFromPool(true)) {
//     // Metadata memory
//     int64_t barrier_signal_bytes = NUM_MAX_NVL_PEERS * sizeof(int);
//     int64_t buffer_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(void*);
//     int64_t barrier_signal_ptr_bytes = NUM_MAX_NVL_PEERS * sizeof(int*);

//     // Common checks
//     EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
//     EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
//     EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
//     EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
//     if (num_rdma_bytes > 0)
//         EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

//     // Get ranks
//     CUDA_CHECK(cudaGetDevice(&device_id));
//     rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
//     num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_NVL_PEERS), num_nvl_ranks = std::min(num_ranks, NUM_MAX_NVL_PEERS);
// #ifdef DISABLE_NVSHMEM
//     EP_HOST_ASSERT(num_rdma_ranks == 1 and not low_latency_mode and "NVSHMEM is disabled during compilation");
// #endif

//     // Get device info
//     cudaDeviceProp device_prop = {};
//     CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
//     num_device_sms = device_prop.multiProcessorCount;

//     if (num_nvl_bytes > 0) {
//         // Local IPC: alloc local memory and set local IPC handles
//         CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank], num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes + barrier_signal_ptr_bytes));
//         CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
//         buffer_ptrs_gpu = reinterpret_cast<void**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes);

//         // Set barrier signals
//         barrier_signal_ptrs[nvl_rank] = reinterpret_cast<int*>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
//         barrier_signal_ptrs_gpu = reinterpret_cast<int**>(static_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + barrier_signal_bytes + buffer_ptr_bytes);

//         // No need to synchronize, will do a full device sync during `sync`
//         CUDA_CHECK(cudaMemsetAsync(barrier_signal_ptrs[nvl_rank], 0, barrier_signal_bytes, comm_stream));
//     }

//     // Create 32 MiB workspace
//     CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
//     CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

//     // MoE counter
//     CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
//     CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
//     *moe_recv_counter = -1;

//     // MoE expert-level counter
//     CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
//     CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
//     for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++ i)
//         moe_recv_expert_counter[i] = -1;

//     // MoE RDMA-level counter
//     if (num_rdma_ranks > 0) {
//         CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
//         CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
//         *moe_recv_rdma_counter = -1;
//     }
// }