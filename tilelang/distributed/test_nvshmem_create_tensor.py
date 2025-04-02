import torch
from tilelang.jit.nvshmem_utils import (
    # init_nvshmem,
    nvshmem_create_tensor,
    ScalarType,
    init_nvshmem
    # nvshmem_my_pe,
    # nvshmem_team_n_pes
)

# init_nvshmem()

def create_sample_tensors():
    float_tensor = nvshmem_create_tensor([1024, 1024], ScalarType.Float)
    print(f"Float tensor created on device {float_tensor.device}, shape: {float_tensor.shape}")

    # half_tensor = nvshmem_create_tensor([512], ScalarType.Half)
    # print(f"Half tensor dtype: {half_tensor.dtype}, elements: {half_tensor.numel()}")

    # double_tensor = nvshmem_create_tensor([256, 256, 3], ScalarType.Double)
    # print(f"Double tensor memory usage: {double_tensor.element_size() * double_tensor.numel()} bytes")

# def cross_node_communication():
#     my_pe = nvshmem_my_pe()
#     world_size = nvshmem_team_n_pes()
    
#     shared_tensor = nvshmem_create_tensor([1024], ScalarType.Float)
    
#     if my_pe == 0:
#         shared_tensor.fill_(1.0)
    
#     torch.cuda.synchronize()
    
#     print(f"PE {my_pe}/{world_size} read value: {shared_tensor[0].item()}")

if __name__ == "__main__":
    init_nvshmem()
    create_sample_tensors()
    # cross_node_communication()