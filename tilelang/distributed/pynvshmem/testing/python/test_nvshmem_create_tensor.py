import torch
import torch.distributed
import pynvshmem
import os
import datetime

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

torch.cuda.set_device(LOCAL_RANK)
torch.distributed.init_process_group(
    backend="nccl",
    world_size=WORLD_SIZE,
    rank=RANK,
    timeout=datetime.timedelta(seconds=1800),
)
assert torch.distributed.is_initialized()
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

torch.cuda.synchronize()
pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

N = 1024
dtype = torch.float32
t = pynvshmem.nvshmem_create_tensor((N,), dtype)

print("nvshmem_create_tensor done")
print(t)
