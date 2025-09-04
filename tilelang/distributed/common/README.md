# IPC Extension
This extension provides APIs for creating IPC-based intranode communiation handles.

## Installation

```bash
cd tilelang/distributed/common
python setup.py build_ext --inplace
cd -
```

## Usage

```python
from tilelang.distributed.utils import init_dist, create_dist_tensor, create_tensor

...

# Initialize the IPC-based distributed module
rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

...

# Instead of `torch.empty`, use `create_tensor` to create a torch tensor for later distributed usage, since cudaGetIpcHandle requires a buffer explicitly allocated by `cudaMalloc`
dist_tensor = create_tensor([M, N], torch.float32)  

# Get handles for distributed tensor for later remote copy
buffer_ptrs_gpu = create_dist_tensor(local_rank, num_local_ranks, dst, rank, group).reshape(1, num_local_ranks)

```

See `examples/distributed/example_remote_copy.py` for more details.