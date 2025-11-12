# Inter-node Examples

Examples in this folder aim to demonstrate the inter-node communication capabilities of TileScale.

- For previous intra-node examples, we can use either NVSHMEM APIs or native communication primitives (e.g. `T.put/get_block`, `T.copy`) provided by TileScale.
- However, for inter-node RDMA communication, currently we rely on NVSHMEM's implementation of IBRC/IBGDA. Hence, it is required to install NVSHMEM and pynvshmem.
    - For detailed installation guide, please refer to [this](../../../docs/get_started/Installation.md#to-use-nvshmem-apis)

## Example Usage 

In order to run inter-node distributed programs, we shall run the launch script simultaneously on multiple nodes.

Example:
```bash
# master 0
NODES=2 NODE_RANK=0 MASTER_IP=ip0 bash tilelang/distributed/launch.sh ./examples/distributed/internode/example_overlapping_allgather.py
# workder 1
NODES=2 NODE_RANK=1 MASTER_IP=ip0 bash tilelang/distributed/launch.sh ./examples/distributed/internode/example_overlapping_allgather.py
```

