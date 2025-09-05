# Benchmarks for IPC communication

This benchmark aims to measure and compare the bandwidth of different implementations of IPC communication:
We launch only one block on each rank to avoid NVLink bandwidth as the bottleneck.

## NVSHMEM-based push/pull
```bash
GPUS=2 bash tilelang/distributed/launch.sh benchmark/distributed/ipc_impls/benchmark_nvshmem_p2p.py
```

## Unrolled-copy implemented in TileScale (*ours*)
```bash
export TILELANG_USE_DISTRIBUTED=1  # 
python benchmark/distributed/ipc_impls/benchmark_unrolledcp_p2p.py
```

## Results on Hopper connected by NVLink
| Size (Bytes) | NVSHMEM Push bandwidth (GB/s) | NVSHMEM Pull bandwidth (GB/s) | TileScale Push bandwidth (GB/s) | TileScale Pull bandwidth (GB/s) |
|--------------|---------------------|---------------------|-----------------------|-----------------------|
| 2,048        |  0.1680             |  0.1755             |  0.0150                 |  0.0151                 |
| 4,096        |  0.3415             |  0.4082              |  0.0307                 |  0.0301                 |
| 8,192        |  0.6836             |  0.8497              |  0.0619                 |  0.0602                 |
| 16,384       |  1.4119             |  1.6178              |  0.1226                 |  0.1199                 |
| 32,768       |  2.4592             |  1.8878              |  0.2434                 |  0.2494                 |
| 65,536       |  4.9380             |  2.0408              |  0.4649                 |  0.5221                 |
| 131,072      |  8.7134             |  2.1465              |  0.9800                 |  0.5468                 |
| 262,144      |  9.0743             |  2.1935              |  1.9186                 |  0.5550                 |
| 524,288      |  10.0191            |  2.2156              |  2.7941                 |  0.5624                 |
| 1,048,576    |  10.4359            |  2.2352              |  2.9053                 |  0.5683                 |
| 2,097,152    |  10.5573            |  2.2456              |  2.9554                 |  0.5704                 |
| 4,194,304    |  10.6560            |  2.2474              |  2.9887                 |  0.5711                 |

