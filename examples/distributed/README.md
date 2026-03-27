# Distributed Examples

This directory contains examples demonstrating distributed computing capabilities using TileLang.
These examples are sorted into two categories:
- Examples under `nvshmem` folder and inter-node examples depend on NVSHMEM library for distributed communication.
- Other examples have no external dependency and only rely on TileScale IPC

## `nvshmem` examples

Before running the examples, you need to build NVSHMEM library for device-side code generation.

```bash
export NVSHMEM_SRC="your_custom_nvshmem_dir" # default to 3rdparty/nvshmem_src
cd tilelang/distributed
source build_nvshmem.sh
```
You also need to install the `pynvshmem` package, which provides wrapped host-side Python API for NVSHMEM.

```bash
cd ./pynvshmem
python setup.py install
export LD_LIBRARY_PATH="$NVSHMEM_SRC/build/src/lib:$LD_LIBRARY_PATH"
```

Then you can test python import:
```bash
python -c "import pynvshmem"
```

Finally, run examples like this:
```bash
TILELANG_USE_DISTRIBUTED=1 bash ./tilelang/distributed/launch.sh examples/distributed/nvshmem/example_allgather.py
```

## IPC-based examples

Simply run via python:
```bash
TILELANG_USE_DISTRIBUTED=1 python examples/distributed/intranode/example_allgather_gemm_overlapped.py
```

> Tips: To disable annoying NCCL IB logs, consider running with: `NCCL_IB_DISABLE=1`