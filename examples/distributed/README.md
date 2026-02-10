# Distributed Examples

This directory contains examples demonstrating distributed computing capabilities using TileLang.

For example,
```
./tilelang/distributed/launch.sh examples/distributed/example_allgather.py
```

## Prerequisites

Before running the examples, you need NVSHMEM (either from source or the pip package) and the `pynvshmem` Python bindings.

**Build NVSHMEM from source (from repo root):**

```bash
pip install mpich
cd tilelang/distributed
bash build_nvshmem.sh   # optional: --arch 90 for H100 (sm_90). Then set NVSHMEM_SRC and LD_LIBRARY_PATH as printed.
```

**Or install the prebuilt NVSHMEM package:**

```bash
pip install nvidia-nvshmem-cu12
```

**Install pynvshmem and set library path:**

```bash
cd tilelang/distributed/pynvshmem
python setup.py install
# If you built NVSHMEM from source:
export LD_LIBRARY_PATH="$NVSHMEM_SRC/build/src/lib:$LD_LIBRARY_PATH"
```

Then verify:

```bash
python -c "import pynvshmem"
```
