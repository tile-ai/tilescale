# Installation Guide

## Building from Source

**Prerequisites for building from source:**

- **Operating System**: Linux
- **Python Version**: >= 3.7
- **CUDA Version**: >= 12.1
- **LLVM**: < 20 if you are using the bundled TVM submodule

We currently provide three methods to install **TileScale**:

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilescale
cd tilescale
```

2. **Install Project**:

```bash
pip install -e . --no-build-isolation
```

3. **Verify Installation**:

Once inside the container, verify that **TileScale** is working correctly:

```bash
python -c "import tilelang; print(tilelang.__version__)"
```

You can now run TileScale examples and develop your applications. 

**Example Usage:**

After accessing the container, you can run TileScale examples:

```bash
cd /home/tilelang
TILELANG_USE_DISTRIBUTED=1 python examples/distributed/example_allgather_gemm_overlapped.py
```


## To use NVSHMEM APIs

Before running the examples using NVSHMEM APIs (e.g., [example_allgather.py](../../examples/distributed/example_allgather.py)), you need to build NVSHMEM library for device-side code generation.

```bash 
pip install mpich  # building NVSHMEM needs MPI
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
