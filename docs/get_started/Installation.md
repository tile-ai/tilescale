# Installation Guide

## Building from Source

**Prerequisites for building from source:**

- **Operating System**: Linux
- **Python Version**: >= 3.7
- **CUDA Version**: >= 12.1
- **LLVM**: < 20 if you are using the bundled TVM submodule

Install **TileScale** with the following steps:

**(optional)Prepare the container**:

```bash
docker pull nvcr.io/nvidia/pytorch:25.03-py3
docker run --name tilescale --ipc=host --network=host --privileged --cap-add=SYS_ADMIN --shm-size=10g --gpus=all -it nvcr.io/nvidia/pytorch:25.03-py3 /bin/bash
echo -n > /etc/pip/constraint.txt
bash Miniconda3-latest-Linux-x86_64.sh # install conda
conda install -c conda-forge libstdcxx-ng
```

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilescale
cd tilescale
```

2. **Install Project**:

```bash
pip install cuda-python==12.9 # should align with your nvcc version
pip install scikit-build-core CMake torch ninja Cython
pip install -e . --no-build-isolation
```

3. **Verify Installation**:

Verify that **TileScale** is working correctly:

```bash
python -c "import tilelang; print(tilelang.__version__)"
```

You can now run TileScale examples and develop your applications.

**Example Usage:**

From the project root:

```bash
TILELANG_USE_DISTRIBUTED=1 python examples/distributed/example_allgather_gemm_overlapped.py
```

## To use NVSHMEM APIs

Device-side code generation (kernels calling `nvshmem_*` on the GPU) requires NVSHMEM built from source (the pip package does not provide `libnvshmem_device`). Build from source and install the Python bindings as follows.

**1. Build NVSHMEM from source**

```bash
pip install mpich   # NVSHMEM build requires MPI
cd tilelang/distributed
# Optional: set NVSHMEM_SRC to a custom path; default is ../../3rdparty/nvshmem_src
# For H100 (sm_90), add: bash build_nvshmem.sh --arch 90
bash build_nvshmem.sh
# Then set the env vars printed at the end (NVSHMEM_SRC, LD_LIBRARY_PATH).
```

The script downloads the NVSHMEM source tarball from NVIDIA; you may need to be logged in at [NVIDIA Developer](https://developer.nvidia.com) for the download to succeed.

**2. Install pynvshmem (host-side Python API)**

From the project root (ensure `NVSHMEM_SRC` is set, e.g. from step 1 in the same shell):

```bash
cd tilelang/distributed/pynvshmem
python setup.py install
export LD_LIBRARY_PATH="${NVSHMEM_SRC}/build/src/lib:$LD_LIBRARY_PATH"
```

**3. Verify**

```bash
python -c "import pynvshmem"
```
