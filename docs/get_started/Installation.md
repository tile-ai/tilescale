# Installation Guide

## Building from Source

**Prerequisites for building from source:**

- **Operating System**: Linux
- **Python Version**: >= 3.7
- **CUDA Version**: >= 12.1
- **LLVM**: < 20 if you are using the bundled TVM submodule

We currently provide three methods to install **TileScale**:

```bash
mkdir -p build
cd build
cmake .. -DUSE_CUDA=ON
make -j
```
Then add the repository root to `PYTHONPATH` before importing `tilelang`, for example:

```bash
export PYTHONPATH=/path/to/tilelang:$PYTHONPATH
python -c "import tilelang; print(tilelang.__version__)"
```

Some useful CMake options you can toggle while configuring:
- `-DUSE_CUDA=ON|OFF` builds against NVIDIA CUDA (default ON when CUDA headers are found).
- `-DUSE_ROCM=ON` selects ROCm support when building on AMD GPUs.
- `-DNO_VERSION_LABEL=ON` disables the backend/git suffix in `tilelang.__version__`.

We currently provide four methods to install **tile-lang**:

1. [Install Using Docker](#install-method-1) (Recommended)
2. [Install from Source (using the bundled TVM submodule)](#install-method-2)
3. [Install from Source (using your own TVM installation)](#install-method-3)

(install-method-1)=

### Method 1: Install Using Docker (Recommended)

For users who prefer a containerized environment with all dependencies pre-configured, **tile-lang** provides Docker images for different CUDA versions. This method is particularly useful for ensuring consistent environments across different systems and is the **recommended approach** for most users.

**Prerequisites:**
- Docker installed on your system
- NVIDIA Docker runtime or GPU is not necessary for building tilelang, you can build on a host without GPU and use that built image on other machine.

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilelang
cd tilelang
```

2. **Build Docker Image**:

Navigate to the docker directory and build the image for your desired CUDA version:

```bash
cd docker
docker build -f Dockerfile.cu120 -t tilelang-cu120 .
```

Available Dockerfiles:
- `Dockerfile.cu120` - For CUDA 12.0
- Other CUDA versions may be available in the docker directory

3. **Run Docker Container**:

Start the container with GPU access and volume mounting:

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /home/tilelang:/home/tilelang \
  --name tilelang_b200 \
  tilelang-cu120 \
  /bin/zsh
```

**Command Parameters Explanation:**
- `--shm-size 32g`: Increases shared memory size for better performance
- `--gpus all`: Enables access to all available GPUs
- `-v /home/tilelang:/home/tilelang`: Mounts host directory to container (adjust path as needed)
- `--name tilelang_b200`: Assigns a name to the container for easy management
- `/bin/zsh`: Uses zsh as the default shell

4. **Access the Container**:

```bash
docker exec -it tilelang_b200 /bin/zsh
```

5. **Verify Installation**:

Once inside the container, verify that **tile-lang** is working correctly:

```bash
python -c "import tilelang; print(tilelang.__version__)"
```

You can now run TileLang examples and develop your applications within the containerized environment. The Docker image comes with all necessary dependencies pre-installed, including CUDA toolkit, TVM, and TileLang itself.

**Example Usage:**

After accessing the container, you can run TileLang examples:

```bash
cd /home/tilelang/examples
python elementwise/test_example_elementwise.py
```

This Docker-based installation method provides a complete, isolated environment that works seamlessly on systems with compatible NVIDIA GPUs like the B200, ensuring optimal performance for TileLang applications.

(install-method-2)=

### Method 2: Install from Source (Using the Bundled TVM Submodule)

If you already have a compatible TVM installation, follow these steps:

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilescale
cd tilescale
```

**Note**: Use the `--recursive` flag to include necessary submodules.

2. **Configure Build Options**:

Create a build directory and specify your existing TVM path:

```bash
pip install . -v
```

(install-method-3)=

### Method 3: Install from Source (Using Your Own TVM Installation)

If you prefer to use the built-in TVM version, follow these instructions:

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilescale
cd tilescale
```

**Note**: Ensure the `--recursive` flag is included to fetch submodules.

2. **Configure Build Options**:

Copy the configuration file and enable the desired backends (e.g., LLVM and CUDA):

```bash
mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build
# echo "set(USE_LLVM ON)"  # set USE_LLVM to ON if using LLVM
echo "set(USE_CUDA ON)" >> config.cmake 
# or echo "set(USE_ROCM ON)" >> config.cmake to enable ROCm runtime
cmake ..
make -j 16
```

The build outputs (e.g., `libtilelang.so`, `libtvm.so`, `libtvm_runtime.so`) will be generated in the `build` directory.

3. **Set Environment Variables**:

Ensure the `tile-lang` Python package is in your `PYTHONPATH`:

```bash
export PYTHONPATH=/your/path/to/tilelang/:$PYTHONPATH
```

(install-method-3)=

### Method 3: Install Using the Provided Script

For a simplified installation, use the provided script:

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilescale
cd tilescale
```

2. **Run the Installation Script**:

```bash
bash install_cuda.sh
# or bash `install_amd.sh` if you want to enable ROCm runtime
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
