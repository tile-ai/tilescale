# Installation Guide

## Building from Source

**Prerequisites for building from source:**

- **Operating System**: Linux
- **Python Version**: >= 3.7
- **CUDA Version**: >= 10.0
- **LLVM**: < 20 if you are using the bundled TVM submodule

We currently provide three methods to install **TileScale**:

1. [Install from Source (using your own TVM installation)](#install-method-1)
2. [Install from Source (using the bundled TVM submodule)](#install-method-2)
3. [Install Using the Provided Script](#install-method-3)

(install-method-1)=

### Method 1: Install from Source (Using Your Own TVM Installation)

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
mkdir build
cd build
cmake .. -DTVM_PREBUILD_PATH=/your/path/to/tvm/build  # e.g., /workspace/tvm/build
make -j 16
```

3. **Set Environment Variables**:

Update `PYTHONPATH` to include the `tile-lang` Python module:

```bash
export PYTHONPATH=/your/path/to/tilelang/:$PYTHONPATH
# TVM_IMPORT_PYTHON_PATH is used by 3rd-party frameworks to import TVM
export TVM_IMPORT_PYTHON_PATH=/your/path/to/tvm/python
```

(install-method-2)=

### Method 2: Install from Source (Using the Bundled TVM Submodule)

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