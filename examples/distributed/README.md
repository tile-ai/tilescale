# Distributed Examples

This directory contains examples demonstrating distributed computing capabilities using TileLang.

For example, 
```
./tilelang/distributed/launch.sh examples/distributed/example_allgather.py
```

## Prerequisites

Before running the examples, you need to install the `pynvshmem` package, which provides wrapped Python API for NVSHMEM.
```bash 
cd tilelang/distributed/pynvshmem
pip install -e . -v # build in editable mode
```
