# TileScale containers

Build every image from the repository root. The Dockerfiles copy and install
the current TileScale checkout, including its pinned submodules.

For NVIDIA GPUs, select a Dockerfile matching the CUDA toolchain you need. B200
requires CUDA 12.8 or newer:

```bash
git clone --recursive https://github.com/tile-ai/tilescale.git
cd tilescale
docker build -f docker/Dockerfile.cu128 -t tilescale-cu128 .
docker run --rm -it --gpus all --network=host --shm-size=32g \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  tilescale-cu128 bash
```

For AMD GPUs:

```bash
docker build -f docker/Dockerfile.rocm -t tilescale-rocm .
docker run --rm -it --network=host --device=/dev/kfd --device=/dev/dri \
  --group-add video --shm-size=32g --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined tilescale-rocm
```

The CUDA images install the `distributed` extra. VMM/fabric and multicast also
require a compatible NVSwitch system, Fabric Manager, and a configured NVIDIA
IMEX channel on the host; CUDA IPC remains the fallback when those capabilities
are unavailable.
