#!/bin/bash
# Build NVSHMEM from source for TileLang device-side use. Run from repo root or set NVSHMEM_SRC.
# Usage: source build_nvshmem.sh [--arch 90] [--jobs N] [--force-download]
# Override at runtime: NVSHMEM_SRC, NVSHMEM_* (see below), CMAKE, NVSHMEM_VERSION.

VER="${NVSHMEM_VERSION:-3.2.5-1}"
TARBALL="nvshmem_src_${VER}.txz"
URL="${NVSHMEM_SOURCE_URL:-https://developer.nvidia.com/downloads/assets/secure/nvshmem/${TARBALL}}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FORCE_DL=""
ARCH=""
JOBS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)  echo "Usage: bash build_nvshmem.sh [--arch ARCH] [--jobs N] [--force-download]"; exit 0 ;;
    --arch)     ARCH="$2"; shift 2 ;;
    --jobs)     JOBS="$2"; shift 2 ;;
    --force-download) FORCE_DL=1; shift ;;
    *)          shift ;;
  esac
done

export NVSHMEM_SRC="$(realpath "${NVSHMEM_SRC:-$SCRIPT_DIR/../../3rdparty/nvshmem_src}")"
export NVSHMEM_PATH="${NVSHMEM_SRC}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
ARCH="${ARCH:-$CMAKE_CUDA_ARCHITECTURES}"

if [[ -n "$FORCE_DL" ]] || [[ ! -f "${NVSHMEM_SRC}/CMakeLists.txt" ]]; then
  if [[ -f "${NVSHMEM_SRC}/CMakeLists.txt" ]]; then
    rm -rf "${NVSHMEM_SRC:?}/"*
    rm -rf "${NVSHMEM_SRC}/".* 2>/dev/null || true
  else
    mkdir -p "${NVSHMEM_SRC}"
  fi
  cd "${SCRIPT_DIR}"
  [[ -f "${TARBALL}" ]] || { wget -q --show-progress "${URL}" -O "${TARBALL}" || { echo "Download failed (login at developer.nvidia.com?)." >&2; exit 1; }; }
  tar -zxf "${TARBALL}"
  rm -f "${TARBALL}"
  [[ -d nvshmem_src ]] || { echo "Missing nvshmem_src after extract." >&2; exit 1; }
  mv nvshmem_src/* "${NVSHMEM_SRC}/"
  mv nvshmem_src/.* "${NVSHMEM_SRC}/" 2>/dev/null || true
  rmdir nvshmem_src
fi

export NVSHMEM_IBGDA_SUPPORT="${NVSHMEM_IBGDA_SUPPORT:-0}"
export NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY="${NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY:-0}"
export NVSHMEM_IBDEVX_SUPPORT="${NVSHMEM_IBDEVX_SUPPORT:-0}"
export NVSHMEM_IBRC_SUPPORT="${NVSHMEM_IBRC_SUPPORT:-1}"
export NVSHMEM_LIBFABRIC_SUPPORT="${NVSHMEM_LIBFABRIC_SUPPORT:-0}"
export NVSHMEM_MPI_SUPPORT="${NVSHMEM_MPI_SUPPORT:-1}"
export NVSHMEM_USE_GDRCOPY="${NVSHMEM_USE_GDRCOPY:-0}"
export NVSHMEM_TORCH_SUPPORT="${NVSHMEM_TORCH_SUPPORT:-1}"
export NVSHMEM_ENABLE_ALL_DEVICE_INLINING="${NVSHMEM_ENABLE_ALL_DEVICE_INLINING:-1}"
[[ -z "${ARCH}" ]] || export CMAKE_CUDA_ARCHITECTURES="${ARCH}"

cd "${NVSHMEM_SRC}"
mkdir -p build && cd build
CMAKE="${CMAKE:-cmake}"
[[ -f CMakeCache.txt ]] || ${CMAKE} .. \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  ${ARCH:+-DCMAKE_CUDA_ARCHITECTURES=${ARCH}} \
  -DNVSHMEM_BUILD_TESTS=OFF \
  -DNVSHMEM_BUILD_EXAMPLES=OFF \
  -DNVSHMEM_BUILD_PACKAGES=OFF

make -j"${JOBS}" VERBOSE=1

echo ""
echo "NVSHMEM built successfully at: ${NVSHMEM_SRC}"
echo ""
echo "To use NVSHMEM, add to your environment (e.g. in ~/.bashrc or before running examples):"
echo "  export NVSHMEM_SRC=\"${NVSHMEM_SRC}\""
echo "  export LD_LIBRARY_PATH=\"${NVSHMEM_SRC}/build/src/lib:\$LD_LIBRARY_PATH\""
echo ""
