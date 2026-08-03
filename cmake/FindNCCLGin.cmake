# Detect an NCCL installation that provides the Device API (GIN).
#
# GIN (GPU-Initiated Networking) is what TileScale's inter-node put/signal path
# compiles against. It is *not* present in every NCCL: the device headers first
# ship in 2.28.7, and a pip `nvidia-nccl-cu12` wheel may be several minor
# versions behind that. Rather than gate on the version macro alone, this module
# requires all three things that the device path actually needs:
#
#   1. nccl.h                    -- host API, and the version macros
#   2. nccl_device/gin.h         -- the ncclGin device class
#   3. ncclDevCommCreate         -- host-side device-comm bootstrap, in libnccl
#
# A tree can satisfy (1) and report a new enough version while missing (2)/(3),
# which is why the symbol check is not skipped when the header is found.
#
# Sets: NCCLGin_FOUND, NCCL_INCLUDE_DIR, NCCL_LIBRARY, NCCL_VERSION_STRING
#
# Hint with -DNCCL_ROOT=<prefix>, or let it fall back to the active Python
# environment's nvidia/nccl wheel and then the CUDA toolkit prefix.

set(_nccl_gin_min_version "2.28.7")

# Candidate prefixes, most specific first.
set(_nccl_hints "")
if(NCCL_ROOT)
  list(APPEND _nccl_hints "${NCCL_ROOT}")
endif()
if(DEFINED ENV{NCCL_ROOT})
  list(APPEND _nccl_hints "$ENV{NCCL_ROOT}")
endif()

# pip-installed NCCL lives under <site-packages>/nvidia/nccl. Ask the
# interpreter rather than globbing, so we track the env actually in use.
if(Python3_EXECUTABLE OR Python_EXECUTABLE)
  if(Python3_EXECUTABLE)
    set(_nccl_py "${Python3_EXECUTABLE}")
  else()
    set(_nccl_py "${Python_EXECUTABLE}")
  endif()
  execute_process(
    COMMAND "${_nccl_py}" -c
      "import os,sysconfig;p=os.path.join(sysconfig.get_paths()['purelib'],'nvidia','nccl');print(p if os.path.isdir(p) else '')"
    OUTPUT_VARIABLE _nccl_pip_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(_nccl_pip_dir)
    list(APPEND _nccl_hints "${_nccl_pip_dir}")
  endif()
endif()

if(CUDAToolkit_LIBRARY_ROOT)
  list(APPEND _nccl_hints "${CUDAToolkit_LIBRARY_ROOT}")
endif()

find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS ${_nccl_hints}
  PATH_SUFFIXES include
  DOC "Directory containing nccl.h")

# pip wheels ship only the versioned soname -- there is no libnccl.so
# development symlink -- so the bare "nccl" name that find_library derives is not
# enough. List the versioned file explicitly, and keep the unversioned name first
# so a real system/dev install still wins.
find_library(NCCL_LIBRARY
  NAMES nccl libnccl.so.2 libnccl.so.2.dylib
  HINTS ${_nccl_hints}
  PATH_SUFFIXES lib lib64
  DOC "NCCL shared library")

set(NCCL_VERSION_STRING "")
set(_nccl_has_gin_header FALSE)
set(_nccl_has_devcomm FALSE)

if(NCCL_INCLUDE_DIR)
  # Version macros. NCCL_VERSION_CODE is not usable here because it is a macro
  # expression, so read the three components directly.
  foreach(_part MAJOR MINOR PATCH)
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _line
      REGEX "^#define NCCL_${_part} +[0-9]+")
    if(_line)
      string(REGEX MATCH "[0-9]+" _nccl_${_part} "${_line}")
    else()
      set(_nccl_${_part} 0)
    endif()
  endforeach()
  set(NCCL_VERSION_STRING "${_nccl_MAJOR}.${_nccl_MINOR}.${_nccl_PATCH}")

  if(EXISTS "${NCCL_INCLUDE_DIR}/nccl_device/gin.h")
    set(_nccl_has_gin_header TRUE)
  endif()
endif()

# ncclDevCommCreate is the host entry point the GIN path needs; a tree can carry
# the header and still be linked against a runtime that does not export it.
if(NCCL_LIBRARY AND _nccl_has_gin_header)
  if(UNIX AND NOT APPLE)
    find_program(_nccl_nm NAMES nm)
    if(_nccl_nm)
      execute_process(
        COMMAND "${_nccl_nm}" -D --defined-only "${NCCL_LIBRARY}"
        OUTPUT_VARIABLE _nccl_syms ERROR_QUIET)
      if(_nccl_syms MATCHES "ncclDevCommCreate")
        set(_nccl_has_devcomm TRUE)
      endif()
    else()
      # No nm available: trust the header plus version gate instead of
      # silently disabling GIN on a stripped-down build image.
      set(_nccl_has_devcomm TRUE)
    endif()
  else()
    set(_nccl_has_devcomm TRUE)
  endif()
endif()

set(NCCLGin_FOUND FALSE)
if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY AND _nccl_has_gin_header
   AND _nccl_has_devcomm
   AND NOT NCCL_VERSION_STRING VERSION_LESS _nccl_gin_min_version)
  set(NCCLGin_FOUND TRUE)
endif()

if(NCCLGin_FOUND)
  message(STATUS "NCCL GIN: enabled (NCCL ${NCCL_VERSION_STRING} at ${NCCL_INCLUDE_DIR})")
elseif(NCCL_INCLUDE_DIR)
  # Found NCCL but cannot use the device path. Say which check failed --
  # "GIN disabled" with no reason is the hard case to debug on a cluster.
  # Order matters: the symbol check is skipped when the library is missing, so
  # test for the library before blaming its exports.
  if(NOT NCCL_LIBRARY)
    set(_why "found headers at ${NCCL_INCLUDE_DIR} but no libnccl alongside them")
  elseif(NOT _nccl_has_gin_header)
    set(_why "no nccl_device/gin.h (needs NCCL >= ${_nccl_gin_min_version})")
  elseif(NOT _nccl_has_devcomm)
    set(_why "libnccl does not export ncclDevCommCreate")
  elseif(NCCL_VERSION_STRING VERSION_LESS _nccl_gin_min_version)
    set(_why "NCCL ${NCCL_VERSION_STRING} < ${_nccl_gin_min_version}")
  else()
    set(_why "incomplete installation")
  endif()
  message(STATUS "NCCL GIN: disabled -- ${_why}. "
                 "Inter-node kernels will fall back to intra-node paths.")
else()
  message(STATUS "NCCL GIN: disabled -- NCCL not found. "
                 "Set -DNCCL_ROOT=<prefix> to enable inter-node support.")
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
