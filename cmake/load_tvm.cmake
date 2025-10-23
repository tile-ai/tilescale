# todo: support prebuilt tvm

set(TVM_BUILD_FROM_SOURCE TRUE)
set(TVM_SOURCE ${CMAKE_SOURCE_DIR}/3rdparty/tvm)

if(DEFINED $ENV{TVM_ROOT})
  if(EXISTS $ENV{TVM_ROOT}/cmake/config.cmake)
    set(TVM_SOURCE $ENV{TVM_ROOT})
  endif()
endif()

set(TVM_INCLUDES
  ${TVM_SOURCE}/include
  ${TVM_SOURCE}/ffi/include
  ${TVM_SOURCE}/src
  ${TVM_SOURCE}/3rdparty/dlpack/include
  ${TVM_SOURCE}/3rdparty/dmlc-core/include
)
