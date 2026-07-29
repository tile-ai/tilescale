# TileScale

[GitHub](https://github.com/tile-ai/tilescale)

TileScale is a single-node multi-GPU extension of TileLang. It preserves the
TileLang compiler, JIT, and Python kernel language while adding CUDA IPC, VMM,
multicast, remote-memory, and cross-GPU synchronization primitives. The Python
distribution is `tilescale`; the compatibility import namespace remains
`tilelang`.

:::{toctree}
:maxdepth: 2
:caption: GET STARTED

get_started/Installation
get_started/overview
get_started/targets
distributed_api_reference
:::

:::{toctree}
:maxdepth: 1
:caption: RELEASE

release_v0_0726
:::

:::{toctree}
:maxdepth: 1
:caption: TUTORIALS

tutorials/debug_tools_for_tilelang
tutorials/auto_tuning
tutorials/logging
:::

:::{toctree}
:maxdepth: 1
:caption: PROGRAMMING GUIDES

programming_guides/overview
programming_guides/language_basics
programming_guides/instructions
programming_guides/control_flow
programming_guides/software_pipeline
programming_guides/python_compatibility
programming_guides/autotuning
programming_guides/type_system
:::

:::{toctree}
:maxdepth: 1
:caption: DEEP LEARNING OPERATORS

deeplearning_operators/elementwise
deeplearning_operators/gemv
deeplearning_operators/matmul
deeplearning_operators/matmul_sparse
deeplearning_operators/deepseek_mla
:::

:::{toctree}
:maxdepth: 1
:caption: COMPILER INTERNALS

compiler_internals/letstmt_inline
compiler_internals/inject_fence_proxy
compiler_internals/tensor_checks
:::

:::{toctree}
:maxdepth: 1
:caption: API Reference

autoapi/tilelang/index
:::

:::{toctree}
:maxdepth: 1
:caption: Privacy

privacy
:::
