# TileScale Codebase 整理与重构建议

## 1. 解决 repo 名称与 package 名称不一致问题

**现状：** repo 叫 `tilescale`，Python 包叫 `tilelang`，二者完全不同。

**建议：**
- 明确一个正式名称。如果 `tilelang` 是对外发布的包名，`tilescale` 是项目/团队代号，则在 repo 根目录的 README、CLAUDE.md 等地方明确说明这一关系。
- 或者，将包名统一为 `tilescale`（如果不考虑向后兼容），避免新加入的开发者混淆。

---

## 2. 消除 `primitives/` 与 `tileop/` 的职责重叠

**现状：**
- `tilelang/primitives/gemm/__init__.py` 是一个几乎为空的目录（只有 `__init__.py`），且其中 import 的模块（`primitives.gemm.base`, `primitives.gemm.gemm_mma`）并不存在于该目录下，实际实现在 `tileop/gemm/` 里。
- `tilelang/tileop/gemm/` 拥有完整实现：`gemm_mma.py`, `gemm_wgmma.py`, `gemm_mfma.py`, `gemm_tcgen05.py` 等。

**建议：**
- 删除空壳 `primitives/` 目录，将其职责完全归入 `tileop/`（或反之统一命名）。
- 在公共 API 层统一暴露，不要让两个名字同时对外可见。

---

## 3. 清理 `language/v2/` —— 明确新旧版本策略

**现状：** 存在 `tilelang/language/` 和 `tilelang/language/v2/` 两套并行实现。

**建议：**
- 如果 v2 是稳定的新实现，应制定迁移计划：deprecate v1 接口，并在 v1 所有入口添加 deprecation warning，最终删除 v1 代码。
- 如果 v2 还在实验阶段，应移入 `language/experimental/` 而非独立的 `v2/` 目录，避免用户误用。
- 不要让两个版本长期共存且没有明确说明。

---

## 4. 合并分散的 `kernel_cache.py`

**现状：** 至少 5 处有各自的 `kernel_cache.py`：
```
tilelang/cache/kernel_cache.py
tilelang/jit/adapter/kernel_cache.py
tilelang/jit/adapter/cutedsl/kernel_cache.py
tilelang/jit/adapter/torch/kernel_cache.py
tilelang/jit/adapter/nvrtc/kernel_cache.py
tilelang/jit/adapter/cython/kernel_cache.py
```

**建议：**
- 设计一个统一的 `tilelang/cache/` 模块作为单一 cache 抽象层。
- 各 adapter（cutedsl/torch/nvrtc/cython）继承或组合此基础实现，而不是各自维护一份。
- 这样可以避免 cache invalidation 逻辑不同步的 bug。

---

## 5. 统一 Distributed 代码的位置

**现状：** 分布式相关代码散落在：
- `tilelang/distributed/` — 顶层 distributed 模块（含 pynvshmem 子包、build 脚本、install 脚本）
- `tilelang/language/distributed/` — language 层面的 distributed 原语
- `tilelang/language/distributed/multi_device/` — nvshmem.py, cpengine.py
- `examples/distributed/` — 示例

**建议：**
- 明确职责边界：language 层只负责 IR/语法原语的声明，runtime 层（nvshmem binding、launch、allocator）归入 `tilelang/distributed/`。
- `tilelang/language/distributed/` 只应包含 IR-level 原语定义，具体实现（nvshmem.py, cpengine.py）应上移至 `tilelang/distributed/`。
- `pynvshmem` 是一个独立的 C 扩展包（有自己的 `setup.py` 和 `CMakeLists.txt`），应提升为顶层子包或独立 repo，而不是嵌套在 `tilelang/distributed/pynvshmem/` 多层深处。

---

## 6. 整理测试代码——测试不应混在 package 内部

**现状：** 测试文件散落在：
- `testing/python/` — 主测试目录（正确）
- `tilelang/distributed/testing/` — 在包内部！
- `tilelang/distributed/pynvshmem/testing/` — 也在包内部！
- `examples/` — 大量 `test_*.py` 和 `regression_*.py` 混在示例里

**建议：**
- 确立唯一的测试根目录：`testing/`（或 `tests/`）。
- 将 `tilelang/distributed/testing/` 和 `tilelang/distributed/pynvshmem/testing/` 的内容迁移到 `testing/python/distributed/`。
- `examples/` 里的 `test_*.py` / `regression_*.py` 要么移入 `testing/`，要么与示例文件分开存放（比如每个 example 子目录只保留 `example_*.py`，测试单独放 `testing/`）。
- 在 `testing/` 下的目录结构应与 `tilelang/` 包结构一一对应，便于查找。

---

## 7. 修复 `__init__.py` 中的重复版本解析逻辑

**现状：** `tilelang/__init__.py` 第 11-65 行有两套 `__version__` 计算逻辑，`__version__` 被赋值两次（第 48 行和第 53-65 行），第一次的计算完全被覆盖。

**建议：**
- 只保留一套版本解析逻辑（推荐 `importlib.metadata` + fallback），删除重复代码。
- 或者将版本逻辑抽到单独的 `_version.py` 文件。

---

## 8. 规范 `language/` 内部命名不一致问题

**现状：** 同类功能有时带 `_op` 后缀，有时不带：
- `language/copy.py` vs `language/copy_op.py`
- `language/fill.py` vs `language/fill_op.py`
- `language/gemm.py` vs `language/gemm_op.py`
- `language/reduce.py` vs `language/reduce_op.py`

**建议：**
- 确定一个命名规范（建议 `_op` 后缀表示操作符对象，无后缀表示 language-level 函数入口），并全面统一。
- 或者合并重复文件，明确每个文件的职责边界。

---

## 9. 整合 `contrib/cutedsl/` 与 `jit/adapter/cutedsl/`

**现状：**
- `tilelang/contrib/cutedsl/` 有底层 CUDA 原语（mbar, ldsm, cpasync, gemm_V1, reduce 等）
- `tilelang/jit/adapter/cutedsl/` 有 adapter/wrapper/libgen 等 JIT 逻辑

**建议：**
- `contrib/cutedsl/` 的角色应该是"第三方或底层 CUDA primitive 封装"，`jit/adapter/cutedsl/` 是"JIT pipeline 对 cutedsl 的适配"，二者职责不同，但命名上容易让人以为是同一件事。
- 建议将 `contrib/cutedsl/` 重命名为 `tilelang/backends/cutedsl/` 或 `tilelang/codegen/cutedsl/`，以更清晰表达其用途。

---

## 10. 整理 `examples/` 目录结构

**现状：** `examples/` 混杂了多种文件类型，缺乏一致结构：
- `example_*.py` — 示例代码
- `test_example_*.py` — 测试
- `regression_*.py` — 回归测试
- `benchmark_*.py` — 性能基准
- 部分有 `README.md`，部分没有
- `examples/pytest.ini` 只有一个，但 examples 子目录结构不统一

**建议：**
- 每个 example 子目录统一结构：`example_*.py`（核心示例）+ `README.md`（必须有）。
- 测试/回归测试移入 `testing/`，不要放在 examples 里。
- Benchmark 单独建 `benchmarks/` 顶级目录（现有 `benchmark/blocksparse_attention/` 也移入此处），与 `examples/` 分开。

---

## 11. 清理重复图片资源

**现状：** 相似图片存在于多个地方：
- `images/`（根目录）
- `docs/_static/img/`
- `examples/deepseek_mla/figures/`
- `docs/_static/img/mla_hopper/`

**建议：**
- 文档用图统一放 `docs/_static/img/`，examples 里的 figures 只保留真正 example 专属的。
- 根目录 `images/` 合并进 `docs/_static/img/`，避免两套图片各自维护。

---

## 12. `tilelang/utils/ts_ext/` 的嵌套 `setup.py` 问题

**现状：** `tilelang/utils/ts_ext/` 内部有独立的 `setup.py`，说明这是一个独立的 C 扩展包，但被嵌套在 `utils/` 深处。

**建议：**
- 将 `ts_ext` 提升为顶级子包（类似 `pynvshmem` 的处理方式，或者更好地统一到同一个构建系统中）。
- 或者把它的构建整合进主 `setup.py` / `CMakeLists.txt`，避免嵌套独立构建脚本。

---

## 优先级总结

| 优先级 | 建议 |
|--------|------|
| 🔴 高 | #7 重复版本逻辑（bug 风险）、#4 kernel_cache 重复（一致性风险）、#6 测试混在包内 |
| 🟡 中 | #2 primitives/tileop 重叠、#5 distributed 代码分散、#8 命名不一致、#3 v2 策略 |
| 🟢 低 | #1 名称统一、#9 contrib vs adapter、#10 examples 结构、#11 图片整理、#12 ts_ext 位置 |

---

# TileScale 专项整理建议

## 13. 统一两个职责重叠的 `init_dist*` 函数

**现状：** `tilelang/distributed/utils.py` 中存在两个功能相似但 API 完全不同的初始化函数：
- `init_dist(local_rank, num_local_ranks)` — 旧式，需要手动传参，配合 `mp.spawn` 使用
- `init_distributed(return_tp_group, init_nvshmem, return_lc_group)` — 新式，从环境变量读取，配合 `torchrun` 使用

两者在不同 example 中被混用，导致新用户不知道该用哪个。

**建议：**
- 统一为一个函数，明确以 `torchrun` 风格（环境变量）为主；
- `init_dist` 如果仍需保留，加 `@deprecated` 装饰器，并在 docstring 说明迁移路径。

---

## 14. 修复 `wait_eq` 名称严重冲突

**现状：** 两处都叫 `wait_eq`，但语义完全不同：
- `tilelang/language/distributed/common.py:wait_eq()` — **IR 层**：生成 NVSHMEM 的 `signal_wait_until` intrinsic，在 GPU 核函数内使用
- `tilelang/distributed/utils.py:wait_eq()` — **Host 层**：调用 `cuStreamWaitValue32`，在 CPU 侧 stream 上等待

两者都被 import 进同一个项目，极易混淆，且 bug 难以定位。

**建议：**
- Host 侧重命名为 `stream_wait_eq` 或 `host_wait_signal`，与 IR 层的 `wait_eq` 区分；
- 在两个函数的 docstring 中明确标注各自的作用层级（host/device）。

---

## 15. 消除 IPC handle 同步逻辑的重复实现

**现状：** 以下两处独立实现了几乎相同的"收集各 rank 的 IPC handle 并映射到本地 GPU 指针"逻辑：
- `distributed/utils.py:create_dist_tensor()` — 通过 `all_gather_object` 收集 IPC handle，调用 `_sync_ipc_handles`
- `distributed/allocator.py:BaseAllocator._init_table()` — 相同流程，只是封装在 Allocator 里

**建议：**
- 将 IPC handle 同步逻辑抽成一个内部函数 `_exchange_ipc_handles(group, local_ptr) -> buffer_ptrs`；
- `create_dist_tensor` 和 `BaseAllocator._init_table` 都调用这个函数，避免两套独立维护。

---

## 16. 集中管理 `cuda-python` 版本兼容 shim

**现状：** 以下代码段在多处重复出现：
```python
cuda_python_version = importlib.metadata.version("cuda-python")
from packaging import version
if version.parse(cuda_python_version) >= version.parse("12.8.0"):
    from cuda.bindings import driver as cuda
else:
    from cuda import cuda
```
出现于 `distributed/utils.py`、`example_allgather_gemm_overlapped.py` 等多处。

**建议：**
- 将此兼容逻辑放在 `tilelang/distributed/_cuda_compat.py` 中，统一 export `cuda, cudart` 对象；
- 其他地方一律 `from tilelang.distributed._cuda_compat import cuda, cudart`，不再各自做版本检测。

---

## 17. 建立统一的对称内存（Symmetric Memory）抽象

**现状：** TileScale 目前有两套完全独立的分布式 tensor 分配路径：
- **NVSHMEM 路径**：`pynvshmem.nvshmem_create_tensor(shape, dtype)` — 通过 NVSHMEM 分配对称内存
- **IPC 路径**：`tilelang.distributed.create_tensor(shape, dtype)` 或 `BaseAllocator` — 通过 `cudaMalloc` + IPC handle 交换

两者语义不同、使用方式不同，但 examples 中混用，用户难以理解应该用哪个，以及何时该回退到 IPC 路径。

**建议：**
- 设计一个统一的 `tilescale.memory.SymmetricBuffer` 抽象，内部根据是否有 nvshmem 自动选择后端；
- 在文档中明确说明两种路径的适用场景（intranode IPC vs internode NVSHMEM）。

---

## 18. 将 shell 脚本从 package 目录移出

**现状：** 以下三个 shell 脚本直接放在 `tilelang/distributed/` 包目录里，会被打包进 Python wheel：
- `tilelang/distributed/launch.sh` — 分布式启动脚本
- `tilelang/distributed/build_nvshmem.sh` — 构建 NVSHMEM 的脚本
- `tilelang/distributed/install_deepep.sh` — 安装 DeepEP 的脚本

**建议：**
- 统一移到 repo 根目录下的 `scripts/` 目录（如 `scripts/launch.sh`, `scripts/build_nvshmem.sh`）；
- `scripts/` 不会被 pip install 打包，符合标准 Python 项目规范。

---

## 19. 拆分过于庞杂的 `distributed/utils.py`

**现状：** `tilelang/distributed/utils.py` 是一个 400 行的"万能工具箱"，混杂了：
- 进程组初始化（`init_dist`, `init_distributed`）
- IPC tensor 创建（`create_tensor`, `create_dist_tensor`）
- CUDA stream 信号操作（`set_signal`, `wait_eq`）
- 性能测量（`perf_fn`）
- 调试打印（`dist_print`）
- NVLink 拓扑检测（`has_fullmesh_nvlink`, `NvidiaSmiUtil`）
- 数据生成（`generate_data`, `_make_tensor`）

**建议：**
```
tilelang/distributed/
├── init.py      # init_dist, init_distributed
├── memory.py    # IPC/tensor allocation
├── signal.py    # CUDA stream signal operations
├── topo.py      # NVLink topology (NvidiaSmiUtil, has_fullmesh_nvlink)
├── perf.py      # perf_fn, dist_print
└── data.py      # generate_data, _make_tensor
```

---

## 20. 将 example 中的复用库代码提升为正式模块

**现状：** 以下文件放在 `examples/distributed/` 目录中，却被其他 example 直接 import，本质上已经是库代码：
- `examples/distributed/sp_ag_attention_intra_node.py` — 被 `example_sp_ag_attention_intra_node.py` import
- `examples/distributed/nvshmem/gemm_rs_utils.py` — `BarrierAllContext`, `ReduceScatter2DContext`
- `examples/distributed/nvshmem/reduce_scatter.py` — `reduce_scatter_2d_op` 等
- `examples/distributed/deepseek_deepep/deepep_utils.py` — `Config`, DeepEP 相关常量

这些文件用相对路径 import，只在同目录运行时有效，无法被其他地方正确引用。

**建议：**
- 将复用代码提升为 `tilelang/distributed/` 下的正式子模块；
- examples 只保留调用代码，不再充当隐式的库。

---

## 21. 清理 `launch.sh` 中泄露的内部集群环境变量

**现状：** `tilelang/distributed/launch.sh` 中有：
```bash
master_addr=${ARNOLD_WORKER_0_HOST:="127.0.0.1"}
master_port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
```
`ARNOLD_*` 是 ByteDance 内部集群（Arnold 训练平台）的环境变量，不应出现在公开代码中。

**建议：**
- 将 `ARNOLD_WORKER_0_HOST` 替换为标准的 `MASTER_ADDR`，`ARNOLD_WORKER_0_PORT` 替换为 `MASTER_PORT`；
- 清理所有内部平台相关的硬编码假设（如 `IB_HCA=mlx5`、`BYTED_TORCH_BYTECCL` 等）。

---

## 22. 为 `tilescale_ext` C 扩展建立清晰的 API 文档与稳定性边界

**现状：** `tilelang/distributed/__init__.py` 直接暴露：
```python
from tilescale_ext import _create_tensor, _create_ipc_handle, _sync_ipc_handles
```
- 以 `_` 开头的私有函数被直接 re-export 为公共 API
- 没有任何 stub 文件（`.pyi`）说明函数签名
- 不清楚哪些是稳定 API，哪些是内部实现细节

**建议：**
- 在 `tilelang/distributed/` 层面做好封装，禁止外部直接调用 `_` 前缀的 `tilescale_ext` 函数；
- 提供 `tilescale_ext.pyi` stub 文件，便于 IDE 补全和类型检查；
- 明确区分稳定公共 API 与内部实现。

---

## 优先级汇总（TileScale 专项）

| 优先级 | 建议 |
|--------|------|
| 🔴 高 | #14 wait_eq 名称冲突（正确性风险）、#15 IPC 逻辑重复（一致性风险）、#21 内部环境变量泄露 |
| 🟡 中 | #13 init_dist 统一、#16 cuda-python shim 集中化、#19 utils.py 拆分、#20 example 库代码提升 |
| 🟢 低 | #17 对称内存抽象、#18 shell 脚本移位、#22 tilescale_ext API 文档 |
