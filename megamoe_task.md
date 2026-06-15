我们现在需要基于 DeepGEMM 中的 SM100 FP8/FP4 MegaMoE kernel 复刻一个
TileScale 版本。TileScale 是 TileLang 的分布式拓展，提供基于 IPC/VMM 的
intranode symmetric memory 通信能力。目标是对齐 DeepGEMM 的 host API、
preprocess 和端到端验证，并继续把通信与计算收敛到一个
DeepGEMM-style MegaMoE megakernel。

当前进度：

- 已对齐 DeepGEMM-style public API、symmetric buffer、权重 transform 合约。
- symmetric distributed buffer 通过 TileScale allocator /
  `tilelang.tensor(..., return_peers=True)` 管理。
- 已保留并验证 `hidden in {128, 256}, intermediate_hidden=128` 的 local
  single-launch megakernel：
  `_make_local_dispatch_l1_l2_combine_sm100_megakernel_prim_func` /
  `_run_local_dispatch_l1_l2_combine_sm100_megakernel`。
- local megakernel 已 fuse：aligned top-k dispatch、L1 expert-pool staging、
  L1 FP8/FP4 GEMM、SwiGLU/top-k weight、FP8 requant、L2 FP8/FP4 GEMM、
  top1 direct write 或 multi-topk BF16 pair atomic combine 到 `y`；hidden=128
  已使用原生 narrow L2 tile，L1 仍保留 256-column gate/up tile；local
  dispatch map 构建已从 `tx==0` 串行 token 扫描改为按 local expert 分片的
  128-thread 并行计数/填充。
- 已保留并验证 `hidden in {128, 256}, intermediate_hidden=128` 的 distributed
  single-launch megakernel：
  `_make_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel_prim_func` /
  `_run_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel`。
- distributed megakernel 已 fuse：intranode remote pull
  `topk_idx/x/x_sf/topk_weights`、本 rank local expert L1/SwiGLU/L2、remote
  BF16 store 到 owner symmetric per-top-k `combine_acts[slot]` buffer、fixed-SM
  release/acquire phase signal、同步后通过独立 shared-memory combine scratch 做
  128-wide chunked top-k reduce 写 `y`，top1 走 direct-copy fast path；
  hidden=128 同样使用原生 narrow L2 tile，不再 pad L2 weight/scale。
- distributed remote dispatch map build 已从单线程串行扫描改为按 local
  expert 分片的 128-thread 远端 `topk_idx` 计数/填充，顺序仍保持
  `(src_rank, token, slot)` 确定性。
- local/distributed megakernel 都使用 `T.Kernel(sm_num)` resident workers，
  在 kernel 内循环处理 `block_id + iter * sm_num` 多 wave active blocks。
- distributed workspace 现在只保留两行 8-rank `workspace_barrier` ABI，
  用于 combine phase 的 system-scope release/acquire 同步。
- 已删除 legacy staged grouped fallback：standalone dispatch+stage kernels、
  grouped L1/L2 pool kernels、resident phase-schedule builder、L1/L2 staged
  scratch、`run_local_mega_moe_sm100_backend`、distributed staged backend、
  `_compile_*`/`lru_cache` wrapper、standalone stats update TileLang kernel 和
  benchmark harness 都已下线。
- public path 现在先尝试 supported SM100 megakernel；unsupported device、
  unsupported shape、explicit all-weight reference calls 都回到 PyTorch
  reference fallback。
- 源码中没有 `T.barrier_blocks`，kernel factory 直接使用 `@tilelang.jit`。

交接用代码结构：

- 主实现：`examples/mega_moe/tilelang_mega_moe_sm100.py`。
- 测试：`examples/mega_moe/test_tilelang_example_mega_moe.py`。
- 目录说明：`examples/mega_moe/README.md`。
- public API 入口：
  `get_symm_buffer_for_mega_moe`、`_copy_inputs_to_buffer`、
  `fp8_fp4_mega_moe`。
- local fused path：
  `_run_local_dispatch_l1_l2_combine_sm100_megakernel` 调用
  `_make_local_dispatch_l1_l2_combine_sm100_megakernel_prim_func`。
- distributed fused path：
  `_run_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel` 调用
  `_make_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel_prim_func`。
- symmetric memory ABI：
  `SymmBuffer` 通过 `tilelang.distributed.allocator.get_allocator` 和
  `tilelang.distributed.tensor.tensor(..., return_peers=True)` 管理
  distributed tensors；`peer_tensor` 仍保留给 CPU tests / fallback view
  检查。distributed megakernel 的主要远端数据是
  `x`、`x_sf`、`topk_idx`、`topk_weights`、`combine_acts` 和
  `workspace_barrier`。
- `workspace_barrier` 是当前仅保留的 distributed workspace ABI：
  2 行 * 8 ranks 的 int32 barrier slots，其中第二行从
  `MEGAMOE_WORKSPACE_COMBINE_BARRIER_ROW` 开始用于 combine phase。
- `combine_acts` shape 是 `[num_topk, max_tokens, hidden]`，每个 top-k
  slot 独立写，最终 combine phase 再 reduce 到 `y`。不要再退回单个
  accumulator buffer，否则多 top-k 和远端写入会互相覆盖。

本轮清理状态：

- 实现文件目前约 1.9k 行，测试约 0.8k 行；旧 staged/grouped/benchmark
  路径已经删掉，剩余 helpers 主要服务 buffer ABI、CPU tests、fallback、
  stats 或 scale-factor layout。
- 不要重新加 `_compile_*` / `lru_cache` wrapper。用户明确要求 kernel
  factory 直接用 `@tilelang.jit`，这样 `num_tokens: T.int32` 作为 kernel
  参数可以避免每个 token 数重复编译。
- 不要动 `examples/mega_moe/reference.py`；当前工作树里它有外部修改，
  不是本 MegaMoE megakernel 清理的一部分。

已验证：

- `py_compile` 覆盖 implementation 和 tests。
- CPU public/reference/control-flow 定向：10 passed。
- local SM100 CUDA 定向：hidden=256 megakernel、hidden=128 megakernel、top1、
  多 wave、CUDA reference fallback，5 passed。
- distributed 2-rank SM100 定向：hidden=128/256 public functional、top1 public
  functional、hidden=128/256 direct single-launch contract，5 passed。

当前机器 GPU 不可用时可跑的验证：

```bash
/home/tong.wu/miniconda3/envs/tilescale/bin/python -m py_compile \
  examples/mega_moe/tilelang_mega_moe_sm100.py \
  examples/mega_moe/test_tilelang_example_mega_moe.py

/home/tong.wu/miniconda3/envs/tilescale/bin/python -m pytest \
  examples/mega_moe/test_tilelang_example_mega_moe.py -q -s -rs \
  -k 'cpu or public_local_path_prefers_megakernel_before_reference_fallback_cpu or public_distributed_path_prefers_megakernel_before_reference_fallback_cpu'

git diff --check -- \
  examples/mega_moe/tilelang_mega_moe_sm100.py \
  examples/mega_moe/test_tilelang_example_mega_moe.py \
  examples/mega_moe/README.md \
  megamoe_task.md
```

GPU 可用时建议重跑：

```bash
CUDA_VISIBLE_DEVICES=0 timeout 300 \
/home/tong.wu/miniconda3/envs/tilescale/bin/python -m pytest \
  examples/mega_moe/test_tilelang_example_mega_moe.py -q -s -rs \
  -k 'functional_mega_moe_megakernel_sm100_cuda or functional_mega_moe_megakernel_top1_sm100_cuda or functional_mega_moe_megakernel_multi_wave_sm100_cuda or functional_mega_moe_megakernel_hidden128_sm100_cuda or functional_mega_moe_sm100_reference_fallback_cuda'

CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=30013 TILESCALE_RUN_MEGAMOE_DIST_TEST=1 \
timeout 420 /home/tong.wu/miniconda3/envs/tilescale/bin/python -m pytest \
  examples/mega_moe/test_tilelang_example_mega_moe.py -q -s -rs \
  -k 'functional_mega_moe_distributed_contract or functional_mega_moe_distributed_fused_hidden256_contract or functional_mega_moe_distributed_top1_contract or distributed_single_launch_dispatch_stage_hidden128_contract or distributed_single_launch_dispatch_stage_contract'
```

DeepGEMM 对照位置：

- `/tmp/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
- `/tmp/DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`
- `/tmp/DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh`
- 重点继续看 L2 epilogue remote combine write、NVLink barrier、combine
  reduce/writeback 三段，把当前 TileLang 的 chunked `T.copy` + shared
  reduce 往 DeepGEMM resident/TMA combine pipeline 推。

下一步：

- 继续把 distributed combine tail 从当前独立 scratch 的 128-wide chunked
  shared reduce 推向更接近 DeepGEMM 的 TMA resident combine schedule。
- 做性能 profiling，按 DeepGEMM MegaMoE 的 tile/scheduler 继续调整 block
  shape、wave order 和通信重叠。
