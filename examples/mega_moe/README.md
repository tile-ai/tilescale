# TileScale FP8/FP4 MegaMoE

This directory mirrors DeepGEMM's MegaMoE public API and preprocessing
contract:

- `get_symm_buffer_for_mega_moe`
- `transform_weights_for_mega_moe`
- `fp8_fp4_mega_moe`
- packed UE8M0 FP8 activation scales and FP4 expert weights

The SM100 path is now organized around single-launch megakernels:

- local `_make_local_dispatch_l1_l2_combine_sm100_megakernel_prim_func`
  fuses aligned top-k dispatch, L1 expert-pool staging, L1 FP8/FP4 GEMM,
  SwiGLU/top-k weighting, FP8 requantization, L2 FP8/FP4 GEMM, and local
  combine directly to `y` without a local accumulator argument. It covers
  `hidden in {128, 256}` with `intermediate_hidden=128`; the 128-wide L2
  output path uses a native narrow L2 tile while L1 keeps the 256-column
  gate/up tile. Its local dispatch map build is thread-parallel across local
  experts instead of a single-thread token scan.
- distributed
  `_make_distributed_dispatch_l1_l2_remote_combine_sm100_megakernel_prim_func`
  fuses intranode remote pull of `topk_idx/x/x_sf/topk_weights`, local expert
  L1/SwiGLU/L2, remote BF16 stores into the owner-rank symmetric per-top-k
  `combine_acts[slot]` buffer, fixed-SM release/acquire phase signaling through
  `workspace_barrier`, and 128-wide chunked top-k combine reduction through a
  dedicated shared-memory combine scratch into `y` with a top1 direct-copy fast
  path. It covers `hidden in {128, 256}` with `intermediate_hidden=128`; the
  128-wide L2 output path uses the same native narrow L2 tile as the local
  megakernel.
- both megakernels use `T.Kernel(sm_num)` resident workers and loop over
  `block_id + iter * sm_num`, so active maps larger than one SM wave remain in
  a single TileLang launch.

The TileScale symmetric allocation path uses `tilelang.tensor(...,
return_peers=True)` through the allocator for distributed buffers. The only
resident workspace ABI kept in this trimmed version is the two-row
`workspace_barrier` used by the distributed combine phase. Legacy staged grouped
L1/L2 compute, resident phase-schedule builders, standalone dispatch+stage
kernels, and `_compile_*` / `lru_cache` wrappers have been removed from this
example.

Unsupported devices, unsupported shapes, and explicit all-weight reference
calls fall back to the PyTorch reference implementation. The remaining kernel
work is to generalize the single-launch megakernels beyond
`hidden in {128, 256}, intermediate_hidden=128` and keep pushing the
distributed communication/combine schedule closer to DeepGEMM's pipelined TMA
combine path.

Run correctness smoke tests:

```bash
python -m pytest examples/mega_moe/test_tilelang_example_mega_moe.py -x
```

Run the main local SM100 tests:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m pytest examples/mega_moe/test_tilelang_example_mega_moe.py -q -s -rs \
  -k 'functional_mega_moe_megakernel_sm100_cuda or functional_mega_moe_megakernel_top1_sm100_cuda or functional_mega_moe_megakernel_multi_wave_sm100_cuda'
```

The 2-rank distributed contract test is opt-in because it initializes NCCL and
requires two usable CUDA devices:

```bash
CUDA_VISIBLE_DEVICES=0,1 TILESCALE_RUN_MEGAMOE_DIST_TEST=1 \
python -m pytest examples/mega_moe/test_tilelang_example_mega_moe.py -q -s -rs \
  -k 'functional_mega_moe_distributed_contract or functional_mega_moe_distributed_fused_hidden256_contract or functional_mega_moe_distributed_top1_contract or distributed_single_launch_dispatch_stage_hidden128_contract or distributed_single_launch_dispatch_stage_contract'
```
