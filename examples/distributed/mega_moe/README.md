# SM90 FP8 MegaMoE

This example implements distributed FP8 MegaMoE with two persistent TileScale
kernels on NVIDIA SM90 GPUs:

```text
inputs -> dispatch + L1 GEMM + SwiGLU -> L2 GEMM + scatter + reduce -> output
```

Let:

- `M`: tokens per rank;
- `H`: hidden size;
- `I`: intermediate hidden size;
- `E`: global expert count;
- `R`: rank count;
- `K`: experts selected per token; and
- `C`: per-expert capacity.

Experts are sharded evenly, so each rank owns `E / R` experts.

## Inputs and Output

The pipeline receives the following tensors on each rank:

| Tensor | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `x` | `[M, H]` | FP8 E4M3 | Local input tokens |
| `x_sf` | `[M, H / 128]` | FP32 | Per-128 activation scales |
| `topk_idx` | `[M, K]` | INT32 | Global expert IDs |
| `topk_weights` | `[M, K]` | FP32 | Route weights |
| `l1_weight` | `[E / R, 2I, H]` | FP8 E4M3 | Local gate/up weights |
| `l1_weight_sf` | `[E / R, 2I / 128, H / 128]` | FP32 | L1 per-128 weight scales |
| `l2_weight` | `[E / R, H, I]` | FP8 E4M3 | Local down-projection weights |
| `l2_weight_sf` | `[E / R, H / 128, I / 128]` | FP32 | L2 per-128 weight scales |

The final output is:

| Tensor | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `out` | `[M, H]` | BF16 | Sum of the `K` routed expert outputs for each local token |

## Kernel Boundary

Kernel 1, `fused_l1_swiglu_manual_warp_kernel`, dispatches tokens to their
expert-owning ranks, computes the gate/up projections and SwiGLU, applies route
weights, and requantizes the intermediate activations. The outputs consumed by kernel 2 are:

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `l2_x` | `[E / R, C, I]` | FP8 E4M3 |
| `l2_x_sf` | `[E / R, C, I / 128]` | FP32 |
| `recv_counts` | `[E / R]` | INT32 |
| `src_ranks` | `[E / R, C]` | INT32 |
| `src_tokens` | `[E / R, C]` | INT32 |
| `src_topk` | `[E / R, C]` | INT32 |

Kernel 2, `fused_l2_scatter_reduce_manual_warp_kernel`, consumes these tensors
and the local L2 weights, scatters each routed result back to its source rank,
and reduces the `K` results into `out`. The `combine[M, K, H]` BF16 tensor
is an internal reduction workspace.

## Run

The distributed runtime requires peer-accessible SM90 GPUs and a configured
NVIDIA IMEX channel.

Run a four-GPU correctness smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python examples/distributed/mega_moe/example_sm90_fp8_mega_moe.py \
  --num-processes 4 \
  --model-config smoke \
  --num-tokens 32 \
  --capacity 64 \
  --check \
  --rep 0
```

Benchmark the Flash configuration on four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  python examples/distributed/mega_moe/example_sm90_fp8_mega_moe.py \
  --num-processes 4 \
  --model-config flash \
  --num-tokens 128 \
  --capacity 64 \
  --warmup 10 \
  --rep 100
```
