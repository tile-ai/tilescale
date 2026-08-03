# Sequence Parallel Examples

This directory contains intranode sequence-parallel communication and attention examples:

- `example_pre_attn_all2all.py`: `[B, NH, S/P, D] -> [B, NH/P, S, D]`
- `example_pre_attn_all2all_transpose.py`: `[B, S/P, NH, D] -> [B, NH/P, S, D]`
- `example_post_attn_all2all_transpose.py`: `[B, NH/P, S, D] -> [B, S/P, NH, D]`
- `example_sp_ag_attention_intra_node.py`: all-gather-based sequence-parallel attention

Run an example on four local GPUs:

```bash
python examples/distributed/experimental/sequence_parallel/example_pre_attn_all2all_transpose.py \
  --num-processes 4 --batch-size 2 --num-heads 32 --seq-len 8192 --head-dim 128
```

The examples require peer-accessible intranode GPUs and the TileScale distributed runtime.
