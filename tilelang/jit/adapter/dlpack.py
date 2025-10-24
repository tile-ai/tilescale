"""The profiler and convert to torch utils"""
from __future__ import annotations

import torch
from tilelang.contrib.dlpack import to_pytorch_func
from .base import BaseKernelAdapter


class TorchDLPackKernelAdapter(BaseKernelAdapter):

    def _convert_torch_func(self) -> callable:
        torch_func = to_pytorch_func(self.mod)

        def func(*ins: list[torch.Tensor]):
            if len(ins) + len(self.result_idx) != len(self.params):
                raise ValueError(
                    f"Expected {len(self.params)} inputs, got {len(ins) + len(self.result_idx)} with {len(ins)} inputs and {len(self.result_idx)} outputs"
                )
            ins_idx = 0
            args = []

            # use the device of the first input tensor if available
            device = ins[0].device if len(ins) > 0 else torch.cuda.current_device()

            for i in range(len(self.params)):
                if i in self.result_idx:
                    dtype = self.params[i].dtype
                    shape = list(map(int, self.params[i].shape))
                    tensor = torch.empty(*shape, dtype=dtype, device=device)
                else:
                    tensor = ins[ins_idx]
                    ins_idx += 1
                args.append(tensor)

            torch_func(*args)

            if len(self.result_idx) == 1:
                return args[self.result_idx[0]]
            else:
                return [args[i] for i in self.result_idx]

        return func
