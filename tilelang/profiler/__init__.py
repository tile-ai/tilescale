"""The profiler and convert to torch utils"""

from typing import List, Optional, Callable, Any
from functools import partial
import torch
from contextlib import suppress
from dataclasses import dataclass
import tvm
from tilelang.utils.tensor import (
    get_tensor_supply,
    TensorSupplyType,
    torch_assert_close,
    adapt_torch2tvm,
)
from tilelang.engine.param import KernelParam
from tilelang.jit.adapter import BaseKernelAdapter
from tilelang.profiler.bench import do_bench
from tilelang import env

import logging

logger = logging.getLogger(__name__)

if env.USE_DISTRIBUTED:
    import pynvshmem
    logger.info("Using distributed profiler")
else:
    logger.info("Not using distributed profiler")


@dataclass
class Profiler:
    """A profiler class for benchmarking and validating kernel implementations.
    
    Attributes:
        params: List of kernel parameters defining the input/output specifications
        result_idx: Indices indicating which parameters are output tensors
        supply_type: Type of tensor supply to use (e.g., random, zeros, etc.)
        adapter: Optional kernel adapter for interfacing with different backends
    """

    params: List[KernelParam]
    result_idx: List[int]
    supply_type: TensorSupplyType
    adapter: Optional[BaseKernelAdapter] = None

    def __post_init__(self):
        """Initialize tensor supply after dataclass initialization"""
        self.result_idx = self._legalize_result_idx(self.result_idx)
        self.supply = get_tensor_supply(self.supply_type)

    def _legalize_result_idx(self, result_idx: Optional[List[int]] = None) -> List[int]:
        params = self.params
        # result_idx is a list of indices of the output tensors
        if result_idx is None:
            result_idx = []
        elif isinstance(result_idx, int):
            if result_idx > len(params) or result_idx < -len(params):
                raise ValueError(
                    f"result_idx should be an integer between {-len(params)} and {len(params) - 1}")
            if result_idx < 0:
                result_idx = len(params) + result_idx
            result_idx = [result_idx]
        elif not isinstance(result_idx, list):
            raise ValueError("result_idx should be a list of integers")

        return result_idx

    def with_default_adapter(self, adapter: BaseKernelAdapter) -> "Profiler":
        self.adapter = adapter
        return self

    def init_distributed(self):
        import os
        import datetime

        WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
        RANK = int(os.environ.get("RANK", 0))
        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(LOCAL_RANK)
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=WORLD_SIZE,
            rank=RANK,
            timeout=datetime.timedelta(seconds=1800),
        )
        assert torch.distributed.is_initialized()
        TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

        torch.cuda.synchronize()
        pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    def _get_inputs(self, with_output=False):
        ins = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                ins.append(self.supply(self.params[i]))
        return ins

    def _get_distributed_inputs(self, with_output=False):
        ins = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                shape = list(map(int, self.params[i].shape))
                tensor = pynvshmem.nvshmem_create_tensor(shape, self.params[i].dtype)
                if self.supply_type == TensorSupplyType.Integer:
                    is_unsigned = self.params[i].is_unsigned()
                    is_float8 = self.params[i].is_float8()
                    if is_unsigned:
                        tensor[:] = torch.randint(
                            low=0, high=3, size=shape, device=tensor.device, dtype=tensor.dtype)
                    elif is_float8:
                        tensor[:] = torch.randint(
                            low=-128, high=128, size=shape, device=tensor.device,
                            dtype=torch.int8).to(dtype=tensor.dtype)
                    else:
                        tensor[:] = torch.randint(
                            low=-2, high=3, size=shape, device=tensor.device, dtype=tensor.dtype)
                elif self.supply_type == TensorSupplyType.Uniform:
                    tensor[:] = torch.empty(
                        *shape, device=tensor.device, dtype=tensor.dtype).uniform_(-1.0, 1.0)
                elif self.supply_type == TensorSupplyType.Normal:
                    tensor[:] = torch.empty(
                        *shape, device=tensor.device, dtype=tensor.dtype).normal_(-1.0, 1.0)
                elif self.supply_type == TensorSupplyType.Randn:
                    tensor[:] = torch.randn(*shape, device=tensor.device).to(dtype=tensor.dtype)
                elif self.supply_type == TensorSupplyType.Zero:
                    tensor[:] = torch.zeros(*shape, device=tensor.device, dtype=tensor.dtype)
                elif self.supply_type == TensorSupplyType.One:
                    tensor[:] = torch.ones(*shape, device=tensor.device, dtype=tensor.dtype)
                else:
                    raise ValueError(f"Unknown supply type: {self.supply_type}")
                ins.append(tensor)
        return ins

    def _get_params(self, with_output=False):
        params = []
        for i in range(len(self.params)):
            if with_output or i not in self.result_idx:
                params.append(self.params[i])
        return params

    def assert_allclose(
        self,
        reference_program: Callable,
        input_tensors: Optional[List[torch.Tensor]] = None,
        atol: float = 1e-2,
        rtol: float = 1e-2,
        max_mismatched_ratio=0.01,
    ):
        """Validates kernel output against a reference implementation.
        
        Args:
            reference_program: Reference implementation to compare against
            input_tensors: Optional pre-generated input tensors
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
            max_mismatched_ratio: Maximum allowed ratio of mismatched elements
        """
        if env.USE_DISTRIBUTED:
            self.init_distributed()
            ins = self._get_distributed_inputs()
        else:
            ins = self._get_inputs() if input_tensors is None else input_tensors
        ref_outs = reference_program(*ins)
        torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        torch.cuda.synchronize()

        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        elif isinstance(lib_outs, tuple):
            lib_outs = list(lib_outs)
        elif lib_outs is None:
            lib_outs = []

        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        elif isinstance(ref_outs, tuple):
            ref_outs = list(ref_outs)
        elif ref_outs is None:
            ref_outs = []

        ref_tensors = ins + ref_outs
        lib_tensors = ins + lib_outs

        assert len(lib_tensors) == len(
            ref_tensors), "len(lib_tensors) not equals to len(ref_tensors) !"
        # torch.set_printoptions(edgeitems=torch.inf)
        for lhs, rhs in zip(lib_tensors, ref_tensors):
            # close_mask = torch.isclose(lhs, rhs, rtol=rtol, atol=atol)
            # total_elements = lhs.numel()
            # num_not_close = (~close_mask).sum().item()
            # percentage_not_close = (num_not_close / total_elements) * 100
            # print(f"{percentage_not_close:.2f}% of the elements are not close.")
            # print(f"Total elements: {total_elements}, Not close elements: {num_not_close}")
            if lhs is not None and rhs is not None:
                # in case of numsplit template, the ref output may be None
                # which means the value is invalid, so we skip the comparison
                torch_assert_close(
                    lhs,
                    rhs,
                    rtol=rtol,
                    atol=atol,
                    max_mismatched_ratio=max_mismatched_ratio,
                    base_name="tilelang",
                    ref_name="ref",
                )

    def manual_assert_close(
        self,
        reference_program: Callable,
        input_tensors: Optional[List[torch.Tensor]] = None,
        manual_check_prog: Callable = None,
    ):
        """Validates kernel output against a reference implementation.
        
        Args:
            reference_program: Reference implementation to compare against
            input_tensors: Optional pre-generated input tensors
            atol: Absolute tolerance for comparison
            rtol: Relative tolerance for comparison
            max_mismatched_ratio: Maximum allowed ratio of mismatched elements
        """
        ins = self._get_inputs() if input_tensors is None else input_tensors
        ref_outs = reference_program(*ins)
        torch.cuda.synchronize()
        lib_outs = self.func(*ins)
        torch.cuda.synchronize()

        if isinstance(lib_outs, torch.Tensor):
            lib_outs = [lib_outs]
        if isinstance(ref_outs, torch.Tensor):
            ref_outs = [ref_outs]
        elif ref_outs is None:
            ref_outs = []
        assert len(lib_outs) == len(ref_outs), f"{len(lib_outs)=} not equals to {len(ref_outs)=} !"
        torch.set_printoptions(edgeitems=torch.inf)
        manual_check_prog(lib_outs, ref_outs)

    def assert_consistent(self, repeat=10):
        """Checks for kernel consistency across multiple runs.
        
        Args:
            repeat: Number of times to repeat the consistency check
        """
        # Used to check no race condition inside the kernel
        if env.USE_DISTRIBUTED:
            self.init_distributed()
            ins = self._get_distributed_inputs()
        else:
            ins = self._get_inputs()
        ref_outs = self.func(*ins)

        for _ in range(repeat):
            lib_outs = self.func(*ins)
            for lhs, rhs in zip(lib_outs, ref_outs):
                assert torch.allclose(lhs, rhs), [
                    "result is not consistent",
                    lhs,
                    rhs,
                ]

    def run_once(self, func: Optional[Callable] = None):
        if env.USE_DISTRIBUTED:  # noqa: SIM108
            # self.init_distributed()
            ins = self._get_distributed_inputs()
        else:
            ins = self._get_inputs()
        if not func:
            func = self.__call__
        return func(*ins)

    def determine_profiler(self, func: Optional[Callable] = None):
        """Determines which profiler backend to use based on function type.
        
        Args:
            func: Function to be profiled
            profiler: Explicitly specified profiler type or "auto" for automatic detection
        
        Returns:
            str: The determined profiler type ("torch" or "tvm")
        """
        if isinstance(func, tvm.runtime.Module):
            return "tvm"
        else:
            return "torch"

    def do_bench(
        self,
        func: Optional[Callable] = None,
        warmup: int = 25,
        rep: int = 100,
        n_warmup: int = 1,
        n_repeat: int = 1,
        input_tensors: List[torch.Tensor] = None,
    ) -> float:
        """Benchmarks the execution time of a given function.
        
        Args:
            func: Function to benchmark (uses adapter if None)
            warmup: Warmup time in milliseconds
            rep: Number of repetitions for timing
            n_warmup: Number of warmup iterations
            n_repeat: Number of timing iterations
            profiler: Which profiling backend to use
            input_tensors: Optional pre-generated input tensors
            
        Returns:
            float: Average execution time in milliseconds
        """
        profiler = self.determine_profiler(func)
        if profiler == "torch":
            if func is None:
                assert self.adapter is not None, "benchmarking function should be provided"
                func = self.adapter
            if env.USE_DISTRIBUTED:
                self.init_distributed()
                ins = self._get_distributed_inputs() if input_tensors is None else input_tensors
            else:
                ins = self._get_inputs() if input_tensors is None else input_tensors
            bench_func = partial(func, *ins)
            return do_bench(
                bench_func,
                warmup=warmup,
                rep=rep,
                _n_warmup=n_warmup,
                _n_repeat=n_repeat,
            )
        elif profiler == "tvm":
            assert func is not None, "func should not be None"
            assert isinstance(
                func, tvm.runtime.Module), f"func should be a TVM module, but got {type(func)}"
            if env.USE_DISTRIBUTED:
                self.init_distributed()
                ins = self._get_distributed_inputs(
                    with_output=True) if input_tensors is None else input_tensors
            else:
                ins = self._get_inputs(with_output=True) if input_tensors is None else input_tensors
            target = "cuda"

            with suppress(Exception):
                target = self.mod.imported_modules[0].type_key

            assert target in ["cuda", "hip"], f"Unknown target: {target}"

            device = tvm.cuda(0) if target == "cuda" else tvm.rocm(0)
            time_evaluator = self.mod.time_evaluator(
                self.mod.entry_name, device, number=rep, repeat=n_repeat)
            tvm_inputs = [adapt_torch2tvm(inp) for inp in ins]
            # Transform Latency to ms
            return time_evaluator(*tvm_inputs).mean * 1e3
        else:
            raise ValueError(f"Unknown profiler: {profiler}")

    @property
    def func(self):
        assert self.adapter is not None, "adapter should be provided"
        return self.adapter

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.func(*args, **kwds)
