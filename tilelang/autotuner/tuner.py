"""The auto-tune module for tilelang programs.

This module provides functionality for auto-tuning tilelang programs, including JIT compilation
and performance optimization through configuration search.
"""

import tilelang
from tilelang import tvm as tvm
from tvm.tir import PrimFunc, Var
from tvm.target import Target
import inspect
from functools import partial
from typing import (Callable, List, Literal, Any, Optional, Union, Dict, overload, Tuple)
from tqdm import tqdm
import logging
import functools
import concurrent.futures
import torch
import os
import sys
import signal
import json
import hashlib
import threading
import traceback
from pathlib import Path

from tilelang import env
from tilelang.autotuner.param import CompileArgs, ProfileArgs, AutotuneResult
from tilelang.autotuner.capture import get_autotune_inputs
from tilelang.jit.param import _P, _RProg
from tilelang.version import __version__


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")


def run_with_timeout(func, timeout, *args, **kwargs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        raise e
    finally:
        signal.alarm(0)
    return result


# Configure logging for the autotuner module
# TODO: Consider creating a common logger in utils
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Lazy handler initialization flag
_logger_handlers_initialized = False


def _init_logger_handlers():
    global _logger_handlers_initialized
    if _logger_handlers_initialized:
        return
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    file_handler = logging.FileHandler('autotuner.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    _logger_handlers_initialized = True


def get_available_cpu_count() -> int:
    """Gets the number of CPU cores available to the current process.
    """
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()

    return cpu_count or 1


class AutoTuner:
    """Auto-tuner for tilelang programs.

    This class handles the auto-tuning process by testing different configurations
    and finding the optimal parameters for program execution.

    Args:
        fn: The function to be auto-tuned.
        configs: List of configurations to try during auto-tuning.
    """
    compile_args = CompileArgs()
    profile_args = ProfileArgs()

    _kernel_parameters: Optional[Tuple[str, ...]] = None
    _lock = threading.Lock()  # For thread safety
    _memory_cache = {}  # In-memory cache dictionary
    cache_dir: Path = Path(env.TILELANG_CACHE_DIR) / "autotuner"

    def __init__(self, fn: Callable, configs):
        self.fn = fn
        self.configs = configs
        self.ref_latency_cache = None
        self.jit_input_tensors = None
        self.ref_input_tensors = None
        self.jit_compile = None

    @classmethod
    def from_kernel(cls, kernel: Callable, configs):
        """Create an AutoTuner instance from a kernel function.

        Args:
            kernel: The kernel function to auto-tune.
            configs: List of configurations to try.

        Returns:
            AutoTuner: A new AutoTuner instance.
        """
        return cls(kernel, configs)

    def set_compile_args(self,
                         out_idx: Union[List[int], int, None] = None,
                         target: Literal['auto', 'cuda', 'hip'] = 'auto',
                         execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
                         target_host: Union[str, Target] = None,
                         verbose: bool = False,
                         pass_configs: Optional[Dict[str, Any]] = None):
        """Set compilation arguments for the auto-tuner.

        Args:
            out_idx: List of output tensor indices.
            target: Target platform.
            execution_backend: Execution backend to use for kernel execution.
            target_host: Target host for cross-compilation.
            verbose: Whether to enable verbose output.
            pass_configs: Additional keyword arguments to pass to the Compiler PassContext.

        Returns:
            AutoTuner: Self for method chaining.
        """
        self.compile_args = CompileArgs(
            out_idx=out_idx,
            target=target,
            execution_backend=execution_backend,
            target_host=target_host,
            verbose=verbose,
            pass_configs=pass_configs)

        return self

    def set_profile_args(self,
                         warmup: int = 25,
                         rep: int = 100,
                         timeout: int = 30,
                         supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
                         ref_prog: Callable = None,
                         supply_prog: Callable = None,
                         rtol: float = 1e-2,
                         atol: float = 1e-2,
                         max_mismatched_ratio: float = 0.01,
                         skip_check: bool = False,
                         manual_check_prog: Callable = None,
                         cache_input_tensors: bool = False):
        """Set profiling arguments for the auto-tuner.

        Args:
            supply_type: Type of tensor supply mechanism. Ignored if `supply_prog` is provided.
            ref_prog: Reference program for validation.
            supply_prog: Supply program for input tensors.
            rtol: Relative tolerance for validation.
            atol: Absolute tolerance for validation.
            max_mismatched_ratio: Maximum allowed mismatch ratio.
            skip_check: Whether to skip validation.
            manual_check_prog: Manual check program for validation.
            cache_input_tensors: Whether to cache input tensors.
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.

        Returns:
            AutoTuner: Self for method chaining.
        """
        # If the program is under `with set_autotune_inputs` context,
        # the `supply_prog` will be ignored and the `get_autotune_inputs` will be used instead.
        if get_autotune_inputs() is not None:
            if supply_prog is not None:
                logger.warning(
                    "`supply_prog` will be ignored as this program is under `with set_autotune_inputs` context."
                )
            supply_prog = lambda _: get_autotune_inputs()  # noqa: E731

        self.profile_args = ProfileArgs(
            supply_type=supply_type,
            ref_prog=ref_prog,
            supply_prog=supply_prog,
            rtol=rtol,
            atol=atol,
            max_mismatched_ratio=max_mismatched_ratio,
            skip_check=skip_check,
            manual_check_prog=manual_check_prog,
            cache_input_tensors=cache_input_tensors,
            warmup=warmup,
            rep=rep,
            timeout=timeout)

        # If a custom `supply_prog` is provided, the profiler's `supply_type` setting
        # becomes ineffective. The custom supply program will be used instead.
        if supply_prog is not None and supply_type != tilelang.TensorSupplyType.Auto:
            logger.warning("Ignoring `supply_type` passed to `set_profile_args` because "
                           "`supply_prog` is not None.")

        return self

    def set_kernel_parameters(self, parameters: Tuple[str, ...]):
        # for cache key generation
        self._kernel_parameters = parameters

    def generate_cache_key(self, parameters: Dict[str, Any]) -> Optional[AutotuneResult]:
        """Generate a cache key for the auto-tuning process.
        """
        # extract parameters from the function signature
        op_parameters = []
        for _, default_value in parameters.items():
            if default_value.default is not inspect.Parameter.empty:
                op_parameters.append(default_value.default)

        if self._kernel_parameters is not None:
            op_parameters += self._kernel_parameters

        func_source = inspect.getsource(self.fn)
        key_data = {
            "version": __version__,
            "op_parameters": tuple(op_parameters),
            "func_source": func_source,
            "configs": self.configs,
            "compile_args": hash(self.compile_args),
            "profile_args": hash(self.profile_args),
        }
        # Sort keys to ensure consistency
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def _save_result_to_disk(self, key, result: AutotuneResult):
        result.save_to_disk(self.cache_dir / key, self.compile_args.verbose)

    def _load_result_from_disk(self, key) -> AutotuneResult:
        result = AutotuneResult.load_from_disk(self.cache_dir / key, self.compile_args)
        return result

    def run(self, warmup: int = 25, rep: int = 100, timeout: int = 30):
        """Run the auto-tuning process.

        Args:
            warmup: Number of warmup iterations.
            rep: Number of repetitions for timing.
            timeout: Maximum time per configuration.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        _init_logger_handlers()

        sig = inspect.signature(self.fn)
        parameters = sig.parameters

        if isinstance(self.configs, Callable):
            self.configs = self.configs(*self._kernel_parameters)

        key = self.generate_cache_key(parameters)

        with self._lock:
            if env.is_cache_enabled():
                # First check in-memory cache
                if key in self._memory_cache:
                    logger.warning("Found kernel in memory cache. For better performance," \
                                        " consider using `@tilelang.autotune` instead of direct AutoTuner.from_kernel.")
                    return self._memory_cache[key]

                # Then check disk cache
                result = self._load_result_from_disk(key)
                if result is not None:
                    # Populate memory cache with disk result
                    self._memory_cache[key] = result
                    return result

        best_latency: float = 1e8
        best_config: Optional[Dict[str, Any]] = None
        best_kernel: Optional[tilelang.JITKernel] = None

        def _compile(**config_arg) -> tilelang.JITKernel:
            compile_args = self.compile_args
            return compile_args.compile_program(self.fn(**config_arg))

        if self.jit_compile is None:
            self.jit_compile = _compile

        def target_fn(jit_kernel: tilelang.JITKernel):
            # Unpack the context
            profile_args = self.profile_args
            supply_type = profile_args.supply_type
            skip_check = profile_args.skip_check
            manual_check_prog = profile_args.manual_check_prog
            cache_input_tensors = profile_args.cache_input_tensors
            ref_prog = profile_args.ref_prog
            supply_prog = profile_args.supply_prog
            rtol = profile_args.rtol
            atol = profile_args.atol
            max_mismatched_ratio = profile_args.max_mismatched_ratio

            profiler = jit_kernel.get_profiler(tensor_supply_type=supply_type)

            # Factory functions for generating input tensors.
            # This encapsulates the logic of using either a custom supply program (`supply_prog`)
            # or the default profiler input generation (`profiler._get_inputs`).
            def get_input_tensors_supply(with_output: bool):

                def func():
                    if supply_prog is not None:
                        return supply_prog(profiler._get_params(with_output=with_output))
                    else:
                        return profiler._get_inputs(with_output=with_output)

                return func

            jit_input_tensors_supply = get_input_tensors_supply(with_output=False)
            ref_input_tensors_supply = get_input_tensors_supply(with_output=False)

            if cache_input_tensors:
                params = profiler._get_params(with_output=False)
                if self.jit_input_tensors is None:
                    self.jit_input_tensors = jit_input_tensors_supply()
                else:
                    # check if the cached tensors are compatible with the current configuration
                    assert len(params) == len(
                        self.jit_input_tensors), "len(params) != len(self.jit_input_tensors)"
                    for p, c in zip(params, self.jit_input_tensors):
                        if not isinstance(c, torch.Tensor):
                            # skip non-tensor inputs checking
                            continue

                        # Check tensor compatibility using generator expression
                        def shape_equal(a, b):
                            return all(
                                a_dim == b_dim or isinstance(a_dim, Var) or isinstance(b_dim, Var)
                                for a_dim, b_dim in zip(a.shape, b.shape))

                        if p.dtype != c.dtype or not shape_equal(p, c):
                            logger.warning(
                                "\nIncompatible input tensor properties detected between cached tensors and "
                                "tensors regenerated for the current configuration trial. "
                                "This can happen if different tuning configurations require different input shapes/dtypes "
                                "and input tensor caching is enabled.\n"
                                "To ensure fresh, compatible inputs are generated for every trial "
                                "you can disable caching by setting:\n"
                                "  `cache_input_tensors=False`\n"
                                "within your `.set_compile_args(...)` call.\n")
                            # otherwise, regenerate the input tensors for safety
                            self.jit_input_tensors = jit_input_tensors_supply()
                            break
            else:
                self.jit_input_tensors = jit_input_tensors_supply()

            if (not skip_check) and (ref_prog is not None):
                if manual_check_prog is not None:
                    profiler.manual_assert_close(
                        ref_prog,
                        input_tensors=self.jit_input_tensors,
                        manual_check_prog=manual_check_prog)
                else:
                    profiler.assert_allclose(
                        ref_prog,
                        input_tensors=self.jit_input_tensors,
                        rtol=rtol,
                        atol=atol,
                        max_mismatched_ratio=max_mismatched_ratio)
            latency = profiler.do_bench(
                warmup=warmup, rep=rep, input_tensors=self.jit_input_tensors)

            if self.ref_latency_cache is None and ref_prog is not None:
                self.ref_input_tensors = ref_input_tensors_supply()
                self.ref_latency_cache = profiler.do_bench(
                    ref_prog, n_warmup=warmup, n_repeat=rep, input_tensors=self.ref_input_tensors)

            return latency, self.ref_latency_cache

        config_args = []
        for config in self.configs:
            new_kwargs = {}
            keys = config.keys()
            for name, _ in parameters.items():
                if name in config:
                    new_kwargs[name] = config[name]
            unused_keys = set(keys) - set(new_kwargs.keys())
            if len(unused_keys) > 0:
                raise ValueError(f"Unused keys in config: {unused_keys}")
            config_args.append(new_kwargs)

        if len(config_args) == 0:
            raise ValueError("No configurations to tune, please check your `@autotune` decorator")

        # check if the tunable arguments has been set.
        # get the back config argument
        top_config, *rest = config_args

        if self._kernel_parameters is not None:
            key_args_tuple, key_kwargs_tuple = self._kernel_parameters
            tunable_arguments = [key for key, _ in top_config.items()]

            # Check if all tunable arguments have been tuned by comparing config keys with key_kwargs_tuple
            if any(key in top_config for key, _ in key_kwargs_tuple):
                logger.warning(
                    f"Tunable parameters {tunable_arguments} already provided during auto-tuning. Skipping compilation and using direct JIT"
                )
                # compile the kernel with the provided parameters
                jit_kernel = self.jit_compile()
                autotuner_result = AutotuneResult(
                    libcode=jit_kernel.get_kernel_source(),
                    func=jit_kernel.prim_func,
                    kernel=jit_kernel)
                self._memory_cache[key] = autotuner_result
                return autotuner_result
        # get the cpu count
        available_cpu_count = get_available_cpu_count()
        cpu_utilizations = float(env.TILELANG_AUTO_TUNING_CPU_UTILITIES)
        cpu_counts = int(env.TILELANG_AUTO_TUNING_CPU_COUNTS)
        max_cpu_count = int(env.TILELANG_AUTO_TUNING_MAX_CPU_COUNT)
        if cpu_counts > 0:
            num_workers = min(cpu_counts, available_cpu_count)
            logger.info(
                f"Auto-tuning with {cpu_counts} CPU counts, {available_cpu_count} CPUs available, {num_workers} CPUs will be used"
            )
        else:
            num_workers = max(1, int(available_cpu_count * cpu_utilizations))
            logger.info(
                f"Auto-tuning with {cpu_utilizations} CPU utilizations, {available_cpu_count} CPUs available, {num_workers} CPUs will be used"
            )

        if max_cpu_count > 0 and num_workers > max_cpu_count:
            logger.warning(
                f"Auto-tuning with {cpu_utilizations} CPU utilizations, {available_cpu_count} CPUs available, {num_workers} CPUs will be used, but the max CPU count is {max_cpu_count}, so we will use {max_cpu_count} CPUs"
            )
            num_workers = max_cpu_count

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        futures = []
        future_to_index = {}

        def device_wrapper(func, device, **config_arg):
            torch.cuda.set_device(device)
            return func(**config_arg)

        for i, config_arg in enumerate(config_args):
            future = pool.submit(
                functools.partial(device_wrapper, self.jit_compile, torch.cuda.current_device()),
                **config_arg,
            )
            futures.append(future)
            future_to_index[future] = i

        results_with_configs = []
        for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Compiling configurations"):
            idx = future_to_index[future]
            config = config_args[idx]
            try:
                result = future.result()
                results_with_configs.append((result, config))
            except Exception as e:
                logger.debug(
                    f"Compilation failed for config {config} at index {idx} with error: {e}")
                continue

        ref_latency = None
        progress_bar = tqdm(range(len(results_with_configs)), desc="Bench configurations")
        for i in progress_bar:
            jit_kernel, config = results_with_configs[i]
            try:
                # Cannot ThreadPoolExecutor to enforce timeout on target_fn execution
                # Because tma init may behave strangely with one thread
                # latency, ref_latency = target_fn(jit_kernel)
                latency, ref_latency = run_with_timeout(target_fn, timeout, jit_kernel)
            except TimeoutException:
                logger.info(
                    f"A timeout occurred while testing config {config}, checkout autotuner.log for more details"
                )
                continue
            except Exception:
                logger.info(
                    f"An error occurred while testing config {config}, checkout autotuner.log for more details"
                )
                logger.debug(f"Error: {traceback.format_exc()}")
                continue

            if latency < best_latency:
                best_latency = latency
                best_config = config
                best_kernel = jit_kernel

            progress_bar.set_postfix({"best_latency": best_latency})
            tqdm.write(f"Tuned Latency {latency} with config {config} at index {i}")

        pool.shutdown()

        if best_kernel is None:
            error_msg = ("Auto-tuning failed: No configuration successfully "
                         "compiled and passed benchmarking/validation.")
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        best_kernel: tilelang.JITKernel = best_kernel.update_tuner_result(
            latency=best_latency,
            config=best_config,
            ref_latency=ref_latency,
        )

        autotuner_result = AutotuneResult(
            latency=best_latency,
            config=best_config,
            ref_latency=ref_latency,
            libcode=best_kernel.get_kernel_source(),
            func=best_kernel.prim_func,
            kernel=best_kernel)

        if self.compile_args.execution_backend == "dlpack":
            logger.warning("DLPack backend does not support cache saving to disk.")
        else:
            with self._lock:
                if env.is_cache_enabled():
                    self._save_result_to_disk(key, autotuner_result)

        self._memory_cache[key] = autotuner_result

        return autotuner_result

    def __call__(self) -> Any:
        """Make the AutoTuner callable, running the auto-tuning process.

        Returns:
            AutotuneResult: Results of the auto-tuning process.
        """
        return self.run()


class _AutoTunerImplementation:
    # Overload __init__ to help type checkers understand the effect of return_program
    # The '-> None' is for __init__ itself. The crucial part is Literal for return_program.

    warmup: int = 25
    rep: int = 100
    timeout: int = 100
    configs: Union[Dict, Callable] = None
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto
    ref_prog: Callable = None
    supply_prog: Callable = None
    rtol: float = 1e-2
    atol: float = 1e-2
    max_mismatched_ratio: float = 0.01
    skip_check: bool = False
    manual_check_prog: Callable = None
    cache_input_tensors: bool = False

    def __init__(self,
                 configs: Union[Dict, Callable],
                 warmup: int = 25,
                 rep: int = 100,
                 timeout: int = 100,
                 supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
                 ref_prog: Callable = None,
                 supply_prog: Callable = None,
                 rtol: float = 1e-2,
                 atol: float = 1e-2,
                 max_mismatched_ratio: float = 0.01,
                 skip_check: bool = False,
                 manual_check_prog: Callable = None,
                 cache_input_tensors: bool = False) -> None:
        """Initialize the AutoTunerImplementation.

        Args:
            configs: Configuration space to explore during auto-tuning.
            warmup: Number of warmup iterations before timing.
            rep: Number of repetitions for timing measurements.
            timeout: Maximum time (in seconds) allowed for each configuration.
            supply_type: Strategy for generating input tensors (random/zeros/etc)
            ref_prog: Reference implementation for validation
            supply_prog: Custom function to provide input tensors
            rtol: Relative tolerance for numerical validation
            atol: Absolute tolerance for numerical validation
            max_mismatched_ratio: Allowed percentage of mismatched values
            skip_check: Bypass validation against reference implementation
            manual_check_prog: Custom validation function
            cache_input_tensors: Reuse input tensors across trials
        """
        # Configuration and benchmarking parameters
        self.configs = configs  # Search space of tuning configurations
        self.warmup = warmup  # Warmup iterations for stable measurements
        self.rep = rep  # Measurement repetitions for statistics
        self.timeout = timeout  # Per-configuration timeout threshold

        # Tensor handling and validation setup
        self.supply_type = supply_type  # Input tensor generation strategy
        self.ref_prog = ref_prog  # Ground truth implementation
        self.supply_prog = supply_prog  # Custom input data provider
        self.rtol = rtol  # Relative error tolerance
        self.atol = atol  # Absolute error tolerance
        self.max_mismatched_ratio = max_mismatched_ratio  # Allowed mismatch

        # Validation control flags
        self.skip_check = skip_check  # Bypass accuracy verification
        self.manual_check_prog = manual_check_prog  # Custom validation
        self.cache_input_tensors = cache_input_tensors  # Reuse inputs

        # Cache for storing tuned kernel implementations
        self._tuner_cache: Dict[tuple, tilelang.JITKernel] = {}  # (args, kwargs) -> compiled kernel

    # This tells the type checker what the *wrapper* function will return.
    # this is for linting, please do not remove it.
    @overload
    def __call__(self, fn: Callable[_P, _RProg]) -> Callable[_P, Tuple[_RProg, AutotuneResult]]:
        ...

    @overload
    def __call__(self, fn: Callable[_P, _RProg]) -> Callable[_P, AutotuneResult]:
        ...

    # Actual implementation of __call__
    def __call__(self, fn: Callable[_P, _RProg]) -> Callable[_P, Any]:
        warmup = self.warmup
        rep = self.rep
        timeout = self.timeout

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):

            key_args_tuple = args
            key_kwargs_tuple = tuple(sorted(kwargs.items()))
            key = (key_args_tuple, key_kwargs_tuple)

            if key not in self._tuner_cache:

                def jit_compile(**config_arg):
                    return fn(*args, **kwargs, __tune_params=config_arg)

                compile_arguments = fn(__return_compile_arguments=True)

                autotuner = AutoTuner(
                    fn, configs=self.configs).set_profile_args(
                        supply_type=self.supply_type,
                        ref_prog=self.ref_prog,
                        supply_prog=self.supply_prog,
                        rtol=self.rtol,
                        atol=self.atol,
                        max_mismatched_ratio=self.max_mismatched_ratio,
                        skip_check=self.skip_check,
                        manual_check_prog=self.manual_check_prog,
                        cache_input_tensors=self.cache_input_tensors,
                    ).set_compile_args(
                        out_idx=compile_arguments['out_idx'],
                        execution_backend=compile_arguments['execution_backend'],
                        target=compile_arguments['target'],
                        target_host=compile_arguments['target_host'],
                        verbose=compile_arguments['verbose'],
                        pass_configs=compile_arguments['pass_configs'],
                    )

                autotuner.jit_compile = jit_compile
                autotuner.set_kernel_parameters(key)

                autotuner.run = partial(autotuner.run, warmup, rep, timeout)

                artifact = autotuner.run()

                self._tuner_cache[key] = artifact.kernel

            return self._tuner_cache[key]

        return wrapper


def autotune(  # This is the new public interface
    func: Union[Callable[_P, _RProg], PrimFunc, None] = None,
    *,  # Indicates subsequent arguments are keyword-only
    configs: Union[Dict, Callable],
    # profile arguments
    warmup: int = 25,
    rep: int = 100,
    timeout: int = 100,
    # compile arguments
    supply_type: tilelang.TensorSupplyType = tilelang.TensorSupplyType.Auto,
    ref_prog: Callable = None,
    supply_prog: Callable = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    max_mismatched_ratio: float = 0.01,
    skip_check: bool = False,
    manual_check_prog: Callable = None,
    cache_input_tensors: bool = False,
):
    """
    Just-In-Time (JIT) compiler decorator for TileLang functions.

    This decorator can be used without arguments (e.g., `@tilelang.jit`):
       Applies JIT compilation with default settings.
    
    Tips:
        - If you want to skip the auto-tuning process, you can set override the tunable parameters in the function signature.
            ```python
                if enable_autotune:
                    kernel = flashattn(batch, heads, seq_len, dim, is_causal)
                else:
                    kernel = flashattn(
                        batch, heads, seq_len, dim, is_causal, groups=groups, block_M=128, block_N=128, num_stages=2, threads=256)
            ```

    Parameters
    ----------
    func_or_out_idx : Any, optional
        If using `@tilelang.jit(...)` to configure, this is the `out_idx` parameter.
        If using `@tilelang.jit` directly on a function, this argument is implicitly
        the function to be decorated (and `out_idx` will be `None`).
    configs : Dict or Callable
        Configuration space to explore during auto-tuning.
    warmup : int, optional
        Number of warmup iterations before timing.
    rep : int, optional
        Number of repetitions for timing measurements.
    timeout : int, optional
    target : Union[str, Target], optional
        Compilation target for TVM (e.g., "cuda", "llvm"). Defaults to "auto".
    target_host : Union[str, Target], optional
        Target host for cross-compilation. Defaults to None.
    execution_backend : Literal["dlpack", "ctypes", "cython"], optional
        Backend for kernel execution and argument passing. Defaults to "cython".
    verbose : bool, optional
        Enables verbose logging during compilation. Defaults to False.
    pass_configs : Optional[Dict[str, Any]], optional
        Configurations for TVM's pass context. Defaults to None.
    debug_root_path : Optional[str], optional
        Directory to save compiled kernel source for debugging. Defaults to None.

    Returns
    -------
    Callable
        Either a JIT-compiled wrapper around the input function, or a configured decorator
        instance that can then be applied to a function.
    """
    if callable(func):
        # Case 1: Used as @autotune (func_or_out_idx is the function, others are defaults)
        # This is a placeholder for a real auto tuner implementation
        raise ValueError(
            "Use tilelang.autotune to decorate func without arguments is not supported yet.")
    elif isinstance(func, PrimFunc):
        raise ValueError("Use tilelang.jit to decorate prim_func is not supported yet.")
    else:
        # Case 2: Used as @autotune(...) to configure, or func_or_out_idx is meant as out_idx.
        # Create a _AutoTunerImplementation instance with the provided/defaulted arguments.
        # This instance is a decorator that will be applied to the function later.
        configured_decorator = _AutoTunerImplementation(
            configs=configs,
            warmup=warmup,
            rep=rep,
            timeout=timeout,
            supply_type=supply_type,
            ref_prog=ref_prog,
            supply_prog=supply_prog,
            rtol=rtol,
            atol=atol,
            max_mismatched_ratio=max_mismatched_ratio,
            skip_check=skip_check,
            manual_check_prog=manual_check_prog,
            cache_input_tensors=cache_input_tensors,
        )
        return configured_decorator
