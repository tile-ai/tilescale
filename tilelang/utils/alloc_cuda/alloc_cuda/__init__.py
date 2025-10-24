from importlib import import_module

__all__ = []

try:
    _C = import_module("alloc_cuda._C")
except Exception as e:
    raise RuntimeError(
        "Failed to load alloc_cuda native extension.\n"
        "Tips:\n"
        "  1) Make sure PyTorch is installed *before* building this package.\n"
        "  2) Match CUDA version between your environment and PyTorch.\n"
        "  3) Reinstall with: pip install -e . --no-build-isolation\n"
        f"Original error: {e}"
    ) from e

for name in dir(_C):
    if not name.startswith("_"):
        globals()[name] = getattr(_C, name)
        __all__.append(name)
