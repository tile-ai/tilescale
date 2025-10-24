from importlib import import_module as _imp

__all__ = []

try:
    _C = _imp("ipc_ext._C")
except Exception as e:
    raise RuntimeError(
        "Failed to load ipc_ext native extension.\n"
        "Tips:\n"
        "  1) Make sure PyTorch is installed *before* building this package.\n"
        "  2) Match CUDA version between your environment and PyTorch.\n"
        "  3) Reinstall with: pip install -e . --no-build-isolation\n"
        f"Original error: {e}"
    ) from e

try:
    _create_ipc_handle = _C._create_ipc_handle
    _sync_ipc_handles  = _C._sync_ipc_handles
    _create_tensor     = _C._create_tensor
    __all__ += ["_create_ipc_handle", "_sync_ipc_handles", "_create_tensor"]
except AttributeError:
    pass

for name in dir(_C):
    if name.startswith("__"):
        continue
    if name in globals():
        continue
    globals()[name] = getattr(_C, name)
    if not name.startswith("_"):
        __all__.append(name)
