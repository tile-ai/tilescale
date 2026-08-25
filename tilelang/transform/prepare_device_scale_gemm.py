"""Compatibility shim for the device-scale tile-op pass.

The generic pass now lives in ``tilelang.transform.prepare_device_scale`` as
``PrepareDeviceScaleTileOps``. This module re-exports it under the original
``PrepareDeviceScaleGemm`` name, plus the GEMM device-template names, so existing
import paths keep working unchanged:

    from tilelang.transform import PrepareDeviceScaleGemm, build_device_gemm_template
    from tilelang.transform.prepare_device_scale_gemm import DeviceGemmInfo
"""

from .prepare_device_scale import (  # noqa: F401
    PrepareScaleTileOps,
    PrepareDeviceScaleTileOps,
    PrepareDeviceScaleGemm,
)
from tilelang.tileop.gemm.device_template import (  # noqa: F401
    DeviceGemmInfo,
    build_device_gemm_template,
)
