"""Compatibility shim for the device-scale tile-op registry.

The registry was generalized to a scale-aware registry in
``tilelang.tileop.scale_template``. This module re-exports the legacy
``device_*`` names so existing callers keep working unchanged:

    from tilelang.tileop.device_template import (
        DeviceTileOpTemplate,
        register_device_template,
        resolve_device_template,
        ensure_default_device_templates_registered,
    )
"""

from tilelang.tileop.scale_template import (  # noqa: F401
    DeviceTileOpTemplate,
    register_device_template,
    resolve_device_template,
    ensure_default_device_templates_registered,
)
