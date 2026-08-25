"""Elementwise tile-op scale templates package.

Import-light: the block-scope fill expansion template in ``scale_expansion`` is
registered lazily via
``tilelang.tileop.scale_expansion.ensure_default_scale_expansion_templates_registered``
to avoid import cycles. Do not import submodules here.
"""
