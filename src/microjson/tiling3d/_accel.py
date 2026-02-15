"""Cython acceleration status for 3D tiling.

Provides a single flag ``CYTHON_AVAILABLE`` that is ``True`` when the
compiled Cython extensions have been built and are importable.
"""

CYTHON_AVAILABLE = False

try:
    from . import clip3d_cy as _clip3d_cy  # noqa: F401
    from . import encoder3d_cy as _encoder3d_cy  # noqa: F401
    from . import tile3d_cy as _tile3d_cy  # noqa: F401

    CYTHON_AVAILABLE = True
except ImportError:
    pass
