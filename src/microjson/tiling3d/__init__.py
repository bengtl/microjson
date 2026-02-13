"""3D vector tile generation and reading for MicroJSON.

Provides octree-based spatial indexing, 3D protobuf encoding,
and a full pipeline from MicroJSON features to 3D vector tiles.
"""

from .generator3d import TileGenerator3D
from .octree import OctreeConfig
from .reader3d import TileReader3D
from .tilejson3d import TileModel3D

__all__ = [
    "TileGenerator3D",
    "OctreeConfig",
    "TileReader3D",
    "TileModel3D",
]
