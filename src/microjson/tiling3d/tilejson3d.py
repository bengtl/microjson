"""TileModel3D — TileJSON with 3D fields."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, conlist

from ..tilemodel import TileModel


class TileModel3D(TileModel):
    """Extended TileJSON metadata for 3D tilesets.

    Adds depth axis, 3D bounds/center, and per-zoom resolution info.
    """

    # 3D bounds: [xmin, ymin, zmin, xmax, ymax, zmax]
    bounds3d: Optional[conlist(float, min_length=6, max_length=6)] = None  # type: ignore

    # 3D center: [x, y, z, zoom]
    center3d: Optional[conlist(float, min_length=4, max_length=4)] = None  # type: ignore

    # Number of depth slices per zoom level (analogous to tile size)
    depthsize: Optional[int] = 256

    # Map of zoom level -> spatial resolution (units per voxel)
    resolution_per_zoom: Optional[dict[int, float]] = None
