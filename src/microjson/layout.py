"""Shared layout module for MicroJSON feature collections.

Computes spatial offsets (row or grid) and applies them by translating
MicroJSON geometry coordinates directly. This makes layout reusable
across all exporters (glTF, Neuroglancer, Arrow/Parquet).
"""

from __future__ import annotations

import copy
from typing import Any

from geojson_pydantic import Polygon, MultiPolygon

from .model import (
    MicroFeature,
    MicroFeatureCollection,
    NeuronMorphology,
    SliceStack,
)
from .transforms import translate_geometry

# (xmin, ymin, zmin, xmax, ymax, zmax) — OGC/GeoJSON bbox ordering
Bounds = tuple[float, float, float, float, float, float]

# Translation offset
Offset3 = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Bounding box computation
# ---------------------------------------------------------------------------

def _collect_xyz_from_coords(
    coords: Any,
    xs: list[float],
    ys: list[float],
    zs: list[float],
) -> None:
    """Recursively extract X/Y/Z from nested GeoJSON coordinate arrays."""
    if not coords:
        return
    if isinstance(coords[0], (int, float)):
        xs.append(float(coords[0]))
        ys.append(float(coords[1]) if len(coords) > 1 else 0.0)
        zs.append(float(coords[2]) if len(coords) > 2 else 0.0)
    else:
        for item in coords:
            _collect_xyz_from_coords(item, xs, ys, zs)


def geometry_bounds(geom: Any) -> Bounds | None:
    """Return 3-D bounding box as (xmin, ymin, zmin, xmax, ymax, zmax).

    Uses ``.bbox3d()`` for 3D types (NeuronMorphology, TIN,
    PolyhedralSurface, SliceStack) and recursive coordinate extraction
    for GeoJSON types.

    Returns None if the geometry is empty or None.
    """
    if geom is None:
        return None

    # 3D types with bbox3d() — returns (xmin, ymin, zmin, xmax, ymax, zmax)
    if hasattr(geom, "bbox3d"):
        try:
            return geom.bbox3d()
        except (ValueError, IndexError):
            return None

    # GeoJSON types with .coordinates
    if not hasattr(geom, "coordinates"):
        return None

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    _collect_xyz_from_coords(geom.coordinates, xs, ys, zs)

    if not xs:
        return None
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


# ---------------------------------------------------------------------------
# Layout algorithms
# ---------------------------------------------------------------------------

def _row_layout(
    bounds: list[Bounds | None],
    spacing: float,
) -> list[Offset3]:
    """Place features side-by-side along X."""
    n = len(bounds)
    widths = [(b[3] - b[0]) if b else 0.0 for b in bounds]  # xmax - xmin
    max_width = max(widths) if widths else 0.0
    gap = spacing if spacing > 0 else max_width * 0.2

    offsets: list[Offset3] = [(0.0, 0.0, 0.0)] * n
    cursor = bounds[0][3] if bounds[0] else gap  # xmax of first feature

    for i in range(1, n):
        b = bounds[i]
        if b is None:
            dx = cursor + gap
            offsets[i] = (dx, 0.0, 0.0)
            cursor = dx
            continue
        dx = (cursor + gap) - b[0]  # shift so xmin aligns to cursor + gap
        offsets[i] = (dx, 0.0, 0.0)
        cursor = b[3] + dx  # new xmax position

    return offsets


def _grid_layout(
    bounds: list[Bounds | None],
    spacing: float,
    grid_max_x: int | None,
    grid_max_y: int | None,
    grid_max_z: int | None,
    n: int,
) -> list[Offset3]:
    """Place features on a uniform grid that wraps X -> Y -> Z.

    ``grid_max_x/y/z`` give the number of cells per axis directly.
    """
    # Bounds format: (xmin, ymin, zmin, xmax, ymax, zmax)
    widths = [(b[3] - b[0]) if b else 0.0 for b in bounds]
    heights = [(b[4] - b[1]) if b else 0.0 for b in bounds]
    depths = [(b[5] - b[2]) if b else 0.0 for b in bounds]

    max_w = max(widths) if widths else 0.0
    max_h = max(heights) if heights else 0.0
    max_d = max(depths) if depths else 0.0

    max_extent = max(max_w, max_h, max_d)

    if spacing > 0:
        # Fixed cell size — center-to-center distance, ignores extent
        cell_x = spacing
        cell_y = spacing
        cell_z = spacing
    else:
        # Auto: 20% gap on top of max extent per axis
        gap = max_extent * 0.2 if max_extent > 0 else 1.0
        cell_x = max_w + gap
        cell_y = max_h + gap
        cell_z = max_d + gap

    # Grid dimensions — values are cell counts, unlimited when None
    cols = grid_max_x if grid_max_x is not None else n
    rows = (
        grid_max_y
        if grid_max_y is not None
        else max(1, -(-n // cols))
    )
    layers = (
        grid_max_z
        if grid_max_z is not None
        else max(1, -(-n // (cols * rows)))
    )

    capacity = cols * rows * layers
    if n > capacity:
        raise ValueError(
            f"Cannot fit {n} features in grid "
            f"({cols} cols \u00d7 {rows} rows \u00d7 {layers} layers "
            f"= {capacity} cells). "
            f"Increase grid_max_x/y/z or reduce feature count."
        )

    # Feature 0 is the reference point; others are placed relative to it
    ref = bounds[0]
    ref_cx = (ref[0] + ref[3]) / 2 if ref else 0.0  # (xmin + xmax) / 2
    ref_cy = (ref[1] + ref[4]) / 2 if ref else 0.0  # (ymin + ymax) / 2
    ref_cz = (ref[2] + ref[5]) / 2 if ref else 0.0  # (zmin + zmax) / 2

    offsets: list[Offset3] = [(0.0, 0.0, 0.0)]

    for i in range(1, n):
        col = i % cols
        row = (i // cols) % rows
        layer = i // (cols * rows)

        target_cx = ref_cx + col * cell_x
        target_cy = ref_cy + row * cell_y
        target_cz = ref_cz + layer * cell_z

        b = bounds[i]
        feat_cx = (b[0] + b[3]) / 2 if b else 0.0
        feat_cy = (b[1] + b[4]) / 2 if b else 0.0
        feat_cz = (b[2] + b[5]) / 2 if b else 0.0

        offsets.append((
            target_cx - feat_cx,
            target_cy - feat_cy,
            target_cz - feat_cz,
        ))

    return offsets


def compute_collection_offsets(
    features: list[MicroFeature],
    spacing: float = 0.0,
    grid_max_x: int | None = None,
    grid_max_y: int | None = None,
    grid_max_z: int | None = None,
) -> list[Offset3]:
    """Compute 3-D translation for each feature in a collection.

    Returns a list of (dx, dy, dz) offsets in source coordinates.
    The first feature always stays at its original position.
    """
    n = len(features)
    if n <= 1:
        return [(0.0, 0.0, 0.0)] * n

    bounds = [
        geometry_bounds(f.geometry) if f.geometry else None
        for f in features
    ]

    has_grid = (
        grid_max_x is not None
        or grid_max_y is not None
        or grid_max_z is not None
    )

    if has_grid:
        return _grid_layout(bounds, spacing, grid_max_x, grid_max_y, grid_max_z, n)
    return _row_layout(bounds, spacing)


def apply_layout(
    collection: MicroFeatureCollection,
    spacing: float = 0.0,
    grid_max_x: int | None = None,
    grid_max_y: int | None = None,
    grid_max_z: int | None = None,
) -> MicroFeatureCollection:
    """Apply spatial layout to a collection, returning a new collection.

    Computes offsets and translates each feature's geometry coordinates
    directly. The returned collection is a copy — the original is not
    modified.
    """
    features = list(collection.features)
    offsets = compute_collection_offsets(
        features, spacing, grid_max_x, grid_max_y, grid_max_z,
    )

    new_features = []
    for feat, (dx, dy, dz) in zip(features, offsets):
        if abs(dx) < 1e-12 and abs(dy) < 1e-12 and abs(dz) < 1e-12:
            new_features.append(feat)
            continue

        if feat.geometry is None:
            new_features.append(feat)
            continue

        new_geom = translate_geometry(feat.geometry, dx, dy, dz)
        # Build a new feature with translated geometry
        new_feat = feat.model_copy(update={"geometry": new_geom})
        new_features.append(new_feat)

    return collection.model_copy(update={"features": new_features})
