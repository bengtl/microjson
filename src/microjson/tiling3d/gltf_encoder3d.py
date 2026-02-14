"""Encode 3D tile features as glTF/GLB for OGC 3D Tiles output.

Converts intermediate tile features (flat arrays in normalized [0,1]³ space)
back to MicroFeatures in world coordinates and delegates to the existing
``gltf/`` module to produce ``.glb`` bytes.
"""

from __future__ import annotations

from typing import Any

from geojson_pydantic import LineString, Point, Polygon

from ..gltf import GltfConfig, to_glb
from ..model import MicroFeature, MicroFeatureCollection, TIN, PolyhedralSurface
from .convert3d import LINESTRING3D, POINT3D, POLYGON3D, POLYHEDRALSURFACE, TIN_TYPE
from .projector3d import CartesianProjector3D


def _unproject_coords(
    feat: dict,
    proj: CartesianProjector3D,
) -> list[list[float]]:
    """Unproject flat tile arrays to world-coordinate position list."""
    xy = feat["geometry"]
    zz = feat["geometry_z"]
    n = len(zz)
    coords: list[list[float]] = []
    for i in range(n):
        wx, wy, wz = proj.unproject(xy[i * 2], xy[i * 2 + 1], zz[i])
        coords.append([wx, wy, wz])
    return coords


def _rebuild_point(feat: dict, proj: CartesianProjector3D) -> MicroFeature:
    coords = _unproject_coords(feat, proj)
    return MicroFeature(
        type="Feature",
        geometry=Point(type="Point", coordinates=coords[0]),
        properties=feat.get("tags", {}),
    )


def _rebuild_line(feat: dict, proj: CartesianProjector3D) -> MicroFeature:
    coords = _unproject_coords(feat, proj)
    return MicroFeature(
        type="Feature",
        geometry=LineString(type="LineString", coordinates=coords),
        properties=feat.get("tags", {}),
    )


def _rebuild_polygon(feat: dict, proj: CartesianProjector3D) -> MicroFeature:
    coords = _unproject_coords(feat, proj)
    rings = _split_rings(coords, feat.get("ring_lengths", [len(coords)]))
    return MicroFeature(
        type="Feature",
        geometry=Polygon(type="Polygon", coordinates=rings),
        properties=feat.get("tags", {}),
    )


def _rebuild_tin(feat: dict, proj: CartesianProjector3D) -> MicroFeature:
    coords = _unproject_coords(feat, proj)
    ring_lengths = feat.get("ring_lengths", [])
    faces = _split_faces(coords, ring_lengths)
    return MicroFeature(
        type="Feature",
        geometry=TIN(type="TIN", coordinates=faces),
        properties=feat.get("tags", {}),
    )


def _rebuild_polyhedral(feat: dict, proj: CartesianProjector3D) -> MicroFeature:
    coords = _unproject_coords(feat, proj)
    ring_lengths = feat.get("ring_lengths", [])
    faces = _split_faces(coords, ring_lengths)
    return MicroFeature(
        type="Feature",
        geometry=PolyhedralSurface(type="PolyhedralSurface", coordinates=faces),
        properties=feat.get("tags", {}),
    )


def _split_rings(
    coords: list[list[float]], ring_lengths: list[int],
) -> list[list[list[float]]]:
    """Split flat coord list into rings by ring_lengths."""
    rings: list[list[list[float]]] = []
    offset = 0
    for length in ring_lengths:
        rings.append(coords[offset:offset + length])
        offset += length
    return rings


def _split_faces(
    coords: list[list[float]], ring_lengths: list[int],
) -> list[list[list[list[float]]]]:
    """Split flat coord list into TIN/PolyhedralSurface faces.

    Each face is [ring], where ring is a list of positions.
    """
    faces: list[list[list[list[float]]]] = []
    offset = 0
    for length in ring_lengths:
        ring = coords[offset:offset + length]
        faces.append([ring])
        offset += length
    return faces


_REBUILDERS = {
    POINT3D: _rebuild_point,
    LINESTRING3D: _rebuild_line,
    POLYGON3D: _rebuild_polygon,
    TIN_TYPE: _rebuild_tin,
    POLYHEDRALSURFACE: _rebuild_polyhedral,
}


def tile_to_glb(
    tile: dict,
    proj: CartesianProjector3D,
    config: GltfConfig | None = None,
) -> bytes:
    """Convert a tile's intermediate features to GLB bytes.

    Parameters
    ----------
    tile : dict
        Tile dict from octree (features in normalized [0,1]³ space).
    proj : CartesianProjector3D
        Projector to convert back to world coordinates.
    config : GltfConfig, optional
        glTF configuration (defaults to no layout, metadata on).

    Returns
    -------
    bytes
        GLB binary content.
    """
    if config is None:
        config = GltfConfig(feature_spacing=None, y_up=False)

    features: list[MicroFeature] = []
    for feat in tile.get("features", []):
        gt = feat.get("type", 0)
        rebuilder = _REBUILDERS.get(gt)
        if rebuilder is not None:
            features.append(rebuilder(feat, proj))

    coll = MicroFeatureCollection(
        type="FeatureCollection",
        features=features,
    )

    return to_glb(coll, config=config)
