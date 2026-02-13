"""Tests for 3D vector tile generation pipeline (tiling3d module).

Covers: proto round-trip, Morton codes, TileModel3D, CartesianProjector3D,
RDP-3D simplification, 3D clipping, octree, generator end-to-end,
and round-trip generate→read→compare.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
from geojson_pydantic import LineString, Point, Polygon

from microjson.model import MicroFeature, MicroFeatureCollection, TIN

# --- Proto imports ---
from microjson.tiling3d.proto import microjson_3d_tile_pb2 as pb

# --- Module imports ---
from microjson.tiling3d.morton import (
    morton_decode_3d,
    morton_encode_3d,
    tile_id_3d,
)
from microjson.tiling3d.projector3d import CartesianProjector3D
from microjson.tiling3d.simplify3d import simplify_3d
from microjson.tiling3d.clip3d import clip_3d
from microjson.tiling3d.convert3d import (
    LINESTRING3D,
    POINT3D,
    POLYGON3D,
    TIN_TYPE,
    compute_bounds_3d,
    convert_collection_3d,
    convert_feature_3d,
)
from microjson.tiling3d.tile3d import create_tile_3d, transform_tile_3d
from microjson.tiling3d.encoder3d import encode_tile_3d
from microjson.tiling3d.reader3d import TileReader3D, decode_tile
from microjson.tiling3d.octree import Octree, OctreeConfig
from microjson.tiling3d.generator3d import TileGenerator3D
from microjson.tiling3d.tilejson3d import TileModel3D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point_feature(x: float, y: float, z: float, **props) -> MicroFeature:
    return MicroFeature(
        type="Feature",
        geometry=Point(type="Point", coordinates=[x, y, z]),
        properties=props if props else {},
    )


def _line_feature(coords: list[list[float]], **props) -> MicroFeature:
    return MicroFeature(
        type="Feature",
        geometry=LineString(type="LineString", coordinates=coords),
        properties=props if props else {},
    )


def _polygon_feature(ring: list[list[float]], **props) -> MicroFeature:
    return MicroFeature(
        type="Feature",
        geometry=Polygon(type="Polygon", coordinates=[ring]),
        properties=props if props else {},
    )


def _tin_feature(**props) -> MicroFeature:
    """A simple TIN with two triangles."""
    tin = TIN(
        type="TIN",
        coordinates=[
            [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.5], [0.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 1.0, 0.5], [1.0, 0.0, 0.0]]],
        ],
    )
    return MicroFeature(
        type="Feature",
        geometry=tin,
        properties=props if props else {},
    )


def _collection(*features: MicroFeature) -> MicroFeatureCollection:
    return MicroFeatureCollection(
        type="FeatureCollection",
        features=list(features),
    )


# ===========================================================================
# 1. Proto round-trip tests
# ===========================================================================

class TestProtoRoundTrip:
    """Protobuf serialization/deserialization."""

    def test_empty_tile(self):
        tile = pb.Tile()
        data = tile.SerializeToString()
        tile2 = pb.Tile()
        tile2.ParseFromString(data)
        assert len(tile2.layers) == 0

    def test_layer_with_feature(self):
        tile = pb.Tile()
        layer = tile.layers.add()
        layer.name = "test"
        layer.version = 3
        layer.extent = 4096
        layer.extent_z = 4096

        feat = layer.features.add()
        feat.id = 42
        feat.type = pb.Tile.POINT3D
        feat.geometry.extend([9, 100, 200])  # MoveTo(1) + zigzag x, y
        feat.geometry_z.extend([50])

        data = tile.SerializeToString()
        tile2 = pb.Tile()
        tile2.ParseFromString(data)

        assert len(tile2.layers) == 1
        assert tile2.layers[0].name == "test"
        assert tile2.layers[0].version == 3
        f = tile2.layers[0].features[0]
        assert f.id == 42
        assert f.type == pb.Tile.POINT3D
        assert list(f.geometry) == [9, 100, 200]
        assert list(f.geometry_z) == [50]

    def test_zigzag_z_encoding(self):
        """Z values are sint32 — negative values should survive."""
        tile = pb.Tile()
        layer = tile.layers.add()
        layer.name = "z_test"
        layer.version = 3
        feat = layer.features.add()
        feat.geometry_z.extend([-100, 50, -50, 0, 100])

        data = tile.SerializeToString()
        tile2 = pb.Tile()
        tile2.ParseFromString(data)
        assert list(tile2.layers[0].features[0].geometry_z) == [-100, 50, -50, 0, 100]

    def test_geom_types(self):
        """All GeomType enum values round-trip."""
        for gt in [pb.Tile.POINT3D, pb.Tile.LINESTRING3D, pb.Tile.POLYGON3D,
                   pb.Tile.POLYHEDRALSURFACE, pb.Tile.TIN]:
            tile = pb.Tile()
            f = tile.layers.add().features.add()
            f.type = gt
            data = tile.SerializeToString()
            tile2 = pb.Tile()
            tile2.ParseFromString(data)
            assert tile2.layers[0].features[0].type == gt

    def test_tags_round_trip(self):
        tile = pb.Tile()
        layer = tile.layers.add()
        layer.name = "tags"
        layer.version = 3
        layer.keys.append("name")
        layer.keys.append("count")
        v1 = layer.values.add()
        v1.string_value = "hello"
        v2 = layer.values.add()
        v2.uint_value = 42

        feat = layer.features.add()
        feat.tags.extend([0, 0, 1, 1])  # name=hello, count=42

        data = tile.SerializeToString()
        tile2 = pb.Tile()
        tile2.ParseFromString(data)
        l = tile2.layers[0]
        assert list(l.keys) == ["name", "count"]
        assert l.values[0].string_value == "hello"
        assert l.values[1].uint_value == 42
        assert list(l.features[0].tags) == [0, 0, 1, 1]

    def test_radii_field(self):
        tile = pb.Tile()
        feat = tile.layers.add().features.add()
        feat.radii.extend([1.5, 2.0, 0.5])
        data = tile.SerializeToString()
        tile2 = pb.Tile()
        tile2.ParseFromString(data)
        assert list(tile2.layers[0].features[0].radii) == pytest.approx([1.5, 2.0, 0.5])


# ===========================================================================
# 2. Morton code tests
# ===========================================================================

class TestMorton:
    def test_encode_decode_origin(self):
        assert morton_decode_3d(morton_encode_3d(0, 0, 0)) == (0, 0, 0)

    def test_encode_decode_various(self):
        for x, y, d in [(1, 2, 3), (7, 7, 7), (0, 0, 1), (100, 200, 50)]:
            code = morton_encode_3d(x, y, d)
            assert morton_decode_3d(code) == (x, y, d)

    def test_monotonic_x(self):
        """Increasing x should increase Morton code (with y=d=0)."""
        codes = [morton_encode_3d(x, 0, 0) for x in range(8)]
        assert codes == sorted(codes)

    def test_tile_id_unique(self):
        """tile_id_3d should produce unique IDs for different tiles."""
        ids = set()
        for z in range(3):
            n = 1 << z
            for x in range(n):
                for y in range(n):
                    for d in range(n):
                        tid = tile_id_3d(z, x, y, d)
                        assert tid not in ids
                        ids.add(tid)


# ===========================================================================
# 3. TileModel3D tests
# ===========================================================================

class TestTileModel3D:
    def test_basic_validation(self):
        m = TileModel3D(
            tilejson="3.0.0",
            tiles=["tiles/{z}/{x}/{y}/{d}.mvt3"],
            vector_layers=[{"id": "test", "fields": {}}],
            bounds3d=[0.0, 0.0, 0.0, 100.0, 100.0, 50.0],
        )
        assert m.bounds3d == [0.0, 0.0, 0.0, 100.0, 100.0, 50.0]

    def test_center3d(self):
        m = TileModel3D(
            tilejson="3.0.0",
            tiles=["t"],
            vector_layers=[{"id": "l"}],
            center3d=[50.0, 50.0, 25.0, 2],
        )
        assert m.center3d == [50.0, 50.0, 25.0, 2]

    def test_depthsize_default(self):
        m = TileModel3D(
            tilejson="3.0.0",
            tiles=["t"],
            vector_layers=[{"id": "l"}],
        )
        assert m.depthsize == 256

    def test_json_serialization(self):
        m = TileModel3D(
            tilejson="3.0.0",
            tiles=["t"],
            vector_layers=[{"id": "l"}],
            bounds3d=[0, 0, 0, 1, 1, 1],
            resolution_per_zoom={0: 1.0, 1: 0.5},
        )
        data = json.loads(m.model_dump_json(exclude_none=True))
        assert "bounds3d" in data
        assert "resolution_per_zoom" in data


# ===========================================================================
# 4. CartesianProjector3D tests
# ===========================================================================

class TestCartesianProjector3D:
    def test_project_origin(self):
        proj = CartesianProjector3D((0, 0, 0, 10, 10, 10))
        assert proj.project(0, 0, 0) == (0.0, 0.0, 0.0)

    def test_project_max(self):
        proj = CartesianProjector3D((0, 0, 0, 10, 10, 10))
        assert proj.project(10, 10, 10) == (1.0, 1.0, 1.0)

    def test_project_mid(self):
        proj = CartesianProjector3D((0, 0, 0, 10, 10, 10))
        assert proj.project(5, 5, 5) == pytest.approx((0.5, 0.5, 0.5))

    def test_unproject_roundtrip(self):
        proj = CartesianProjector3D((10, 20, 30, 110, 120, 130))
        for x, y, z in [(10, 20, 30), (60, 70, 80), (110, 120, 130)]:
            nx, ny, nz = proj.project(x, y, z)
            rx, ry, rz = proj.unproject(nx, ny, nz)
            assert (rx, ry, rz) == pytest.approx((x, y, z))


# ===========================================================================
# 5. RDP-3D simplification tests
# ===========================================================================

class TestSimplify3D:
    def test_no_simplification_short(self):
        """Two-point line should not simplify."""
        xy = [0.0, 0.0, 1.0, 1.0]
        z = [0.0, 1.0]
        out_xy, out_z = simplify_3d(xy, z, 0.01)
        assert len(out_z) == 2

    def test_collinear_simplifies(self):
        """Collinear 3D points should simplify to endpoints."""
        xy = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]
        z = [0.0, 0.5, 1.0]
        out_xy, out_z = simplify_3d(xy, z, 0.01)
        assert len(out_z) == 2  # only endpoints

    def test_non_collinear_kept(self):
        """Point far from segment should be kept."""
        xy = [0.0, 0.0, 0.5, 10.0, 1.0, 0.0]
        z = [0.0, 0.0, 0.0]
        out_xy, out_z = simplify_3d(xy, z, 0.01)
        assert len(out_z) == 3

    def test_high_tolerance_collapses(self):
        """Very high tolerance collapses to endpoints."""
        xy = [0.0, 0.0, 0.5, 0.1, 1.0, 0.0]
        z = [0.0, 0.1, 0.0]
        out_xy, out_z = simplify_3d(xy, z, 100.0, min_vertices=2)
        assert len(out_z) == 2

    def test_min_vertices(self):
        """min_vertices prevents over-simplification."""
        # Non-collinear points so RDP can pick intermediates
        xy = [0.0, 0.0, 0.2, 5.0, 0.5, 0.0, 0.8, 5.0, 1.0, 0.0]
        z = [0.0, 1.0, 0.0, 1.0, 0.0]
        out_xy, out_z = simplify_3d(xy, z, 100.0, min_vertices=4)
        assert len(out_z) >= 4


# ===========================================================================
# 6. 3D clipping tests
# ===========================================================================

class TestClip3D:
    def _make_point(self, x, y, z):
        return {
            "geometry": [x, y], "geometry_z": [z],
            "type": POINT3D, "tags": {},
            "minX": x, "minY": y, "minZ": z,
            "maxX": x, "maxY": y, "maxZ": z,
        }

    def _make_line(self, xy, z):
        n = len(z)
        return {
            "geometry": xy, "geometry_z": z,
            "type": LINESTRING3D, "tags": {},
            "minX": min(xy[i * 2] for i in range(n)),
            "minY": min(xy[i * 2 + 1] for i in range(n)),
            "minZ": min(z),
            "maxX": max(xy[i * 2] for i in range(n)),
            "maxY": max(xy[i * 2 + 1] for i in range(n)),
            "maxZ": max(z),
        }

    def test_trivial_accept(self):
        """Point fully inside clip bounds."""
        pt = self._make_point(0.5, 0.5, 0.5)
        result = clip_3d([pt], 0.0, 1.0, 0)
        assert len(result) == 1

    def test_trivial_reject_x(self):
        """Point outside X bounds."""
        pt = self._make_point(1.5, 0.5, 0.5)
        result = clip_3d([pt], 0.0, 1.0, 0)
        assert len(result) == 0

    def test_trivial_reject_z(self):
        """Point outside Z bounds."""
        pt = self._make_point(0.5, 0.5, 1.5)
        result = clip_3d([pt], 0.0, 1.0, 2)
        assert len(result) == 0

    def test_line_clip_x(self):
        """Line crossing X boundary gets clipped."""
        line = self._make_line(
            [0.0, 0.5, 0.3, 0.5, 0.8, 0.5],
            [0.0, 0.3, 0.8],
        )
        result = clip_3d([line], 0.0, 0.6, 0)
        assert len(result) == 1
        assert result[0]["type"] == LINESTRING3D

    def test_line_fully_outside(self):
        """Line entirely outside Y bounds."""
        line = self._make_line(
            [0.0, 0.8, 1.0, 0.9],
            [0.0, 1.0],
        )
        result = clip_3d([line], 0.0, 0.5, 1)
        assert len(result) == 0

    def test_tin_straddling_included(self):
        """TIN face straddling boundary is included whole."""
        tin_feat = {
            "geometry": [0.0, 0.0, 1.0, 0.0, 0.5, 1.0],
            "geometry_z": [0.0, 0.0, 0.5],
            "type": TIN_TYPE, "tags": {},
            "minX": 0.0, "minY": 0.0, "minZ": 0.0,
            "maxX": 1.0, "maxY": 1.0, "maxZ": 0.5,
        }
        result = clip_3d([tin_feat], 0.0, 0.5, 0)
        assert len(result) == 1  # included whole

    def test_tin_fully_outside(self):
        """TIN face fully outside is rejected."""
        tin_feat = {
            "geometry": [2.0, 2.0, 3.0, 2.0, 2.5, 3.0],
            "geometry_z": [0.0, 0.0, 0.5],
            "type": TIN_TYPE, "tags": {},
            "minX": 2.0, "minY": 2.0, "minZ": 0.0,
            "maxX": 3.0, "maxY": 3.0, "maxZ": 0.5,
        }
        result = clip_3d([tin_feat], 0.0, 1.0, 0)
        assert len(result) == 0

    def test_clip_per_axis(self):
        """Clip works for all three axes."""
        pt = self._make_point(0.3, 0.7, 0.5)
        # X: in range
        assert len(clip_3d([pt], 0.0, 0.5, 0)) == 1
        # Y: out of range
        assert len(clip_3d([pt], 0.0, 0.5, 1)) == 0
        # Z: in range
        assert len(clip_3d([pt], 0.0, 0.6, 2)) == 1


# ===========================================================================
# 7. Octree tests
# ===========================================================================

class TestOctree:
    def test_single_point_builds(self):
        """Single point should produce tiles at all zoom levels."""
        feat = {
            "geometry": [0.5, 0.5], "geometry_z": [0.5],
            "type": POINT3D, "tags": {},
            "minX": 0.5, "minY": 0.5, "minZ": 0.5,
            "maxX": 0.5, "maxY": 0.5, "maxZ": 0.5,
        }
        config = OctreeConfig(max_zoom=2)
        tree = Octree([feat], config)
        assert tree.get_tile(0, 0, 0, 0) is not None

    def test_zoom_levels(self):
        """Tiles should exist at each zoom up to max_zoom."""
        feat = {
            "geometry": [0.5, 0.5], "geometry_z": [0.5],
            "type": POINT3D, "tags": {},
            "minX": 0.5, "minY": 0.5, "minZ": 0.5,
            "maxX": 0.5, "maxY": 0.5, "maxZ": 0.5,
        }
        config = OctreeConfig(max_zoom=3)
        tree = Octree([feat], config)
        for z in range(4):
            tiles = tree.tiles_at_zoom(z)
            assert len(tiles) >= 1, f"No tiles at zoom {z}"

    def test_empty_features(self):
        """Empty feature list produces no tiles."""
        config = OctreeConfig(max_zoom=2)
        tree = Octree([], config)
        assert len(tree.all_tiles) == 0

    def test_multiple_points_split(self):
        """Two distant points should end up in different octants at zoom 1."""
        f1 = {
            "geometry": [0.1, 0.1], "geometry_z": [0.1],
            "type": POINT3D, "tags": {"id": "a"},
            "minX": 0.1, "minY": 0.1, "minZ": 0.1,
            "maxX": 0.1, "maxY": 0.1, "maxZ": 0.1,
        }
        f2 = {
            "geometry": [0.9, 0.9], "geometry_z": [0.9],
            "type": POINT3D, "tags": {"id": "b"},
            "minX": 0.9, "minY": 0.9, "minZ": 0.9,
            "maxX": 0.9, "maxY": 0.9, "maxZ": 0.9,
        }
        config = OctreeConfig(max_zoom=1)
        tree = Octree([f1, f2], config)
        tiles_z1 = tree.tiles_at_zoom(1)
        # At zoom 1 we should have at least 2 tiles (the two points are in different octants)
        assert len(tiles_z1) >= 2

    def test_min_zoom_respected(self):
        """Tiles below min_zoom should not be stored."""
        feat = {
            "geometry": [0.5, 0.5], "geometry_z": [0.5],
            "type": POINT3D, "tags": {},
            "minX": 0.5, "minY": 0.5, "minZ": 0.5,
            "maxX": 0.5, "maxY": 0.5, "maxZ": 0.5,
        }
        config = OctreeConfig(min_zoom=1, max_zoom=2)
        tree = Octree([feat], config)
        assert tree.get_tile(0, 0, 0, 0) is None
        assert len(tree.tiles_at_zoom(1)) >= 1

    def test_eight_octants(self):
        """8 points in 8 corners should produce 8 tiles at zoom 1."""
        features = []
        for x in [0.1, 0.9]:
            for y in [0.1, 0.9]:
                for d in [0.1, 0.9]:
                    features.append({
                        "geometry": [x, y], "geometry_z": [d],
                        "type": POINT3D, "tags": {},
                        "minX": x, "minY": y, "minZ": d,
                        "maxX": x, "maxY": y, "maxZ": d,
                    })
        config = OctreeConfig(max_zoom=1)
        tree = Octree(features, config)
        assert len(tree.tiles_at_zoom(1)) == 8


# ===========================================================================
# 8. Convert + Tile + Transform tests
# ===========================================================================

class TestConvert3D:
    def test_point_conversion(self):
        coll = _collection(_point_feature(5, 5, 5))
        proj = CartesianProjector3D((0, 0, 0, 10, 10, 10))
        feats = convert_collection_3d(coll, proj)
        assert len(feats) == 1
        assert feats[0]["type"] == POINT3D
        assert feats[0]["geometry"] == pytest.approx([0.5, 0.5])
        assert feats[0]["geometry_z"] == pytest.approx([0.5])

    def test_line_conversion(self):
        coll = _collection(_line_feature([[0, 0, 0], [10, 10, 10]]))
        proj = CartesianProjector3D((0, 0, 0, 10, 10, 10))
        feats = convert_collection_3d(coll, proj)
        assert len(feats) == 1
        assert feats[0]["type"] == LINESTRING3D
        assert len(feats[0]["geometry_z"]) == 2

    def test_polygon_conversion(self):
        ring = [[0, 0, 0], [10, 0, 0], [10, 10, 5], [0, 10, 5], [0, 0, 0]]
        coll = _collection(_polygon_feature(ring))
        proj = CartesianProjector3D((0, 0, 0, 10, 10, 10))
        feats = convert_collection_3d(coll, proj)
        assert len(feats) == 1
        assert feats[0]["type"] == POLYGON3D

    def test_tin_conversion(self):
        coll = _collection(_tin_feature())
        proj = CartesianProjector3D((0, 0, 0, 2, 1, 0.5))
        feats = convert_collection_3d(coll, proj)
        assert len(feats) == 1
        assert feats[0]["type"] == TIN_TYPE

    def test_compute_bounds(self):
        coll = _collection(
            _point_feature(0, 0, 0),
            _point_feature(10, 20, 30),
        )
        bounds = compute_bounds_3d(coll)
        assert bounds == (0, 0, 0, 10, 20, 30)

    def test_empty_collection(self):
        coll = _collection()
        bounds = compute_bounds_3d(coll)
        assert bounds == (0, 0, 0, 1, 1, 1)  # default


class TestTile3D:
    def test_create_tile(self):
        feat = {
            "geometry": [0.5, 0.5], "geometry_z": [0.5],
            "type": POINT3D, "tags": {},
            "minX": 0.5, "minY": 0.5, "minZ": 0.5,
            "maxX": 0.5, "maxY": 0.5, "maxZ": 0.5,
        }
        tile = create_tile_3d([feat], 0, 0, 0, 0)
        assert tile["num_features"] == 1
        assert tile["z"] == 0

    def test_transform_tile(self):
        feat = {
            "geometry": [0.5, 0.5], "geometry_z": [0.5],
            "type": POINT3D, "tags": {},
            "minX": 0.5, "minY": 0.5, "minZ": 0.5,
            "maxX": 0.5, "maxY": 0.5, "maxZ": 0.5,
        }
        tile = create_tile_3d([feat], 0, 0, 0, 0)
        transformed = transform_tile_3d(tile, extent=4096, extent_z=4096)
        # At zoom 0, tile covers [0,1], so 0.5 -> 2048
        f = transformed["features"][0]
        assert f["geometry"][0] == 2048
        assert f["geometry"][1] == 2048
        assert f["geometry_z"][0] == 2048

    def test_transform_zoom1(self):
        """At zoom 1, tile (1,1,1) covers [0.5,1.0] in all axes."""
        feat = {
            "geometry": [0.75, 0.75], "geometry_z": [0.75],
            "type": POINT3D, "tags": {},
            "minX": 0.75, "minY": 0.75, "minZ": 0.75,
            "maxX": 0.75, "maxY": 0.75, "maxZ": 0.75,
        }
        tile = create_tile_3d([feat], 1, 1, 1, 1)
        transformed = transform_tile_3d(tile, extent=4096, extent_z=4096)
        f = transformed["features"][0]
        # 0.75 in tile (1,1,1) at zoom 1: local = (0.75 - 0.5) * 2 = 0.5 -> 2048
        assert f["geometry"][0] == 2048


# ===========================================================================
# 9. Encoder tests
# ===========================================================================

class TestEncoder3D:
    def test_encode_point(self):
        tile_data = {
            "features": [{
                "geometry": [2048, 2048],
                "geometry_z": [2048],
                "type": POINT3D,
                "tags": {"name": "test"},
            }],
        }
        data = encode_tile_3d(tile_data)
        assert isinstance(data, bytes)
        assert len(data) > 0

        # Decode and verify
        layers = decode_tile(data)
        assert len(layers) == 1
        assert layers[0]["name"] == "default"
        assert len(layers[0]["features"]) == 1

    def test_encode_line(self):
        tile_data = {
            "features": [{
                "geometry": [0, 0, 1000, 2000, 4096, 4096],
                "geometry_z": [0, 2048, 4096],
                "type": LINESTRING3D,
                "tags": {},
            }],
        }
        data = encode_tile_3d(tile_data)
        layers = decode_tile(data)
        feat = layers[0]["features"][0]
        assert len(feat["xy"]) == 3
        # Verify absolute coordinates
        assert feat["xy"][0] == (0, 0)
        assert feat["xy"][1] == (1000, 2000)
        assert feat["xy"][2] == (4096, 4096)

    def test_encode_z_delta(self):
        """Z values are delta-encoded and decoded correctly."""
        tile_data = {
            "features": [{
                "geometry": [0, 0, 100, 100, 200, 200],
                "geometry_z": [100, 300, 200],
                "type": LINESTRING3D,
                "tags": {},
            }],
        }
        data = encode_tile_3d(tile_data)
        layers = decode_tile(data)
        feat = layers[0]["features"][0]
        assert feat["z"] == [100, 300, 200]

    def test_encode_tags(self):
        tile_data = {
            "features": [{
                "geometry": [100, 200],
                "geometry_z": [50],
                "type": POINT3D,
                "tags": {"category": "neuron", "count": 42},
            }],
        }
        data = encode_tile_3d(tile_data)
        layers = decode_tile(data)
        feat = layers[0]["features"][0]
        assert feat["tags"]["category"] == "neuron"
        assert feat["tags"]["count"] == 42

    def test_encode_polygon(self):
        tile_data = {
            "features": [{
                "geometry": [0, 0, 4096, 0, 4096, 4096, 0, 4096, 0, 0],
                "geometry_z": [0, 0, 4096, 4096, 0],
                "ring_lengths": [5],
                "type": POLYGON3D,
                "tags": {},
            }],
        }
        data = encode_tile_3d(tile_data)
        layers = decode_tile(data)
        assert len(layers[0]["features"]) == 1

    def test_extent_z_in_layer(self):
        tile_data = {"features": []}
        data = encode_tile_3d(tile_data, extent=8192, extent_z=2048)
        layers = decode_tile(data)
        assert layers[0]["extent"] == 8192
        assert layers[0]["extent_z"] == 2048


# ===========================================================================
# 10. Generator end-to-end tests
# ===========================================================================

class TestGenerator3D:
    def test_point_pipeline(self, tmp_path):
        """Generate tiles from 3D points."""
        coll = _collection(
            _point_feature(0, 0, 0, name="origin"),
            _point_feature(10, 10, 10, name="far"),
        )
        config = OctreeConfig(max_zoom=1)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        count = gen.generate(tmp_path)
        assert count > 0

        # Verify files exist
        mvt3_files = list(tmp_path.rglob("*.mvt3"))
        assert len(mvt3_files) == count

    def test_line_pipeline(self, tmp_path):
        """Generate tiles from 3D lines."""
        coll = _collection(
            _line_feature([[0, 0, 0], [5, 5, 5], [10, 10, 10]]),
        )
        config = OctreeConfig(max_zoom=1)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        count = gen.generate(tmp_path)
        assert count > 0

    def test_tin_pipeline(self, tmp_path):
        """Generate tiles from TIN geometry."""
        coll = _collection(_tin_feature(mesh="tin1"))
        config = OctreeConfig(max_zoom=1)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        count = gen.generate(tmp_path)
        assert count > 0

    def test_tilejson_written(self, tmp_path):
        """TileJSON metadata file is written correctly."""
        coll = _collection(_point_feature(5, 5, 5))
        config = OctreeConfig(max_zoom=2)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tilejson(tmp_path / "tilejson3d.json")

        assert (tmp_path / "tilejson3d.json").exists()
        meta = json.loads((tmp_path / "tilejson3d.json").read_text())
        assert "bounds3d" in meta
        assert meta["tilejson"] == "3.0.0"

    def test_mixed_geometry(self, tmp_path):
        """Pipeline handles mixed geometry types."""
        coll = _collection(
            _point_feature(1, 1, 1),
            _line_feature([[0, 0, 0], [10, 10, 10]]),
            _tin_feature(),
        )
        config = OctreeConfig(max_zoom=1)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        count = gen.generate(tmp_path)
        assert count > 0

    def test_custom_extent(self, tmp_path):
        """Custom extent values propagate to tiles."""
        coll = _collection(_point_feature(5, 5, 5))
        config = OctreeConfig(max_zoom=0, extent=8192, extent_z=2048)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        gen.generate(tmp_path)

        tile_file = list(tmp_path.rglob("*.mvt3"))[0]
        layers = decode_tile(tile_file.read_bytes())
        assert layers[0]["extent"] == 8192
        assert layers[0]["extent_z"] == 2048

    def test_no_features_error(self, tmp_path):
        """Generate without add_features raises RuntimeError."""
        gen = TileGenerator3D()
        with pytest.raises(RuntimeError):
            gen.generate(tmp_path)

    def test_layer_name(self, tmp_path):
        """Custom layer name is used in tiles."""
        coll = _collection(_point_feature(5, 5, 5))
        config = OctreeConfig(max_zoom=0)
        gen = TileGenerator3D(config)
        gen.add_features(coll, layer_name="neurons")
        gen.generate(tmp_path)

        tile_file = list(tmp_path.rglob("*.mvt3"))[0]
        layers = decode_tile(tile_file.read_bytes())
        assert layers[0]["name"] == "neurons"


# ===========================================================================
# 11. Round-trip tests (generate → read → compare)
# ===========================================================================

class TestRoundTrip:
    def test_point_roundtrip(self, tmp_path):
        """Points survive generate → read → reconstruct."""
        coll = _collection(
            _point_feature(5, 5, 5, label="center"),
        )
        config = OctreeConfig(max_zoom=0)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tilejson(tmp_path / "tilejson3d.json")

        reader = TileReader3D(tmp_path / "tilejson3d.json")
        result = reader.tiles2microjson(0)
        assert len(result.features) == 1

    def test_line_roundtrip(self, tmp_path):
        """Lines survive generate → read → reconstruct."""
        coll = _collection(
            _line_feature([[0, 0, 0], [10, 10, 10]]),
        )
        config = OctreeConfig(max_zoom=0)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tilejson(tmp_path / "tilejson3d.json")

        reader = TileReader3D(tmp_path / "tilejson3d.json")
        result = reader.tiles2microjson(0)
        assert len(result.features) >= 1

    def test_reader_metadata(self, tmp_path):
        """Reader correctly loads TileJSON metadata."""
        coll = _collection(_point_feature(5, 5, 5))
        config = OctreeConfig(max_zoom=1)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tilejson(tmp_path / "tilejson3d.json")

        reader = TileReader3D(tmp_path / "tilejson3d.json")
        assert reader.metadata["tilejson"] == "3.0.0"
        assert "bounds3d" in reader.metadata

    def test_reader_tiles_at_zoom(self, tmp_path):
        """Reader can enumerate tiles at a zoom level."""
        coll = _collection(
            _point_feature(2, 2, 2),
            _point_feature(8, 8, 8),
        )
        config = OctreeConfig(max_zoom=1)
        gen = TileGenerator3D(config)
        gen.add_features(coll)
        gen.generate(tmp_path)
        gen.write_tilejson(tmp_path / "tilejson3d.json")

        reader = TileReader3D(tmp_path / "tilejson3d.json")
        tiles = reader.tiles_at_zoom(1)
        assert len(tiles) >= 2
