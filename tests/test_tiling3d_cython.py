"""Tests for Cython-accelerated 3D tiling functions.

Parametrized over "python" and "cython" backends. The cython backend
is skipped when the extensions haven't been compiled.

Each test verifies functional correctness and, where both backends
are available, bit-identical output.
"""

import pytest
import struct

from microjson.tiling3d._accel import CYTHON_AVAILABLE
from microjson.tiling3d.clip3d import (
    _clip_surface_py,
    _clip_line_py,
    clip_3d,
)
from microjson.tiling3d.encoder3d import _build_indexed_mesh_py, encode_tile_3d_py
from microjson.tiling3d.tile3d import transform_tile_3d_py


# --- Helpers to get Cython functions when available ---

def _get_cy_clip_surface():
    if not CYTHON_AVAILABLE:
        return None
    from microjson.tiling3d.clip3d_cy import _clip_surface
    return _clip_surface


def _get_cy_clip_line():
    if not CYTHON_AVAILABLE:
        return None
    from microjson.tiling3d.clip3d_cy import _clip_line
    return _clip_line


def _get_cy_build_indexed_mesh():
    if not CYTHON_AVAILABLE:
        return None
    from microjson.tiling3d.encoder3d_cy import _build_indexed_mesh
    return _build_indexed_mesh


def _get_cy_encode_tile_3d():
    if not CYTHON_AVAILABLE:
        return None
    from microjson.tiling3d.encoder3d_cy import encode_tile_3d
    return encode_tile_3d


def _get_cy_transform_tile_3d():
    if not CYTHON_AVAILABLE:
        return None
    from microjson.tiling3d.tile3d_cy import transform_tile_3d
    return transform_tile_3d


# --- Fixtures ---

@pytest.fixture(params=["python", "cython"])
def backend(request):
    if request.param == "cython" and not CYTHON_AVAILABLE:
        pytest.skip("Cython extensions not compiled")
    return request.param


def _clip_surface_fn(backend):
    if backend == "cython":
        return _get_cy_clip_surface()
    return _clip_surface_py


def _clip_line_fn(backend):
    if backend == "cython":
        return _get_cy_clip_line()
    return _clip_line_py


def _build_indexed_mesh_fn(backend):
    if backend == "cython":
        return _get_cy_build_indexed_mesh()
    return _build_indexed_mesh_py


def _transform_tile_3d_fn(backend):
    if backend == "cython":
        return _get_cy_transform_tile_3d()
    return transform_tile_3d_py


# --- _clip_surface tests ---

class TestClipSurface:
    """Tests for _clip_surface (TIN/PS per-face clipping)."""

    def _make_surface(self, xy, z, ring_lengths):
        """Build a minimal surface feature dict."""
        return {
            "geometry": xy,
            "geometry_z": z,
            "ring_lengths": ring_lengths,
            "type": 5,  # TIN_TYPE
            "tags": {"id": 1},
            "minX": min(xy[i * 2] for i in range(len(z))),
            "minY": min(xy[i * 2 + 1] for i in range(len(z))),
            "minZ": min(z),
            "maxX": max(xy[i * 2] for i in range(len(z))),
            "maxY": max(xy[i * 2 + 1] for i in range(len(z))),
            "maxZ": max(z),
        }

    def test_trivial_accept(self, backend):
        """Face fully inside [k1, k2) is kept."""
        fn = _clip_surface_fn(backend)
        feat = self._make_surface(
            [0.2, 0.2, 0.4, 0.2, 0.3, 0.4, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3],
            [4],
        )
        result = fn(feat, 0.0, 1.0, 0)
        assert result is not None
        assert result["ring_lengths"] == [4]
        assert result["geometry"] == feat["geometry"]

    def test_trivial_reject(self, backend):
        """Face fully outside is discarded."""
        fn = _clip_surface_fn(backend)
        feat = self._make_surface(
            [0.8, 0.2, 0.9, 0.2, 0.85, 0.4, 0.8, 0.2],
            [0.3, 0.3, 0.3, 0.3],
            [4],
        )
        result = fn(feat, 0.0, 0.5, 0)  # X range [0, 0.5), face at X 0.8-0.9
        assert result is None

    def test_partial_clip(self, backend):
        """Two faces, one inside, one outside."""
        fn = _clip_surface_fn(backend)
        # Face 1: X in [0.1, 0.3] — inside [0, 0.5)
        # Face 2: X in [0.6, 0.8] — outside [0, 0.5)
        feat = self._make_surface(
            [0.1, 0.1, 0.3, 0.1, 0.2, 0.3, 0.1, 0.1,
             0.6, 0.1, 0.8, 0.1, 0.7, 0.3, 0.6, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4, 4],
        )
        result = fn(feat, 0.0, 0.5, 0)
        assert result is not None
        assert result["ring_lengths"] == [4]
        assert len(result["geometry_z"]) == 4

    def test_bounds_updated(self, backend):
        """Output bounds reflect only kept faces."""
        fn = _clip_surface_fn(backend)
        feat = self._make_surface(
            [0.1, 0.2, 0.3, 0.2, 0.2, 0.4, 0.1, 0.2],
            [0.5, 0.5, 0.5, 0.5],
            [4],
        )
        result = fn(feat, 0.0, 1.0, 0)
        assert result is not None
        assert result["minX"] == pytest.approx(0.1)
        assert result["maxX"] == pytest.approx(0.3)

    def test_z_axis_clip(self, backend):
        """Clip along Z axis."""
        fn = _clip_surface_fn(backend)
        feat = self._make_surface(
            [0.1, 0.2, 0.3, 0.2, 0.2, 0.4, 0.1, 0.2],
            [0.1, 0.1, 0.1, 0.1],
            [4],
        )
        # Z range [0.5, 1.0) — face at Z=0.1 is outside
        result = fn(feat, 0.5, 1.0, 2)
        assert result is None


# --- _clip_line tests ---

class TestClipLine:
    """Tests for _clip_line (LineString3D clipping)."""

    def _make_line(self, xy, z):
        n = len(z)
        return {
            "geometry": xy,
            "geometry_z": z,
            "type": 2,  # LINESTRING3D
            "tags": {},
            "minX": min(xy[i * 2] for i in range(n)),
            "minY": min(xy[i * 2 + 1] for i in range(n)),
            "minZ": min(z),
            "maxX": max(xy[i * 2] for i in range(n)),
            "maxY": max(xy[i * 2 + 1] for i in range(n)),
            "maxZ": max(z),
        }

    def test_fully_inside(self, backend):
        """Line fully inside — one segment returned."""
        fn = _clip_line_fn(backend)
        feat = self._make_line([0.2, 0.3, 0.4, 0.5], [0.1, 0.2])
        result = fn(feat, 0.0, 1.0, 0)
        assert len(result) == 1
        assert len(result[0]["geometry_z"]) == 2

    def test_fully_outside(self, backend):
        """Line fully outside — empty."""
        fn = _clip_line_fn(backend)
        feat = self._make_line([0.6, 0.3, 0.8, 0.5], [0.1, 0.2])
        result = fn(feat, 0.0, 0.5, 0)
        assert len(result) == 0

    def test_crossing_produces_segment(self, backend):
        """Line crossing clip boundary produces clipped segment."""
        fn = _clip_line_fn(backend)
        # X goes from 0.3 to 0.7, clipped to [0, 0.5)
        feat = self._make_line([0.3, 0.0, 0.7, 0.0], [0.0, 0.0])
        result = fn(feat, 0.0, 0.5, 0)
        assert len(result) == 1
        # Should have entry point and intersection
        assert len(result[0]["geometry_z"]) >= 2


# --- _build_indexed_mesh tests ---

class TestBuildIndexedMesh:
    """Tests for _build_indexed_mesh (vertex dedup + bytes packing)."""

    def test_single_triangle(self, backend):
        """Single triangle: 3 unique vertices, 3 indices."""
        fn = _build_indexed_mesh_fn(backend)
        xy = [0, 0, 100, 0, 50, 100, 0, 0]  # closed ring: 4 verts
        z = [0, 0, 0, 0]
        pos_bytes, idx_bytes = fn(xy, z, [4])

        n_floats = len(pos_bytes) // 4
        assert n_floats == 9  # 3 vertices × 3 coords
        n_indices = len(idx_bytes) // 4
        assert n_indices == 3

        positions = struct.unpack(f"<{n_floats}f", pos_bytes)
        indices = struct.unpack(f"<{n_indices}I", idx_bytes)
        assert set(indices) == {0, 1, 2}

    def test_shared_vertex_dedup(self, backend):
        """Two triangles sharing an edge: 4 unique vertices, 6 indices."""
        fn = _build_indexed_mesh_fn(backend)
        # Triangle 1: (0,0,0), (100,0,0), (50,100,0)
        # Triangle 2: (100,0,0), (150,100,0), (50,100,0)
        xy = [
            0, 0, 100, 0, 50, 100, 0, 0,       # ring 1
            100, 0, 150, 100, 50, 100, 100, 0,  # ring 2
        ]
        z = [0, 0, 0, 0, 0, 0, 0, 0]
        pos_bytes, idx_bytes = fn(xy, z, [4, 4])

        n_floats = len(pos_bytes) // 4
        n_verts = n_floats // 3
        assert n_verts == 4  # deduped: 4 unique vertices

        n_indices = len(idx_bytes) // 4
        assert n_indices == 6  # 2 triangles × 3

    def test_empty_input(self, backend):
        """Empty ring lengths: empty bytes."""
        fn = _build_indexed_mesh_fn(backend)
        pos_bytes, idx_bytes = fn([], [], [])
        assert pos_bytes == b""
        assert idx_bytes == b""


# --- transform_tile_3d tests ---

class TestTransformTile3D:
    """Tests for transform_tile_3d (coordinate transform)."""

    def _make_tile(self, features, z=1, x=0, y=0, d=0):
        return {
            "features": features,
            "z": z,
            "x": x,
            "y": y,
            "d": d,
            "num_features": len(features),
            "num_points": sum(len(f["geometry_z"]) for f in features),
        }

    def test_midpoint_transform(self, backend):
        """Midpoint of tile maps to extent/2."""
        fn = _transform_tile_3d_fn(backend)
        # Zoom=1, tile (0,0,0) covers [0, 0.5) in each axis
        # Midpoint: (0.25, 0.25, 0.25) → tile-local (0.5, 0.5, 0.5) → 2048
        feat = {
            "geometry": [0.25, 0.25],
            "geometry_z": [0.25],
            "type": 1,  # POINT3D
            "tags": {},
        }
        tile = self._make_tile([feat], z=1, x=0, y=0, d=0)
        result = fn(tile, extent=4096, extent_z=4096)

        rf = result["features"][0]
        assert rf["geometry"] == [2048, 2048]
        assert rf["geometry_z"] == [2048]

    def test_preserves_tags(self, backend):
        """Tags are preserved through transform."""
        fn = _transform_tile_3d_fn(backend)
        feat = {
            "geometry": [0.0, 0.0],
            "geometry_z": [0.0],
            "type": 1,
            "tags": {"name": "test", "id": 42},
        }
        tile = self._make_tile([feat])
        result = fn(tile)
        assert result["features"][0]["tags"] == {"name": "test", "id": 42}

    def test_preserves_ring_lengths(self, backend):
        """ring_lengths survives transform."""
        fn = _transform_tile_3d_fn(backend)
        feat = {
            "geometry": [0.1, 0.1, 0.2, 0.1, 0.15, 0.2, 0.1, 0.1],
            "geometry_z": [0.1, 0.1, 0.1, 0.1],
            "type": 5,  # TIN
            "tags": {},
            "ring_lengths": [4],
        }
        tile = self._make_tile([feat])
        result = fn(tile)
        assert result["features"][0]["ring_lengths"] == [4]

    def test_output_metadata(self, backend):
        """Output tile has extent, num_features, etc."""
        fn = _transform_tile_3d_fn(backend)
        feat = {
            "geometry": [0.0, 0.0],
            "geometry_z": [0.0],
            "type": 1,
            "tags": {},
        }
        tile = self._make_tile([feat])
        result = fn(tile)
        assert result["extent"] == 4096
        assert result["extent_z"] == 4096
        assert result["num_features"] == 1


# --- Bit-identical comparison tests ---

class TestBitIdentical:
    """Verify Python and Cython backends produce identical output."""

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython not compiled")
    def test_clip_surface_identical(self):
        """_clip_surface: Python == Cython."""
        py_fn = _clip_surface_py
        cy_fn = _get_cy_clip_surface()

        xy = [0.1, 0.2, 0.3, 0.2, 0.2, 0.4, 0.1, 0.2,
              0.6, 0.2, 0.8, 0.2, 0.7, 0.4, 0.6, 0.2]
        z = [0.1, 0.2, 0.3, 0.1, 0.5, 0.6, 0.7, 0.5]
        feat = {
            "geometry": xy, "geometry_z": z,
            "ring_lengths": [4, 4], "type": 5,
            "tags": {"id": 1},
            "minX": 0.1, "minY": 0.2, "minZ": 0.1,
            "maxX": 0.8, "maxY": 0.4, "maxZ": 0.7,
        }

        py_result = py_fn(feat, 0.0, 0.5, 0)
        cy_result = cy_fn(feat, 0.0, 0.5, 0)

        assert py_result == cy_result

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython not compiled")
    def test_clip_line_identical(self):
        """_clip_line: Python == Cython."""
        py_fn = _clip_line_py
        cy_fn = _get_cy_clip_line()

        feat = {
            "geometry": [0.1, 0.5, 0.3, 0.5, 0.6, 0.5, 0.9, 0.5],
            "geometry_z": [0.0, 0.0, 0.0, 0.0],
            "type": 2, "tags": {"line": True},
            "minX": 0.1, "minY": 0.5, "minZ": 0.0,
            "maxX": 0.9, "maxY": 0.5, "maxZ": 0.0,
        }

        py_result = py_fn(feat, 0.0, 0.5, 0)
        cy_result = cy_fn(feat, 0.0, 0.5, 0)

        assert len(py_result) == len(cy_result)
        for py_seg, cy_seg in zip(py_result, cy_result):
            assert py_seg["geometry"] == pytest.approx(cy_seg["geometry"])
            assert py_seg["geometry_z"] == pytest.approx(cy_seg["geometry_z"])

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython not compiled")
    def test_build_indexed_mesh_identical(self):
        """_build_indexed_mesh: Python == Cython."""
        py_fn = _build_indexed_mesh_py
        cy_fn = _get_cy_build_indexed_mesh()

        xy = [0, 0, 100, 0, 50, 100, 0, 0,
              100, 0, 150, 100, 50, 100, 100, 0]
        z = [0, 0, 0, 0, 0, 0, 0, 0]

        py_pos, py_idx = py_fn(xy, z, [4, 4])
        cy_pos, cy_idx = cy_fn(xy, z, [4, 4])

        assert py_pos == cy_pos
        assert py_idx == cy_idx

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython not compiled")
    def test_encode_tile_3d_identical_tin(self):
        """encode_tile_3d: Python == Cython for TIN features."""
        py_fn = encode_tile_3d_py
        cy_fn = _get_cy_encode_tile_3d()

        tile_data = {
            "features": [
                {
                    "geometry": [0, 0, 100, 0, 50, 100, 0, 0],
                    "geometry_z": [0, 0, 0, 0],
                    "ring_lengths": [4],
                    "type": 5,  # TIN
                    "tags": {"name": "tri", "area": 5000},
                },
            ],
        }

        py_bytes = py_fn(tile_data)
        cy_bytes = cy_fn(tile_data)
        assert py_bytes == cy_bytes

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython not compiled")
    def test_encode_tile_3d_identical_mixed(self):
        """encode_tile_3d: Python == Cython for mixed geometry."""
        py_fn = encode_tile_3d_py
        cy_fn = _get_cy_encode_tile_3d()

        tile_data = {
            "features": [
                {
                    "geometry": [100, 200],
                    "geometry_z": [50],
                    "type": 1,  # POINT3D
                    "tags": {"id": 1},
                },
                {
                    "geometry": [0, 0, 100, 100, 200, 50],
                    "geometry_z": [10, 20, 30],
                    "type": 2,  # LINESTRING3D
                    "tags": {"id": 2, "name": "line"},
                },
                {
                    "geometry": [0, 0, 100, 0, 50, 100, 0, 0,
                                 100, 0, 200, 0, 150, 100, 100, 0],
                    "geometry_z": [0, 0, 0, 0, 0, 0, 0, 0],
                    "ring_lengths": [4, 4],
                    "type": 5,  # TIN
                    "tags": {"name": "mesh", "count": 2},
                },
            ],
        }

        py_bytes = py_fn(tile_data)
        cy_bytes = cy_fn(tile_data)
        assert py_bytes == cy_bytes

    @pytest.mark.skipif(not CYTHON_AVAILABLE, reason="Cython not compiled")
    def test_transform_tile_3d_identical(self):
        """transform_tile_3d: Python == Cython."""
        py_fn = transform_tile_3d_py
        cy_fn = _get_cy_transform_tile_3d()

        feat = {
            "geometry": [0.25, 0.25, 0.1, 0.3],
            "geometry_z": [0.25, 0.1],
            "type": 1,
            "tags": {"id": 1},
            "ring_lengths": None,
        }
        tile = {
            "features": [feat],
            "z": 2, "x": 1, "y": 1, "d": 1,
            "num_features": 1,
            "num_points": 2,
        }

        py_result = py_fn(tile)
        cy_result = cy_fn(tile)

        assert py_result == cy_result
