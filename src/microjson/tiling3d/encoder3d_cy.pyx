# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-optimized 3D tile encoder.

Drop-in replacements for _build_indexed_mesh and encode_tile_3d
in encoder3d.py.  All encoding helpers (_zigzag, _command,
_encode_*_geometry, _encode_z) are inlined as cdef for zero
function-call overhead.
"""

import struct
from microjson.tiling3d.proto import microjson_3d_tile_pb2 as pb


# ---------------------------------------------------------------------------
# cdef helpers — invisible to Python, zero call overhead
# ---------------------------------------------------------------------------

cdef inline int _zigzag(int n):
    return (n << 1) ^ (n >> 31)


cdef inline int _command(int cmd, int count):
    return (count << 3) | (cmd & 0x7)


cdef list _encode_point_geometry(list xy):
    cdef int n = len(xy) // 2
    cdef list result = [_command(1, n)]
    cdef int x = 0, y = 0, dx, dy, i
    for i in range(n):
        dx = <int>xy[i * 2] - x
        dy = <int>xy[i * 2 + 1] - y
        result.append(_zigzag(dx))
        result.append(_zigzag(dy))
        x += dx
        y += dy
    return result


cdef list _encode_line_geometry(list xy):
    cdef int n = len(xy) // 2
    if n < 2:
        return []
    cdef list result = []
    cdef int x = 0, y = 0, dx, dy, i

    result.append(_command(1, 1))
    dx = <int>xy[0] - x
    dy = <int>xy[1] - y
    result.append(_zigzag(dx))
    result.append(_zigzag(dy))
    x += dx
    y += dy

    result.append(_command(2, n - 1))
    for i in range(1, n):
        dx = <int>xy[i * 2] - x
        dy = <int>xy[i * 2 + 1] - y
        result.append(_zigzag(dx))
        result.append(_zigzag(dy))
        x += dx
        y += dy
    return result


cdef list _encode_polygon_geometry(list xy, list ring_lengths):
    cdef list result = []
    cdef int x = 0, y = 0, dx, dy
    cdef int offset = 0, ring_len, line_count, i, idx

    if ring_lengths is None:
        ring_lengths = [len(xy) // 2]

    for ring_len in ring_lengths:
        if ring_len < 3:
            offset += ring_len
            continue

        result.append(_command(1, 1))
        dx = <int>xy[offset * 2] - x
        dy = <int>xy[offset * 2 + 1] - y
        result.append(_zigzag(dx))
        result.append(_zigzag(dy))
        x += dx
        y += dy

        line_count = ring_len - 2
        if line_count > 0:
            result.append(_command(2, line_count))
            for i in range(1, ring_len - 1):
                idx = offset + i
                dx = <int>xy[idx * 2] - x
                dy = <int>xy[idx * 2 + 1] - y
                result.append(_zigzag(dx))
                result.append(_zigzag(dy))
                x += dx
                y += dy

        result.append(_command(7, 1))
        offset += ring_len
    return result


cdef list _encode_z(list z_coords):
    cdef int n = len(z_coords)
    if n == 0:
        return []
    cdef list result = [z_coords[0]]
    cdef int i
    for i in range(1, n):
        result.append(<int>z_coords[i] - <int>z_coords[i - 1])
    return result


cdef void _write_value(object pb_value, object value):
    if isinstance(value, bool):
        pb_value.bool_value = value
    elif isinstance(value, str):
        pb_value.string_value = value
    elif isinstance(value, float):
        pb_value.double_value = value
    elif isinstance(value, int):
        if <int>value < 0:
            pb_value.sint_value = value
        else:
            pb_value.uint_value = value


# ---------------------------------------------------------------------------
# _build_indexed_mesh — unchanged from previous version
# ---------------------------------------------------------------------------

def _build_indexed_mesh(
    list xy,
    list z_coords,
    list ring_lengths_arg,
):
    """Build indexed triangle mesh from ring-based face data."""
    cdef list ring_lengths
    if ring_lengths_arg is None:
        ring_lengths = [len(xy) // 2]
    else:
        ring_lengths = ring_lengths_arg

    cdef dict vertex_map = {}
    cdef list positions = []
    cdef list indices = []

    cdef int offset = 0
    cdef int ring_len, n_verts, i, vi
    cdef int x, y, z
    cdef tuple key
    cdef int idx
    cdef int ti0, ti1, ti2

    for ring_len in ring_lengths:
        if ring_len >= 4:
            n_verts = 3
        else:
            n_verts = ring_len

        if n_verts == 3:
            ti0 = -1; ti1 = -1; ti2 = -1

            vi = offset
            x = <int>xy[vi * 2]
            y = <int>xy[vi * 2 + 1]
            z = <int>z_coords[vi] if vi < len(z_coords) else 0
            key = (x, y, z)
            if key not in vertex_map:
                idx = len(vertex_map)
                vertex_map[key] = idx
                positions.append(<double>x)
                positions.append(<double>y)
                positions.append(<double>z)
            else:
                idx = <int>vertex_map[key]
            ti0 = idx

            vi = offset + 1
            x = <int>xy[vi * 2]
            y = <int>xy[vi * 2 + 1]
            z = <int>z_coords[vi] if vi < len(z_coords) else 0
            key = (x, y, z)
            if key not in vertex_map:
                idx = len(vertex_map)
                vertex_map[key] = idx
                positions.append(<double>x)
                positions.append(<double>y)
                positions.append(<double>z)
            else:
                idx = <int>vertex_map[key]
            ti1 = idx

            vi = offset + 2
            x = <int>xy[vi * 2]
            y = <int>xy[vi * 2 + 1]
            z = <int>z_coords[vi] if vi < len(z_coords) else 0
            key = (x, y, z)
            if key not in vertex_map:
                idx = len(vertex_map)
                vertex_map[key] = idx
                positions.append(<double>x)
                positions.append(<double>y)
                positions.append(<double>z)
            else:
                idx = <int>vertex_map[key]
            ti2 = idx

            indices.append(ti0)
            indices.append(ti1)
            indices.append(ti2)

        offset += ring_len

    cdef bytes pos_bytes, idx_bytes

    if positions:
        pos_bytes = struct.pack(f"<{len(positions)}f", *positions)
    else:
        pos_bytes = b""

    if indices:
        idx_bytes = struct.pack(f"<{len(indices)}I", *indices)
    else:
        idx_bytes = b""

    return pos_bytes, idx_bytes


# ---------------------------------------------------------------------------
# Protobuf enum constants (matches GeomType in proto)
# ---------------------------------------------------------------------------

cdef int _PB_UNKNOWN = 0
cdef int _PB_POINT3D = 1
cdef int _PB_LINESTRING3D = 2
cdef int _PB_POLYGON3D = 3
cdef int _PB_POLYHEDRALSURFACE = 4
cdef int _PB_TIN = 5


# GeomType mapping (internal type int → protobuf enum)
# Internal types match proto enum values exactly (1-5)
_GEOM_TYPE_MAP = {
    1: _PB_POINT3D,
    2: _PB_LINESTRING3D,
    3: _PB_POLYGON3D,
    4: _PB_POLYHEDRALSURFACE,
    5: _PB_TIN,
}


# ---------------------------------------------------------------------------
# encode_tile_3d — Cython version
# ---------------------------------------------------------------------------

def encode_tile_3d(
    dict tile_data,
    str layer_name="default",
    int extent=4096,
    int extent_z=4096,
):
    """Encode a transformed tile to protobuf bytes.

    Cython version with typed loops, inlined encoding helpers,
    and minimized Python object creation.
    """
    tile_pb = pb.Tile()
    layer = tile_pb.layers.add()
    layer.name = layer_name
    layer.version = 3
    layer.extent = extent
    layer.extent_z = extent_z

    cdef dict key_indices = {}
    cdef dict value_indices = {}

    cdef list features = tile_data.get("features", [])
    cdef int n_features = len(features)
    cdef int feat_idx, gt
    cdef dict feat, tags
    cdef list xy, z_coords
    cdef list ring_lengths
    cdef list encoded
    cdef bytes pos_bytes, idx_bytes
    cdef str k
    cdef object v
    cdef int ki, vi_idx

    # Cache protobuf container references (avoid repeated attribute lookups)
    layer_keys = layer.keys
    layer_values = layer.values
    layer_features = layer.features

    for feat_idx in range(n_features):
        feat = <dict>features[feat_idx]

        pb_feat = layer_features.add()
        pb_feat.id = feat_idx

        gt = feat.get("type", 0)
        pb_feat.type = _GEOM_TYPE_MAP.get(gt, _PB_UNKNOWN)

        # Encode tags
        tags = feat.get("tags", {})
        pb_feat_tags = pb_feat.tags
        for k, v in tags.items():
            if v is None:
                continue

            if k not in key_indices:
                ki = len(key_indices)
                key_indices[k] = ki
                layer_keys.append(k)
            else:
                ki = <int>key_indices[k]
            pb_feat_tags.append(ki)

            if v not in value_indices:
                vi_idx = len(value_indices)
                value_indices[v] = vi_idx
                _write_value(layer_values.add(), v)
            else:
                vi_idx = <int>value_indices[v]
            pb_feat_tags.append(vi_idx)

        # Encode geometry
        xy = feat.get("geometry", [])
        z_coords = feat.get("geometry_z", [])
        ring_lengths = feat.get("ring_lengths")

        if gt == 1:  # POINT3D
            pb_feat.geometry.extend(_encode_point_geometry(xy))
            pb_feat.geometry_z.extend(_encode_z(z_coords))
        elif gt == 2:  # LINESTRING3D
            pb_feat.geometry.extend(_encode_line_geometry(xy))
            pb_feat.geometry_z.extend(_encode_z(z_coords))
        elif gt == 4 or gt == 5:  # POLYHEDRALSURFACE, TIN — indexed mesh
            pos_bytes, idx_bytes = _build_indexed_mesh(xy, z_coords, ring_lengths)
            pb_feat.mesh_positions = pos_bytes
            pb_feat.mesh_indices = idx_bytes
        elif gt == 3:  # POLYGON3D — ring-based
            pb_feat.geometry.extend(_encode_polygon_geometry(xy, ring_lengths))
            pb_feat.geometry_z.extend(_encode_z(z_coords))
        else:
            pb_feat.geometry.extend(_encode_line_geometry(xy))
            pb_feat.geometry_z.extend(_encode_z(z_coords))

        # Per-vertex radii
        radii = feat.get("radii")
        if radii:
            pb_feat.radii.extend(radii)

    return tile_pb.SerializeToString()
