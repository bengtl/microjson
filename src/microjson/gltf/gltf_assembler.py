"""Assemble MicroJSON geometries into a glTF scene graph.

Dispatches each geometry type to the appropriate mesh/primitive builder
and wires them into pygltflib's GLTF2 object.
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np
from pygltflib import (
    ARRAY_BUFFER,
    ELEMENT_ARRAY_BUFFER,
    FLOAT,
    LINES,
    POINTS,
    TRIANGLES,
    UNSIGNED_INT,
    Asset,
    Attributes,
    Buffer,
    GLTF2,
    Material,
    Mesh,
    Node,
    PbrMetallicRoughness,
    Primitive,
    Scene,
)

from geojson_pydantic import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    MultiPolygon,
)

from ..model import (
    MicroFeature,
    MicroFeatureCollection,
    NeuronMorphology,
    PolyhedralSurface,
    SliceStack,
    TIN,
)
from ._buffers import create_accessor
from .mesh_builder import neuron_to_tube_mesh, neuron_to_typed_meshes
from .models import DEFAULT_SWC_FALLBACK_COLOR, GltfConfig
from .triangulator import multipolygon_to_mesh, polygon_to_mesh


# Z-up → Y-up rotation matrix (rotate -90° around X)
_Z_UP_TO_Y_UP = np.array(
    [1, 0, 0, 0,
     0, 0, 1, 0,
     0, -1, 0, 0,
     0, 0, 0, 1],
    dtype=np.float64,
).reshape(4, 4)


def _apply_y_up(positions: np.ndarray) -> np.ndarray:
    """Rotate Nx3 positions from Z-up to Y-up (swap Y↔Z, negate new Z)."""
    out = np.empty_like(positions)
    out[:, 0] = positions[:, 0]
    out[:, 1] = positions[:, 2]
    out[:, 2] = -positions[:, 1]
    return out


def _init_gltf(config: GltfConfig) -> GLTF2:
    """Create a fresh GLTF2 object with default material."""
    gltf = GLTF2()
    gltf._glb_data = None
    gltf.asset = Asset(version="2.0", generator="microjson-gltf")
    gltf.scenes = [Scene(nodes=[])]
    gltf.scene = 0

    r, g, b, a = config.default_color
    gltf.materials = [
        Material(
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=[r, g, b, a],
                metallicFactor=0.1,
                roughnessFactor=0.8,
            ),
            doubleSided=True,
        )
    ]
    return gltf


def _add_triangle_mesh(
    gltf: GLTF2,
    vertices: np.ndarray,
    indices: np.ndarray,
    normals: np.ndarray | None,
    config: GltfConfig,
    material_idx: int = 0,
) -> int:
    """Add a triangle mesh and return the mesh index."""
    if config.y_up:
        vertices = _apply_y_up(vertices)
        if normals is not None:
            normals = _apply_y_up(normals)

    verts_f32 = vertices.astype(np.float32)
    pos_acc = create_accessor(gltf, verts_f32, FLOAT, "VEC3", ARRAY_BUFFER)

    attrs = Attributes(POSITION=pos_acc)
    if normals is not None:
        norms_f32 = normals.astype(np.float32)
        norm_acc = create_accessor(gltf, norms_f32, FLOAT, "VEC3", ARRAY_BUFFER)
        attrs.NORMAL = norm_acc

    idx_flat = indices.ravel().astype(np.uint32)
    idx_acc = create_accessor(gltf, idx_flat, UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(
        Mesh(primitives=[Primitive(attributes=attrs, indices=idx_acc, material=material_idx, mode=TRIANGLES)])
    )
    return mesh_idx


def _add_line_primitive(
    gltf: GLTF2,
    vertices: np.ndarray,
    indices: np.ndarray,
    config: GltfConfig,
) -> int:
    """Add a line primitive and return the mesh index."""
    if config.y_up:
        vertices = _apply_y_up(vertices)

    verts_f32 = vertices.astype(np.float32)
    pos_acc = create_accessor(gltf, verts_f32, FLOAT, "VEC3", ARRAY_BUFFER)
    idx_flat = indices.ravel().astype(np.uint32)
    idx_acc = create_accessor(gltf, idx_flat, UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(
        Mesh(primitives=[Primitive(attributes=Attributes(POSITION=pos_acc), indices=idx_acc, material=0, mode=LINES)])
    )
    return mesh_idx


def _add_point_primitive(
    gltf: GLTF2,
    vertices: np.ndarray,
    config: GltfConfig,
) -> int:
    """Add a point primitive and return the mesh index."""
    if config.y_up:
        vertices = _apply_y_up(vertices)

    verts_f32 = vertices.astype(np.float32)
    pos_acc = create_accessor(gltf, verts_f32, FLOAT, "VEC3", ARRAY_BUFFER)

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(
        Mesh(primitives=[Primitive(attributes=Attributes(POSITION=pos_acc), material=0, mode=POINTS)])
    )
    return mesh_idx


def _add_node(
    gltf: GLTF2,
    mesh_idx: int,
    name: str | None = None,
    extras: dict | None = None,
    translation: list[float] | None = None,
) -> int:
    """Add a node referencing a mesh and return its index."""
    node = Node(mesh=mesh_idx)
    if name:
        node.name = name
    if extras:
        node.extras = extras
    if translation is not None:
        node.translation = translation
    node_idx = len(gltf.nodes)
    gltf.nodes.append(node)
    gltf.scenes[0].nodes.append(node_idx)
    return node_idx


# ---------------------------------------------------------------------------
# Geometry dispatchers
# ---------------------------------------------------------------------------

_SWC_TYPE_NAMES = {
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
}


def _ensure_type_material(
    gltf: GLTF2,
    swc_type: int,
    config: GltfConfig,
    material_cache: dict[int, int],
) -> int:
    """Get or create a material for the given SWC type. Returns material index."""
    if swc_type in material_cache:
        return material_cache[swc_type]

    color = config.swc_type_colors.get(swc_type, DEFAULT_SWC_FALLBACK_COLOR)
    r, g, b, a = color
    mat_idx = len(gltf.materials)
    gltf.materials.append(
        Material(
            name=_SWC_TYPE_NAMES.get(swc_type, f"type_{swc_type}"),
            pbrMetallicRoughness=PbrMetallicRoughness(
                baseColorFactor=[r, g, b, a],
                metallicFactor=0.1,
                roughnessFactor=0.8,
            ),
            doubleSided=True,
        )
    )
    material_cache[swc_type] = mat_idx
    return mat_idx


def _convert_neuron(
    gltf: GLTF2,
    geom: NeuronMorphology,
    config: GltfConfig,
) -> int | list[int]:
    """Convert neuron geometry, optionally colored by SWC type.

    Returns a single mesh index, or a list of mesh indices when
    ``config.color_by_type`` is True.
    """
    if not config.color_by_type:
        verts, normals, indices = neuron_to_tube_mesh(
            geom,
            segments=config.tube_segments,
            min_radius=config.tube_min_radius,
            smooth_subdivisions=config.smooth_factor,
            mesh_quality=config.mesh_quality,
        )
        return _add_triangle_mesh(gltf, verts, indices, normals, config)

    # Per-type meshes with distinct materials
    typed = neuron_to_typed_meshes(
        geom,
        segments=config.tube_segments,
        min_radius=config.tube_min_radius,
        smooth_subdivisions=config.smooth_factor,
        mesh_quality=config.mesh_quality,
    )
    material_cache: dict[int, int] = {}
    mesh_indices = []

    for swc_type in sorted(typed):
        verts, normals, indices = typed[swc_type]
        mat_idx = _ensure_type_material(gltf, swc_type, config, material_cache)
        mesh_idx = _add_triangle_mesh(gltf, verts, indices, normals, config, mat_idx)
        mesh_indices.append(mesh_idx)

    return mesh_indices


def _convert_tin(
    gltf: GLTF2,
    geom: TIN,
    config: GltfConfig,
) -> int:
    all_verts = []
    all_indices = []
    offset = 0
    for face_coords in geom.coordinates:
        ring = face_coords[0]  # TIN: exactly 1 ring per face, 4 positions (closed triangle)
        tri_verts = [list(p) for p in ring[:3]]  # drop closing vertex
        # Pad to 3D if needed
        for v in tri_verts:
            while len(v) < 3:
                v.append(0.0)
        all_verts.extend(tri_verts)
        all_indices.append([offset, offset + 1, offset + 2])
        offset += 3

    if not all_verts:
        return -1
    verts = np.array(all_verts, dtype=np.float64)
    indices = np.array(all_indices, dtype=np.uint32)
    return _add_triangle_mesh(gltf, verts, indices, None, config)


def _convert_polyhedral_surface(
    gltf: GLTF2,
    geom: PolyhedralSurface,
    config: GltfConfig,
) -> int:
    all_verts: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    offset = 0

    for face_coords in geom.coordinates:
        # Each face is a list of rings (outer + optional holes)
        from shapely.geometry import Polygon as ShapelyPolygon

        outer = [(p[0], p[1]) for p in face_coords[0]]
        holes = [[(p[0], p[1]) for p in ring] for ring in face_coords[1:]]
        z = face_coords[0][0][2] if len(face_coords[0][0]) > 2 else 0.0

        poly = ShapelyPolygon(outer, holes)
        verts, idxs = polygon_to_mesh(poly, z=z)
        if verts.shape[0] == 0:
            continue

        # If original coords had 3D, use their Z values
        if len(face_coords[0][0]) > 2:
            # Map unique 2D positions back to 3D from original coords
            all_pts_3d = []
            for ring in face_coords:
                for p in ring:
                    all_pts_3d.append(list(p) + [0.0] * (3 - len(p)))
            # For each vertex in the triangulated output, find nearest original Z
            for i in range(verts.shape[0]):
                vx, vy = verts[i, 0], verts[i, 1]
                best_z = z
                best_dist = float("inf")
                for pt in all_pts_3d:
                    d = (pt[0] - vx) ** 2 + (pt[1] - vy) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_z = pt[2]
                verts[i, 2] = best_z

        all_verts.append(verts)
        all_indices.append(idxs + offset)
        offset += verts.shape[0]

    if not all_verts:
        return -1
    return _add_triangle_mesh(
        gltf,
        np.concatenate(all_verts),
        np.concatenate(all_indices),
        None,
        config,
    )


def _convert_polygon(
    gltf: GLTF2,
    geom: Polygon,
    config: GltfConfig,
) -> int:
    from shapely.geometry import Polygon as ShapelyPolygon

    coords = geom.coordinates
    outer = [(p[0], p[1]) for p in coords[0]]
    holes = [[(p[0], p[1]) for p in ring] for ring in coords[1:]]
    z = coords[0][0][2] if len(coords[0][0]) > 2 else 0.0

    poly = ShapelyPolygon(outer, holes)
    verts, indices = polygon_to_mesh(poly, z=z)
    if verts.shape[0] == 0:
        return -1
    return _add_triangle_mesh(gltf, verts, indices, None, config)


def _convert_multipolygon(
    gltf: GLTF2,
    geom: MultiPolygon,
    config: GltfConfig,
) -> int:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import MultiPolygon as ShapelyMultiPolygon

    polys = []
    for poly_coords in geom.coordinates:
        outer = [(p[0], p[1]) for p in poly_coords[0]]
        holes = [[(p[0], p[1]) for p in ring] for ring in poly_coords[1:]]
        polys.append(ShapelyPolygon(outer, holes))

    mp = ShapelyMultiPolygon(polys)
    z = geom.coordinates[0][0][0][2] if len(geom.coordinates[0][0][0]) > 2 else 0.0
    verts, indices = multipolygon_to_mesh(mp, z=z)
    if verts.shape[0] == 0:
        return -1
    return _add_triangle_mesh(gltf, verts, indices, None, config)


def _convert_slice_stack(
    gltf: GLTF2,
    geom: SliceStack,
    config: GltfConfig,
) -> int:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import MultiPolygon as ShapelyMultiPolygon

    all_verts: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    offset = 0

    for slc in geom.slices:
        slice_geom = slc.geometry
        if isinstance(slice_geom, Polygon):
            outer = [(p[0], p[1]) for p in slice_geom.coordinates[0]]
            holes = [[(p[0], p[1]) for p in ring] for ring in slice_geom.coordinates[1:]]
            poly = ShapelyPolygon(outer, holes)
            verts, idxs = polygon_to_mesh(poly, z=slc.z)
        elif isinstance(slice_geom, MultiPolygon):
            polys = []
            for pc in slice_geom.coordinates:
                outer = [(p[0], p[1]) for p in pc[0]]
                holes = [[(p[0], p[1]) for p in ring] for ring in pc[1:]]
                polys.append(ShapelyPolygon(outer, holes))
            mp = ShapelyMultiPolygon(polys)
            verts, idxs = multipolygon_to_mesh(mp, z=slc.z)
        else:
            continue

        if verts.shape[0] == 0:
            continue
        all_verts.append(verts)
        all_indices.append(idxs + offset)
        offset += verts.shape[0]

    if not all_verts:
        return -1
    return _add_triangle_mesh(
        gltf,
        np.concatenate(all_verts),
        np.concatenate(all_indices),
        None,
        config,
    )


def _convert_linestring(
    gltf: GLTF2,
    geom: LineString,
    config: GltfConfig,
) -> int:
    coords = geom.coordinates
    verts = []
    for p in coords:
        verts.append([p[0], p[1], p[2] if len(p) > 2 else 0.0])
    if len(verts) < 2:
        return -1
    verts_arr = np.array(verts, dtype=np.float64)

    # Build line segment indices (pairs of consecutive vertices)
    indices = []
    for i in range(len(verts) - 1):
        indices.extend([i, i + 1])
    indices_arr = np.array(indices, dtype=np.uint32)

    return _add_line_primitive(gltf, verts_arr, indices_arr, config)


def _convert_multilinestring(
    gltf: GLTF2,
    geom: MultiLineString,
    config: GltfConfig,
) -> int:
    all_verts = []
    all_indices = []
    offset = 0
    for line_coords in geom.coordinates:
        verts = [[p[0], p[1], p[2] if len(p) > 2 else 0.0] for p in line_coords]
        if len(verts) < 2:
            continue
        for i in range(len(verts) - 1):
            all_indices.extend([offset + i, offset + i + 1])
        all_verts.extend(verts)
        offset += len(verts)

    if not all_verts:
        return -1
    return _add_line_primitive(
        gltf,
        np.array(all_verts, dtype=np.float64),
        np.array(all_indices, dtype=np.uint32),
        config,
    )


def _convert_point(
    gltf: GLTF2,
    geom: Point,
    config: GltfConfig,
) -> int:
    c = geom.coordinates
    z = c[2] if len(c) > 2 else 0.0
    verts = np.array([[c[0], c[1], z]], dtype=np.float64)
    return _add_point_primitive(gltf, verts, config)


def _convert_multipoint(
    gltf: GLTF2,
    geom: MultiPoint,
    config: GltfConfig,
) -> int:
    verts = []
    for c in geom.coordinates:
        z = c[2] if len(c) > 2 else 0.0
        verts.append([c[0], c[1], z])
    if not verts:
        return -1
    return _add_point_primitive(gltf, np.array(verts, dtype=np.float64), config)


# ---------------------------------------------------------------------------
# Bounding box helpers (for feature spacing / grid layout)
# ---------------------------------------------------------------------------

# Type alias: (x_min, x_max, y_min, y_max, z_min, z_max)
_Bounds = tuple[float, float, float, float, float, float]


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


def _geometry_bounds(geom: Any) -> _Bounds | None:
    """Return 3-D bounding box or *None* if the geometry is empty."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    if isinstance(geom, NeuronMorphology):
        for s in geom.tree:
            xs.append(s.x)
            ys.append(s.y)
            zs.append(s.z)
    elif isinstance(geom, SliceStack):
        for slc in geom.slices:
            if hasattr(slc.geometry, "coordinates"):
                _collect_xyz_from_coords(
                    slc.geometry.coordinates, xs, ys, zs,
                )
            # Include slice Z
            zs.append(slc.z)
    elif hasattr(geom, "coordinates"):
        _collect_xyz_from_coords(geom.coordinates, xs, ys, zs)

    if not xs:
        return None
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

_Offset3 = tuple[float, float, float]


def _row_layout(
    bounds: list[_Bounds | None],
    config: GltfConfig,
) -> list[_Offset3]:
    """Place features side-by-side along X (original behaviour)."""
    n = len(bounds)
    widths = [(b[1] - b[0]) if b else 0.0 for b in bounds]
    max_width = max(widths) if widths else 0.0
    gap = (
        config.feature_spacing
        if config.feature_spacing > 0
        else max_width * 0.2
    )

    offsets: list[_Offset3] = [(0.0, 0.0, 0.0)] * n
    cursor = bounds[0][1] if bounds[0] else gap

    for i in range(1, n):
        b = bounds[i]
        if b is None:
            dx = cursor + gap
            offsets[i] = (dx, 0.0, 0.0)
            cursor = dx
            continue
        dx = (cursor + gap) - b[0]
        offsets[i] = (dx, 0.0, 0.0)
        cursor = b[1] + dx

    return offsets


def _grid_layout(
    bounds: list[_Bounds | None],
    config: GltfConfig,
    n: int,
) -> list[_Offset3]:
    """Place features on a uniform grid that wraps X → Y → Z.

    ``grid_max_x/y/z`` give the number of cells per axis directly
    (e.g. 3 × 3 × 3 = 27 cells).
    """
    widths = [(b[1] - b[0]) if b else 0.0 for b in bounds]
    heights = [(b[3] - b[2]) if b else 0.0 for b in bounds]
    depths = [(b[5] - b[4]) if b else 0.0 for b in bounds]

    max_w = max(widths) if widths else 0.0
    max_h = max(heights) if heights else 0.0
    max_d = max(depths) if depths else 0.0

    max_extent = max(max_w, max_h, max_d)
    gap = (
        config.feature_spacing
        if config.feature_spacing > 0
        else (max_extent * 0.2 if max_extent > 0 else 1.0)
    )

    cell_x = max_w + gap
    cell_y = max_h + gap
    cell_z = max_d + gap

    # Grid dimensions — values are cell counts, unlimited when None
    cols = config.grid_max_x if config.grid_max_x is not None else n
    rows = (
        config.grid_max_y
        if config.grid_max_y is not None
        else max(1, -(-n // cols))
    )
    layers = (
        config.grid_max_z
        if config.grid_max_z is not None
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
    ref_cx = (ref[0] + ref[1]) / 2 if ref else 0.0
    ref_cy = (ref[2] + ref[3]) / 2 if ref else 0.0
    ref_cz = (ref[4] + ref[5]) / 2 if ref else 0.0

    offsets: list[_Offset3] = [(0.0, 0.0, 0.0)]

    for i in range(1, n):
        col = i % cols
        row = (i // cols) % rows
        layer = i // (cols * rows)

        target_cx = ref_cx + col * cell_x
        target_cy = ref_cy + row * cell_y
        target_cz = ref_cz + layer * cell_z

        b = bounds[i]
        feat_cx = (b[0] + b[1]) / 2 if b else 0.0
        feat_cy = (b[2] + b[3]) / 2 if b else 0.0
        feat_cz = (b[4] + b[5]) / 2 if b else 0.0

        offsets.append((
            target_cx - feat_cx,
            target_cy - feat_cy,
            target_cz - feat_cz,
        ))

    return offsets


def _compute_feature_offsets(
    features: list,
    config: GltfConfig,
) -> list[_Offset3]:
    """Compute 3-D translation for each feature in a collection.

    Returns a list of ``(dx, dy, dz)`` offsets in **source** coordinates.
    The first feature always stays at its original position.

    Raises ``ValueError`` when all three ``grid_max_*`` values are set and
    the features cannot fit (checked before any mesh generation).
    """
    n = len(features)
    if n <= 1:
        return [(0.0, 0.0, 0.0)] * n

    bounds = [
        _geometry_bounds(f.geometry) if f.geometry else None
        for f in features
    ]

    has_grid = (
        config.grid_max_x is not None
        or config.grid_max_y is not None
        or config.grid_max_z is not None
    )

    if has_grid:
        return _grid_layout(bounds, config, n)
    return _row_layout(bounds, config)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _convert_geometry(
    gltf: GLTF2,
    geom: Any,
    config: GltfConfig,
) -> int | list[int]:
    """Dispatch a geometry to its converter. Returns mesh index(es) or -1."""
    if isinstance(geom, NeuronMorphology):
        return _convert_neuron(gltf, geom, config)
    elif isinstance(geom, TIN):
        return _convert_tin(gltf, geom, config)
    elif isinstance(geom, PolyhedralSurface):
        return _convert_polyhedral_surface(gltf, geom, config)
    elif isinstance(geom, SliceStack):
        return _convert_slice_stack(gltf, geom, config)
    elif isinstance(geom, Polygon):
        return _convert_polygon(gltf, geom, config)
    elif isinstance(geom, MultiPolygon):
        return _convert_multipolygon(gltf, geom, config)
    elif isinstance(geom, LineString):
        return _convert_linestring(gltf, geom, config)
    elif isinstance(geom, MultiLineString):
        return _convert_multilinestring(gltf, geom, config)
    elif isinstance(geom, Point):
        return _convert_point(gltf, geom, config)
    elif isinstance(geom, MultiPoint):
        return _convert_multipoint(gltf, geom, config)
    return -1


def _add_nodes_for_mesh(
    gltf: GLTF2,
    mesh_result: int | list[int],
    name: str | None = None,
    extras: dict | None = None,
    translation: list[float] | None = None,
) -> None:
    """Add node(s) for one or more meshes returned by a converter."""
    if isinstance(mesh_result, list):
        for j, midx in enumerate(mesh_result):
            node_name = f"{name}_{j}" if name else None
            _add_node(gltf, midx, name=node_name,
                      extras=extras if j == 0 else None,
                      translation=translation)
    elif mesh_result >= 0:
        _add_node(gltf, mesh_result, name=name, extras=extras,
                  translation=translation)


def feature_to_gltf(
    feature: MicroFeature,
    config: GltfConfig | None = None,
) -> GLTF2:
    """Convert a single MicroFeature to a glTF scene.

    Args:
        feature: The MicroJSON feature to convert.
        config: Export configuration (uses defaults if None).

    Returns:
        A pygltflib GLTF2 object.
    """
    if config is None:
        config = GltfConfig()

    gltf = _init_gltf(config)

    geom = feature.geometry
    if geom is None:
        return gltf

    mesh_result = _convert_geometry(gltf, geom, config)

    extras = None
    if config.include_metadata and feature.properties:
        extras = dict(feature.properties)

    _add_nodes_for_mesh(gltf, mesh_result, extras=extras)
    return gltf


def collection_to_gltf(
    collection: MicroFeatureCollection,
    config: GltfConfig | None = None,
) -> GLTF2:
    """Convert a MicroFeatureCollection to a glTF scene.

    Each feature becomes a separate node in the scene.

    Args:
        collection: The feature collection.
        config: Export configuration.

    Returns:
        A pygltflib GLTF2 object.
    """
    if config is None:
        config = GltfConfig()

    gltf = _init_gltf(config)

    # Compute layout offsets *before* any mesh generation so that a
    # capacity error is raised early.
    offsets = _compute_feature_offsets(list(collection.features), config)

    for i, feature in enumerate(collection.features):
        geom = feature.geometry
        if geom is None:
            continue

        mesh_result = _convert_geometry(gltf, geom, config)

        extras = None
        if config.include_metadata and feature.properties:
            extras = dict(feature.properties)

        # Build glTF-space translation from the source-space offset
        dx, dy, dz = offsets[i]
        translation = None
        if abs(dx) > 1e-12 or abs(dy) > 1e-12 or abs(dz) > 1e-12:
            if config.y_up:
                translation = [dx, dz, -dy]
            else:
                translation = [dx, dy, dz]

        _add_nodes_for_mesh(gltf, mesh_result, name=f"feature_{i}",
                            extras=extras, translation=translation)

    # Store collection-level metadata in scene extras
    if config.include_metadata and collection.properties:
        gltf.scenes[0].extras = dict(collection.properties)

    return gltf
