"""Generate triangle meshes from NeuronMorphology trees.

Produces continuous tubes along each branch path using parallel transport
frames (no gaps at intermediate nodes), icospheres at soma and branch
junctions.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..model import NeuronMorphology


def _catmull_rom(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    n: int,
) -> list[np.ndarray]:
    """Evaluate Catmull-Rom spline between p1 and p2.

    Returns *n* evenly spaced points **excluding** p1 and **including** p2.
    """
    pts: list[np.ndarray] = []
    for i in range(1, n + 1):
        t = i / n
        t2 = t * t
        t3 = t2 * t
        v = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        pts.append(v)
    return pts


def smooth_path(
    points: list[np.ndarray],
    radii: list[float],
    subdivisions: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Smooth a polyline path using Catmull-Rom spline interpolation.

    Inserts *subdivisions* intermediate samples between each pair of
    original points.  Radii are linearly interpolated.  The output always
    passes through the original control points.

    Args:
        points: Original 3D positions (length >= 2).
        radii: Radius at each original position.
        subdivisions: Number of points to insert per segment (0 = no-op).

    Returns:
        ``(smooth_points, smooth_radii)``
    """
    if subdivisions <= 0 or len(points) < 2:
        return points, radii

    n = len(points)
    out_pts: list[np.ndarray] = [points[0]]
    out_r: list[float] = [radii[0]]

    for i in range(n - 1):
        # Catmull-Rom needs 4 control points; clamp at boundaries
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, n - 1)]

        interp = _catmull_rom(p0, p1, p2, p3, subdivisions + 1)
        for j, pt in enumerate(interp):
            t = (j + 1) / (subdivisions + 1)
            r = radii[i] * (1 - t) + radii[i + 1] * t
            out_pts.append(pt)
            out_r.append(r)

    return out_pts, out_r


def thin_path(
    points: list[np.ndarray],
    radii: list[float],
    keep_fraction: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Uniformly subsample a path to reduce vertex count.

    Intended to run *after* :func:`smooth_path` so the dense spline is
    thinned while the remaining points still lie on the smooth curve.

    Args:
        points: Path positions (length >= 2).
        radii: Radius at each position.
        keep_fraction: Fraction of points to retain (0.0–1.0).
            ``1.0`` keeps all points.

    Returns:
        ``(thinned_points, thinned_radii)``
    """
    n = len(points)
    if keep_fraction >= 1.0 or n <= 2:
        return points, radii

    target = max(2, round(n * keep_fraction))
    if target >= n:
        return points, radii

    indices = np.round(np.linspace(0, n - 1, target)).astype(int)
    return [points[i] for i in indices], [radii[i] for i in indices]


def _find_perp(tangent: np.ndarray) -> np.ndarray:
    """Find an arbitrary unit vector perpendicular to *tangent*."""
    if abs(tangent[0]) < 0.9:
        perp = np.cross(tangent, np.array([1.0, 0.0, 0.0]))
    else:
        perp = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
    return perp / np.linalg.norm(perp)


def _tube_along_path(
    points: list[np.ndarray],
    radii: list[float],
    segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a continuous tube mesh along a polyline.

    Uses parallel transport frame for smooth, consistent cross-section
    orientation.  Ring vertices are shared at intermediate nodes,
    eliminating gaps between consecutive segments.

    Args:
        points: Ordered 3D positions along the path (length >= 2).
        radii: Radius at each point.
        segments: Number of sides in the cross-section polygon.

    Returns:
        ``(vertices, normals, indices)`` — Nx3, Nx3, Mx3 arrays.
    """
    n = len(points)
    if n < 2:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint32),
        )

    # Compute tangent at each point
    tangents = []
    for i in range(n):
        if i == 0:
            t = points[1] - points[0]
        elif i == n - 1:
            t = points[-1] - points[-2]
        else:
            # Average of incoming and outgoing directions
            t = points[i + 1] - points[i - 1]
        length = np.linalg.norm(t)
        if length < 1e-12:
            t = tangents[-1] if tangents else np.array([0.0, 0.0, 1.0])
        else:
            t = t / length
        tangents.append(t)

    # Build rings using parallel transport frame
    angles = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    all_verts = np.empty((n * segments, 3), dtype=np.float64)
    all_normals = np.empty((n * segments, 3), dtype=np.float64)

    perp = _find_perp(tangents[0])

    for i in range(n):
        if i > 0:
            # Parallel transport: project previous perp onto the plane
            # perpendicular to the new tangent
            perp = perp - np.dot(perp, tangents[i]) * tangents[i]
            norm = np.linalg.norm(perp)
            if norm < 1e-10:
                perp = _find_perp(tangents[i])
            else:
                perp = perp / norm

        perp2 = np.cross(tangents[i], perp)
        base = i * segments

        for j in range(segments):
            direction = perp * cos_a[j] + perp2 * sin_a[j]
            all_verts[base + j] = points[i] + direction * radii[i]
            all_normals[base + j] = direction

    # Connect consecutive rings with triangle strips
    indices = []
    for ring_i in range(n - 1):
        base0 = ring_i * segments
        base1 = (ring_i + 1) * segments
        for j in range(segments):
            j_next = (j + 1) % segments
            indices.append([base0 + j, base1 + j, base1 + j_next])
            indices.append([base0 + j, base1 + j_next, base0 + j_next])

    return all_verts, all_normals, np.array(indices, dtype=np.uint32)


def _extract_paths(
    tree: list,
    id_to_sample: dict,
) -> list[list[int]]:
    """Extract linear branch paths from the neuron tree.

    Each path is a maximal sequence of nodes where internal nodes have
    exactly one child.  Paths share endpoints at branch points.

    Returns:
        List of paths, where each path is a list of sample IDs.
    """
    children: dict[int, list[int]] = defaultdict(list)
    roots: list[int] = []

    for s in tree:
        if s.parent == -1:
            roots.append(s.id)
        elif s.parent in id_to_sample:
            children[s.parent].append(s.id)

    paths: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()
    stack = list(roots)

    while stack:
        node_id = stack.pop()
        kids = children[node_id]

        for kid in kids:
            if (node_id, kid) in visited_edges:
                continue

            # Trace a chain from node_id through kid
            chain = [node_id, kid]
            visited_edges.add((node_id, kid))
            current = kid

            while len(children[current]) == 1:
                child = children[current][0]
                visited_edges.add((current, child))
                chain.append(child)
                current = child

            paths.append(chain)

            # If chain ends at a branch point, continue from there
            if len(children[current]) > 1:
                stack.append(current)

    return paths


def _icosphere(
    center: np.ndarray,
    radius: float,
    subdivisions: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an icosphere mesh.

    Args:
        center: Center point (3,).
        radius: Sphere radius.
        subdivisions: Number of subdivision iterations (0=icosahedron).

    Returns:
        ``(vertices, normals, indices)`` — Nx3, Nx3, Mx3 arrays.
    """
    if radius < 1e-12:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint32),
        )

    # Start with icosahedron
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw_verts = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    verts_list = []
    for v in raw_verts:
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        verts_list.append([v[0] / n, v[1] / n, v[2] / n])

    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_faces = []
        for tri in faces:
            mids = []
            for edge in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
                key = (min(edge), max(edge))
                if key not in midpoint_cache:
                    v0 = verts_list[edge[0]]
                    v1 = verts_list[edge[1]]
                    mid = [(v0[i] + v1[i]) / 2 for i in range(3)]
                    n = math.sqrt(mid[0] ** 2 + mid[1] ** 2 + mid[2] ** 2)
                    mid = [mid[i] / n for i in range(3)]
                    midpoint_cache[key] = len(verts_list)
                    verts_list.append(mid)
                mids.append(midpoint_cache[key])
            new_faces.append([tri[0], mids[0], mids[2]])
            new_faces.append([tri[1], mids[1], mids[0]])
            new_faces.append([tri[2], mids[2], mids[1]])
            new_faces.append([mids[0], mids[1], mids[2]])
        faces = new_faces

    verts_arr = np.array(verts_list, dtype=np.float64)
    normals_arr = verts_arr.copy()  # On unit sphere, vertex == normal
    verts_arr = verts_arr * radius + center

    return verts_arr, normals_arr, np.array(faces, dtype=np.uint32)


def neuron_to_tube_mesh(
    morphology: NeuronMorphology,
    segments: int = 8,
    min_radius: float = 0.1,
    smooth_subdivisions: int = 0,
    mesh_quality: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a NeuronMorphology into a tube mesh with sphere soma.

    Traces linear branch paths through the tree and generates continuous
    tubes using parallel transport frames (no gaps at intermediate nodes).
    Adds icospheres at soma and non-soma branch junctions.

    Args:
        morphology: The neuron morphology tree.
        segments: Cross-section polygon sides for tubes.
        min_radius: Minimum radius clamp.
        smooth_subdivisions: Catmull-Rom subdivisions per segment (0 = off).
        mesh_quality: Fraction of smoothed path points to keep (0.0–1.0).

    Returns:
        ``(vertices, normals, indices)`` — Nx3, Nx3, Mx3 arrays.
    """
    tree = morphology.tree
    id_to_sample = {s.id: s for s in tree}

    # Build children map for junction detection
    children: dict[int, list[int]] = defaultdict(list)
    for s in tree:
        if s.parent != -1 and s.parent in id_to_sample:
            children[s.parent].append(s.id)

    all_verts: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    vertex_offset = 0

    # 1. Soma spheres (type==1, root)
    for sample in tree:
        if sample.type == 1 and sample.parent == -1:
            center = np.array([sample.x, sample.y, sample.z])
            r = max(sample.r, min_radius)
            v, n, idx = _icosphere(center, r, subdivisions=2)
            if v.shape[0] > 0:
                all_verts.append(v)
                all_normals.append(n)
                all_indices.append(idx + vertex_offset)
                vertex_offset += v.shape[0]

    # 2. Junction spheres at non-soma branch points (>1 child)
    for sample in tree:
        if sample.type == 1 and sample.parent == -1:
            continue  # Already handled as soma
        if len(children[sample.id]) > 1:
            center = np.array([sample.x, sample.y, sample.z])
            r = max(sample.r, min_radius)
            v, n, idx = _icosphere(center, r, subdivisions=1)
            if v.shape[0] > 0:
                all_verts.append(v)
                all_normals.append(n)
                all_indices.append(idx + vertex_offset)
                vertex_offset += v.shape[0]

    # 3. Continuous tubes along each branch path
    paths = _extract_paths(tree, id_to_sample)
    for path_ids in paths:
        points = []
        radii = []
        for sid in path_ids:
            s = id_to_sample[sid]
            points.append(np.array([s.x, s.y, s.z]))
            radii.append(max(s.r, min_radius))

        if smooth_subdivisions > 0:
            points, radii = smooth_path(points, radii, smooth_subdivisions)
        if mesh_quality < 1.0:
            points, radii = thin_path(points, radii, mesh_quality)

        v, n, idx = _tube_along_path(points, radii, segments)
        if v.shape[0] > 0:
            all_verts.append(v)
            all_normals.append(n)
            all_indices.append(idx + vertex_offset)
            vertex_offset += v.shape[0]

    if not all_verts:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.uint32),
        )

    return (
        np.concatenate(all_verts),
        np.concatenate(all_normals),
        np.concatenate(all_indices),
    )


# ---------------------------------------------------------------------------
# Type-segmented mesh generation (for per-compartment coloring)
# ---------------------------------------------------------------------------

_MeshTuple = tuple[np.ndarray, np.ndarray, np.ndarray]


def _split_path_by_type(
    path_ids: list[int],
    id_to_sample: dict,
) -> list[tuple[int, list[int]]]:
    """Split a path into sub-paths of consecutive same-type edges.

    Each edge inherits the SWC type of its child node.  The returned
    sub-paths overlap by one node at type boundaries so tubes remain
    gap-free.

    Returns:
        List of ``(swc_type, node_id_list)`` pairs.
    """
    if len(path_ids) < 2:
        return []

    result: list[tuple[int, list[int]]] = []
    current_type = id_to_sample[path_ids[1]].type
    current_sub: list[int] = [path_ids[0], path_ids[1]]

    for i in range(2, len(path_ids)):
        child_type = id_to_sample[path_ids[i]].type
        if child_type != current_type:
            result.append((current_type, current_sub))
            # Overlap: include the last node of the previous sub-path
            current_sub = [path_ids[i - 1], path_ids[i]]
            current_type = child_type
        else:
            current_sub.append(path_ids[i])

    result.append((current_type, current_sub))
    return result


def neuron_to_typed_meshes(
    morphology: NeuronMorphology,
    segments: int = 8,
    min_radius: float = 0.1,
    smooth_subdivisions: int = 0,
    mesh_quality: float = 1.0,
) -> dict[int, _MeshTuple]:
    """Convert a NeuronMorphology into per-SWC-type meshes.

    Like :func:`neuron_to_tube_mesh`, but returns separate
    ``(verts, normals, indices)`` for each SWC type so that each
    compartment can be assigned a distinct material/color.

    Args:
        morphology: The neuron morphology tree.
        segments: Cross-section polygon sides for tubes.
        min_radius: Minimum radius clamp.
        smooth_subdivisions: Catmull-Rom subdivisions per segment (0 = off).
        mesh_quality: Fraction of smoothed path points to keep (0.0–1.0).

    Returns:
        Dict mapping SWC type code → ``(vertices, normals, indices)``.
    """
    tree = morphology.tree
    id_to_sample = {s.id: s for s in tree}

    children: dict[int, list[int]] = defaultdict(list)
    for s in tree:
        if s.parent != -1 and s.parent in id_to_sample:
            children[s.parent].append(s.id)

    # Accumulators per type
    typed_verts: dict[int, list[np.ndarray]] = defaultdict(list)
    typed_normals: dict[int, list[np.ndarray]] = defaultdict(list)
    typed_indices: dict[int, list[np.ndarray]] = defaultdict(list)
    typed_offsets: dict[int, int] = defaultdict(int)

    def _append(swc_type: int, v: np.ndarray, n: np.ndarray, idx: np.ndarray) -> None:
        if v.shape[0] == 0:
            return
        typed_verts[swc_type].append(v)
        typed_normals[swc_type].append(n)
        typed_indices[swc_type].append(idx + typed_offsets[swc_type])
        typed_offsets[swc_type] += v.shape[0]

    # 1. Soma spheres
    for sample in tree:
        if sample.type == 1 and sample.parent == -1:
            center = np.array([sample.x, sample.y, sample.z])
            r = max(sample.r, min_radius)
            v, n, idx = _icosphere(center, r, subdivisions=2)
            _append(1, v, n, idx)

    # 2. Junction spheres at non-soma branch points
    for sample in tree:
        if sample.type == 1 and sample.parent == -1:
            continue
        if len(children[sample.id]) > 1:
            center = np.array([sample.x, sample.y, sample.z])
            r = max(sample.r, min_radius)
            v, n, idx = _icosphere(center, r, subdivisions=1)
            _append(sample.type, v, n, idx)

    # 3. Continuous tubes, split by SWC type
    paths = _extract_paths(tree, id_to_sample)
    for path_ids in paths:
        typed_subs = _split_path_by_type(path_ids, id_to_sample)
        for swc_type, sub_ids in typed_subs:
            points = [np.array([id_to_sample[s].x, id_to_sample[s].y, id_to_sample[s].z]) for s in sub_ids]
            radii = [max(id_to_sample[s].r, min_radius) for s in sub_ids]
            if smooth_subdivisions > 0:
                points, radii = smooth_path(points, radii, smooth_subdivisions)
            if mesh_quality < 1.0:
                points, radii = thin_path(points, radii, mesh_quality)
            v, n, idx = _tube_along_path(points, radii, segments)
            _append(swc_type, v, n, idx)

    # Concatenate per type
    result: dict[int, _MeshTuple] = {}
    for swc_type in typed_verts:
        result[swc_type] = (
            np.concatenate(typed_verts[swc_type]),
            np.concatenate(typed_normals[swc_type]),
            np.concatenate(typed_indices[swc_type]),
        )
    return result
