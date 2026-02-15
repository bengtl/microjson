# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-optimized 3D clipping for _clip_surface and _clip_line.

Drop-in replacements for the pure-Python versions in clip3d.py.
"""

from libc.math cimport INFINITY


def _clip_surface(dict feat, double k1, double k2, int axis):
    """Clip a TIN/PolyhedralSurface per-face — keep faces overlapping [k1, k2).

    Avoids temporary Python list allocations per face by using typed C loops
    for min/max computation.
    """
    cdef list xy = feat["geometry"]
    cdef list z = feat["geometry_z"]
    cdef list ring_lengths = feat.get("ring_lengths") or [len(z)]

    cdef list out_xy = []
    cdef list out_z = []
    cdef list out_ring_lengths = []

    cdef int offset = 0
    cdef int rl, j, vi
    cdef double f_min, f_max, val
    cdef int n

    for rl in ring_lengths:
        # Compute face min/max along clip axis with typed loop
        f_min = INFINITY
        f_max = -INFINITY

        if axis == 0:
            for j in range(rl):
                vi = (offset + j) * 2
                val = <double>xy[vi]
                if val < f_min:
                    f_min = val
                if val > f_max:
                    f_max = val
        elif axis == 1:
            for j in range(rl):
                vi = (offset + j) * 2 + 1
                val = <double>xy[vi]
                if val < f_min:
                    f_min = val
                if val > f_max:
                    f_max = val
        else:
            for j in range(rl):
                val = <double>z[offset + j]
                if val < f_min:
                    f_min = val
                if val > f_max:
                    f_max = val

        # Half-open interval [k1, k2): reject if fully outside
        if f_min >= k2 or f_max < k1:
            offset += rl
            continue

        # Face overlaps — keep it whole
        out_xy.extend(xy[offset * 2:(offset + rl) * 2])
        out_z.extend(z[offset:offset + rl])
        out_ring_lengths.append(rl)
        offset += rl

    if not out_ring_lengths:
        return None

    # Compute output bounds with typed loops
    n = len(out_z)
    cdef double min_x = INFINITY, max_x = -INFINITY
    cdef double min_y = INFINITY, max_y = -INFINITY
    cdef double min_z = INFINITY, max_z = -INFINITY
    cdef double vx, vy, vz

    for j in range(n):
        vx = <double>out_xy[j * 2]
        vy = <double>out_xy[j * 2 + 1]
        vz = <double>out_z[j]
        if vx < min_x:
            min_x = vx
        if vx > max_x:
            max_x = vx
        if vy < min_y:
            min_y = vy
        if vy > max_y:
            max_y = vy
        if vz < min_z:
            min_z = vz
        if vz > max_z:
            max_z = vz

    return {
        "geometry": out_xy,
        "geometry_z": out_z,
        "ring_lengths": out_ring_lengths,
        "type": feat["type"],
        "tags": feat.get("tags", {}),
        "minX": min_x, "minY": min_y, "minZ": min_z,
        "maxX": max_x, "maxY": max_y, "maxZ": max_z,
    }


cdef inline double _axis_val(list xy, list z, int idx, int axis):
    """Get value along axis for vertex at index idx."""
    if axis == 0:
        return <double>xy[idx * 2]
    elif axis == 1:
        return <double>xy[idx * 2 + 1]
    else:
        return <double>z[idx]


cdef inline tuple _intersect_inline(
    list xy, list z,
    int i, int j,
    double k, int axis,
):
    """Interpolate intersection point where line (i->j) crosses axis=k."""
    cdef double ax = <double>xy[i * 2]
    cdef double ay = <double>xy[i * 2 + 1]
    cdef double az = <double>z[i]
    cdef double bx = <double>xy[j * 2]
    cdef double by = <double>xy[j * 2 + 1]
    cdef double bz = <double>z[j]
    cdef double d, t

    if axis == 0:
        d = bx - ax
        t = (k - ax) / d if d != 0.0 else 0.0
    elif axis == 1:
        d = by - ay
        t = (k - ay) / d if d != 0.0 else 0.0
    else:
        d = bz - az
        t = (k - az) / d if d != 0.0 else 0.0

    return (
        ax + (bx - ax) * t,
        ay + (by - ay) * t,
        az + (bz - az) * t,
    )


def _clip_line(dict feat, double k1, double k2, int axis):
    """Clip a LineString3D to [k1, k2) along axis.

    Returns zero or more clipped line segments.  Inlines helper functions
    and avoids closure overhead compared to the pure-Python version.
    """
    cdef list xy = feat["geometry"]
    cdef list z = feat["geometry_z"]
    cdef int n = len(z)

    if n < 2:
        return []

    cdef list segments = []
    cdef list out_xy = []
    cdef list out_z = []
    cdef dict tags = feat.get("tags", {})
    cdef object ring_lengths = feat.get("ring_lengths")

    cdef int i, nn, jj
    cdef double a_val, b_val
    cdef bint a_in
    cdef bint cross_k1, cross_k2
    cdef double ix, iy, iz, d, t1, t2
    cdef double ix1, iy1, iz1, ix2, iy2, iz2
    cdef double last_val
    cdef double min_x, min_y, min_z_val, max_x, max_y, max_z_val
    cdef double vx, vy, vz

    for i in range(n - 1):
        a_val = _axis_val(xy, z, i, axis)
        b_val = _axis_val(xy, z, i + 1, axis)
        a_in = k1 <= a_val < k2

        if a_in:
            out_xy.append(xy[i * 2])
            out_xy.append(xy[i * 2 + 1])
            out_z.append(z[i])

        # Detect boundary crossings
        cross_k1 = (a_val < k1 and b_val > k1) or (b_val < k1 and a_val > k1)
        cross_k2 = (a_val < k2 and b_val >= k2) or (b_val < k2 and a_val >= k2)

        # Handle crossings — inline for 0, 1, or 2 crossings
        if cross_k1 and cross_k2:
            # Both crossings — sort by parameter t
            d = b_val - a_val
            if d != 0.0:
                t1 = (k1 - a_val) / d
                t2 = (k2 - a_val) / d
            else:
                t1 = 0.0
                t2 = 0.0

            ix1, iy1, iz1 = _intersect_inline(xy, z, i, i + 1, k1, axis)
            ix2, iy2, iz2 = _intersect_inline(xy, z, i, i + 1, k2, axis)

            # Determine enter/exit for k1 and k2
            # k1: entering if going upward (b_val > a_val)
            # k2: entering if going downward (b_val < a_val)
            if t1 < t2:
                # k1 first, then k2
                if b_val > a_val:
                    # Upward: enter at k1, exit at k2
                    # Flush previous, start new at k1, add k2, flush
                    if len(out_z) >= 2:
                        nn = len(out_z)
                        min_x = INFINITY; max_x = -INFINITY
                        min_y = INFINITY; max_y = -INFINITY
                        min_z_val = INFINITY; max_z_val = -INFINITY
                        for jj in range(nn):
                            vx = <double>out_xy[jj * 2]
                            vy = <double>out_xy[jj * 2 + 1]
                            vz = <double>out_z[jj]
                            if vx < min_x: min_x = vx
                            if vx > max_x: max_x = vx
                            if vy < min_y: min_y = vy
                            if vy > max_y: max_y = vy
                            if vz < min_z_val: min_z_val = vz
                            if vz > max_z_val: max_z_val = vz
                        segments.append({
                            "geometry": out_xy, "geometry_z": out_z,
                            "type": 2, "tags": tags,
                            "ring_lengths": ring_lengths,
                            "minX": min_x, "minY": min_y, "minZ": min_z_val,
                            "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                        })
                    out_xy = [ix1, iy1]
                    out_z = [iz1]
                    out_xy.append(ix2)
                    out_xy.append(iy2)
                    out_z.append(iz2)
                    # Flush exit at k2
                    if len(out_z) >= 2:
                        nn = len(out_z)
                        min_x = INFINITY; max_x = -INFINITY
                        min_y = INFINITY; max_y = -INFINITY
                        min_z_val = INFINITY; max_z_val = -INFINITY
                        for jj in range(nn):
                            vx = <double>out_xy[jj * 2]
                            vy = <double>out_xy[jj * 2 + 1]
                            vz = <double>out_z[jj]
                            if vx < min_x: min_x = vx
                            if vx > max_x: max_x = vx
                            if vy < min_y: min_y = vy
                            if vy > max_y: max_y = vy
                            if vz < min_z_val: min_z_val = vz
                            if vz > max_z_val: max_z_val = vz
                        segments.append({
                            "geometry": out_xy, "geometry_z": out_z,
                            "type": 2, "tags": tags,
                            "ring_lengths": ring_lengths,
                            "minX": min_x, "minY": min_y, "minZ": min_z_val,
                            "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                        })
                    out_xy = []
                    out_z = []
                else:
                    # Downward: exit at k1, enter at k2
                    out_xy.append(ix1)
                    out_xy.append(iy1)
                    out_z.append(iz1)
                    # Flush exit at k1
                    if len(out_z) >= 2:
                        nn = len(out_z)
                        min_x = INFINITY; max_x = -INFINITY
                        min_y = INFINITY; max_y = -INFINITY
                        min_z_val = INFINITY; max_z_val = -INFINITY
                        for jj in range(nn):
                            vx = <double>out_xy[jj * 2]
                            vy = <double>out_xy[jj * 2 + 1]
                            vz = <double>out_z[jj]
                            if vx < min_x: min_x = vx
                            if vx > max_x: max_x = vx
                            if vy < min_y: min_y = vy
                            if vy > max_y: max_y = vy
                            if vz < min_z_val: min_z_val = vz
                            if vz > max_z_val: max_z_val = vz
                        segments.append({
                            "geometry": out_xy, "geometry_z": out_z,
                            "type": 2, "tags": tags,
                            "ring_lengths": ring_lengths,
                            "minX": min_x, "minY": min_y, "minZ": min_z_val,
                            "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                        })
                    # Start new at k2 entry
                    out_xy = [ix2, iy2]
                    out_z = [iz2]
            else:
                # k2 first, then k1 (t2 < t1)
                if b_val < a_val:
                    # Downward: enter at k2, exit at k1
                    if len(out_z) >= 2:
                        nn = len(out_z)
                        min_x = INFINITY; max_x = -INFINITY
                        min_y = INFINITY; max_y = -INFINITY
                        min_z_val = INFINITY; max_z_val = -INFINITY
                        for jj in range(nn):
                            vx = <double>out_xy[jj * 2]
                            vy = <double>out_xy[jj * 2 + 1]
                            vz = <double>out_z[jj]
                            if vx < min_x: min_x = vx
                            if vx > max_x: max_x = vx
                            if vy < min_y: min_y = vy
                            if vy > max_y: max_y = vy
                            if vz < min_z_val: min_z_val = vz
                            if vz > max_z_val: max_z_val = vz
                        segments.append({
                            "geometry": out_xy, "geometry_z": out_z,
                            "type": 2, "tags": tags,
                            "ring_lengths": ring_lengths,
                            "minX": min_x, "minY": min_y, "minZ": min_z_val,
                            "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                        })
                    out_xy = [ix2, iy2]
                    out_z = [iz2]
                    out_xy.append(ix1)
                    out_xy.append(iy1)
                    out_z.append(iz1)
                    if len(out_z) >= 2:
                        nn = len(out_z)
                        min_x = INFINITY; max_x = -INFINITY
                        min_y = INFINITY; max_y = -INFINITY
                        min_z_val = INFINITY; max_z_val = -INFINITY
                        for jj in range(nn):
                            vx = <double>out_xy[jj * 2]
                            vy = <double>out_xy[jj * 2 + 1]
                            vz = <double>out_z[jj]
                            if vx < min_x: min_x = vx
                            if vx > max_x: max_x = vx
                            if vy < min_y: min_y = vy
                            if vy > max_y: max_y = vy
                            if vz < min_z_val: min_z_val = vz
                            if vz > max_z_val: max_z_val = vz
                        segments.append({
                            "geometry": out_xy, "geometry_z": out_z,
                            "type": 2, "tags": tags,
                            "ring_lengths": ring_lengths,
                            "minX": min_x, "minY": min_y, "minZ": min_z_val,
                            "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                        })
                    out_xy = []
                    out_z = []
                else:
                    # Upward: exit at k2, enter at k1
                    out_xy.append(ix2)
                    out_xy.append(iy2)
                    out_z.append(iz2)
                    if len(out_z) >= 2:
                        nn = len(out_z)
                        min_x = INFINITY; max_x = -INFINITY
                        min_y = INFINITY; max_y = -INFINITY
                        min_z_val = INFINITY; max_z_val = -INFINITY
                        for jj in range(nn):
                            vx = <double>out_xy[jj * 2]
                            vy = <double>out_xy[jj * 2 + 1]
                            vz = <double>out_z[jj]
                            if vx < min_x: min_x = vx
                            if vx > max_x: max_x = vx
                            if vy < min_y: min_y = vy
                            if vy > max_y: max_y = vy
                            if vz < min_z_val: min_z_val = vz
                            if vz > max_z_val: max_z_val = vz
                        segments.append({
                            "geometry": out_xy, "geometry_z": out_z,
                            "type": 2, "tags": tags,
                            "ring_lengths": ring_lengths,
                            "minX": min_x, "minY": min_y, "minZ": min_z_val,
                            "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                        })
                    out_xy = [ix1, iy1]
                    out_z = [iz1]

        elif cross_k1:
            ix, iy, iz = _intersect_inline(xy, z, i, i + 1, k1, axis)
            if b_val > a_val:
                # Entering at k1
                if len(out_z) >= 2:
                    nn = len(out_z)
                    min_x = INFINITY; max_x = -INFINITY
                    min_y = INFINITY; max_y = -INFINITY
                    min_z_val = INFINITY; max_z_val = -INFINITY
                    for jj in range(nn):
                        vx = <double>out_xy[jj * 2]
                        vy = <double>out_xy[jj * 2 + 1]
                        vz = <double>out_z[jj]
                        if vx < min_x: min_x = vx
                        if vx > max_x: max_x = vx
                        if vy < min_y: min_y = vy
                        if vy > max_y: max_y = vy
                        if vz < min_z_val: min_z_val = vz
                        if vz > max_z_val: max_z_val = vz
                    segments.append({
                        "geometry": out_xy, "geometry_z": out_z,
                        "type": 2, "tags": tags,
                        "ring_lengths": ring_lengths,
                        "minX": min_x, "minY": min_y, "minZ": min_z_val,
                        "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                    })
                out_xy = [ix, iy]
                out_z = [iz]
            else:
                # Exiting at k1
                out_xy.append(ix)
                out_xy.append(iy)
                out_z.append(iz)
                if len(out_z) >= 2:
                    nn = len(out_z)
                    min_x = INFINITY; max_x = -INFINITY
                    min_y = INFINITY; max_y = -INFINITY
                    min_z_val = INFINITY; max_z_val = -INFINITY
                    for jj in range(nn):
                        vx = <double>out_xy[jj * 2]
                        vy = <double>out_xy[jj * 2 + 1]
                        vz = <double>out_z[jj]
                        if vx < min_x: min_x = vx
                        if vx > max_x: max_x = vx
                        if vy < min_y: min_y = vy
                        if vy > max_y: max_y = vy
                        if vz < min_z_val: min_z_val = vz
                        if vz > max_z_val: max_z_val = vz
                    segments.append({
                        "geometry": out_xy, "geometry_z": out_z,
                        "type": 2, "tags": tags,
                        "ring_lengths": ring_lengths,
                        "minX": min_x, "minY": min_y, "minZ": min_z_val,
                        "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                    })
                out_xy = []
                out_z = []

        elif cross_k2:
            ix, iy, iz = _intersect_inline(xy, z, i, i + 1, k2, axis)
            if b_val < a_val:
                # Entering at k2 (downward)
                if len(out_z) >= 2:
                    nn = len(out_z)
                    min_x = INFINITY; max_x = -INFINITY
                    min_y = INFINITY; max_y = -INFINITY
                    min_z_val = INFINITY; max_z_val = -INFINITY
                    for jj in range(nn):
                        vx = <double>out_xy[jj * 2]
                        vy = <double>out_xy[jj * 2 + 1]
                        vz = <double>out_z[jj]
                        if vx < min_x: min_x = vx
                        if vx > max_x: max_x = vx
                        if vy < min_y: min_y = vy
                        if vy > max_y: max_y = vy
                        if vz < min_z_val: min_z_val = vz
                        if vz > max_z_val: max_z_val = vz
                    segments.append({
                        "geometry": out_xy, "geometry_z": out_z,
                        "type": 2, "tags": tags,
                        "ring_lengths": ring_lengths,
                        "minX": min_x, "minY": min_y, "minZ": min_z_val,
                        "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                    })
                out_xy = [ix, iy]
                out_z = [iz]
            else:
                # Exiting at k2 (upward)
                out_xy.append(ix)
                out_xy.append(iy)
                out_z.append(iz)
                if len(out_z) >= 2:
                    nn = len(out_z)
                    min_x = INFINITY; max_x = -INFINITY
                    min_y = INFINITY; max_y = -INFINITY
                    min_z_val = INFINITY; max_z_val = -INFINITY
                    for jj in range(nn):
                        vx = <double>out_xy[jj * 2]
                        vy = <double>out_xy[jj * 2 + 1]
                        vz = <double>out_z[jj]
                        if vx < min_x: min_x = vx
                        if vx > max_x: max_x = vx
                        if vy < min_y: min_y = vy
                        if vy > max_y: max_y = vy
                        if vz < min_z_val: min_z_val = vz
                        if vz > max_z_val: max_z_val = vz
                    segments.append({
                        "geometry": out_xy, "geometry_z": out_z,
                        "type": 2, "tags": tags,
                        "ring_lengths": ring_lengths,
                        "minX": min_x, "minY": min_y, "minZ": min_z_val,
                        "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
                    })
                out_xy = []
                out_z = []

    # Last point
    if n > 0:
        last_val = _axis_val(xy, z, n - 1, axis)
        if k1 <= last_val < k2:
            out_xy.append(xy[(n - 1) * 2])
            out_xy.append(xy[(n - 1) * 2 + 1])
            out_z.append(z[n - 1])

    # Final flush
    if len(out_z) >= 2:
        nn = len(out_z)
        min_x = INFINITY; max_x = -INFINITY
        min_y = INFINITY; max_y = -INFINITY
        min_z_val = INFINITY; max_z_val = -INFINITY
        for jj in range(nn):
            vx = <double>out_xy[jj * 2]
            vy = <double>out_xy[jj * 2 + 1]
            vz = <double>out_z[jj]
            if vx < min_x: min_x = vx
            if vx > max_x: max_x = vx
            if vy < min_y: min_y = vy
            if vy > max_y: max_y = vy
            if vz < min_z_val: min_z_val = vz
            if vz > max_z_val: max_z_val = vz
        segments.append({
            "geometry": out_xy, "geometry_z": out_z,
            "type": 2, "tags": tags,
            "ring_lengths": ring_lengths,
            "minX": min_x, "minY": min_y, "minZ": min_z_val,
            "maxX": max_x, "maxY": max_y, "maxZ": max_z_val,
        })

    return segments
