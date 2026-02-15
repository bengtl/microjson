# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython-optimized coordinate transform for 3D tiles.

Drop-in replacement for transform_tile_3d in tile3d.py.
"""


def transform_tile_3d(dict tile, int extent=4096, int extent_z=4096):
    """Transform tile features from normalized [0,1] to integer coords.

    Pre-computes scale/offset and uses typed inner loop for arithmetic.
    Uses Python round() (not C round) for bit-identical banker's rounding.
    """
    cdef int z = tile["z"]
    cdef int tx = tile["x"]
    cdef int ty = tile["y"]
    cdef int td = tile["d"]

    # Number of tiles per axis at this zoom
    cdef int n = 1 << z  # 2^z

    # Sub-range this tile covers in [0,1]
    cdef double x0 = <double>tx / <double>n
    cdef double y0 = <double>ty / <double>n
    cdef double z0 = <double>td / <double>n
    cdef double scale_x = <double>n
    cdef double scale_y = <double>n
    cdef double scale_z = <double>n

    cdef list features = tile["features"]
    cdef list new_features = []
    cdef int nf = len(features)

    cdef int fi, nv, i
    cdef dict feat, new_feat
    cdef list xy, zz, new_xy, new_z
    cdef double lx, ly, lz

    for fi in range(nf):
        feat = <dict>features[fi]
        xy = feat["geometry"]
        zz = feat["geometry_z"]
        nv = len(zz)

        # Pre-allocate output lists
        new_xy = [0] * (nv * 2)
        new_z = [0] * nv

        for i in range(nv):
            # Normalize to tile-local [0, 1]
            lx = (<double>xy[i * 2] - x0) * scale_x
            ly = (<double>xy[i * 2 + 1] - y0) * scale_y
            lz = (<double>zz[i] - z0) * scale_z

            # Scale to integer extent — Python round() for banker's rounding
            new_xy[i * 2] = round(lx * extent)
            new_xy[i * 2 + 1] = round(ly * extent)
            new_z[i] = round(lz * extent_z)

        new_feat = {
            "geometry": new_xy,
            "geometry_z": new_z,
            "type": feat["type"],
            "tags": feat.get("tags", {}),
        }
        if "ring_lengths" in feat:
            new_feat["ring_lengths"] = feat["ring_lengths"]
        if "radii" in feat:
            new_feat["radii"] = feat["radii"]
        new_features.append(new_feat)

    return {
        "features": new_features,
        "z": z,
        "x": tx,
        "y": ty,
        "d": td,
        "num_features": len(new_features),
        "num_points": sum(len(f["geometry_z"]) for f in new_features),
        "extent": extent,
        "extent_z": extent_z,
    }
