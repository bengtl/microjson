"""3D coordinate transforms for MicroJSON.

Provides AffineTransform (4x4 matrix), VoxelCoordinateSystem, and helper
functions for coordinate conversion. Extends the existing transformation
patterns in tilemodel.py.
"""

from __future__ import annotations

from typing import List, Literal, Tuple, Union

from pydantic import BaseModel, field_validator
from geojson_pydantic import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)

from .tilemodel import CoordinateTransformation
from .model import (
    NeuronMorphology,
    SWCSample,
    TIN,
    PolyhedralSurface,
    SliceStack,
    Slice,
)


class AffineTransform(CoordinateTransformation):
    """A 4x4 affine transformation matrix for 3D coordinates.

    The matrix is stored in row-major order:
        [[r00, r01, r02, tx],
         [r10, r11, r12, ty],
         [r20, r21, r22, tz],
         [0,   0,   0,   1 ]]
    """

    type: Literal["affine"] = "affine"
    matrix: List[List[float]]

    @field_validator("matrix")
    @classmethod
    def _validate_4x4(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != 4:
            raise ValueError(f"Affine matrix must have 4 rows, got {len(v)}")
        for i, row in enumerate(v):
            if len(row) != 4:
                raise ValueError(
                    f"Affine matrix row {i} must have 4 columns, got {len(row)}"
                )
        return v


class VoxelCoordinateSystem(BaseModel):
    """Defines voxel-to-physical coordinate mapping.

    Physical coordinate = voxel * resolution + origin
    """

    axes: List[str]
    units: List[str]
    resolution: List[float]
    origin: List[float] = [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _transform_position(
    pos: tuple, matrix: List[List[float]]
) -> tuple:
    """Apply a 4x4 affine matrix to a single position (2D or 3D)."""
    x = float(pos[0])
    y = float(pos[1]) if len(pos) > 1 else 0.0
    z = float(pos[2]) if len(pos) > 2 else 0.0
    nx = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3]
    ny = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3]
    nz = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]
    if len(pos) <= 2:
        return (nx, ny)
    return (nx, ny, nz)


def _transform_coords(coords, matrix: List[List[float]]):
    """Recursively transform nested coordinate arrays."""
    if not coords:
        return coords
    # Single position (tuple of numbers)
    if isinstance(coords[0], (int, float)):
        return _transform_position(tuple(coords), matrix)
    # List of positions or deeper nesting
    return [_transform_coords(c, matrix) for c in coords]


# ---------------------------------------------------------------------------
# 3D type transform helpers
# ---------------------------------------------------------------------------

def _transform_neuron(
    geom: NeuronMorphology, matrix: List[List[float]]
) -> NeuronMorphology:
    """Transform each SWCSample position, preserving r/id/type/parent."""
    new_tree = []
    for s in geom.tree:
        nx, ny, nz = _transform_position((s.x, s.y, s.z), matrix)
        new_tree.append(SWCSample(
            id=s.id, type=s.type, x=nx, y=ny, z=nz, r=s.r, parent=s.parent,
        ))
    return NeuronMorphology(type="NeuronMorphology", tree=new_tree)


def _transform_tin(geom: TIN, matrix: List[List[float]]) -> TIN:
    """Transform nested polygon coordinates in a TIN."""
    new_coords = _transform_coords(list(geom.coordinates), matrix)
    return TIN(type="TIN", coordinates=new_coords)


def _transform_polyhedral(
    geom: PolyhedralSurface, matrix: List[List[float]]
) -> PolyhedralSurface:
    """Transform nested polygon coordinates in a PolyhedralSurface."""
    new_coords = _transform_coords(list(geom.coordinates), matrix)
    return PolyhedralSurface(type="PolyhedralSurface", coordinates=new_coords)


def _transform_slice_stack(
    geom: SliceStack, matrix: List[List[float]]
) -> SliceStack:
    """Transform each slice's geometry and z position."""
    new_slices = []
    for slc in geom.slices:
        # Transform the slice's 2D geometry coordinates
        new_geom_coords = _transform_coords(
            list(slc.geometry.coordinates), matrix,
        )
        new_geom = type(slc.geometry)(
            type=slc.geometry.type, coordinates=new_geom_coords,
        )
        # Transform the z position through the matrix
        _, _, new_z = _transform_position((0, 0, slc.z), matrix)
        new_slices.append(Slice(
            z=new_z, geometry=new_geom, properties=slc.properties,
        ))
    # Sort slices by z to satisfy validator
    new_slices.sort(key=lambda s: s.z)
    return SliceStack(
        type="SliceStack",
        slices=new_slices,
        axis=geom.axis,
        units=geom.units,
        interpolation=geom.interpolation,
    )


# ---------------------------------------------------------------------------
# Geometry union + dispatch
# ---------------------------------------------------------------------------

Geometry = Union[
    Point, MultiPoint, LineString, MultiLineString, Polygon, MultiPolygon,
    NeuronMorphology, TIN, PolyhedralSurface, SliceStack,
]


def apply_transform(geometry: Geometry, transform: AffineTransform) -> Geometry:
    """Apply an affine transform to a geometry, returning a new geometry.

    Supports all GeoJSON geometry types plus MicroJSON 3D types
    (NeuronMorphology, TIN, PolyhedralSurface, SliceStack).

    Args:
        geometry: A geometry with 3D coordinates.
        transform: A 4x4 affine transformation.

    Returns:
        A new geometry of the same type with transformed coordinates.
    """
    matrix = transform.matrix

    if isinstance(geometry, NeuronMorphology):
        return _transform_neuron(geometry, matrix)
    if isinstance(geometry, TIN):
        return _transform_tin(geometry, matrix)
    if isinstance(geometry, PolyhedralSurface):
        return _transform_polyhedral(geometry, matrix)
    if isinstance(geometry, SliceStack):
        return _transform_slice_stack(geometry, matrix)

    # GeoJSON types — all have .coordinates
    new_coords = _transform_coords(list(geometry.coordinates), matrix)
    return type(geometry)(type=geometry.type, coordinates=new_coords)


def translate_geometry(
    geometry: Geometry, dx: float, dy: float, dz: float
) -> Geometry:
    """Translate a geometry by (dx, dy, dz).

    Convenience wrapper that builds a translation matrix and calls
    ``apply_transform()``.
    """
    t = AffineTransform(
        type="affine",
        matrix=[
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1],
        ],
    )
    return apply_transform(geometry, t)


# ---------------------------------------------------------------------------
# Coordinate system conversions
# ---------------------------------------------------------------------------

def voxel_to_physical(
    coords: Tuple[float, float, float],
    vcs: VoxelCoordinateSystem,
) -> Tuple[float, float, float]:
    """Convert voxel coordinates to physical coordinates.

    Physical = voxel * resolution + origin
    """
    return (
        coords[0] * vcs.resolution[0] + vcs.origin[0],
        coords[1] * vcs.resolution[1] + vcs.origin[1],
        coords[2] * vcs.resolution[2] + vcs.origin[2],
    )


def physical_to_voxel(
    coords: Tuple[float, float, float],
    vcs: VoxelCoordinateSystem,
) -> Tuple[float, float, float]:
    """Convert physical coordinates to voxel coordinates.

    Voxel = (physical - origin) / resolution
    """
    return (
        (coords[0] - vcs.origin[0]) / vcs.resolution[0],
        (coords[1] - vcs.origin[1]) / vcs.resolution[1],
        (coords[2] - vcs.origin[2]) / vcs.resolution[2],
    )
