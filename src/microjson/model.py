"""MicroJSON and GeoJSON models, defined manually using pydantic."""

from typing import Any, List, Literal, Optional, Tuple, Union, Dict
from pydantic import BaseModel, StrictInt, StrictStr, RootModel, field_validator
from .provenance import Workflow
from .provenance import WorkflowCollection
from .provenance import Artifact
from .provenance import ArtifactCollection
from geojson_pydantic import Feature, FeatureCollection, GeometryCollection
from geojson_pydantic import Point, MultiPoint, LineString, MultiLineString
from geojson_pydantic import Polygon, MultiPolygon
from geojson_pydantic.types import (
    Position,
    LinearRing,
    PolygonCoords,
    MultiPolygonCoords,
)


# ---------------------------------------------------------------------------
# Helper: extract all 3D positions from nested coordinate structures
# ---------------------------------------------------------------------------

def _iter_positions(coords: list) -> list[Position]:
    """Recursively flatten nested coordinate arrays to a list of positions."""
    if not coords:
        return []
    # If the first element is a number, this is a single position
    if isinstance(coords[0], (int, float)):
        return [tuple(coords)]  # type: ignore[return-value]
    # If the first element is a tuple/namedtuple, this is a list of positions
    if isinstance(coords[0], tuple) and isinstance(coords[0][0], (int, float)):
        return list(coords)  # type: ignore[return-value]
    # Otherwise, recurse
    result: list[Position] = []
    for item in coords:
        result.extend(_iter_positions(item))
    return result


def _bbox3d(coords: list) -> Tuple[float, float, float, float, float, float]:
    """Compute 3D bounding box from nested coordinate arrays."""
    positions = _iter_positions(coords)
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _centroid3d(coords: list) -> Tuple[float, float, float]:
    """Compute centroid of unique 3D positions from nested coordinate arrays."""
    positions = _iter_positions(coords)
    unique = list({(float(p[0]), float(p[1]), float(p[2])) for p in positions})
    n = len(unique)
    return (
        sum(p[0] for p in unique) / n,
        sum(p[1] for p in unique) / n,
        sum(p[2] for p in unique) / n,
    )


# ---------------------------------------------------------------------------
# 3D Geometry Types
# ---------------------------------------------------------------------------

class PolyhedralSurface(BaseModel):
    """A closed surface mesh consisting of polygonal faces (ISO 19107).

    Each face has the same structure as a Polygon: a list of linear rings,
    where each ring is a list of 3D positions.
    """

    type: Literal["PolyhedralSurface"]
    coordinates: List[PolygonCoords]

    @field_validator("coordinates")
    @classmethod
    def _at_least_one_face(cls, v: List[PolygonCoords]) -> List[PolygonCoords]:
        if len(v) == 0:
            raise ValueError("PolyhedralSurface must have at least one face")
        return v

    def bbox3d(self) -> Tuple[float, float, float, float, float, float]:
        return _bbox3d(self.coordinates)

    def centroid3d(self) -> Tuple[float, float, float]:
        return _centroid3d(self.coordinates)


class TIN(BaseModel):
    """A Triangulated Irregular Network — triangle mesh surface (ISO 19107).

    Each face must be a single closed ring of exactly 4 positions
    (3 vertices + repeated first vertex).
    """

    type: Literal["TIN"]
    coordinates: List[PolygonCoords]

    @field_validator("coordinates")
    @classmethod
    def _validate_triangles(cls, v: List[PolygonCoords]) -> List[PolygonCoords]:
        if len(v) == 0:
            raise ValueError("TIN must have at least one face")
        for i, face in enumerate(v):
            if len(face) != 1:
                raise ValueError(
                    f"TIN face {i} must have exactly one ring, got {len(face)}"
                )
            ring = face[0]
            if len(ring) != 4:
                raise ValueError(
                    f"TIN face {i} ring must have exactly 4 positions "
                    f"(closed triangle), got {len(ring)}"
                )
        return v

    def bbox3d(self) -> Tuple[float, float, float, float, float, float]:
        return _bbox3d(self.coordinates)

    def centroid3d(self) -> Tuple[float, float, float]:
        return _centroid3d(self.coordinates)


# ---------------------------------------------------------------------------
# SliceStack — ordered Z-stack of 2D slices (the key "2.5D" type)
# ---------------------------------------------------------------------------

class Slice(BaseModel):
    """A single 2D slice at a given position along the stack axis."""

    z: float
    geometry: Union[Polygon, MultiPolygon]
    properties: Optional[Dict[str, Any]] = None


class SliceStack(BaseModel):
    """An ordered collection of 2D slices at specified positions along an axis.

    Represents the MicroJSON 2.5D goal: 2D contours stacked in 3D.
    """

    type: Literal["SliceStack"]
    slices: List[Slice]
    axis: Literal["x", "y", "z"] = "z"
    units: Optional[str] = None
    interpolation: Optional[Literal["linear", "nearest", "cubic"]] = None

    @field_validator("slices")
    @classmethod
    def _validate_slices(cls, v: List[Slice]) -> List[Slice]:
        if len(v) == 0:
            raise ValueError("SliceStack must have at least one slice")
        z_values = [s.z for s in v]
        if z_values != sorted(z_values):
            raise ValueError("SliceStack slices must be sorted by z value")
        if len(z_values) != len(set(z_values)):
            raise ValueError("SliceStack slices must not have duplicate z values")
        return v

    def bbox3d(self) -> Tuple[float, float, float, float, float, float]:
        """Compute 3D bbox from all slice geometries + z positions."""
        all_positions: list[Position] = []
        for s in self.slices:
            all_positions.extend(_iter_positions(list(s.geometry.coordinates)))
        xs = [p[0] for p in all_positions]
        ys = [p[1] for p in all_positions]
        z_values = [s.z for s in self.slices]
        return (
            min(xs), min(ys), min(z_values),
            max(xs), max(ys), max(z_values),
        )


# ---------------------------------------------------------------------------
# NeuronMorphology — SWC tree structure for neuron morphologies
# ---------------------------------------------------------------------------

# Standard SWC type codes
SWC_UNDEFINED = 0
SWC_SOMA = 1
SWC_AXON = 2
SWC_BASAL_DENDRITE = 3
SWC_APICAL_DENDRITE = 4
SWC_FORK_POINT = 5
SWC_END_POINT = 6
SWC_CUSTOM = 7


class SWCSample(BaseModel):
    """A single sample point in an SWC neuron morphology tree."""

    id: int
    type: int
    x: float
    y: float
    z: float
    r: float
    parent: int


class NeuronMorphology(BaseModel):
    """A neuron morphology represented as an SWC tree.

    Each node has a 3D position, radius, type code, and parent reference.
    """

    type: Literal["NeuronMorphology"]
    tree: List[SWCSample]

    @field_validator("tree")
    @classmethod
    def _validate_tree(cls, v: List[SWCSample]) -> List[SWCSample]:
        if len(v) == 0:
            raise ValueError("NeuronMorphology tree must have at least one node")
        ids = {s.id for s in v}
        has_root = any(s.parent == -1 for s in v)
        if not has_root:
            raise ValueError(
                "NeuronMorphology tree must have at least one root node "
                "(parent == -1)"
            )
        for s in v:
            if s.parent != -1 and s.parent not in ids:
                raise ValueError(
                    f"Node {s.id} has parent {s.parent} which does not exist "
                    f"in the tree"
                )
        return v

    def bbox3d(self) -> Tuple[float, float, float, float, float, float]:
        xs = [s.x for s in self.tree]
        ys = [s.y for s in self.tree]
        zs = [s.z for s in self.tree]
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def centroid3d(self) -> Tuple[float, float, float]:
        n = len(self.tree)
        return (
            sum(s.x for s in self.tree) / n,
            sum(s.y for s in self.tree) / n,
            sum(s.z for s in self.tree) / n,
        )


GeometryType = Union[  # type: ignore
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
    GeometryCollection,
    PolyhedralSurface,
    TIN,
    SliceStack,
    NeuronMorphology,
    type(None),
]


class GeoJSON(RootModel):
    """The root object of a GeoJSON file"""

    root: Union[Feature, FeatureCollection, GeometryType]  # type: ignore


class MicroFeature(Feature):
    """A MicroJSON feature, which is a GeoJSON feature with additional
    metadata

    Args:
        geometry (Optional[GeometryType]): Extended geometry supporting 3D types
        multiscale (Optional[Multiscale]): The coordinate system of the feature
        ref (Optional[Union[StrictStr, StrictInt]]):
            A reference to the parent feature
        parentId (Optional[Union[StrictStr, StrictInt]]):
            A reference to the parent feature
        featureClass (Optional[str]): The class of the feature
    """

    # Override geometry to accept MicroJSON 3D types in addition to GeoJSON types
    geometry: Union[GeometryType, None]  # type: ignore[assignment]
    ref: Optional[Union[StrictStr, StrictInt]] = None
    # reference to the parent feature
    parentId: Optional[Union[StrictStr, StrictInt]] = None
    # for now, only string feature class is supported
    # in the future, it may be expanded with a class registry
    featureClass: Optional[str] = None


class MicroFeatureCollection(FeatureCollection):
    """A MicroJSON feature collection, which is a GeoJSON feature
    collection with additional metadata.

    Args:
        features (List[MicroFeature]): Features with extended 3D geometry support
        properties (Optional[Props]): The properties of the feature collection
        id (Optional[Union[StrictStr, StrictInt]]): The ID of the feature coll.
        provenance (Optional[Union[Workflow,
            WorkflowCollection,
            Artifact,
            ArtifactCollection]]): The provenance of the feature collection
    """

    # Override features to use MicroFeature (supports 3D geometry types)
    features: List[MicroFeature]  # type: ignore[assignment]
    properties: Optional[Dict[str, Any]] = None
    id: Optional[Union[StrictStr, StrictInt]] = None
    provenance: Optional[
        Union[Workflow, WorkflowCollection, Artifact, ArtifactCollection]
    ] = None


class MicroJSON(RootModel):
    """The root object of a MicroJSON file"""

    root: Union[MicroFeature, MicroFeatureCollection, GeometryType]  # type: ignore
