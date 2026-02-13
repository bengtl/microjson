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
# Ontology Vocabulary Support
# ---------------------------------------------------------------------------

class OntologyTerm(BaseModel):
    """A reference to a formal ontology term.

    Attributes:
        uri: Full URI of the ontology term (e.g. "http://purl.obolibrary.org/obo/CL_0000598").
        label: Human-readable label (e.g. "pyramidal neuron").
        description: Optional longer description of the term.
    """
    uri: str
    label: Optional[str] = None
    description: Optional[str] = None


class Vocabulary(BaseModel):
    """Maps property values to formal ontology terms.

    Attributes:
        namespace: Common URI prefix for the ontology (e.g. "http://purl.obolibrary.org/obo/CL_").
        description: Optional description of this vocabulary.
        terms: Mapping from property values to ontology terms.
    """
    namespace: Optional[str] = None
    description: Optional[str] = None
    terms: Dict[str, OntologyTerm]


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
    vocabularies: Optional[Union[Dict[str, Vocabulary], str]] = None


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
    vocabularies: Optional[Union[Dict[str, Vocabulary], str]] = None


class MicroJSON(RootModel):
    """The root object of a MicroJSON file"""

    root: Union[MicroFeature, MicroFeatureCollection, GeometryType]  # type: ignore
