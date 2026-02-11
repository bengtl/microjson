"""Shapely → MicroJSON/GeoJSON geometry conversion (reverse of _geometry.py)."""

from __future__ import annotations

import json
from typing import Any

import shapely
from shapely.geometry import (
    GeometryCollection as ShapelyGeometryCollection,
    LineString as ShapelyLineString,
    MultiLineString as ShapelyMultiLineString,
    MultiPoint as ShapelyMultiPoint,
    MultiPolygon as ShapelyMultiPolygon,
    Point as ShapelyPoint,
    Polygon as ShapelyPolygon,
)

from geojson_pydantic import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from ..model import (
    NeuronMorphology,
    Slice,
    SliceStack,
    SWCSample,
)


def _pos(coord: tuple) -> tuple:
    """Convert a Shapely coordinate tuple to a GeoJSON position."""
    return tuple(float(c) for c in coord)


def _ring_coords(ring) -> list[tuple]:
    """Extract ring coordinates from a Shapely LinearRing or coord sequence."""
    return [_pos(c) for c in ring.coords]


def shapely_to_microjson(geom: Any) -> Any:
    """Convert a Shapely geometry to a GeoJSON-pydantic model.

    Returns None for None input.
    """
    if geom is None:
        return None

    if isinstance(geom, ShapelyPoint):
        return Point(type="Point", coordinates=_pos(geom.coords[0]))

    if isinstance(geom, ShapelyMultiPoint):
        return MultiPoint(
            type="MultiPoint",
            coordinates=[_pos(p.coords[0]) for p in geom.geoms],
        )

    if isinstance(geom, ShapelyLineString):
        return LineString(
            type="LineString",
            coordinates=[_pos(c) for c in geom.coords],
        )

    if isinstance(geom, ShapelyMultiLineString):
        return MultiLineString(
            type="MultiLineString",
            coordinates=[
                [_pos(c) for c in line.coords] for line in geom.geoms
            ],
        )

    if isinstance(geom, ShapelyPolygon):
        rings = [_ring_coords(geom.exterior)]
        for interior in geom.interiors:
            rings.append(_ring_coords(interior))
        return Polygon(type="Polygon", coordinates=rings)

    if isinstance(geom, ShapelyMultiPolygon):
        polys = []
        for poly in geom.geoms:
            rings = [_ring_coords(poly.exterior)]
            for interior in poly.interiors:
                rings.append(_ring_coords(interior))
            polys.append(rings)
        return MultiPolygon(type="MultiPolygon", coordinates=polys)

    if isinstance(geom, ShapelyGeometryCollection):
        parts = [shapely_to_microjson(g) for g in geom.geoms]
        return GeometryCollection(type="GeometryCollection", geometries=parts)

    raise TypeError(f"Unsupported Shapely geometry type: {type(geom)}")


def neuron_from_tree_json(tree_json: str) -> NeuronMorphology:
    """Reconstruct a NeuronMorphology from its JSON-serialized tree."""
    samples = [SWCSample(**s) for s in json.loads(tree_json)]
    return NeuronMorphology(type="NeuronMorphology", tree=samples)


def slicestack_from_rows(
    rows: list[dict[str, Any]],
) -> SliceStack:
    """Reconstruct a SliceStack from exploded rows.

    Each row must have ``_slice_z``, a Shapely geometry, and optionally
    ``_slice_properties``.  Rows are sorted by ``_slice_z``.
    """
    sorted_rows = sorted(rows, key=lambda r: r["_slice_z"])
    slices = []
    for r in sorted_rows:
        geom = shapely_to_microjson(r["_shapely_geom"])
        props = None
        sp = r.get("_slice_properties")
        if sp:
            props = json.loads(sp)
        slices.append(Slice(z=r["_slice_z"], geometry=geom, properties=props))
    return SliceStack(type="SliceStack", slices=slices)
