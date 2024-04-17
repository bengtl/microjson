"""Microjson using manual MicroJSON pydantic models,
but autogenerated GeoJSON models."""
from typing import List, Optional, Union, Dict
from pydantic import BaseModel, StrictInt, StrictStr
from geojson.Feature import GeojsonFeature
from geojson.FeatureCollection import GeojsonFeaturecollection
from geojson.Geometry import GeojsonGeometry
from geojson.GeometryCollection import GeojsonGeometrycollection
from microjson.model import Axis, CoordinateSystem
from microjson.model import Properties, ValueRange


class GeoJSONAuto(BaseModel):
    """The root object of a GeoJSON file"""

    __root__: Union[
        GeojsonFeature,
        GeojsonFeaturecollection,
        GeojsonGeometry,
        GeojsonGeometrycollection,
    ]


class MicroFeature(GeojsonFeature):
    """A MicroJSON feature, which is a GeoJSON feature with additional
    metadata"""

    coordinateSystem: Optional[List[Axis]]
    ref: Optional[Union[StrictStr, StrictInt]]
    properties: Properties


class MicroFeatureCollection(GeojsonFeaturecollection):
    """A MicroJSON feature collection, which is a GeoJSON feature
    collection with additional metadata"""

    coordinateSystem: Optional[CoordinateSystem]
    valueRange: Optional[Dict[str, ValueRange]]
    descriptiveFields: Optional[List[str]]


class MicroJSONAuto(BaseModel):
    """The root object of a MicroJSON file"""

    __root__: Union[
        MicroFeature, MicroFeatureCollection, GeojsonGeometry, GeojsonGeometrycollection
    ]
