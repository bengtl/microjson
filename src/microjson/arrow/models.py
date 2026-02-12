"""Pydantic configuration models for Arrow/GeoParquet export."""

from __future__ import annotations

from pydantic import BaseModel


class ArrowConfig(BaseModel):
    """Configuration for MicroJSON -> Arrow/GeoParquet conversion.

    Attributes:
        explode_slicestacks: When True, each slice in a SliceStack becomes
            a separate row with ``_slice_z`` and ``_slice_properties`` columns.
            When False, the whole SliceStack is serialized as a GeometryCollection.
        primary_geometry_column: Name of the WKB geometry column.
    """

    explode_slicestacks: bool = True
    primary_geometry_column: str = "geometry"
