"""Pydantic configuration models for Arrow/GeoParquet export."""

from __future__ import annotations

from pydantic import BaseModel


class ArrowConfig(BaseModel):
    """Configuration for MicroJSON → Arrow/GeoParquet conversion.

    Attributes:
        explode_slicestacks: When True, each slice in a SliceStack becomes
            a separate row with ``_slice_z`` and ``_slice_properties`` columns.
            When False, the whole SliceStack is serialized as a GeometryCollection.
        include_neuron_tree: When True, NeuronMorphology features get a
            ``_neuron_tree`` column with the full SWC tree as JSON.
        primary_geometry_column: Name of the WKB geometry column.
    """

    explode_slicestacks: bool = True
    include_neuron_tree: bool = True
    primary_geometry_column: str = "geometry"
