"""Read GeoParquet / Arrow tables back into MicroJSON models."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Union

import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from ..model import MicroFeature, MicroFeatureCollection
from ._from_geometry import neuron_from_tree_json, shapely_to_microjson, slicestack_from_rows


# Columns with special meaning — not copied to feature.properties
_RESERVED_COLUMNS = {"id", "featureClass", "geometry", "_neuron_tree",
                      "_slice_z", "_slice_properties"}


def _row_to_dict(table: pa.Table, idx: int) -> dict[str, Any]:
    """Extract a single row from an Arrow table as a plain dict."""
    return {col: table[col][idx].as_py() for col in table.column_names}


def _geometry_from_row(row: dict[str, Any], geom_col: str = "geometry") -> Any:
    """Decode WKB geometry from a row dict, returning a Shapely object or None."""
    wkb = row.get(geom_col)
    if wkb is None:
        return None
    return shapely.from_wkb(wkb)


def _feature_from_row(row: dict[str, Any], geom_col: str = "geometry") -> MicroFeature:
    """Build a MicroFeature from a plain row (no SliceStack grouping)."""
    shapely_geom = _geometry_from_row(row, geom_col)

    # NeuronMorphology: prefer _neuron_tree over WKB geometry
    tree_json = row.get("_neuron_tree")
    if tree_json is not None:
        geometry = neuron_from_tree_json(tree_json)
    else:
        geometry = shapely_to_microjson(shapely_geom)

    # Collect user properties (everything not reserved)
    props: dict[str, Any] = {}
    for k, v in row.items():
        if k not in _RESERVED_COLUMNS and k != geom_col:
            props[k] = v

    return MicroFeature(
        type="Feature",
        id=row.get("id"),
        geometry=geometry,
        properties=props if props else {},
        featureClass=row.get("featureClass"),
    )


def _group_slicestack_rows(
    rows: list[dict[str, Any]],
    geom_col: str,
) -> MicroFeature:
    """Re-aggregate exploded slice rows into a single MicroFeature with SliceStack geometry."""
    # Attach decoded shapely geometry to each row
    for r in rows:
        r["_shapely_geom"] = _geometry_from_row(r, geom_col)

    stack = slicestack_from_rows(rows)

    # Feature-level metadata comes from the first row
    first = rows[0]
    props: dict[str, Any] = {}
    for k, v in first.items():
        if k not in _RESERVED_COLUMNS and k != geom_col and k != "_shapely_geom":
            props[k] = v

    return MicroFeature(
        type="Feature",
        id=first.get("id"),
        geometry=stack,
        properties=props if props else {},
        featureClass=first.get("featureClass"),
    )


def from_arrow_table(
    table: pa.Table,
    geometry_column: str = "geometry",
) -> MicroFeatureCollection:
    """Convert a pyarrow.Table (with WKB geometry) to a MicroFeatureCollection.

    Rows with non-null ``_slice_z`` are re-grouped by ``id`` into
    SliceStack features.  Rows with ``_neuron_tree`` are reconstructed
    as NeuronMorphology.

    Args:
        table: A pyarrow.Table, typically read from a GeoParquet file.
        geometry_column: Name of the WKB geometry column.

    Returns:
        A MicroFeatureCollection.
    """
    has_slice_z = "_slice_z" in table.column_names

    # Separate slice rows from non-slice rows
    slice_groups: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    plain_rows: list[dict[str, Any]] = []

    for i in range(len(table)):
        row = _row_to_dict(table, i)
        if has_slice_z and row.get("_slice_z") is not None:
            slice_groups[row.get("id")].append(row)
        else:
            plain_rows.append(row)

    features: list[MicroFeature] = []

    # Build plain features (preserving order)
    for row in plain_rows:
        features.append(_feature_from_row(row, geometry_column))

    # Build SliceStack features (one per group)
    for _fid, rows in slice_groups.items():
        features.append(_group_slicestack_rows(rows, geometry_column))

    return MicroFeatureCollection(
        type="FeatureCollection",
        features=features,
    )


def from_geoparquet(
    path: str | Path,
    geometry_column: str | None = None,
) -> MicroFeatureCollection:
    """Read a GeoParquet file into a MicroFeatureCollection.

    Args:
        path: Path to the ``.parquet`` file.
        geometry_column: Override the geometry column name.  If None, it
            is read from the GeoParquet ``geo`` metadata.

    Returns:
        A MicroFeatureCollection.
    """
    import json

    table = pq.read_table(str(path))

    # Auto-detect geometry column from GeoParquet metadata
    if geometry_column is None:
        geo_meta = table.schema.metadata.get(b"geo")
        if geo_meta:
            meta = json.loads(geo_meta)
            geometry_column = meta.get("primary_column", "geometry")
        else:
            geometry_column = "geometry"

    return from_arrow_table(table, geometry_column)
