#!/usr/bin/env python3
"""Export MicroJSON data to GeoParquet for ML/data-science pipelines.

Demonstrates three scenarios:
  1. SWC neuron files → GeoParquet (skeleton geometry + tree metadata)
  2. Mixed 2D/3D geometry → GeoParquet
  3. SliceStack → exploded GeoParquet (one row per slice)

Usage:
    # Export SWC files from swcs/ directory
    .venv/bin/python scripts/export_parquet.py

    # Export specific SWC files
    .venv/bin/python scripts/export_parquet.py file1.swc file2.swc

    # Custom output path
    .venv/bin/python scripts/export_parquet.py -o neurons.parquet

    # Run the built-in mixed-geometry demo
    .venv/bin/python scripts/export_parquet.py --demo

Then inspect with:
    python -c "import geopandas; print(geopandas.read_parquet('output.parquet'))"
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from microjson.arrow import ArrowConfig, to_arrow_table, to_geoparquet
from microjson.model import (
    MicroFeature,
    MicroFeatureCollection,
    NeuronMorphology,
    Slice,
    SliceStack,
    SWCSample,
)

from geojson_pydantic import MultiPolygon, Point, Polygon

DEFAULT_SWC_DIR = "swcs"
DEFAULT_OUT = "output.parquet"


def _pop_flag(args, flag):
    if flag in args:
        idx = args.index(flag)
        val = args[idx + 1]
        del args[idx : idx + 2]
        return val
    return None


def _demo_collection():
    """Build a mixed-geometry MicroFeatureCollection for demonstration."""
    # A 3D point
    point_feat = MicroFeature(
        type="Feature",
        id="cell_001",
        geometry=Point(type="Point", coordinates=(100.5, 200.3, 15.0)),
        properties={"cell_type": "pyramidal", "layer": 5, "area_um2": 312.7},
        featureClass="cell",
    )

    # A polygon (2D region)
    region_feat = MicroFeature(
        type="Feature",
        id="region_001",
        geometry=Polygon(
            type="Polygon",
            coordinates=[
                [(0, 0), (500, 0), (500, 500), (0, 500), (0, 0)],
                [(100, 100), (400, 100), (400, 400), (100, 400), (100, 100)],
            ],
        ),
        properties={"region_name": "cortex", "area_um2": 210000.0},
        featureClass="region",
    )

    # A small neuron morphology
    neuron_feat = MicroFeature(
        type="Feature",
        id="neuron_001",
        geometry=NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=250, y=250, z=10, r=8.0, parent=-1),
                SWCSample(id=2, type=2, x=280, y=250, z=12, r=1.5, parent=1),
                SWCSample(id=3, type=2, x=310, y=260, z=15, r=1.0, parent=2),
                SWCSample(id=4, type=3, x=250, y=220, z=8, r=2.0, parent=1),
                SWCSample(id=5, type=3, x=240, y=190, z=5, r=1.5, parent=4),
                SWCSample(id=6, type=4, x=250, y=280, z=12, r=1.8, parent=1),
                SWCSample(id=7, type=4, x=260, y=310, z=18, r=1.2, parent=6),
            ],
        ),
        properties={"species": "mouse", "brain_region": "hippocampus"},
        featureClass="neuron",
    )

    # A slice stack (2.5D contour)
    stack_feat = MicroFeature(
        type="Feature",
        id="stack_001",
        geometry=SliceStack(
            type="SliceStack",
            slices=[
                Slice(
                    z=0.0,
                    geometry=Polygon(
                        type="Polygon",
                        coordinates=[
                            [(50, 50), (150, 50), (150, 150), (50, 150), (50, 50)]
                        ],
                    ),
                    properties={"stain_intensity": 0.8},
                ),
                Slice(
                    z=5.0,
                    geometry=Polygon(
                        type="Polygon",
                        coordinates=[
                            [(60, 60), (140, 60), (140, 140), (60, 140), (60, 60)]
                        ],
                    ),
                    properties={"stain_intensity": 0.6},
                ),
                Slice(
                    z=10.0,
                    geometry=Polygon(
                        type="Polygon",
                        coordinates=[
                            [(70, 70), (130, 70), (130, 130), (70, 130), (70, 70)]
                        ],
                    ),
                    properties={"stain_intensity": 0.3},
                ),
            ],
        ),
        properties={"structure": "soma_contour"},
        featureClass="contour",
    )

    return MicroFeatureCollection(
        type="FeatureCollection",
        features=[point_feat, region_feat, neuron_feat, stack_feat],
    )


def _print_table_info(path):
    """Print a summary of the written GeoParquet file."""
    table = pq.read_table(str(path))
    meta = json.loads(table.schema.metadata[b"geo"])

    print(f"\n  Rows: {len(table)}")
    print(f"  Columns: {table.column_names}")
    print(f"  GeoParquet version: {meta['version']}")
    print(f"  Geometry types: {meta['columns']['geometry']['geometry_types']}")
    if "bbox" in meta["columns"]["geometry"]:
        bbox = meta["columns"]["geometry"]["bbox"]
        print(f"  Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    print(f"  File size: {path.stat().st_size / 1024:.1f} KB")


def main():
    args = sys.argv[1:]

    out_path = _pop_flag(args, "-o") or _pop_flag(args, "--output") or DEFAULT_OUT
    demo = "--demo" in args
    if demo:
        args.remove("--demo")

    out = Path(out_path)

    if demo:
        # Mixed geometry demo
        print("Building mixed-geometry demo collection...")
        coll = _demo_collection()
        to_geoparquet(coll, out)
        print(f"Wrote {out}")
        _print_table_info(out)

        # Also show the Arrow table in-memory
        table = to_arrow_table(coll)
        print(f"\n  Arrow table schema:")
        for field in table.schema:
            print(f"    {field.name}: {field.type}")

        print(f"\n  Inspect with:")
        print(f"    python -c \"import geopandas; print(geopandas.read_parquet('{out}'))\"")
        return

    # SWC mode
    from microjson.swc import swc_to_microjson

    if args:
        swc_paths = [Path(a) for a in args]
    else:
        swc_dir = Path(DEFAULT_SWC_DIR)
        swc_paths = sorted(swc_dir.glob("*.swc"))
        if not swc_paths:
            print(f"No SWC files in {swc_dir}/. Use --demo for a built-in example.")
            sys.exit(1)

    for p in swc_paths:
        if not p.exists():
            print(f"Error: file not found: {p}")
            sys.exit(1)

    features = []
    for swc_path in swc_paths:
        feat = swc_to_microjson(str(swc_path))
        if feat.properties is None:
            feat.properties = {}
        feat.properties["source_file"] = swc_path.name
        features.append(feat)

    print(f"Loaded {len(features)} SWC file(s)")

    coll = MicroFeatureCollection(
        type="FeatureCollection",
        features=features,
    )

    to_geoparquet(coll, out)
    print(f"Wrote {out}")
    _print_table_info(out)


if __name__ == "__main__":
    main()
