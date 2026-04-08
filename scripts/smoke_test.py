#!/usr/bin/env python3
"""Smoke test for mudm package installation.

Usage:
    python scripts/smoke_test.py
"""

import json

from mudm.model import MuDM, MuDMFeature, GeoJSON
from mudm._rs import StreamingTileGenerator, StreamingTileGenerator2D

# Validate a simple feature
data = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [10, 20]},
        "properties": {"label": "test"},
    }],
}
obj = MuDM.model_validate(data)
print(f"Model OK: {len(obj.root.features)} feature(s)")

# 2D tiling round-trip
gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=2, buffer=0.0)
bounds = (0.0, 0.0, 100.0, 100.0)
gen.add_geojson(json.dumps(data), bounds)
print(f"2D generator OK: {gen.feature_count_val()} feature(s)")

# 3D generator instantiation
gen3d = StreamingTileGenerator(min_zoom=0, max_zoom=2)
print("3D generator OK")

print("All checks passed.")
