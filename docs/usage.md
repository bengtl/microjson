# Using MicroJSON

This page covers common usage patterns for the MicroJSON library, from basic validation to high-performance tiling pipelines.

## Requirements

- Python >= 3.11, < 3.14
- Install: `uv add microjson` (or `pip install microjson`)
- For Rust acceleration (tiling pipelines): built automatically via maturin when installing from source

## Validating MicroJSON and GeoJSON

```python
import microjson.model as mj
import json

# Validate a MicroJSON file
with open("annotations.json") as f:
    data = json.load(f)
microjson_obj = mj.MicroJSON.model_validate(data)

# Validate a GeoJSON file (any GeoJSON is valid MicroJSON)
with open("features.geojson") as f:
    data = json.load(f)
geojson_obj = mj.GeoJSON.model_validate(data)
```

## Creating MicroJSON from a DataFrame

The `df_to_microjson` function converts a pandas DataFrame into a MicroJSON FeatureCollection.

::: microjson.examples.df_to_microjson.df_to_microjson
    :docstring:

```python
import pandas as pd
from microjson.examples.df_to_microjson import df_to_microjson

data = [
    {
        "type": "Feature",
        "geometryType": "Point",
        "coordinates": [0, 0],
        "name": "Point 1",
        "value": 1,
        "values": [1, 2, 3],
    },
    {
        "type": "Feature",
        "geometryType": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        "name": "Polygon 1",
        "value": 2,
        "values": [4, 5, 6],
    },
]

df = pd.DataFrame(data)
feature_collection = df_to_microjson(df)
print(feature_collection.model_dump_json(indent=2, exclude_unset=True))
```

## 2D Tiling (Rust-Accelerated)

Generate vector tiles from GeoJSON data using the Rust pipeline. See the [Tiling](tiling.md) page for the full specification.

### PBF (MVT) Output

```python
from microjson._rs import StreamingTileGenerator2D
from microjson.tiling2d import generate_pbf, read_pbf

gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64/4096)

geojson_str = open("data.json").read()
bounds = (0.0, 0.0, 10000.0, 10000.0)
gen.add_geojson(geojson_str, bounds)

# Write PBF tiles to a directory tree ({z}/{x}/{y}.pbf)
n_tiles = generate_pbf(gen, "tiles/", bounds, simplify=True)

# Read tiles back
features = read_pbf("tiles/", bounds, zoom=0)
```

### Tiled Parquet Output

```python
from microjson._rs import StreamingTileGenerator2D
from microjson.tiling2d import generate_parquet, read_parquet

gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64/4096)

geojson_str = open("data.json").read()
bounds = (0.0, 0.0, 10000.0, 10000.0)
gen.add_geojson(geojson_str, bounds)

# Write tiled Parquet (ZSTD compressed)
n_rows = generate_parquet(gen, "output.parquet", bounds, simplify=True)

# Read with optional zoom/tile filtering
rows = read_parquet("output.parquet", zoom=0)
```

A complete example script is at `src/microjson/examples/tiling_rust.py`.

## 3D Tiling

Generate 3D Tiles, Parquet, or Neuroglancer output from OBJ mesh files.

```python
from microjson._rs import StreamingTileGenerator

gen = StreamingTileGenerator(
    min_zoom=0, max_zoom=5,
    extent=4096, extent_z=4096,
    buffer=0.0, base_cells=16,
)

# Ingest OBJ files
obj_paths = ["neuron_001.obj", "neuron_002.obj"]
bounds = (xmin, ymin, zmin, xmax, ymax, zmax)
gen.add_obj_files(obj_paths, bounds)

# Generate 3D Tiles with meshopt compression
gen.generate_3dtiles("output/3dtiles", bounds, compression="meshopt")
```

## GeoParquet Import/Export

```python
from microjson import to_geoparquet, from_geoparquet, ArrowConfig

# Export MicroJSON to GeoParquet
config = ArrowConfig()
to_geoparquet(feature_collection, "output.parquet", config)

# Import from GeoParquet
fc = from_geoparquet("output.parquet")
```

## glTF/GLB Export

```python
from microjson import to_glb, GltfConfig

config = GltfConfig()
to_glb(feature_collection, "output.glb", config)
```
