# muDM

muDM (micro Data Model) is a JSON-based data model inspired by [GeoJSON](https://geojson.org), designed to encode a variety of data structures related to microscopy images. It supports 2D and 3D annotations — from reference points and regions of interest to triangle mesh surfaces — and includes a high-performance Rust-accelerated tiling engine for scalable visualization and ML training pipelines.

For more extensive documentation, please refer to the [online documentation](https://polusai.github.io/microjson/).

## Features

- **2D and 3D Geometry**: Points, LineStrings, Polygons, MultiPolygons, PolyhedralSurfaces, and TIN (Triangulated Irregular Network) mesh surfaces.
- **Pydantic Validation**: Strict schema validation via [Pydantic v2](https://docs.pydantic.dev/) and [geojson-pydantic](https://developmentseed.org/geojson-pydantic/). Any GeoJSON is valid muDM.
- **Rust-Accelerated Tiling**: Parallel 2D (quadtree) and 3D (octree) tile generation via PyO3 and rayon, with geometry clipping and simplification (Douglas-Peucker for 2D, QEM for 3D meshes).
- **Multiple Output Formats**:
  - **PBF (MVT)**: Standard Mapbox Vector Tiles for web map viewers (Leaflet, MapLibre).
  - **3D Tiles (GLB)**: OGC 3D Tiles with meshopt or Draco mesh compression for Three.js / Cesium.
  - **Tiled Parquet**: ZSTD-compressed columnar format for ML training with zero-copy reads.
  - **Neuroglancer**: Precomputed mesh format for browser-based 3D visualization.
- **GeoParquet/Arrow**: Import and export via Apache Arrow for data science interoperability.
- **Coordinate Systems**: OME-compatible multiscale metadata, affine transforms, and voxel-to-physical conversions.
- **Binary Image Utilities**: Convert between binary/label images and muDM polygon annotations.

## Requirements

- Python >= 3.11, < 3.14
- Rust toolchain (for building from source via maturin)

## Installation

Install from PyPI:

```bash
pip install mudm
```

For optional Draco compression support:

```bash
pip install mudm[draco]
```

For the full set of dependencies (including binary image utilities):

```bash
pip install mudm[all]
```

## Quick Start

### Validate muDM / GeoJSON

```python
import mudm.model as mj
import json

with open("annotations.json") as f:
    data = json.load(f)

obj = mj.MuDM.model_validate(data)
```

### 2D Tiling (Rust-accelerated)

```python
from mudm._rs import StreamingTileGenerator2D
from mudm.tiling2d import generate_pbf

gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64/4096)
gen.add_geojson(open("data.json").read(), bounds)
generate_pbf(gen, "tiles/", bounds)
```

### 3D Tiling (OBJ meshes to 3D Tiles)

```python
from mudm._rs import StreamingTileGenerator

gen = StreamingTileGenerator(
    min_zoom=0, max_zoom=5,
    extent=4096, extent_z=4096,
    buffer=0.0, base_cells=16,
)
gen.add_obj_files(obj_paths, bounds)
gen.generate_3dtiles("output/3dtiles", bounds, compression="meshopt")
```

See `src/mudm/examples/` for more complete examples.

## Specification

For detailed information about the muDM structure, see the [Specification](docs/index.md).

## External Resources

The GeoJSON test files are copied from the [GeoJSON Schema GitHub repository](https://github.com/geojson/schema), and are Copyright (c) 2018 Tim Schaub under MIT License.

## Contribution

We welcome contributions to the development and enhancement of muDM. Whether you're reporting bugs, suggesting enhancements, or contributing to the code, your input is highly appreciated.

## License

muDM is primarily licensed under the [MIT License](./LICENSE).

Portions of this project are derived from 'geojson2vt' and are located in 'src/mudm/mudm2vt'.
These portions are licensed under the [ISC License](./src/mudm/mudm2vt/LICENSE).

---

This project is maintained by NovaGen Research Fund. For any queries or further discussion, please contact [Novagen](info@novagenresearch.org).
