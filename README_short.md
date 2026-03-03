# MicroJSON

MicroJSON is a JSON-based format inspired by [GeoJSON](https://geojson.org), designed to encode 2D and 3D annotations for microscopy images. It includes a Rust-accelerated tiling engine for scalable visualization and ML training pipelines.

## Features

- **2D and 3D Geometry**: Points, LineStrings, Polygons, PolyhedralSurfaces, and TIN mesh surfaces.
- **Pydantic Validation**: Strict schema validation via Pydantic v2 and geojson-pydantic.
- **Rust-Accelerated Tiling**: Parallel quadtree (2D) and octree (3D) tile generation with geometry simplification.
- **Multiple Output Formats**: PBF (MVT) vector tiles, 3D Tiles (GLB) with meshopt/Draco compression, tiled Parquet (ZSTD), and Neuroglancer precomputed meshes.
- **GeoParquet/Arrow**: Import and export via Apache Arrow.
- **Coordinate Systems**: OME-compatible multiscale metadata and affine transforms.

## Installation

```bash
pip install microjson
```

For optional Draco compression: `pip install microjson[draco]`

For all dependencies: `pip install microjson[all]`

## Requirements

- Python >= 3.11, < 3.14
- Rust toolchain (for building from source)

## License

MicroJSON is licensed under the MIT License.

---

This project is maintained by NovaGen Research Fund.
