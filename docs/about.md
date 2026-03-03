# About MicroJSON

MicroJSON is a format and Python library for representing microscopy annotations, regions of interest, and spatial metadata. Inspired by [GeoJSON](https://geojson.org), it extends the GeoJSON specification with microscopy-specific features while maintaining full backwards compatibility — any GeoJSON is valid MicroJSON, and any MicroJSON is valid GeoJSON.

## Key Capabilities

- **Format Specification**: A JSON-based format for 2D and 3D microscopy annotations, with support for coordinate systems, multiscale metadata, and provenance tracking.
- **Pydantic Validation**: Python models built on [Pydantic v2](https://docs.pydantic.dev/) and [geojson-pydantic](https://developmentseed.org/geojson-pydantic/) for strict schema validation.
- **Rust-Accelerated Tiling**: High-performance 2D and 3D vector tile generation via PyO3, with parallel processing (rayon), quadtree/octree spatial indexing, and geometry simplification.
- **Multiple Output Formats**: PBF (MVT) vector tiles, 3D Tiles (GLB) with meshopt or Draco compression, tiled Parquet (ZSTD) for ML training, and Neuroglancer precomputed format.
- **GeoParquet/Arrow**: Import and export via Apache Arrow and GeoParquet for interoperability with data science tools.

## Architecture

The library has two layers:

1. **Python layer** (`microjson`): Pydantic models, coordinate transforms, GeoParquet I/O, glTF assembly, and high-level pipeline orchestration.
2. **Rust layer** (`microjson._rs`): Hot-path tile generation, geometry clipping, mesh simplification (QEM), protobuf/MVT encoding, and compression (meshopt, Draco, ZSTD). Built with [maturin](https://www.maturin.rs/) and [PyO3](https://pyo3.rs/).

## Requirements

- Python >= 3.11, < 3.14
- Rust toolchain (for building from source)

## Links

- **Repository**: [github.com/PolusAI/microjson](https://github.com/PolusAI/microjson)
- **License**: MIT
