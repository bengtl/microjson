#!/usr/bin/env python3
"""End-to-end MouseLight HortaObj benchmark pipeline.

Downloads OBJ brain region meshes, converts to MicroJSON TIN features with
Allen CCF metadata, generates mjb and 3D Tiles, and reports benchmarks.

Usage::

    # Download data first (one-time, ~537 MB):
    aws s3 sync s3://janelia-mouselight-imagery/registration/2021-09-16/HortaObj/ \\
      data/mouselight/2021-09-16/ --no-sign-request

    # Run full pipeline:
    .venv/bin/python scripts/benchmark_mouselight.py \\
      --obj-dir data/mouselight/2021-09-16/ \\
      --output-dir data/mouselight/tiles/ \\
      --max-zoom 3 --workers 0

    # Skip 3D Tiles (much faster):
    .venv/bin/python scripts/benchmark_mouselight.py \\
      --obj-dir data/mouselight/2021-09-16/ \\
      --output-dir data/mouselight/tiles/ \\
      --skip-3dtiles

    # With CSV export and DataLoader benchmark:
    .venv/bin/python scripts/benchmark_mouselight.py \\
      --obj-dir data/mouselight/2021-09-16/ \\
      --output-dir data/mouselight/tiles/ \\
      --csv results.csv --dataloader
"""

from __future__ import annotations

import argparse
import gzip
import os
import random
import shutil
import statistics
import struct
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Force unbuffered stdout for progress output
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# Ensure src/ and scripts/ are importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from microjson.tiling3d import RUST_AVAILABLE
from microjson.tiling3d.convert3d import compute_bounds_3d, convert_feature_3d
from microjson.tiling3d.generator3d import TileGenerator3D
from microjson.tiling3d.octree import OctreeConfig
from microjson.tiling3d.projector3d import CartesianProjector3D
from microjson.tiling3d.reader3d import decode_tile

# Import obj_to_microjson functions
from obj_to_microjson import (
    build_collection,
    fetch_allen_ontology,
    match_region,
    obj_to_feature,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    """Total bytes of all files under *path*."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _dir_size_gzipped(path: Path) -> int:
    """Total gzip-compressed size of all tile files."""
    total = 0
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        total += len(gzip.compress(f.read_bytes(), compresslevel=6))
    return total


def _collect_tile_files(path: Path, suffix: str) -> list[Path]:
    return sorted(path.rglob(f"*{suffix}"))


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


# ---------------------------------------------------------------------------
# Step 1: OBJ → MicroJSON
# ---------------------------------------------------------------------------

def load_obj_collection(
    obj_dir: Path,
    *,
    no_ontology: bool = False,
    max_files: int | None = None,
):
    """Load OBJ files and convert to MicroFeatureCollection."""
    obj_paths = sorted(obj_dir.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No .obj files found in {obj_dir}", file=sys.stderr)
        sys.exit(1)

    if max_files and max_files < len(obj_paths):
        obj_paths = obj_paths[:max_files]

    print(f"Found {len(obj_paths)} OBJ files in {obj_dir}")

    # Fetch ontology
    ontology = None
    if not no_ontology:
        print("Fetching Allen CCF ontology...")
        t0 = time.perf_counter()
        ontology = fetch_allen_ontology()
        print(f"  Loaded {len(ontology)} ontology entries ({_fmt_time(time.perf_counter() - t0)})")

    # Convert OBJ → MicroFeature
    print(f"Converting {len(obj_paths)} OBJ files to MicroJSON TIN features...")
    t0 = time.perf_counter()
    features = []
    total_verts = 0
    total_faces = 0
    for i, obj_path in enumerate(obj_paths, 1):
        if i % 50 == 0 or i == len(obj_paths):
            print(f"  [{i}/{len(obj_paths)}] {obj_path.name}", end="\r", file=sys.stderr)
        feat = obj_to_feature(str(obj_path), ontology)
        features.append(feat)
        total_verts += feat.properties.get("vertex_count", 0)
        total_faces += feat.properties.get("face_count", 0)

    print(file=sys.stderr)  # newline after \r
    convert_time = time.perf_counter() - t0

    collection = build_collection(features, ontology)

    print(f"  {len(features)} features, {total_verts:,} vertices, {total_faces:,} faces")
    print(f"  Conversion time: {_fmt_time(convert_time)}")

    return collection, convert_time


# ---------------------------------------------------------------------------
# Step 2: Tile Generation
# ---------------------------------------------------------------------------

def generate_tiles(
    collection,
    output_dir: Path,
    *,
    max_zoom: int = 3,
    workers: int | None = None,
    skip_3dtiles: bool = False,
) -> dict[str, Any]:
    """Generate mjb (and optionally 3dtiles) from collection."""
    results: dict[str, Any] = {}

    # mjb
    mjb_dir = output_dir / "mjb"
    if mjb_dir.exists():
        shutil.rmtree(mjb_dir)
    mjb_dir.mkdir(parents=True, exist_ok=True)

    config = OctreeConfig(max_zoom=max_zoom)
    gen = TileGenerator3D(config, output_format="mjb", workers=workers)

    print(f"\nGenerating mjb tiles (zoom 0-{max_zoom})...")
    t0 = time.perf_counter()
    gen.add_features(collection)
    t_index = time.perf_counter() - t0
    print(f"  Indexing: {_fmt_time(t_index)}")

    t0 = time.perf_counter()
    n_tiles = gen.generate(mjb_dir)
    t_gen = time.perf_counter() - t0
    gen.write_metadata(mjb_dir)

    mjb_size = _dir_size(mjb_dir)
    mjb_gzip = _dir_size_gzipped(mjb_dir)

    results["mjb_tiles"] = n_tiles
    results["mjb_index_time"] = t_index
    results["mjb_gen_time"] = t_gen
    results["mjb_size_raw"] = mjb_size
    results["mjb_size_gzip"] = mjb_gzip
    results["mjb_dir"] = mjb_dir

    print(f"  {n_tiles} tiles in {_fmt_time(t_gen)}")
    print(f"  Size: {_fmt_bytes(mjb_size)} raw, {_fmt_bytes(mjb_gzip)} gzipped")
    print(f"  Throughput: {n_tiles / t_gen:.0f} tiles/s")

    # 3dtiles (optional)
    if not skip_3dtiles:
        tiles3d_dir = output_dir / "3dtiles"
        if tiles3d_dir.exists():
            shutil.rmtree(tiles3d_dir)
        tiles3d_dir.mkdir(parents=True, exist_ok=True)

        gen3d = TileGenerator3D(
            OctreeConfig(max_zoom=max_zoom),
            output_format="3dtiles",
            workers=workers,
        )

        print(f"\nGenerating 3D Tiles (zoom 0-{max_zoom})...")
        t0 = time.perf_counter()
        gen3d.add_features(collection)
        t_index_3d = time.perf_counter() - t0

        t0 = time.perf_counter()
        n_tiles_3d = gen3d.generate(tiles3d_dir)
        t_gen_3d = time.perf_counter() - t0
        gen3d.write_metadata(tiles3d_dir)

        tiles3d_size = _dir_size(tiles3d_dir)
        tiles3d_gzip = _dir_size_gzipped(tiles3d_dir)

        results["3dtiles_tiles"] = n_tiles_3d
        results["3dtiles_index_time"] = t_index_3d
        results["3dtiles_gen_time"] = t_gen_3d
        results["3dtiles_size_raw"] = tiles3d_size
        results["3dtiles_size_gzip"] = tiles3d_gzip
        results["3dtiles_dir"] = tiles3d_dir

        print(f"  {n_tiles_3d} tiles in {_fmt_time(t_gen_3d)}")
        print(f"  Size: {_fmt_bytes(tiles3d_size)} raw, {_fmt_bytes(tiles3d_gzip)} gzipped")
        print(f"  Throughput: {n_tiles_3d / t_gen_3d:.0f} tiles/s")

    return results


def _build_mouselight_tags(obj_path: Path, ontology: dict | None) -> dict:
    """Build a tags dict for a MouseLight OBJ file from the Allen CCF ontology."""
    stem = obj_path.stem
    tags: dict = {"mesh_name": stem, "source": obj_path.name}

    if ontology is not None:
        region = match_region(stem, ontology)
        if region:
            tags["ccf_id"] = region["id"]
            tags["name"] = region["name"]
            tags["acronym"] = region["acronym"]
            if region["parent_structure_id"] is not None:
                tags["parent_id"] = region["parent_structure_id"]
            if region["color_hex_triplet"]:
                tags["color"] = f"#{region['color_hex_triplet']}"
            if region["structure_id_path"]:
                tags["hierarchy_path"] = region["structure_id_path"]

    return tags


def generate_tiles_streaming(
    obj_dir: Path,
    output_dir: Path,
    *,
    max_zoom: int = 3,
    skip_3dtiles: bool = False,
    no_ontology: bool = False,
) -> tuple[dict[str, Any], float]:
    """Generate tiles using parallel Rust OBJ ingest (no Python MicroFeatureCollection)."""
    from microjson._rs import StreamingTileGenerator, scan_obj_bounds

    results: dict[str, Any] = {}

    obj_paths = sorted(obj_dir.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No .obj files found in {obj_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(obj_paths)} OBJ files in {obj_dir}")

    # Fetch ontology for tags
    ontology = None
    if not no_ontology:
        print("Fetching Allen CCF ontology...")
        t0 = time.perf_counter()
        ontology = fetch_allen_ontology()
        print(f"  Loaded {len(ontology)} ontology entries ({_fmt_time(time.perf_counter() - t0)})")

    # Build tags list (Python, fast — just dict construction)
    tags_list = [_build_mouselight_tags(p, ontology) for p in obj_paths]
    path_strs = [str(p) for p in obj_paths]

    # Scan bounds (parallel Rust, GIL released)
    print(f"Scanning bounds ({len(obj_paths)} files, parallel)...")
    t0 = time.perf_counter()
    bounds = scan_obj_bounds(path_strs)
    t_bounds = time.perf_counter() - t0
    print(f"  Bounds: x=[{bounds[0]:.0f}, {bounds[3]:.0f}] "
          f"y=[{bounds[1]:.0f}, {bounds[4]:.0f}] "
          f"z=[{bounds[2]:.0f}, {bounds[5]:.0f}]  ({_fmt_time(t_bounds)})")

    n_cores = os.cpu_count() or 1
    print(f"  Parallel ingest using {n_cores} cores (rayon)")

    # --- mjb ---
    mjb_dir = output_dir / "mjb"
    if mjb_dir.exists():
        shutil.rmtree(mjb_dir)
    mjb_dir.mkdir(parents=True, exist_ok=True)

    gen = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    print(f"\nIngesting {len(obj_paths)} OBJ files (parallel Rust, zoom 0-{max_zoom})...")
    t0 = time.perf_counter()
    gen.add_obj_files(path_strs, bounds, tags_list)
    t_ingest = time.perf_counter() - t0
    print(f"  Ingest: {_fmt_time(t_ingest)} ({len(obj_paths) / t_ingest:.0f} files/s)")

    print("Encoding mjb tiles (parallel rayon)...")
    t0 = time.perf_counter()
    n_tiles = gen.generate_mjb(str(mjb_dir), "default")
    t_gen = time.perf_counter() - t0

    tilejson_path = mjb_dir / "tilejson3d.json"
    gen.write_tilejson3d(str(tilejson_path), bounds, "default")

    mjb_size = _dir_size(mjb_dir)
    mjb_gzip = _dir_size_gzipped(mjb_dir)

    results["mjb_tiles"] = n_tiles
    results["mjb_index_time"] = t_ingest
    results["mjb_gen_time"] = t_gen
    results["mjb_size_raw"] = mjb_size
    results["mjb_size_gzip"] = mjb_gzip
    results["mjb_dir"] = mjb_dir

    print(f"  {n_tiles} tiles in {_fmt_time(t_gen)}")
    print(f"  Size: {_fmt_bytes(mjb_size)} raw, {_fmt_bytes(mjb_gzip)} gzipped")
    if t_gen > 0:
        print(f"  Throughput: {n_tiles / t_gen:.0f} tiles/s")

    # --- 3dtiles (optional) ---
    if not skip_3dtiles:
        tiles3d_dir = output_dir / "3dtiles"
        if tiles3d_dir.exists():
            shutil.rmtree(tiles3d_dir)
        tiles3d_dir.mkdir(parents=True, exist_ok=True)

        gen3d = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
        print(f"\nIngesting for 3D Tiles (parallel Rust)...")
        t0 = time.perf_counter()
        gen3d.add_obj_files(path_strs, bounds, tags_list)
        t_ingest_3d = time.perf_counter() - t0
        print(f"  Ingest: {_fmt_time(t_ingest_3d)}")

        print("Encoding 3D Tiles (parallel rayon)...")
        t0 = time.perf_counter()
        n_tiles_3d = gen3d.generate_3dtiles(str(tiles3d_dir), bounds)
        t_gen_3d = time.perf_counter() - t0

        tiles3d_size = _dir_size(tiles3d_dir)
        tiles3d_gzip = _dir_size_gzipped(tiles3d_dir)

        results["3dtiles_tiles"] = n_tiles_3d
        results["3dtiles_index_time"] = t_ingest_3d
        results["3dtiles_gen_time"] = t_gen_3d
        results["3dtiles_size_raw"] = tiles3d_size
        results["3dtiles_size_gzip"] = tiles3d_gzip
        results["3dtiles_dir"] = tiles3d_dir

        print(f"  {n_tiles_3d} tiles in {_fmt_time(t_gen_3d)}")
        print(f"  Size: {_fmt_bytes(tiles3d_size)} raw, {_fmt_bytes(tiles3d_gzip)} gzipped")
        if t_gen_3d > 0:
            print(f"  Throughput: {n_tiles_3d / t_gen_3d:.0f} tiles/s")

    # --- feature-centric mjb ---
    feat_mjb_dir = output_dir / "feature_mjb"
    if feat_mjb_dir.exists():
        shutil.rmtree(feat_mjb_dir)
    feat_mjb_dir.mkdir(parents=True, exist_ok=True)

    gen_fmjb = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    print(f"\nIngesting for feature-centric MJB (parallel Rust)...")
    t0 = time.perf_counter()
    gen_fmjb.add_obj_files(path_strs, bounds, tags_list)
    t_ingest_fmjb = time.perf_counter() - t0
    print(f"  Ingest: {_fmt_time(t_ingest_fmjb)}")

    print("Encoding per-feature MJB (parallel rayon)...")
    t0 = time.perf_counter()
    n_features_fmjb = gen_fmjb.generate_feature_mjb(str(feat_mjb_dir), bounds)
    t_gen_fmjb = time.perf_counter() - t0

    feat_mjb_size = _dir_size(feat_mjb_dir)
    results["feature_mjb_features"] = n_features_fmjb
    results["feature_mjb_gen_time"] = t_gen_fmjb
    results["feature_mjb_size_raw"] = feat_mjb_size
    results["feature_mjb_dir"] = feat_mjb_dir

    print(f"  {n_features_fmjb} features in {_fmt_time(t_gen_fmjb)}")
    print(f"  Size: {_fmt_bytes(feat_mjb_size)} raw")

    # --- neuroglancer ---
    ng_dir = output_dir / "neuroglancer"
    if ng_dir.exists():
        shutil.rmtree(ng_dir)
    ng_dir.mkdir(parents=True, exist_ok=True)

    gen_ng = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    print(f"\nIngesting for Neuroglancer (parallel Rust)...")
    t0 = time.perf_counter()
    gen_ng.add_obj_files(path_strs, bounds, tags_list)
    t_ingest_ng = time.perf_counter() - t0
    print(f"  Ingest: {_fmt_time(t_ingest_ng)}")

    print("Encoding Neuroglancer multilod_draco meshes (parallel rayon)...")
    t0 = time.perf_counter()
    n_features_ng = gen_ng.generate_neuroglancer_multilod(str(ng_dir), bounds)
    t_gen_ng = time.perf_counter() - t0

    ng_size = _dir_size(ng_dir)
    results["neuroglancer_features"] = n_features_ng
    results["neuroglancer_gen_time"] = t_gen_ng
    results["neuroglancer_size_raw"] = ng_size
    results["neuroglancer_dir"] = ng_dir

    print(f"  {n_features_ng} features in {_fmt_time(t_gen_ng)}")
    print(f"  Size: {_fmt_bytes(ng_size)} raw")

    # --- parquet ---
    from microjson.tiling3d.parquet_writer import generate_parquet as _gen_pq

    pq_path = output_dir / "tiles.parquet"
    gen_pq = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    print(f"\nIngesting for Parquet (parallel Rust)...")
    t0 = time.perf_counter()
    gen_pq.add_obj_files(path_strs, bounds, tags_list)
    t_ingest_pq = time.perf_counter() - t0
    print(f"  Ingest: {_fmt_time(t_ingest_pq)}")

    print("Writing Parquet (ZSTD)...")
    t0 = time.perf_counter()
    n_rows_pq = _gen_pq(gen_pq, pq_path, bounds)
    t_gen_pq = time.perf_counter() - t0

    pq_size = pq_path.stat().st_size if pq_path.exists() else 0
    results["parquet_rows"] = n_rows_pq
    results["parquet_gen_time"] = t_gen_pq
    results["parquet_size_raw"] = pq_size
    results["parquet_path"] = pq_path

    print(f"  {n_rows_pq} rows in {_fmt_time(t_gen_pq)}")
    print(f"  Size: {_fmt_bytes(pq_size)}")

    return results, t_ingest


# ---------------------------------------------------------------------------
# Step 3: Decode Benchmarks
# ---------------------------------------------------------------------------

def bench_decode(
    mjb_dir: Path,
    *,
    n_iterations: int = 20,
    max_sample: int = 50,
) -> dict[str, Any]:
    """Benchmark mjb decode latency on sample tiles."""
    mjb_files = _collect_tile_files(mjb_dir, ".mjb")
    if not mjb_files:
        return {}

    if len(mjb_files) > max_sample:
        mjb_files = random.sample(mjb_files, max_sample)

    # Pre-load bytes
    tile_bytes = [f.read_bytes() for f in mjb_files]
    tile_sizes = [len(b) for b in tile_bytes]

    # Warmup
    for data in tile_bytes[:5]:
        decode_tile(data)

    # Benchmark
    times: list[float] = []
    for _ in range(n_iterations):
        for data in tile_bytes:
            t0 = time.perf_counter()
            decode_tile(data)
            times.append(time.perf_counter() - t0)

    # Per-tile feature counts from one pass
    total_features = 0
    total_mesh_bytes = 0
    for data in tile_bytes:
        layers = decode_tile(data)
        for layer in layers:
            for feat in layer["features"]:
                total_features += 1
                total_mesh_bytes += len(feat.get("mesh_positions", b""))

    return {
        "sampled_tiles": len(tile_bytes),
        "median_tile_bytes": statistics.median(tile_sizes),
        "total_features": total_features,
        "total_mesh_bytes": total_mesh_bytes,
        "decode_median_us": statistics.median(times) * 1_000_000,
        "decode_p95_us": sorted(times)[int(len(times) * 0.95)] * 1_000_000,
        "decode_p99_us": sorted(times)[int(len(times) * 0.99)] * 1_000_000,
    }


def bench_decode_3dtiles(
    tiles3d_dir: Path,
    *,
    n_iterations: int = 20,
    max_sample: int = 50,
) -> dict[str, Any]:
    """Benchmark 3D Tiles (GLB) decode latency."""
    try:
        import pygltflib
    except ImportError:
        print("  (pygltflib not installed, skipping 3D Tiles decode benchmark)")
        return {}

    glb_files = _collect_tile_files(tiles3d_dir, ".glb")
    if not glb_files:
        return {}

    if len(glb_files) > max_sample:
        glb_files = random.sample(glb_files, max_sample)

    tile_bytes = [f.read_bytes() for f in glb_files]

    # Warmup
    for data in tile_bytes[:5]:
        pygltflib.GLTF2.load_from_bytes(data)

    times: list[float] = []
    for _ in range(n_iterations):
        for data in tile_bytes:
            t0 = time.perf_counter()
            pygltflib.GLTF2.load_from_bytes(data)
            times.append(time.perf_counter() - t0)

    return {
        "sampled_tiles": len(tile_bytes),
        "decode_median_us": statistics.median(times) * 1_000_000,
        "decode_p95_us": sorted(times)[int(len(times) * 0.95)] * 1_000_000,
    }


# ---------------------------------------------------------------------------
# Step 4: Memory Benchmark
# ---------------------------------------------------------------------------

def bench_memory(mjb_dir: Path, tiles3d_dir: Path | None = None) -> dict[str, Any]:
    """Measure peak memory during full-tileset decode."""
    results: dict[str, Any] = {}

    mjb_bytes = [f.read_bytes() for f in _collect_tile_files(mjb_dir, ".mjb")]

    tracemalloc.start()
    for data in mjb_bytes:
        decode_tile(data)
    _, mjb_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results["mjb_peak_mb"] = mjb_peak / (1024 * 1024)

    if tiles3d_dir:
        try:
            import pygltflib
        except ImportError:
            return results

        glb_bytes = [f.read_bytes() for f in _collect_tile_files(tiles3d_dir, ".glb")]
        tracemalloc.start()
        for data in glb_bytes:
            pygltflib.GLTF2.load_from_bytes(data)
        _, glb_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results["3dtiles_peak_mb"] = glb_peak / (1024 * 1024)

    return results


# ---------------------------------------------------------------------------
# Step 5: DataLoader Benchmark (optional)
# ---------------------------------------------------------------------------

def bench_dataloader(
    mjb_dir: Path,
    tiles3d_dir: Path | None = None,
    *,
    n_epochs: int = 3,
    batch_size: int = 4,
) -> dict[str, Any] | None:
    """Measure tiles/sec through a PyTorch DataLoader."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        print("  (torch not installed, skipping DataLoader benchmark)")
        return None

    class Mvt3Dataset(Dataset):
        def __init__(self, tile_dir: Path):
            self._data = [f.read_bytes() for f in _collect_tile_files(tile_dir, ".mjb")]

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, idx: int) -> torch.Tensor:
            layers = decode_tile(self._data[idx])
            parts: list[torch.Tensor] = []
            for layer in layers:
                for feat in layer["features"]:
                    mesh_pos = feat.get("mesh_positions", b"")
                    if mesh_pos:
                        parts.append(torch.frombuffer(
                            bytearray(mesh_pos), dtype=torch.float32,
                        ))
            if not parts:
                return torch.zeros(1)
            return torch.cat(parts) if len(parts) > 1 else parts[0]

    def _collate(batch: list) -> list:
        return batch

    results: dict[str, Any] = {}

    ds = Mvt3Dataset(mjb_dir)
    if len(ds) == 0:
        return results

    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=_collate)

    total_tiles = 0
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        for batch in loader:
            total_tiles += len(batch)
    elapsed = time.perf_counter() - t0
    results["mjb_tiles_per_sec"] = total_tiles / elapsed if elapsed > 0 else 0
    results["mjb_elapsed_sec"] = elapsed

    if tiles3d_dir:
        try:
            import pygltflib

            class GlbDataset(Dataset):
                def __init__(self, tile_dir: Path):
                    self._data = [f.read_bytes() for f in _collect_tile_files(tile_dir, ".glb")]

                def __len__(self) -> int:
                    return len(self._data)

                def __getitem__(self, idx: int) -> torch.Tensor:
                    gltf = pygltflib.GLTF2.load_from_bytes(self._data[idx])
                    n = sum(len(m.primitives) for m in gltf.meshes)
                    return torch.tensor([float(n)], dtype=torch.float32)

            ds3d = GlbDataset(tiles3d_dir)
            if len(ds3d) > 0:
                loader3d = DataLoader(ds3d, batch_size=batch_size, num_workers=0, collate_fn=_collate)
                total_tiles_3d = 0
                t0 = time.perf_counter()
                for _ in range(n_epochs):
                    for batch in loader3d:
                        total_tiles_3d += len(batch)
                elapsed_3d = time.perf_counter() - t0
                results["3dtiles_tiles_per_sec"] = total_tiles_3d / elapsed_3d if elapsed_3d > 0 else 0
                results["3dtiles_elapsed_sec"] = elapsed_3d
        except ImportError:
            pass

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    convert_time: float,
    tile_results: dict[str, Any],
    decode_mjb: dict[str, Any],
    decode_3dt: dict[str, Any],
    memory: dict[str, Any],
    dataloader: dict[str, Any] | None,
) -> None:
    """Print formatted summary table."""
    print(f"\n{'=' * 72}")
    print(f"  MouseLight HortaObj — Benchmark Results")
    print(f"{'=' * 72}")

    backend = "Rust" if RUST_AVAILABLE else "Python"
    print(f"\n  Backend: {backend}")

    # --- Conversion ---
    print(f"\n  OBJ -> MicroJSON Conversion")
    print(f"  {'─' * 50}")
    print(f"  {'Time':30s} {_fmt_time(convert_time):>15s}")

    # --- Tile Generation ---
    print(f"\n  Tile Generation")
    has_3dt = "3dtiles_tiles" in tile_results
    if has_3dt:
        print(f"  {'':30s} {'mjb':>15s} {'3dtiles':>15s}")
        print(f"  {'─' * 62}")
        print(f"  {'Tile count':30s} {tile_results['mjb_tiles']:>15,d} {tile_results['3dtiles_tiles']:>15,d}")
        print(f"  {'Index time':30s} {_fmt_time(tile_results['mjb_index_time']):>15s} {_fmt_time(tile_results['3dtiles_index_time']):>15s}")
        print(f"  {'Generation time':30s} {_fmt_time(tile_results['mjb_gen_time']):>15s} {_fmt_time(tile_results['3dtiles_gen_time']):>15s}")
        print(f"  {'Raw size':30s} {_fmt_bytes(tile_results['mjb_size_raw']):>15s} {_fmt_bytes(tile_results['3dtiles_size_raw']):>15s}")
        print(f"  {'Gzipped size':30s} {_fmt_bytes(tile_results['mjb_size_gzip']):>15s} {_fmt_bytes(tile_results['3dtiles_size_gzip']):>15s}")
        if tile_results["3dtiles_size_raw"] > 0:
            ratio = tile_results["3dtiles_size_raw"] / tile_results["mjb_size_raw"]
            print(f"  {'Size ratio (3dtiles/mjb)':30s} {ratio:>15.1f}x")
        if tile_results["3dtiles_gen_time"] > 0:
            speedup = tile_results["3dtiles_gen_time"] / tile_results["mjb_gen_time"]
            print(f"  {'Gen speedup (mjb vs 3dt)':30s} {speedup:>15.1f}x")
    else:
        print(f"  {'─' * 50}")
        print(f"  {'Tile count':30s} {tile_results['mjb_tiles']:>15,d}")
        print(f"  {'Index time':30s} {_fmt_time(tile_results['mjb_index_time']):>15s}")
        print(f"  {'Generation time':30s} {_fmt_time(tile_results['mjb_gen_time']):>15s}")
        print(f"  {'Raw size':30s} {_fmt_bytes(tile_results['mjb_size_raw']):>15s}")
        print(f"  {'Gzipped size':30s} {_fmt_bytes(tile_results['mjb_size_gzip']):>15s}")

    # --- Decode Latency ---
    if decode_mjb:
        print(f"\n  Decode Latency (per tile)")
        if decode_3dt:
            print(f"  {'':30s} {'mjb':>15s} {'3dtiles':>15s}")
            print(f"  {'─' * 62}")
            print(f"  {'Median':30s} {decode_mjb['decode_median_us']:>13.0f}us {decode_3dt['decode_median_us']:>13.0f}us")
            print(f"  {'P95':30s} {decode_mjb['decode_p95_us']:>13.0f}us {decode_3dt['decode_p95_us']:>13.0f}us")
            if decode_3dt["decode_median_us"] > 0:
                speedup = decode_3dt["decode_median_us"] / decode_mjb["decode_median_us"]
                print(f"  {'Speedup (mjb vs 3dt)':30s} {speedup:>15.1f}x")
        else:
            print(f"  {'─' * 50}")
            print(f"  {'Sampled tiles':30s} {decode_mjb['sampled_tiles']:>15,d}")
            print(f"  {'Features in sample':30s} {decode_mjb['total_features']:>15,d}")
            print(f"  {'Mesh bytes in sample':30s} {_fmt_bytes(decode_mjb['total_mesh_bytes']):>15s}")
            print(f"  {'Median':30s} {decode_mjb['decode_median_us']:>13.0f}us")
            print(f"  {'P95':30s} {decode_mjb['decode_p95_us']:>13.0f}us")
            print(f"  {'P99':30s} {decode_mjb['decode_p99_us']:>13.0f}us")

    # --- Memory ---
    if memory:
        print(f"\n  Peak Memory (full decode)")
        if "3dtiles_peak_mb" in memory:
            print(f"  {'':30s} {'mjb':>15s} {'3dtiles':>15s}")
            print(f"  {'─' * 62}")
            print(f"  {'Peak':30s} {memory['mjb_peak_mb']:>13.1f}MB {memory['3dtiles_peak_mb']:>13.1f}MB")
            ratio = memory["3dtiles_peak_mb"] / memory["mjb_peak_mb"] if memory["mjb_peak_mb"] > 0 else 0
            print(f"  {'Ratio (3dtiles/mjb)':30s} {ratio:>15.1f}x")
        else:
            print(f"  {'─' * 50}")
            print(f"  {'mjb peak':30s} {memory['mjb_peak_mb']:>13.1f}MB")

    # --- DataLoader ---
    if dataloader:
        print(f"\n  ML DataLoader Throughput")
        if "3dtiles_tiles_per_sec" in dataloader:
            print(f"  {'':30s} {'mjb':>15s} {'3dtiles':>15s}")
            print(f"  {'─' * 62}")
            print(f"  {'Tiles/sec':30s} {dataloader['mjb_tiles_per_sec']:>15.1f} {dataloader['3dtiles_tiles_per_sec']:>15.1f}")
            if dataloader["3dtiles_tiles_per_sec"] > 0:
                speedup = dataloader["mjb_tiles_per_sec"] / dataloader["3dtiles_tiles_per_sec"]
                print(f"  {'Speedup (mjb vs 3dt)':30s} {speedup:>15.1f}x")
        else:
            print(f"  {'─' * 50}")
            print(f"  {'mjb tiles/sec':30s} {dataloader['mjb_tiles_per_sec']:>15.1f}")

    print()


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_csv(
    path: Path,
    convert_time: float,
    tile_results: dict[str, Any],
    decode_mjb: dict[str, Any],
    decode_3dt: dict[str, Any],
    memory: dict[str, Any],
    dataloader: dict[str, Any] | None,
) -> None:
    """Write all results to a CSV file."""
    import csv

    row: dict[str, Any] = {"convert_time_s": convert_time}

    # Tile gen (exclude Path objects)
    for k, v in tile_results.items():
        if not isinstance(v, Path):
            row[f"tilegen_{k}"] = v

    for k, v in decode_mjb.items():
        row[f"decode_mjb_{k}"] = v
    for k, v in decode_3dt.items():
        row[f"decode_3dt_{k}"] = v
    for k, v in memory.items():
        row[f"memory_{k}"] = v
    if dataloader:
        for k, v in dataloader.items():
            row[f"dataloader_{k}"] = v

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"Results written to {path}")


# ---------------------------------------------------------------------------
# All-brains pyramid generation
# ---------------------------------------------------------------------------

def generate_all_brains(
    brains_dir: Path,
    output_dir: Path,
    *,
    max_zoom: int = 3,
) -> None:
    """Generate 3D Tiles for all brain directories with pyramid manifest."""
    import json as _json
    from microjson._rs import StreamingTileGenerator, scan_obj_bounds

    # Import build_feature_index
    sys.path.insert(0, str(_ROOT / "scripts"))
    from build_feature_index import build_index

    brain_dirs = sorted(
        d for d in brains_dir.iterdir()
        if d.is_dir() and sorted(d.glob("*.obj"))
    )
    if not brain_dirs:
        print(f"ERROR: No brain directories with OBJ files found in {brains_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(brain_dirs)} brain directories in {brains_dir}")

    # Fetch ontology once for all brains
    print("Fetching Allen CCF ontology...")
    t0 = time.perf_counter()
    ontology = fetch_allen_ontology()
    print(f"  Loaded {len(ontology)} ontology entries ({_fmt_time(time.perf_counter() - t0)})")

    pyramids = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, brain_dir in enumerate(brain_dirs):
        brain_id = brain_dir.name
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(brain_dirs)}] Processing brain: {brain_id}")
        print(f"{'='*60}")

        obj_paths = sorted(brain_dir.glob("*.obj"))
        if not obj_paths:
            print(f"  Skipping (no OBJ files)")
            continue

        print(f"  {len(obj_paths)} OBJ files")
        tags_list = [_build_mouselight_tags(p, ontology) for p in obj_paths]
        path_strs = [str(p) for p in obj_paths]

        # Scan bounds
        t0 = time.perf_counter()
        bounds = scan_obj_bounds(path_strs)
        print(f"  Bounds scanned ({_fmt_time(time.perf_counter() - t0)})")

        # 3D Tiles output
        tiles3d_dir = output_dir / brain_id / "3dtiles"
        if tiles3d_dir.exists():
            shutil.rmtree(tiles3d_dir)
        tiles3d_dir.mkdir(parents=True, exist_ok=True)

        gen = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
        t0 = time.perf_counter()
        gen.add_obj_files(path_strs, bounds, tags_list)
        t_ingest = time.perf_counter() - t0
        print(f"  Ingest: {_fmt_time(t_ingest)} ({len(obj_paths) / max(t_ingest, 0.001):.0f} files/s)")

        t0 = time.perf_counter()
        n_tiles = gen.generate_3dtiles(str(tiles3d_dir), bounds)
        t_gen = time.perf_counter() - t0
        del gen

        tiles_size = _dir_size(tiles3d_dir)
        print(f"  {n_tiles} tiles in {_fmt_time(t_gen)} ({_fmt_bytes(tiles_size)})")

        # Build features.json
        print(f"  Building features.json...")
        index = build_index(tiles3d_dir)
        features_path = tiles3d_dir / "features.json"
        features_path.write_text(_json.dumps(index, indent=2))
        n_features = len(index.get("features", {}))
        print(f"  {n_features} features indexed")

        # Feature-centric MJB
        feat_mjb_dir = output_dir / brain_id / "feature_mjb"
        if feat_mjb_dir.exists():
            shutil.rmtree(feat_mjb_dir)
        feat_mjb_dir.mkdir(parents=True, exist_ok=True)

        gen_fmjb = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
        gen_fmjb.add_obj_files(path_strs, bounds, tags_list)
        t0 = time.perf_counter()
        n_feat_mjb = gen_fmjb.generate_feature_mjb(str(feat_mjb_dir), bounds)
        t_feat_mjb = time.perf_counter() - t0
        feat_mjb_size = _dir_size(feat_mjb_dir)
        del gen_fmjb
        print(f"  Feature MJB: {n_feat_mjb} features in {_fmt_time(t_feat_mjb)} ({_fmt_bytes(feat_mjb_size)})")

        # Neuroglancer
        ng_dir = output_dir / brain_id / "neuroglancer"
        if ng_dir.exists():
            shutil.rmtree(ng_dir)
        ng_dir.mkdir(parents=True, exist_ok=True)

        gen_ng = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
        gen_ng.add_obj_files(path_strs, bounds, tags_list)
        t0 = time.perf_counter()
        n_feat_ng = gen_ng.generate_neuroglancer_multilod(str(ng_dir), bounds)
        t_feat_ng = time.perf_counter() - t0
        ng_size = _dir_size(ng_dir)
        del gen_ng
        print(f"  Neuroglancer: {n_feat_ng} features in {_fmt_time(t_feat_ng)} ({_fmt_bytes(ng_size)})")

        # Parquet
        from microjson.tiling3d.parquet_writer import generate_parquet as _gen_pq

        pq_path = output_dir / brain_id / "tiles.parquet"
        gen_pq = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
        gen_pq.add_obj_files(path_strs, bounds, tags_list)
        t0 = time.perf_counter()
        n_rows_pq = _gen_pq(gen_pq, pq_path, bounds)
        t_pq = time.perf_counter() - t0
        pq_size = pq_path.stat().st_size if pq_path.exists() else 0
        del gen_pq
        print(f"  Parquet: {n_rows_pq} rows in {_fmt_time(t_pq)} ({_fmt_bytes(pq_size)})")

        pyramids.append({
            "id": brain_id,
            "label": f"MouseLight {brain_id} ({n_features} regions)",
            "tiles": n_tiles,
            "features": n_features,
            "max_zoom": max_zoom,
            "size_bytes": tiles_size,
            "feature_mjb_size": feat_mjb_size,
            "neuroglancer_size": ng_size,
            "parquet_size": pq_size,
        })

    # Write pyramids.json
    manifest = {"pyramids": pyramids}
    manifest_path = output_dir / "pyramids.json"
    manifest_path.write_text(_json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path} ({len(pyramids)} pyramids)")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MouseLight HortaObj end-to-end benchmark pipeline",
    )
    parser.add_argument(
        "--obj-dir", type=Path, default=None,
        help="Directory containing OBJ files (e.g. data/mouselight/2021-09-16/)",
    )
    parser.add_argument(
        "--all-brains", type=Path, default=None,
        help="Parent dir with brain subdirs (e.g. data/mouselight/). "
             "Generates 3D Tiles for every brain + pyramids.json manifest.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/mouselight/tiles"),
        help="Output directory for tiles (default: data/mouselight/tiles/)",
    )
    parser.add_argument(
        "--max-zoom", type=int, default=3,
        help="Max zoom level for tiling (default: 3)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes: None/0=auto, 1=serial (default: auto)",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit number of OBJ files to process (for testing)",
    )
    parser.add_argument(
        "--no-ontology", action="store_true",
        help="Skip Allen CCF ontology lookup",
    )
    parser.add_argument(
        "--skip-3dtiles", action="store_true",
        help="Skip 3D Tiles generation (only generate mjb)",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="Use Rust StreamingTileGenerator instead of batch TileGenerator3D",
    )
    parser.add_argument(
        "--decode-iters", type=int, default=20,
        help="Iterations for decode benchmark (default: 20)",
    )
    parser.add_argument(
        "--dataloader", action="store_true",
        help="Run PyTorch DataLoader throughput benchmark",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Export results to CSV file",
    )
    args = parser.parse_args()

    # All-brains mode: generate pyramids for every brain directory
    if args.all_brains:
        if not args.all_brains.is_dir():
            print(f"ERROR: {args.all_brains} is not a directory", file=sys.stderr)
            sys.exit(1)
        generate_all_brains(
            args.all_brains,
            args.output_dir,
            max_zoom=args.max_zoom,
        )
        return

    if not args.obj_dir or not args.obj_dir.is_dir():
        print(f"ERROR: --obj-dir is required (or use --all-brains)", file=sys.stderr)
        sys.exit(1)

    effective_workers = args.workers if args.workers and args.workers >= 1 else (os.cpu_count() or 1)
    print(f"Backend: {'Rust' if RUST_AVAILABLE else 'Python'}")
    print(f"Workers: {effective_workers} ({'auto' if args.workers is None else 'explicit'})")
    print(f"Max zoom: {args.max_zoom}")
    print()

    # Step 1+2: OBJ → Tiles
    if args.streaming:
        print("Mode: Parallel Streaming (Rust rayon — OBJ-direct, no Python intermediates)")
        tile_results, convert_time = generate_tiles_streaming(
            args.obj_dir,
            args.output_dir,
            max_zoom=args.max_zoom,
            skip_3dtiles=args.skip_3dtiles,
            no_ontology=args.no_ontology,
        )
    else:
        collection, convert_time = load_obj_collection(
            args.obj_dir,
            no_ontology=args.no_ontology,
            max_files=args.max_files,
        )
        tile_results = generate_tiles(
            collection,
            args.output_dir,
            max_zoom=args.max_zoom,
            workers=args.workers,
            skip_3dtiles=args.skip_3dtiles,
        )

    # Step 3: Decode benchmarks
    mjb_dir = tile_results["mjb_dir"]
    tiles3d_dir = tile_results.get("3dtiles_dir")

    print(f"\nBenchmarking mjb decode ({args.decode_iters} iterations)...")
    decode_mjb = bench_decode(mjb_dir, n_iterations=args.decode_iters)

    decode_3dt: dict[str, Any] = {}
    if tiles3d_dir:
        print(f"Benchmarking 3D Tiles decode ({args.decode_iters} iterations)...")
        decode_3dt = bench_decode_3dtiles(tiles3d_dir, n_iterations=args.decode_iters)

    # Step 4: Memory
    print("Measuring peak memory...")
    memory = bench_memory(mjb_dir, tiles3d_dir)

    # Step 5: DataLoader (optional)
    dataloader_results = None
    if args.dataloader:
        print("Benchmarking DataLoader throughput...")
        dataloader_results = bench_dataloader(mjb_dir, tiles3d_dir)

    # Report
    print_report(
        convert_time, tile_results,
        decode_mjb, decode_3dt,
        memory, dataloader_results,
    )

    # CSV export
    if args.csv:
        export_csv(
            args.csv, convert_time, tile_results,
            decode_mjb, decode_3dt,
            memory, dataloader_results,
        )

    print("Done.")


if __name__ == "__main__":
    main()
