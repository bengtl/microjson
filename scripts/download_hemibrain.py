#!/usr/bin/env python3
"""Download Hemibrain v1.2.1 neuron meshes and metadata for benchmarking.

Downloads neuron meshes from the Janelia Hemibrain dataset via CloudVolume,
queries neuPrint for cell type metadata, exports as OBJ files, converts to
MicroJSON, tiles with TileGenerator3D, and benchmarks.

Prerequisites:
    uv pip install --python .venv/bin/python cloud-volume requests

Usage::

    # Set neuPrint auth token (get from https://neuprint.janelia.org → Account):
    export NEUPRINT_TOKEN="eyJhbGciOi..."

    # Full pipeline:
    .venv/bin/python scripts/download_hemibrain.py --download --convert --tile --benchmark

    # Download only (top 1000 neurons):
    .venv/bin/python scripts/download_hemibrain.py --download --max-neurons 1000

    # From existing OBJ files:
    .venv/bin/python scripts/download_hemibrain.py --convert --tile --benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_DATA_DIR = _ROOT / "data" / "hemibrain"
_MESH_DIR = _DATA_DIR / "meshes"
_META_PATH = _DATA_DIR / "metadata.json"
_TILES_DIR = _DATA_DIR / "tiles"

_HEMIBRAIN_SEG = (
    "precomputed://gs://neuroglancer-janelia-flyem-hemibrain/v1.2/segmentation"
)
_NEUPRINT_URL = "https://neuprint.janelia.org"
_NEUPRINT_DATASET = "hemibrain:v1.2.1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


# ---------------------------------------------------------------------------
# Step 1: Query neuPrint for metadata
# ---------------------------------------------------------------------------

def query_neuprint(token: str, max_neurons: int | None = None) -> list[dict]:
    """Query neuPrint for neuron metadata via REST API.

    Returns list of dicts with bodyId, type, instance, status, etc.
    """
    import requests

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Query for all traced neurons with cell type annotations
    cypher = """
    MATCH (n :Neuron)
    WHERE n.status = "Traced"
    RETURN n.bodyId AS bodyId,
           n.type AS cellType,
           n.instance AS instance,
           n.status AS status,
           n.size AS size,
           n.somaLocation AS somaLocation
    ORDER BY n.size DESC
    """
    if max_neurons:
        cypher += f"\nLIMIT {max_neurons}"

    payload = {"cypher": cypher, "dataset": _NEUPRINT_DATASET}
    print(f"Querying neuPrint for neuron metadata...")
    resp = requests.post(
        f"{_NEUPRINT_URL}/api/custom/custom",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()

    columns = result["columns"]
    rows = result["data"]
    neurons = [dict(zip(columns, row)) for row in rows]
    print(f"  Got {len(neurons)} neurons from neuPrint")
    return neurons


# ---------------------------------------------------------------------------
# Step 2: Download meshes via CloudVolume
# ---------------------------------------------------------------------------

def download_meshes(
    body_ids: list[int],
    output_dir: Path,
    *,
    skip_existing: bool = True,
) -> int:
    """Download neuron meshes as OBJ files via CloudVolume.

    Returns number of successfully downloaded meshes.
    """
    from cloudvolume import CloudVolume

    output_dir.mkdir(parents=True, exist_ok=True)

    cv = CloudVolume(
        _HEMIBRAIN_SEG,
        use_https=True,
        progress=False,
    )

    downloaded = 0
    errors = 0
    for i, body_id in enumerate(body_ids, 1):
        obj_path = output_dir / f"{body_id}.obj"
        if skip_existing and obj_path.exists():
            downloaded += 1
            if i % 100 == 0:
                print(f"  [{i}/{len(body_ids)}] {body_id} — exists, skipping")
            continue

        try:
            mesh = cv.mesh.get(body_id, lod=0)[body_id]
            vertices = mesh.vertices
            faces = mesh.faces.reshape(-1, 3)

            # Write OBJ
            with open(obj_path, "w") as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in faces:
                    f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

            downloaded += 1
            if i % 10 == 0 or i == len(body_ids):
                print(
                    f"  [{i}/{len(body_ids)}] {body_id} — "
                    f"{len(vertices):,} verts, {len(faces):,} faces"
                )
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [{i}/{len(body_ids)}] {body_id} — ERROR: {e}")
            elif errors == 6:
                print("  (suppressing further errors)")

    print(f"  Downloaded {downloaded}/{len(body_ids)} meshes ({errors} errors)")
    return downloaded


# ---------------------------------------------------------------------------
# Step 3: Convert OBJ → MicroJSON
# ---------------------------------------------------------------------------

def convert_to_microjson(
    mesh_dir: Path,
    metadata_path: Path,
    *,
    max_files: int | None = None,
):
    """Convert Hemibrain OBJ meshes to MicroFeatureCollection."""
    import numpy as np

    from microjson.model import (
        MicroFeature,
        MicroFeatureCollection,
        OntologyTerm,
        Vocabulary,
    )
    from microjson.swc import _mesh_to_tin
    from obj_to_microjson import parse_obj

    obj_paths = sorted(mesh_dir.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No .obj files in {mesh_dir}", file=sys.stderr)
        sys.exit(1)

    if max_files and max_files < len(obj_paths):
        obj_paths = obj_paths[:max_files]

    # Load metadata if available
    meta_lookup: dict[str, dict] = {}
    if metadata_path.exists():
        raw = json.loads(metadata_path.read_text())
        for neuron in raw.get("neurons", []):
            meta_lookup[str(neuron["bodyId"])] = neuron

    print(f"Converting {len(obj_paths)} OBJ files to MicroJSON...")
    t0 = time.perf_counter()
    features: list[MicroFeature] = []
    total_verts = 0
    total_faces = 0

    for i, obj_path in enumerate(obj_paths, 1):
        if i % 50 == 0 or i == len(obj_paths):
            print(f"  [{i}/{len(obj_paths)}] {obj_path.name}", end="\r", file=sys.stderr)

        vertices, faces = parse_obj(str(obj_path))
        tin = _mesh_to_tin(vertices, faces)

        body_id = obj_path.stem
        props: dict = {
            "body_id": int(body_id) if body_id.isdigit() else body_id,
            "source": obj_path.name,
            "vertex_count": int(vertices.shape[0]),
            "face_count": int(faces.shape[0]),
        }

        # Add neuPrint metadata if available
        meta = meta_lookup.get(body_id, {})
        feature_class = body_id
        if meta:
            if meta.get("cellType"):
                props["cell_type"] = meta["cellType"]
                feature_class = meta["cellType"]
            if meta.get("instance"):
                props["instance"] = meta["instance"]
            if meta.get("status"):
                props["status"] = meta["status"]
            if meta.get("somaLocation"):
                props["soma_location"] = meta["somaLocation"]

        features.append(MicroFeature(
            type="Feature",
            geometry=tin,
            properties=props,
            featureClass=feature_class,
        ))
        total_verts += vertices.shape[0]
        total_faces += faces.shape[0]

    print(file=sys.stderr)
    convert_time = time.perf_counter() - t0

    # Build vocabulary from cell types
    cell_types = set()
    for f in features:
        ct = (f.properties or {}).get("cell_type")
        if ct:
            cell_types.add(ct)

    vocabs = None
    if cell_types:
        terms = {
            ct: OntologyTerm(
                uri=f"https://neuprint.janelia.org/view/celltype/{ct}",
                label=ct,
            )
            for ct in sorted(cell_types)
        }
        vocabs = {
            "hemibrain_cell_types": Vocabulary(
                namespace="https://neuprint.janelia.org/",
                description="Hemibrain v1.2.1 cell type annotations",
                terms=terms,
            ),
        }

    collection = MicroFeatureCollection(
        type="FeatureCollection",
        features=features,
        properties={
            "dataset": "hemibrain_v1.2.1",
            "mesh_count": len(features),
            "total_vertices": total_verts,
            "total_faces": total_faces,
            "cell_types": len(cell_types),
        },
        vocabularies=vocabs,
    )

    print(f"  {len(features)} features, {total_verts:,} vertices, {total_faces:,} faces")
    print(f"  {len(cell_types)} unique cell types")
    print(f"  Conversion time: {_fmt_time(convert_time)}")

    return collection, convert_time


# ---------------------------------------------------------------------------
# Step 4 & 5: Tile + Benchmark (reuse from benchmark_mouselight)
# ---------------------------------------------------------------------------

def tile_and_benchmark(
    collection,
    output_dir: Path,
    *,
    max_zoom: int = 3,
    workers: int | None = None,
    do_tile: bool = True,
    do_benchmark: bool = True,
    skip_3dtiles: bool = False,
    csv_path: Path | None = None,
) -> dict:
    """Tile the collection and run benchmarks."""
    from benchmark_mouselight import (
        bench_decode,
        bench_decode_3dtiles,
        bench_memory,
        export_csv,
        generate_tiles,
        print_report,
    )

    results: dict = {}

    if do_tile:
        tile_results = generate_tiles(
            collection,
            output_dir,
            max_zoom=max_zoom,
            workers=workers,
            skip_3dtiles=skip_3dtiles,
        )
        results["tile"] = tile_results

    if do_benchmark:
        mvt3_dir = output_dir / "mvt3"
        tiles3d_dir = output_dir / "3dtiles" if not skip_3dtiles else None

        if mvt3_dir.exists():
            print(f"\nBenchmarking mvt3 decode...")
            decode_mvt3 = bench_decode(mvt3_dir)
            results["decode_mvt3"] = decode_mvt3
        else:
            decode_mvt3 = {}

        decode_3dt: dict = {}
        if tiles3d_dir and tiles3d_dir.exists():
            print(f"Benchmarking 3D Tiles decode...")
            decode_3dt = bench_decode_3dtiles(tiles3d_dir)
            results["decode_3dt"] = decode_3dt

        print("Measuring peak memory...")
        memory = bench_memory(
            mvt3_dir if mvt3_dir.exists() else Path("/dev/null"),
            tiles3d_dir if tiles3d_dir and tiles3d_dir.exists() else None,
        )
        results["memory"] = memory

        if do_tile:
            print_report(
                results.get("convert_time", 0),
                tile_results,
                decode_mvt3,
                decode_3dt,
                memory,
                None,
            )

        if csv_path:
            export_csv(
                csv_path,
                results.get("convert_time", 0),
                tile_results if do_tile else {},
                decode_mvt3,
                decode_3dt,
                memory,
                None,
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hemibrain v1.2.1 download, conversion, tiling, and benchmark",
    )
    parser.add_argument("--download", action="store_true", help="Download meshes from CloudVolume + metadata from neuPrint")
    parser.add_argument("--convert", action="store_true", help="Convert OBJ to MicroJSON")
    parser.add_argument("--tile", action="store_true", help="Generate tiles (mvt3 + 3dtiles)")
    parser.add_argument("--benchmark", action="store_true", help="Run decode/memory benchmarks")
    parser.add_argument("--max-neurons", type=int, default=1000, help="Max neurons to download (default: 1000)")
    parser.add_argument("--max-zoom", type=int, default=3, help="Max zoom level (default: 3)")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes")
    parser.add_argument("--skip-3dtiles", action="store_true", help="Skip 3D Tiles generation")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Data directory")
    parser.add_argument("--csv", type=Path, default=None, help="Export results to CSV")
    args = parser.parse_args()

    if not any([args.download, args.convert, args.tile, args.benchmark]):
        parser.print_help()
        sys.exit(1)

    data_dir = args.data_dir
    mesh_dir = data_dir / "meshes"
    meta_path = data_dir / "metadata.json"
    tiles_dir = data_dir / "tiles"

    # --- Download ---
    if args.download:
        data_dir.mkdir(parents=True, exist_ok=True)

        token = os.environ.get("NEUPRINT_TOKEN", "")
        neurons: list[dict] = []

        if token:
            neurons = query_neuprint(token, max_neurons=args.max_neurons)

            # Save metadata
            meta_path.write_text(json.dumps(
                {"neurons": neurons, "dataset": _NEUPRINT_DATASET},
                indent=2,
            ))
            print(f"  Saved metadata to {meta_path}")
        else:
            print("WARNING: NEUPRINT_TOKEN not set. Downloading without metadata.")
            print("  Set it via: export NEUPRINT_TOKEN='your-token-here'")
            print("  Get token from https://neuprint.janelia.org → Account")

            if meta_path.exists():
                raw = json.loads(meta_path.read_text())
                neurons = raw.get("neurons", [])
                print(f"  Using existing metadata ({len(neurons)} neurons)")

        if not neurons:
            print("ERROR: No neuron IDs to download. Set NEUPRINT_TOKEN.", file=sys.stderr)
            sys.exit(1)

        body_ids = [n["bodyId"] for n in neurons]
        print(f"\nDownloading {len(body_ids)} neuron meshes...")
        t0 = time.perf_counter()
        downloaded = download_meshes(body_ids, mesh_dir)
        dl_time = time.perf_counter() - t0
        print(f"  Download time: {_fmt_time(dl_time)}")

    # --- Convert ---
    collection = None
    convert_time = 0.0
    if args.convert:
        collection, convert_time = convert_to_microjson(
            mesh_dir, meta_path, max_files=args.max_neurons,
        )

    # --- Tile + Benchmark ---
    if args.tile or args.benchmark:
        if collection is None:
            print("Loading MicroJSON collection (convert step)...")
            collection, convert_time = convert_to_microjson(
                mesh_dir, meta_path, max_files=args.max_neurons,
            )

        tile_and_benchmark(
            collection,
            tiles_dir,
            max_zoom=args.max_zoom,
            workers=args.workers,
            do_tile=args.tile,
            do_benchmark=args.benchmark,
            skip_3dtiles=args.skip_3dtiles,
            csv_path=args.csv,
        )

    print("Done.")


if __name__ == "__main__":
    main()
