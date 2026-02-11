#!/usr/bin/env python3
"""Import a GeoParquet file into MicroJSON and export as Draco-compressed GLB.

Usage:
    .venv/bin/python scripts/parquet_to_glb.py INPUT.parquet [OPTIONS]

Options:
    -o OUTPUT           Output GLB path (default: <input_stem>.glb)
    --no-draco          Disable Draco compression
    --grid X,Y,Z        Grid layout, e.g. --grid 5,4 or --grid 5,4,2
    --spacing N         Gap between features in source units (default: 0 = auto)
    --smooth N          Catmull-Rom subdivisions per segment (default: 10)
    --quality N         Mesh quality 0.0–1.0 (default: 0.3)

Examples:
    # Basic: read parquet, write Draco GLB
    .venv/bin/python scripts/parquet_to_glb.py neurons.parquet

    # Custom output and grid layout
    .venv/bin/python scripts/parquet_to_glb.py neurons.parquet -o grid.glb --grid 5,4

    # Without Draco (plain GLB)
    .venv/bin/python scripts/parquet_to_glb.py neurons.parquet --no-draco

Then drag the .glb file into https://gltf-viewer.donmccurdy.com/
"""

import sys
from pathlib import Path

from microjson.arrow import from_geoparquet
from microjson.gltf import GltfConfig, to_glb


def _pop_flag(args: list[str], flag: str) -> str | None:
    """Remove ``flag VALUE`` from *args*, return VALUE or None."""
    if flag in args:
        idx = args.index(flag)
        val = args[idx + 1]
        del args[idx : idx + 2]
        return val
    return None


def _pop_bool(args: list[str], flag: str) -> bool:
    """Remove a boolean flag from *args*, return True if present."""
    if flag in args:
        args.remove(flag)
        return True
    return False


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    # Parse options (before consuming positional arg)
    out_path_str = _pop_flag(args, "-o") or _pop_flag(args, "--output")
    grid_str = _pop_flag(args, "--grid")
    spacing_str = _pop_flag(args, "--spacing")
    smooth_str = _pop_flag(args, "--smooth")
    quality_str = _pop_flag(args, "--quality")
    no_draco = _pop_bool(args, "--no-draco")

    if not args:
        print("Error: no input parquet file specified")
        sys.exit(1)

    input_path = Path(args[0])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    out_path = Path(out_path_str) if out_path_str else input_path.with_suffix(".glb")

    # Parse grid spec
    grid_x = grid_y = grid_z = None
    if grid_str:
        parts = [int(x) for x in grid_str.split(",")]
        grid_x = parts[0]
        if len(parts) > 1:
            grid_y = parts[1]
        if len(parts) > 2:
            grid_z = parts[2]

    # --- Read Parquet → MicroJSON ---
    print(f"Reading {input_path} ...")
    fc = from_geoparquet(input_path)
    print(f"  {len(fc.features)} feature(s)")

    geom_types = set()
    for f in fc.features:
        if f.geometry is not None:
            geom_types.add(type(f.geometry).__name__)
    print(f"  Geometry types: {', '.join(sorted(geom_types)) or 'none'}")

    # --- Configure GLB export ---
    draco = not no_draco
    if draco:
        try:
            import DracoPy  # noqa: F401
        except ImportError:
            print("  Warning: DracoPy not installed, falling back to plain GLB")
            draco = False

    config = GltfConfig(
        draco=draco,
        smooth_factor=int(smooth_str) if smooth_str else 10,
        mesh_quality=float(quality_str) if quality_str else 0.3,
        feature_spacing=float(spacing_str) if spacing_str else 0.0,
        grid_max_x=grid_x,
        grid_max_y=grid_y,
        grid_max_z=grid_z,
    )

    # --- Export to GLB ---
    print(f"Exporting to {out_path} (draco={'on' if draco else 'off'}) ...")
    glb_bytes = to_glb(fc, output_path=out_path, config=config)

    mb = len(glb_bytes) / 1024 / 1024
    grid_info = ""
    if grid_x:
        dims = [str(grid_x)]
        if grid_y:
            dims.append(str(grid_y))
        if grid_z:
            dims.append(str(grid_z))
        grid_info = f"  grid={'x'.join(dims)}"

    print(f"  Wrote {out_path} ({mb:.2f} MB){grid_info}")
    print(f"  Open https://gltf-viewer.donmccurdy.com/ and drag in {out_path}")


if __name__ == "__main__":
    main()
