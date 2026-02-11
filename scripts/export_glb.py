#!/usr/bin/env python3
"""Export SWC file(s) to a single GLB for visualization.

Usage:
    .venv/bin/python scripts/export_glb.py [SWC_FILES...] [OPTIONS]

Options:
    -o OUTPUT       Output file path (default: neuron.glb)
    --grid X,Y,Z    Grid layout, e.g. --grid 5,4 or --grid 5,4,2
    --spacing N     Gap between features in source units (default: 0 = auto 20%)

Examples:
    # All SWC files in swcs/ (single row)
    .venv/bin/python scripts/export_glb.py

    # 5 columns, unlimited rows
    .venv/bin/python scripts/export_glb.py --grid 5

    # 5×4 grid
    .venv/bin/python scripts/export_glb.py --grid 5,4

    # 5×4×2 grid (max 40 cells)
    .venv/bin/python scripts/export_glb.py --grid 5,4,2

Then drag the .glb file into https://gltf-viewer.donmccurdy.com/
"""

import sys
from pathlib import Path

from microjson.model import MicroFeatureCollection
from microjson.swc import swc_to_microjson
from microjson.gltf import to_glb, GltfConfig

DEFAULT_SWC_DIR = "swcs"
DEFAULT_OUT = "neuron.glb"


def _pop_flag(args, flag):
    """Remove ``flag VALUE`` from *args*, return VALUE or None."""
    if flag in args:
        idx = args.index(flag)
        val = args[idx + 1]
        del args[idx:idx + 2]
        return val
    return None


def main():
    args = sys.argv[1:]

    out_path = _pop_flag(args, "-o") or DEFAULT_OUT
    out_path = _pop_flag(args, "--output") or out_path
    grid_str = _pop_flag(args, "--grid")
    spacing_str = _pop_flag(args, "--spacing")
    feature_spacing = float(spacing_str) if spacing_str else 0.0

    # Parse grid spec
    grid_x = grid_y = grid_z = None
    if grid_str:
        parts = [int(x) for x in grid_str.split(",")]
        grid_x = parts[0]
        if len(parts) > 1:
            grid_y = parts[1]
        if len(parts) > 2:
            grid_z = parts[2]

    # Collect SWC paths
    if args:
        swc_paths = [Path(a) for a in args]
    else:
        swc_dir = Path(DEFAULT_SWC_DIR)
        swc_paths = sorted(swc_dir.glob("*.swc"))
        if not swc_paths:
            print(f"No SWC files found in {swc_dir}/")
            sys.exit(1)

    for p in swc_paths:
        if not p.exists():
            print(f"Error: SWC file not found: {p}")
            sys.exit(1)

    RED = (1.0, 0.0, 0.0, 1.0)
    GREEN = (0.0, 1.0, 0.0, 1.0)
    MAGENTA = (1.0, 0.0, 1.0, 1.0)
    GREY = (0.5, 0.5, 0.5, 1.0)

    config = GltfConfig(
        swc_type_colors={
            1: RED,
            2: GREY,
            3: GREEN,
            4: MAGENTA,
        },
        color_by_type=True,
        smooth_factor=10,
        mesh_quality=0.3,
        grid_max_x=grid_x,
        grid_max_y=grid_y,
        grid_max_z=grid_z,
        feature_spacing=feature_spacing,
    )

    features = []
    for swc_path in swc_paths:
        feat = swc_to_microjson(str(swc_path))
        if feat.properties is None:
            feat.properties = {}
        feat.properties["source"] = swc_path.name
        features.append(feat)

    print(f"Loaded {len(features)} SWC files")

    if len(features) == 1:
        data = to_glb(
            features[0], config=config, output_path=out_path,
        )
    else:
        coll = MicroFeatureCollection(
            type="FeatureCollection",
            features=features,
            properties={"neuron_count": len(features)},
        )
        data = to_glb(coll, config=config, output_path=out_path)

    mb = len(data) / 1024 / 1024
    grid_info = ""
    if grid_x:
        dims = [str(grid_x)]
        if grid_y:
            dims.append(str(grid_y))
        if grid_z:
            dims.append(str(grid_z))
        grid_info = f"  grid={'x'.join(dims)}"

    print(
        f"Wrote {out_path} ({mb:.1f} MB)"
        f" — {len(features)} neuron(s){grid_info}"
    )
    print(
        "Open https://gltf-viewer.donmccurdy.com/"
        f" and drag in {out_path}"
    )


if __name__ == "__main__":
    main()
