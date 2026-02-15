#!/usr/bin/env python3
"""Scan .glb tiles and build a feature index for the Three.js viewer.

Reads the JSON chunk from each GLB file to extract feature names and
properties from node extras, then writes a features.json manifest.

Output format:
{
  "features": {
    "Caudoputamen": {
      "color": "#98D9AA",
      "acronym": "CP",
      "ccf_id": 672,
      "tiles": {
        "0": ["0/0/0/0.glb"],
        "2": ["2/0/0/0.glb", "2/0/0/1.glb"],
        "3": ["3/0/0/0.glb", ...]
      }
    },
    ...
  },
  "zoom_counts": {"0": 1, "1": 8, "2": 64, "3": 414},
  "max_zoom": 3
}
"""

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path


def read_glb_json(path: Path) -> dict:
    """Read only the JSON chunk from a GLB file (fast, skips binary data)."""
    with open(path, "rb") as f:
        header = f.read(12)
        if len(header) < 12:
            return {}
        magic, version, total_len = struct.unpack("<III", header)
        if magic != 0x46546C67:  # 'glTF'
            return {}

        chunk_header = f.read(8)
        if len(chunk_header) < 8:
            return {}
        chunk_len, chunk_type = struct.unpack("<II", chunk_header)
        if chunk_type != 0x4E4F534A:  # 'JSON'
            return {}

        json_bytes = f.read(chunk_len)
        return json.loads(json_bytes)


def extract_features(glb_json: dict) -> list[dict]:
    """Extract feature info from glTF node extras."""
    features = []
    for node in glb_json.get("nodes", []):
        extras = node.get("extras")
        if extras and ("name" in extras or "acronym" in extras):
            features.append(extras)
    return features


def build_index(tiles_dir: Path) -> dict:
    """Build feature index from all .glb files, grouped by zoom level."""
    # feature_name → {color, acronym, ccf_id, tiles: {zoom_str: [uri, ...]}}
    feature_map: dict[str, dict] = {}
    zoom_counts: dict[int, int] = defaultdict(int)
    max_zoom = 0

    glb_files = sorted(tiles_dir.rglob("*.glb"))
    print(f"Scanning {len(glb_files)} .glb files...")

    for glb_path in glb_files:
        rel = glb_path.relative_to(tiles_dir)
        parts = rel.parts  # e.g., ('2', '0', '1', '0.glb')
        if len(parts) != 4:
            continue

        z = int(parts[0])
        zoom_counts[z] += 1
        max_zoom = max(max_zoom, z)

        uri = str(rel)
        glb_json = read_glb_json(glb_path)
        features = extract_features(glb_json)

        for feat in features:
            name = feat.get("name", feat.get("acronym", ""))
            if not name:
                continue

            if name not in feature_map:
                feature_map[name] = {
                    "color": feat.get("color", "#888888"),
                    "acronym": feat.get("acronym", ""),
                    "ccf_id": feat.get("ccf_id"),
                    "tiles": {},
                }

            zstr = str(z)
            if zstr not in feature_map[name]["tiles"]:
                feature_map[name]["tiles"][zstr] = []
            tile_list = feature_map[name]["tiles"][zstr]
            if uri not in tile_list:
                tile_list.append(uri)

    print(f"Found {len(feature_map)} unique features")
    for z in sorted(zoom_counts):
        print(f"  Zoom {z}: {zoom_counts[z]} tiles")

    return {
        "features": dict(sorted(feature_map.items())),
        "zoom_counts": {str(k): v for k, v in sorted(zoom_counts.items())},
        "max_zoom": max_zoom,
    }


def main():
    parser = argparse.ArgumentParser(description="Build feature index for 3D viewer")
    parser.add_argument(
        "--tiles-dir",
        default="data/mouselight/tiles/3dtiles",
        help="3D Tiles directory",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: <tiles-dir>/features.json)",
    )
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir)
    output = Path(args.output) if args.output else tiles_dir / "features.json"

    index = build_index(tiles_dir)
    output.write_text(json.dumps(index, indent=2))
    print(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
