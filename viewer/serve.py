#!/usr/bin/env python3
"""Development HTTP server for the Three.js 3D Tiles viewer.

Routes:
    /                                  → viewer/index.html
    /js/*                              → viewer/js/*
    /tiles/pyramids.json               → manifest file
    /tiles/{pyramid_id}/*              → {tiles_base}/{pyramid_id}/3dtiles/*
    /neuroglancer/{pyramid_id}/info    → Neuroglancer mesh info
    /neuroglancer/{pyramid_id}/<id>    → Neuroglancer binary mesh
    /neuroglancer/{pyramid_id}/<id>:0  → Neuroglancer fragment manifest
    /neuroglancer/{pyramid_id}/segment_properties/info → segment properties
"""

import argparse
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIEWER_DIR.parent


class ViewerHandler(SimpleHTTPRequestHandler):
    """Route requests to viewer assets or tile data."""

    tiles_base: str = ""

    def translate_path(self, path: str) -> str:
        # Strip query string
        path = path.split("?", 1)[0].split("#", 1)[0]

        if path == "/tiles/pyramids.json":
            return os.path.join(self.tiles_base, "pyramids.json")

        if path.startswith("/neuroglancer/"):
            # /neuroglancer/{pyramid_id}/rest → {tiles_base}/{pyramid_id}/neuroglancer/rest
            rel = path[len("/neuroglancer/"):]
            parts = rel.split("/", 1)
            pyramid_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            return os.path.join(self.tiles_base, pyramid_id, "neuroglancer", rest)

        if path.startswith("/tiles/"):
            # /tiles/{pyramid_id}/rest → {tiles_base}/{pyramid_id}/3dtiles/rest
            rel = path[len("/tiles/"):]
            parts = rel.split("/", 1)
            pyramid_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            return os.path.join(self.tiles_base, pyramid_id, "3dtiles", rest)

        if path == "/":
            return str(VIEWER_DIR / "index.html")

        # Serve viewer assets (js/, etc.)
        rel = path.lstrip("/")
        return str(VIEWER_DIR / rel)

    def end_headers(self):
        # CORS for local development
        self.send_header("Access-Control-Allow-Origin", "*")
        # Cache .glb files aggressively (they don't change)
        if self.path.endswith(".glb"):
            self.send_header("Cache-Control", "public, max-age=86400")
        elif self.path.endswith(".js") or self.path.endswith(".html"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description="Serve 3D Tiles viewer")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--tiles-base",
        default=str(PROJECT_ROOT / "data" / "mouselight" / "tiles"),
        help="Base directory containing pyramid subdirectories (default: data/mouselight/tiles/)",
    )
    args = parser.parse_args()

    tiles_base = os.path.abspath(args.tiles_base)
    manifest = os.path.join(tiles_base, "pyramids.json")
    if not os.path.isfile(manifest):
        print(f"WARNING: No pyramids.json found in {tiles_base}")

    ViewerHandler.tiles_base = tiles_base

    server = HTTPServer(("", args.port), ViewerHandler)
    print(f"Viewer:  http://localhost:{args.port}")
    print(f"Tiles:   {tiles_base}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
