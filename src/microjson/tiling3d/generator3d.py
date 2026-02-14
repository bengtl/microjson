"""TileGenerator3D — high-level API for 3D vector tile generation.

Orchestrates the full pipeline: convert → index (octree) → transform →
encode → write to disk as ``{z}/{x}/{y}/{d}.mvt3`` files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model import MicroFeatureCollection, Vocabulary
from .convert3d import compute_bounds_3d, convert_collection_3d
from .encoder3d import encode_tile_3d
from .octree import Octree, OctreeConfig
from .projector3d import CartesianProjector3D
from .tile3d import transform_tile_3d
from .tilejson3d import TileModel3D


def _infer_field_type(value: Any) -> str | None:
    """Map a Python value to a TileJSON field type string."""
    if isinstance(value, bool):
        return "Boolean"
    if isinstance(value, (int, float)):
        return "Number"
    if isinstance(value, str):
        return "String"
    return None


class TileGenerator3D:
    """Generate 3D vector tiles from a MicroJSON FeatureCollection.

    Usage::

        gen = TileGenerator3D(config)
        gen.add_features(collection)
        gen.generate(Path("output/tiles"))
        gen.write_tilejson(Path("output/tiles/tilejson3d.json"))
    """

    def __init__(self, config: OctreeConfig | None = None) -> None:
        self.config = config or OctreeConfig()
        self._octree: Octree | None = None
        self._bounds: tuple[float, float, float, float, float, float] | None = None
        self._layer_name = "default"
        # Metadata collected during add_features()
        self._field_types: dict[str, str] = {}
        self._field_ranges: dict[str, list[int | float]] = {}
        self._field_enums: dict[str, list[str]] = {}
        self._vocabularies: dict[str, Vocabulary] | None = None

    def _scan_properties(self, collection: MicroFeatureCollection) -> None:
        """Scan all feature properties to compute field types, ranges, enums."""
        numeric_mins: dict[str, float] = {}
        numeric_maxs: dict[str, float] = {}
        string_sets: dict[str, set[str]] = {}

        for feat in collection.features:
            if not feat.properties:
                continue
            for key, value in feat.properties.items():
                if value is None:
                    continue
                ftype = _infer_field_type(value)
                if ftype is None:
                    continue

                # Record type (first non-None wins; Number beats String if mixed)
                if key not in self._field_types:
                    self._field_types[key] = ftype
                elif self._field_types[key] != ftype:
                    # Mixed types — keep Number if either is Number
                    if ftype == "Number" or self._field_types[key] == "Number":
                        self._field_types[key] = "Number"

                # Track numeric ranges
                if ftype == "Number":
                    v = float(value)
                    if key not in numeric_mins:
                        numeric_mins[key] = v
                        numeric_maxs[key] = v
                    else:
                        if v < numeric_mins[key]:
                            numeric_mins[key] = v
                        if v > numeric_maxs[key]:
                            numeric_maxs[key] = v

                # Track string enumerations
                elif ftype == "String":
                    if key not in string_sets:
                        string_sets[key] = set()
                    string_sets[key].add(value)

        # Build ranges for numeric fields
        for key in numeric_mins:
            self._field_ranges[key] = [numeric_mins[key], numeric_maxs[key]]

        # Build enums for string fields (sorted for deterministic output)
        for key, vals in string_sets.items():
            self._field_enums[key] = sorted(vals)

    def add_features(
        self,
        collection: MicroFeatureCollection,
        layer_name: str = "default",
    ) -> None:
        """Convert and index features into the octree.

        Parameters
        ----------
        collection : MicroFeatureCollection
            Input features.
        layer_name : str
            Layer name for tile encoding.
        """
        self._layer_name = layer_name

        # Scan properties for metadata (field types, ranges, enums)
        self._scan_properties(collection)

        # Capture vocabularies from collection (dict only, not URI strings)
        if isinstance(collection.vocabularies, dict):
            self._vocabularies = collection.vocabularies

        # Compute world bounds
        bounds = compute_bounds_3d(collection)
        self._bounds = bounds

        # Update config bounds
        self.config.bounds = bounds

        # Project features to normalized [0,1]³
        proj = CartesianProjector3D(bounds)
        features = convert_collection_3d(collection, proj)

        # Build octree
        self._octree = Octree(features, self.config)

    def generate(self, output_dir: Path | str) -> int:
        """Write tiles to disk as {z}/{x}/{y}/{d}.mvt3 files.

        Returns the number of tiles written.
        """
        if self._octree is None:
            raise RuntimeError("Call add_features() before generate()")

        output_dir = Path(output_dir)
        count = 0

        for (z, x, y, d), tile in self._octree.all_tiles.items():
            if z < self.config.min_zoom:
                continue

            # Transform to integer coords
            transformed = transform_tile_3d(
                tile,
                extent=self.config.extent,
                extent_z=self.config.extent_z,
            )

            # Encode to protobuf
            data = encode_tile_3d(
                transformed,
                layer_name=self._layer_name,
                extent=self.config.extent,
                extent_z=self.config.extent_z,
            )

            # Write file
            tile_path = output_dir / str(z) / str(x) / str(y) / f"{d}.mvt3"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            tile_path.write_bytes(data)
            count += 1

        return count

    def write_tilejson(self, path: Path | str) -> None:
        """Write TileJSON 3D metadata file."""
        if self._bounds is None:
            raise RuntimeError("Call add_features() before write_tilejson()")

        path = Path(path)
        bounds = self._bounds

        model = TileModel3D(
            tilejson="3.0.0",
            tiles=["{z}/{x}/{y}/{d}.mvt3"],
            name=self._layer_name,
            minzoom=self.config.min_zoom,
            maxzoom=self.config.max_zoom,
            bounds3d=list(bounds),
            center3d=[
                (bounds[0] + bounds[3]) / 2,
                (bounds[1] + bounds[4]) / 2,
                (bounds[2] + bounds[5]) / 2,
                self.config.min_zoom,
            ],
            depthsize=self.config.extent_z,
            vector_layers=[{
                "id": self._layer_name,
                "fields": self._field_types or {},
                "minzoom": self.config.min_zoom,
                "maxzoom": self.config.max_zoom,
                **({"fieldranges": self._field_ranges}
                   if self._field_ranges else {}),
                **({"fieldenums": self._field_enums}
                   if self._field_enums else {}),
                **({"vocabularies": {
                        k: v.model_dump(exclude_none=True)
                        for k, v in self._vocabularies.items()
                    }} if self._vocabularies else {}),
            }],
        )

        path.write_text(model.model_dump_json(indent=2, exclude_none=True))
