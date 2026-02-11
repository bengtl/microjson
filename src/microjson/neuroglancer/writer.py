"""Unified Neuroglancer export orchestrator.

Auto-dispatches MicroJSON data to the appropriate Neuroglancer writer:
- NeuronMorphology → skeleton (precomputed)
- Point features   → point annotations
- LineString features → line annotations
- Properties → segment_properties
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from geojson_pydantic import LineString, Point

from ..model import (
    MicroFeature,
    MicroFeatureCollection,
    NeuronMorphology,
)
from ..transforms import AffineTransform
from .annotation_writer import write_annotations
from .properties_writer import write_segment_properties
from .skeleton_writer import write_skeleton
from .state import (
    build_annotation_layer,
    build_skeleton_layer,
    build_viewer_state,
    viewer_state_to_url,
)


def to_neuroglancer(
    data: Union[MicroFeature, MicroFeatureCollection],
    output_dir: str | Path,
    *,
    transform: Optional[AffineTransform] = None,
    base_url: Optional[str] = None,
) -> dict[str, Any]:
    """Export MicroJSON data to Neuroglancer precomputed format.

    Auto-dispatches based on geometry type:
    - NeuronMorphology → skeleton binary
    - Point → point annotations
    - LineString → line annotations
    - Feature properties → segment_properties

    Args:
        data: A MicroFeature or MicroFeatureCollection.
        output_dir: Root output directory.
        transform: Optional affine transform for skeletons.
        base_url: Optional Neuroglancer instance URL for viewer state.

    Returns:
        Dict with keys: ``paths`` (written dirs), ``viewer_state`` (if base_url).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Normalize to list of features
    if isinstance(data, MicroFeatureCollection):
        features = list(data.features)
    else:
        features = [data]

    # Classify features
    skeleton_features: list[MicroFeature] = []
    point_features: list[MicroFeature] = []
    line_features: list[MicroFeature] = []

    for feat in features:
        geom = feat.geometry
        if isinstance(geom, NeuronMorphology):
            skeleton_features.append(feat)
        elif isinstance(geom, Point):
            point_features.append(feat)
        elif isinstance(geom, LineString):
            line_features.append(feat)

    paths: dict[str, str] = {}
    layers: list[dict[str, Any]] = []

    # Write skeletons
    if skeleton_features:
        skel_dir = out / "skeletons"
        has_props = any(f.properties for f in skeleton_features)

        for i, feat in enumerate(skeleton_features):
            segment_id = i + 1
            write_skeleton(
                skel_dir,
                segment_id,
                feat.geometry,  # type: ignore[arg-type]
                transform=transform,
                segment_properties="seg_props" if has_props else None,
            )

        # Write segment_properties if any features have properties
        if has_props:
            seg_ids = list(range(1, len(skeleton_features) + 1))
            write_segment_properties(
                skel_dir / "seg_props",
                skeleton_features,
                seg_ids,
            )

        paths["skeletons"] = str(skel_dir)
        if base_url:
            layers.append(
                build_skeleton_layer(
                    "skeletons",
                    f"precomputed://{base_url}/skeletons",
                )
            )

    # Write point annotations
    if point_features:
        ann_dir = out / "point_annotations"
        write_annotations(ann_dir, point_features, "point")
        paths["point_annotations"] = str(ann_dir)
        if base_url:
            layers.append(
                build_annotation_layer(
                    "point_annotations",
                    f"precomputed://{base_url}/point_annotations",
                )
            )

    # Write line annotations
    if line_features:
        ann_dir = out / "line_annotations"
        write_annotations(ann_dir, line_features, "line")
        paths["line_annotations"] = str(ann_dir)
        if base_url:
            layers.append(
                build_annotation_layer(
                    "line_annotations",
                    f"precomputed://{base_url}/line_annotations",
                )
            )

    result: dict[str, Any] = {"paths": paths}

    if base_url and layers:
        state = build_viewer_state(layers)
        result["viewer_state"] = state
        result["viewer_url"] = viewer_state_to_url(state, base_url)

    return result
