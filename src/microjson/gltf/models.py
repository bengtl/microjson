"""Pydantic configuration models for glTF export."""

from __future__ import annotations

from pydantic import BaseModel


# Standard SWC compartment type codes
SWC_SOMA = 1
SWC_AXON = 2
SWC_BASAL_DENDRITE = 3
SWC_APICAL_DENDRITE = 4

# Default RGBA colors per SWC type (neuroscience convention)
DEFAULT_SWC_COLORS: dict[int, tuple[float, float, float, float]] = {
    SWC_SOMA: (0.55, 0.0, 0.55, 1.0),             # purple
    SWC_AXON: (0.2, 0.4, 0.9, 1.0),               # blue
    SWC_BASAL_DENDRITE: (0.9, 0.25, 0.1, 1.0),     # red-orange
    SWC_APICAL_DENDRITE: (0.9, 0.2, 0.65, 1.0),    # magenta-pink
}

# Fallback for undefined / custom types
DEFAULT_SWC_FALLBACK_COLOR: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)


class GltfConfig(BaseModel):
    """Configuration for MicroJSON → glTF conversion.

    Attributes:
        tube_segments: Number of sides for tube cross-section polygons.
        tube_min_radius: Minimum radius for neuron tube segments.
        smooth_factor: Catmull-Rom subdivisions per segment (0 = no smoothing).
        include_metadata: Store MicroJSON properties in glTF ``extras``.
        y_up: Apply Z-up → Y-up rotation (glTF standard is Y-up).
        default_color: RGBA color for default PBR material.
        color_by_type: Color neuron compartments by SWC type.
        swc_type_colors: Override RGBA colors per SWC type code.
        feature_spacing: Gap between features when exporting a
            collection.  ``0`` = auto (20 % of widest feature).  A positive
            value sets a fixed gap in source coordinate units.
        mesh_quality: Fraction of smoothed path vertices to keep (0.0–1.0).
            ``1.0`` = full detail, ``0.5`` ≈ half the rings, ``0.1`` =
            aggressive reduction.  Smoothing runs first at full resolution,
            then the resulting path is uniformly thinned.  This preserves
            smooth curves while cutting vertex count.
        grid_max_x: Max number of columns (X direction) before wrapping
            to a new row.  ``None`` = no limit.  When set, features are
            placed on a uniform grid.  E.g. ``grid_max_x=3,
            grid_max_y=3, grid_max_z=3`` allows up to 27 features.
        grid_max_y: Max number of rows (Y direction) before wrapping
            to a new layer.  ``None`` = no limit.
        grid_max_z: Max number of layers (Z direction).  When all three
            ``grid_max_*`` values are set and the features cannot fit,
            a ``ValueError`` is raised before any mesh generation.
        draco: Enable Draco mesh compression (``KHR_draco_mesh_compression``).
            Only TRIANGLES primitives are compressed; lines and points
            are left uncompressed.  Requires the optional ``DracoPy``
            package (``pip install DracoPy``).
        draco_quantization_position: Quantization bits for vertex positions
            (1–30).  Higher = more precise but larger.  14 bits ≈ 0.6 µm
            precision for typical microscopy volumes.
        draco_quantization_normal: Quantization bits for normal vectors
            (1–30).  10 bits is usually sufficient for direction vectors.
        draco_compression_level: Draco compression level (0–10).  Higher
            values produce smaller output but take longer to encode.
    """

    tube_segments: int = 8
    tube_min_radius: float = 0.1
    smooth_factor: int = 3
    include_metadata: bool = True
    y_up: bool = True
    default_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    color_by_type: bool = True
    swc_type_colors: dict[int, tuple[float, float, float, float]] = DEFAULT_SWC_COLORS
    feature_spacing: float = 0.0
    mesh_quality: float = 1.0
    grid_max_x: int | None = None
    grid_max_y: int | None = None
    grid_max_z: int | None = None
    draco: bool = False
    draco_quantization_position: int = 14
    draco_quantization_normal: int = 10
    draco_compression_level: int = 1
