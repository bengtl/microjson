"""Unified glTF / GLB export API.

Provides ``to_gltf()`` and ``to_glb()`` as the primary entry points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from pygltflib import BufferFormat, GLTF2

from ..model import MicroFeature, MicroFeatureCollection
from .gltf_assembler import collection_to_gltf, feature_to_gltf
from .models import GltfConfig


def _assemble(
    data: Union[MicroFeature, MicroFeatureCollection],
    config: GltfConfig | None,
) -> GLTF2:
    """Dispatch to the appropriate assembler."""
    if isinstance(data, MicroFeatureCollection):
        return collection_to_gltf(data, config)
    return feature_to_gltf(data, config)


def to_gltf(
    data: Union[MicroFeature, MicroFeatureCollection],
    output_path: str | Path | None = None,
    config: GltfConfig | None = None,
) -> GLTF2:
    """Convert MicroJSON data to glTF 2.0.

    Args:
        data: A MicroFeature or MicroFeatureCollection.
        output_path: If given, save as ``.gltf`` (JSON + embedded base64).
        config: Export configuration.

    Returns:
        The pygltflib GLTF2 object.
    """
    gltf = _assemble(data, config)

    if output_path is not None:
        gltf.convert_buffers(BufferFormat.DATAURI)
        gltf.save(str(output_path))

    return gltf


def to_glb(
    data: Union[MicroFeature, MicroFeatureCollection],
    output_path: str | Path | None = None,
    config: GltfConfig | None = None,
) -> bytes:
    """Convert MicroJSON data to GLB (binary glTF).

    Args:
        data: A MicroFeature or MicroFeatureCollection.
        output_path: If given, save as ``.glb`` file.
        config: Export configuration.

    Returns:
        The GLB file content as bytes.
    """
    gltf = _assemble(data, config)
    gltf.convert_buffers(BufferFormat.BINARYBLOB)
    glb_bytes = b"".join(gltf.save_to_bytes())

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(glb_bytes)

    return glb_bytes
