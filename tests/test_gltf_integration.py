"""End-to-end integration tests for glTF/GLB export pipeline."""

import struct
from pathlib import Path

import pytest
from pygltflib import GLTF2

from microjson.model import (
    MicroFeature,
    MicroFeatureCollection,
    NeuronMorphology,
    SWCSample,
    TIN,
    Slice,
    SliceStack,
)
from microjson.swc import swc_to_microjson
from microjson.gltf import GltfConfig, to_glb, to_gltf


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SWC_FILE = FIXTURE_DIR / "sample_neuron.swc"


class TestSWCToGLB:
    """SWC file → swc_to_microjson → to_glb → validate."""

    def test_swc_to_glb_pipeline(self, tmp_path):
        feat = swc_to_microjson(str(SWC_FILE))
        assert isinstance(feat.geometry, NeuronMorphology)

        glb_bytes = to_glb(feat)
        assert glb_bytes[:4] == b"glTF"
        version = struct.unpack_from("<I", glb_bytes, 4)[0]
        assert version == 2

    def test_swc_to_glb_roundtrip(self, tmp_path):
        feat = swc_to_microjson(str(SWC_FILE))
        out = tmp_path / "neuron.glb"
        to_glb(feat, output_path=out)

        loaded = GLTF2.load(str(out))
        assert len(loaded.meshes) > 0
        assert loaded.asset.version == "2.0"
        assert loaded.asset.generator == "microjson-gltf"

    def test_swc_to_gltf_roundtrip(self, tmp_path):
        feat = swc_to_microjson(str(SWC_FILE))
        out = tmp_path / "neuron.gltf"
        to_gltf(feat, output_path=out)

        loaded = GLTF2.load(str(out))
        assert len(loaded.meshes) > 0

    def test_swc_collection_to_glb(self, tmp_path):
        feat = swc_to_microjson(str(SWC_FILE))
        collection = MicroFeatureCollection(
            type="FeatureCollection",
            features=[feat],
            properties={"source": "test"},
        )
        glb = to_glb(collection)
        assert glb[:4] == b"glTF"

        out = tmp_path / "collection.glb"
        to_glb(collection, output_path=out)
        loaded = GLTF2.load(str(out))
        assert loaded.scenes[0].extras["source"] == "test"


class TestMixedGeometryCollection:
    """Test a collection with multiple geometry types."""

    def test_mixed_collection(self, tmp_path):
        from geojson_pydantic import LineString, Point, Polygon

        neuron = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1),
                SWCSample(id=2, type=3, x=10, y=0, z=0, r=2, parent=1),
            ],
        )
        point = Point(type="Point", coordinates=[1.0, 2.0, 3.0])
        line = LineString(type="LineString", coordinates=[[0, 0, 0], [5, 5, 5]])
        poly = Polygon(
            type="Polygon",
            coordinates=[[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
        )
        tin = TIN(
            type="TIN",
            coordinates=[[[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]]],
        )

        features = [
            MicroFeature(type="Feature", geometry=neuron, properties={"kind": "neuron"}),
            MicroFeature(type="Feature", geometry=point, properties={"kind": "point"}),
            MicroFeature(type="Feature", geometry=line, properties={"kind": "line"}),
            MicroFeature(type="Feature", geometry=poly, properties={"kind": "polygon"}),
            MicroFeature(type="Feature", geometry=tin, properties={"kind": "tin"}),
        ]
        collection = MicroFeatureCollection(
            type="FeatureCollection",
            features=features,
        )

        out = tmp_path / "mixed.glb"
        glb = to_glb(collection, output_path=out)

        loaded = GLTF2.load(str(out))
        # Neuron → 2 meshes (soma + dendrite), rest → 1 each = 6 total
        assert len(loaded.meshes) == 6
        assert len(loaded.nodes) == 6


class TestSliceStackToGLB:
    def test_slice_stack_glb(self, tmp_path):
        from geojson_pydantic import Polygon as GeoPolygon

        ss = SliceStack(
            type="SliceStack",
            slices=[
                Slice(z=0.0, geometry=GeoPolygon(
                    type="Polygon",
                    coordinates=[[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                )),
                Slice(z=5.0, geometry=GeoPolygon(
                    type="Polygon",
                    coordinates=[[[1, 1], [9, 1], [9, 9], [1, 9], [1, 1]]],
                )),
                Slice(z=10.0, geometry=GeoPolygon(
                    type="Polygon",
                    coordinates=[[[2, 2], [8, 2], [8, 8], [2, 8], [2, 2]]],
                )),
            ],
        )
        feat = MicroFeature(type="Feature", geometry=ss, properties={"type": "stack"})
        out = tmp_path / "stack.glb"
        to_glb(feat, output_path=out)

        loaded = GLTF2.load(str(out))
        assert len(loaded.meshes) == 1


class TestPublicAPIImports:
    """Verify the public API is importable from the top-level package."""

    def test_import_from_microjson(self):
        from microjson import to_gltf, to_glb, GltfConfig

        assert callable(to_gltf)
        assert callable(to_glb)
        assert GltfConfig is not None

    def test_import_from_gltf_subpackage(self):
        from microjson.gltf import to_gltf, to_glb, GltfConfig

        assert callable(to_gltf)
        assert callable(to_glb)
