"""Tests for glTF scene graph assembly."""

import numpy as np
import pytest
from pygltflib import LINES, POINTS, TRIANGLES

from microjson.model import (
    MicroFeature,
    MicroFeatureCollection,
    NeuronMorphology,
    PolyhedralSurface,
    Slice,
    SliceStack,
    SWCSample,
    TIN,
)
from microjson.gltf.gltf_assembler import (
    collection_to_gltf,
    feature_to_gltf,
)
from microjson.gltf.models import GltfConfig


def _neuron_feature(**props) -> MicroFeature:
    neuron = NeuronMorphology(
        type="NeuronMorphology",
        tree=[
            SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1),
            SWCSample(id=2, type=3, x=10, y=0, z=0, r=2, parent=1),
        ],
    )
    return MicroFeature(type="Feature", geometry=neuron, properties=props or None)


class TestFeatureToGltf:
    def test_neuron_morphology(self):
        feat = _neuron_feature()
        gltf = feature_to_gltf(feat)

        # Neuron with soma(type=1) + dendrite(type=3) → 2 meshes (one per type)
        assert len(gltf.meshes) == 2
        assert len(gltf.nodes) == 2
        for mesh in gltf.meshes:
            assert mesh.primitives[0].mode == TRIANGLES
        assert len(gltf.accessors) > 0
        assert gltf.asset.generator == "microjson-gltf"

    def test_neuron_no_color_by_type(self):
        feat = _neuron_feature()
        config = GltfConfig(color_by_type=False)
        gltf = feature_to_gltf(feat, config=config)

        # Single mesh when coloring disabled
        assert len(gltf.meshes) == 1
        assert len(gltf.nodes) == 1

    def test_point_geometry(self):
        from geojson_pydantic import Point

        feat = MicroFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=[1.0, 2.0, 3.0]),
            properties=None,
        )
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1
        assert gltf.meshes[0].primitives[0].mode == POINTS

    def test_linestring_geometry(self):
        from geojson_pydantic import LineString

        feat = MicroFeature(
            type="Feature",
            geometry=LineString(
                type="LineString",
                coordinates=[[0, 0, 0], [1, 1, 1], [2, 0, 0]],
            ),
            properties=None,
        )
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1
        assert gltf.meshes[0].primitives[0].mode == LINES

    def test_polygon_geometry(self):
        from geojson_pydantic import Polygon

        feat = MicroFeature(
            type="Feature",
            geometry=Polygon(
                type="Polygon",
                coordinates=[[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
            ),
            properties=None,
        )
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1
        assert gltf.meshes[0].primitives[0].mode == TRIANGLES

    def test_tin_geometry(self):
        tin = TIN(
            type="TIN",
            coordinates=[
                [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]],
                [[[1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]]],
            ],
        )
        feat = MicroFeature(type="Feature", geometry=tin, properties=None)
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1
        assert gltf.meshes[0].primitives[0].mode == TRIANGLES

    def test_polyhedral_surface(self):
        ps = PolyhedralSurface(
            type="PolyhedralSurface",
            coordinates=[
                [[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]]],
                [[[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]]],
            ],
        )
        feat = MicroFeature(type="Feature", geometry=ps, properties=None)
        gltf = feature_to_gltf(feat)

        assert len(gltf.meshes) == 1

    def test_slice_stack(self):
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
                    coordinates=[[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                )),
            ],
        )
        feat = MicroFeature(type="Feature", geometry=ss, properties=None)
        gltf = feature_to_gltf(feat)
        assert len(gltf.meshes) == 1

    def test_multipoint(self):
        from geojson_pydantic import MultiPoint

        feat = MicroFeature(
            type="Feature",
            geometry=MultiPoint(type="MultiPoint", coordinates=[[0, 0, 0], [1, 1, 1]]),
            properties=None,
        )
        gltf = feature_to_gltf(feat)
        assert gltf.meshes[0].primitives[0].mode == POINTS

    def test_multilinestring(self):
        from geojson_pydantic import MultiLineString

        feat = MicroFeature(
            type="Feature",
            geometry=MultiLineString(
                type="MultiLineString",
                coordinates=[[[0, 0], [1, 1]], [[2, 2], [3, 3]]],
            ),
            properties=None,
        )
        gltf = feature_to_gltf(feat)
        assert gltf.meshes[0].primitives[0].mode == LINES

    def test_null_geometry(self):
        feat = MicroFeature(type="Feature", geometry=None, properties=None)
        gltf = feature_to_gltf(feat)
        assert len(gltf.meshes) == 0
        assert len(gltf.nodes) == 0


class TestMetadata:
    def test_feature_properties_in_extras(self):
        feat = _neuron_feature(cell_type="pyramidal", layer=5)
        gltf = feature_to_gltf(feat)

        assert gltf.nodes[0].extras == {"cell_type": "pyramidal", "layer": 5}

    def test_metadata_disabled(self):
        feat = _neuron_feature(cell_type="pyramidal")
        config = GltfConfig(include_metadata=False)
        gltf = feature_to_gltf(feat, config=config)

        assert not gltf.nodes[0].extras  # empty dict or None

    def test_collection_metadata(self):
        feat = _neuron_feature()
        collection = MicroFeatureCollection(
            type="FeatureCollection",
            features=[feat],
            properties={"study": "test_study"},
        )
        gltf = collection_to_gltf(collection)
        assert gltf.scenes[0].extras == {"study": "test_study"}


class TestCoordinateTransform:
    def test_y_up_swaps_yz(self):
        """With y_up=True, Z values should become Y values."""
        from geojson_pydantic import Point

        feat = MicroFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=[1.0, 2.0, 3.0]),
            properties=None,
        )
        gltf = feature_to_gltf(feat, GltfConfig(y_up=True))

        # Extract position from buffer
        acc = gltf.accessors[0]
        bv = gltf.bufferViews[acc.bufferView]
        data = gltf._glb_data[bv.byteOffset : bv.byteOffset + bv.byteLength]
        pos = np.frombuffer(data, dtype="<f4").reshape(-1, 3)

        # Original (1, 2, 3) → Y-up (1, 3, -2)
        np.testing.assert_allclose(pos[0], [1.0, 3.0, -2.0], atol=1e-6)

    def test_y_up_disabled(self):
        from geojson_pydantic import Point

        feat = MicroFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=[1.0, 2.0, 3.0]),
            properties=None,
        )
        gltf = feature_to_gltf(feat, GltfConfig(y_up=False))

        acc = gltf.accessors[0]
        bv = gltf.bufferViews[acc.bufferView]
        data = gltf._glb_data[bv.byteOffset : bv.byteOffset + bv.byteLength]
        pos = np.frombuffer(data, dtype="<f4").reshape(-1, 3)

        np.testing.assert_allclose(pos[0], [1.0, 2.0, 3.0], atol=1e-6)


class TestCollectionToGltf:
    def test_multiple_features(self):
        feat1 = _neuron_feature()
        from geojson_pydantic import Point

        feat2 = MicroFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=[5.0, 5.0, 5.0]),
            properties=None,
        )
        collection = MicroFeatureCollection(
            type="FeatureCollection",
            features=[feat1, feat2],
        )
        gltf = collection_to_gltf(collection)

        # Neuron → 2 meshes (soma + dendrite), point → 1 mesh
        assert len(gltf.meshes) == 3
        assert len(gltf.nodes) == 3
        assert gltf.nodes[0].name == "feature_0_0"
        assert gltf.nodes[1].name == "feature_0_1"
        assert gltf.nodes[2].name == "feature_1"

    def test_empty_collection(self):
        collection = MicroFeatureCollection(
            type="FeatureCollection",
            features=[],
        )
        gltf = collection_to_gltf(collection)
        assert len(gltf.meshes) == 0
        assert len(gltf.nodes) == 0

    def test_default_material(self):
        feat = _neuron_feature()
        config = GltfConfig(color_by_type=False)
        gltf = feature_to_gltf(feat, config=config)

        assert len(gltf.materials) == 1
        pbr = gltf.materials[0].pbrMetallicRoughness
        assert pbr.metallicFactor == 0.1
        assert pbr.roughnessFactor == 0.8

    def test_type_colored_materials(self):
        feat = _neuron_feature()
        gltf = feature_to_gltf(feat)

        # Default + soma(1) + dendrite(3) = 3 materials
        assert len(gltf.materials) == 3
        # Type-specific materials have names
        type_mat_names = {m.name for m in gltf.materials if m.name}
        assert "soma" in type_mat_names
        assert "basal_dendrite" in type_mat_names


class TestFeatureSpacing:
    """Feature spacing in collections."""

    def _two_neuron_collection(self):
        """Two neurons at the same position."""
        n1 = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1),
                SWCSample(id=2, type=3, x=100, y=0, z=0, r=2, parent=1),
            ],
        )
        n2 = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1),
                SWCSample(id=2, type=3, x=80, y=0, z=0, r=2, parent=1),
            ],
        )
        return MicroFeatureCollection(
            type="FeatureCollection",
            features=[
                MicroFeature(type="Feature", geometry=n1, properties=None),
                MicroFeature(type="Feature", geometry=n2, properties=None),
            ],
        )

    def test_auto_spacing_offsets_second_feature(self):
        """With default spacing=0, second feature gets a positive X translation."""
        coll = self._two_neuron_collection()
        gltf = collection_to_gltf(coll)

        # First feature's nodes should have no translation
        assert gltf.nodes[0].translation is None
        # Second feature's nodes should be translated along X
        second_nodes = [n for n in gltf.nodes if n.name and n.name.startswith("feature_1")]
        assert len(second_nodes) > 0
        for node in second_nodes:
            assert node.translation is not None
            assert node.translation[0] > 0  # shifted right
            assert node.translation[1] == 0.0
            assert node.translation[2] == 0.0

    def test_fixed_spacing(self):
        """Explicit feature_spacing sets the gap."""
        coll = self._two_neuron_collection()
        config = GltfConfig(feature_spacing=50.0, color_by_type=False)
        gltf = collection_to_gltf(coll, config=config)

        # First feature: x range [0,100], second feature: x range [0,80]
        # Second should be shifted so its left edge (0) is at 100 + 50 = 150
        second_node = [n for n in gltf.nodes if n.name == "feature_1"][0]
        assert second_node.translation is not None
        assert abs(second_node.translation[0] - 150.0) < 1e-6

    def test_single_feature_no_offset(self):
        """A single-feature collection should not get translated."""
        feat = _neuron_feature()
        coll = MicroFeatureCollection(
            type="FeatureCollection",
            features=[feat],
        )
        gltf = collection_to_gltf(coll)
        for node in gltf.nodes:
            assert node.translation is None

    def test_all_typed_nodes_share_translation(self):
        """When color_by_type=True, all nodes of a feature share the same offset."""
        coll = self._two_neuron_collection()
        gltf = collection_to_gltf(coll)

        second_nodes = [n for n in gltf.nodes if n.name and n.name.startswith("feature_1")]
        assert len(second_nodes) >= 2  # soma + dendrite type
        translations = [tuple(n.translation) for n in second_nodes]
        assert len(set(translations)) == 1  # all identical


class TestGridLayout:
    """Grid placement when grid_max_* is set."""

    def _make_neuron(self, x_offset=0.0, width=100.0):
        return NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=x_offset, y=0, z=0, r=5, parent=-1),
                SWCSample(id=2, type=3, x=x_offset + width, y=0, z=0, r=2, parent=1),
            ],
        )

    def _collection(self, n, **neuron_kw):
        features = [
            MicroFeature(
                type="Feature",
                geometry=self._make_neuron(**neuron_kw),
                properties=None,
            )
            for _ in range(n)
        ]
        return MicroFeatureCollection(
            type="FeatureCollection",
            features=features,
        )

    def test_wraps_to_rows(self):
        """4 features, 2 cols → 2 rows."""
        coll = self._collection(4, width=100)
        config = GltfConfig(
            grid_max_x=2,
            feature_spacing=10,
            color_by_type=False,
        )
        gltf = collection_to_gltf(coll, config=config)

        # Feature 0: no translation
        f0 = [n for n in gltf.nodes if n.name == "feature_0"][0]
        assert f0.translation is None

        # Feature 1: shifted right (same row)
        f1 = [n for n in gltf.nodes if n.name == "feature_1"][0]
        assert f1.translation is not None
        assert f1.translation[0] > 0   # X offset
        assert f1.translation[1] == 0.0  # no layer (glTF Y)

        # Feature 2: second row (Y in source → -Z in glTF Y-up)
        f2 = [n for n in gltf.nodes if n.name == "feature_2"][0]
        assert f2.translation is not None
        assert abs(f2.translation[0]) < 1e-6  # back to col 0
        assert f2.translation[2] < 0  # negative Z = positive source Y

    def test_wraps_to_layers(self):
        """5 features, 2 cols × 2 rows → 5th goes to layer 1."""
        coll = self._collection(5, width=100)
        config = GltfConfig(
            grid_max_x=2,
            grid_max_y=2,
            feature_spacing=10,
            color_by_type=False,
        )
        gltf = collection_to_gltf(coll, config=config)

        # Feature 4: grid pos (0, 0, 1) → source dz = 1*cell_z
        # glTF Y-up: source Z → glTF Y
        f4 = [n for n in gltf.nodes if n.name == "feature_4"][0]
        assert f4.translation is not None
        assert f4.translation[1] > 0  # positive glTF Y = positive source Z

    def test_capacity_error(self):
        """All three grid_max_* set with too many features → ValueError."""
        coll = self._collection(10, width=100)
        # 2 × 2 × 2 = 8 < 10
        config = GltfConfig(
            grid_max_x=2,
            grid_max_y=2,
            grid_max_z=2,
            feature_spacing=10,
            color_by_type=False,
        )
        with pytest.raises(ValueError, match="Cannot fit 10 features"):
            collection_to_gltf(coll, config=config)

    def test_exact_capacity_fits(self):
        """Exactly filling the grid should not raise."""
        coll = self._collection(8, width=100)
        config = GltfConfig(
            grid_max_x=2,
            grid_max_y=2,
            grid_max_z=2,
            color_by_type=False,
        )
        gltf = collection_to_gltf(coll, config=config)
        assert len(gltf.nodes) == 8

    def test_capacity_error_before_generation(self):
        """Error should happen before any mesh is built (fast fail)."""
        coll = self._collection(100, width=100)
        config = GltfConfig(
            grid_max_x=3,
            grid_max_y=3,
            grid_max_z=3,
            color_by_type=False,
        )
        # 3×3×3 = 27 < 100 → ValueError fires instantly
        with pytest.raises(ValueError):
            collection_to_gltf(coll, config=config)

    def test_unconstrained_z_always_fits(self):
        """With only grid_max_x set, any number of features should fit."""
        coll = self._collection(20, width=100)
        config = GltfConfig(
            grid_max_x=3,
            feature_spacing=10,
            color_by_type=False,
        )
        # Should not raise — rows/layers expand as needed
        gltf = collection_to_gltf(coll, config=config)
        assert len(gltf.nodes) == 20

    def test_grid_typed_nodes(self):
        """Grid layout with color_by_type: all sub-nodes share offset."""
        coll = self._collection(4, width=100)
        config = GltfConfig(
            grid_max_x=2,
            feature_spacing=10,
            color_by_type=True,
        )
        gltf = collection_to_gltf(coll, config=config)

        # Feature 2 (row 1, col 0) should have multiple nodes
        f2_nodes = [
            n for n in gltf.nodes
            if n.name and n.name.startswith("feature_2")
        ]
        assert len(f2_nodes) >= 2
        translations = [tuple(n.translation) for n in f2_nodes]
        assert len(set(translations)) == 1
