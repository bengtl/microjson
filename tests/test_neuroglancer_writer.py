"""Tests for Neuroglancer unified orchestrator."""

import json
from pathlib import Path

import pytest

from microjson.model import (
    MicroFeature,
    MicroFeatureCollection,
    NeuronMorphology,
    SWCSample,
)
from microjson.neuroglancer.writer import to_neuroglancer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def neuron_feature():
    morph = NeuronMorphology(
        type="NeuronMorphology",
        tree=[
            SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1),
            SWCSample(id=2, type=2, x=10, y=0, z=0, r=2, parent=1),
        ],
    )
    return MicroFeature(
        type="Feature",
        geometry=morph,
        properties={"name": "test_neuron"},
    )


@pytest.fixture
def point_feature():
    return MicroFeature(
        type="Feature",
        geometry={"type": "Point", "coordinates": [1, 2, 3]},
        properties={"label": "marker"},
    )


@pytest.fixture
def line_feature():
    return MicroFeature(
        type="Feature",
        geometry={"type": "LineString", "coordinates": [[0, 0, 0], [10, 10, 10]]},
        properties={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToNeuroglancerSingleNeuron:
    def test_creates_skeleton_dir(self, neuron_feature, tmp_path):
        result = to_neuroglancer(neuron_feature, tmp_path)
        assert "skeletons" in result["paths"]
        skel_dir = Path(result["paths"]["skeletons"])
        assert (skel_dir / "info").exists()
        assert (skel_dir / "1").exists()

    def test_segment_properties(self, neuron_feature, tmp_path):
        result = to_neuroglancer(neuron_feature, tmp_path)
        skel_dir = Path(result["paths"]["skeletons"])
        seg_props = skel_dir / "seg_props" / "info"
        assert seg_props.exists()
        info = json.loads(seg_props.read_text())
        assert info["@type"] == "neuroglancer_segment_properties"

    def test_no_viewer_url_without_base(self, neuron_feature, tmp_path):
        result = to_neuroglancer(neuron_feature, tmp_path)
        assert "viewer_url" not in result

    def test_viewer_url_with_base(self, neuron_feature, tmp_path):
        result = to_neuroglancer(
            neuron_feature, tmp_path, base_url="http://localhost:8080"
        )
        assert "viewer_url" in result
        assert "viewer_state" in result
        assert result["viewer_url"].startswith("http://localhost:8080/#!")


class TestToNeuroglancerMixedCollection:
    def test_mixed_types(self, neuron_feature, point_feature, line_feature, tmp_path):
        collection = MicroFeatureCollection(
            type="FeatureCollection",
            features=[neuron_feature, point_feature, line_feature],
        )
        result = to_neuroglancer(collection, tmp_path)
        assert "skeletons" in result["paths"]
        assert "point_annotations" in result["paths"]
        assert "line_annotations" in result["paths"]

    def test_directory_structure(self, neuron_feature, point_feature, tmp_path):
        collection = MicroFeatureCollection(
            type="FeatureCollection",
            features=[neuron_feature, point_feature],
        )
        to_neuroglancer(collection, tmp_path)
        assert (tmp_path / "skeletons" / "info").exists()
        assert (tmp_path / "skeletons" / "1").exists()
        assert (tmp_path / "point_annotations" / "info").exists()
        assert (tmp_path / "point_annotations" / "spatial0" / "0_0_0").exists()


class TestToNeuroglancerPointsOnly:
    def test_point_only(self, point_feature, tmp_path):
        result = to_neuroglancer(point_feature, tmp_path)
        assert "point_annotations" in result["paths"]
        assert "skeletons" not in result["paths"]


class TestToNeuroglancerLinesOnly:
    def test_line_only(self, line_feature, tmp_path):
        result = to_neuroglancer(line_feature, tmp_path)
        assert "line_annotations" in result["paths"]
        assert "skeletons" not in result["paths"]
