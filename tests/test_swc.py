"""Tests for NeuronMorphology model and SWC interoperability."""

import pytest
from pathlib import Path
from pydantic import ValidationError
from microjson.model import SWCSample, NeuronMorphology


FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE_SWC = FIXTURE_DIR / "sample_neuron.swc"


# ---------------------------------------------------------------------------
# SWCSample model tests
# ---------------------------------------------------------------------------


class TestSWCSample:
    def test_create(self):
        s = SWCSample(id=1, type=1, x=0.0, y=0.0, z=0.0, r=8.0, parent=-1)
        assert s.id == 1
        assert s.type == 1
        assert s.r == 8.0
        assert s.parent == -1

    def test_roundtrip(self):
        s = SWCSample(id=2, type=3, x=10.0, y=5.0, z=2.0, r=2.0, parent=1)
        data = s.model_dump()
        s2 = SWCSample.model_validate(data)
        assert s2.x == 10.0
        assert s2.parent == 1


# ---------------------------------------------------------------------------
# NeuronMorphology model tests
# ---------------------------------------------------------------------------


class TestNeuronMorphology:
    def _simple_tree(self):
        return [
            SWCSample(id=1, type=1, x=0, y=0, z=0, r=8.0, parent=-1),
            SWCSample(id=2, type=3, x=10, y=5, z=2, r=2.0, parent=1),
            SWCSample(id=3, type=3, x=20, y=10, z=5, r=1.5, parent=2),
        ]

    def test_create(self):
        nm = NeuronMorphology(type="NeuronMorphology", tree=self._simple_tree())
        assert nm.type == "NeuronMorphology"
        assert len(nm.tree) == 3

    def test_bbox3d(self):
        nm = NeuronMorphology(type="NeuronMorphology", tree=self._simple_tree())
        bb = nm.bbox3d()
        assert bb == (0.0, 0.0, 0.0, 20.0, 10.0, 5.0)

    def test_centroid3d(self):
        nm = NeuronMorphology(type="NeuronMorphology", tree=self._simple_tree())
        c = nm.centroid3d()
        assert c[0] == pytest.approx(10.0)
        assert c[1] == pytest.approx(5.0)
        assert c[2] == pytest.approx(7.0 / 3.0)

    def test_no_root_rejected(self):
        """At least one node must have parent == -1."""
        tree = [
            SWCSample(id=1, type=1, x=0, y=0, z=0, r=1, parent=2),
            SWCSample(id=2, type=3, x=1, y=1, z=1, r=1, parent=1),
        ]
        with pytest.raises(ValidationError, match="root"):
            NeuronMorphology(type="NeuronMorphology", tree=tree)

    def test_invalid_parent_rejected(self):
        """Non-root parent ids must reference existing nodes."""
        tree = [
            SWCSample(id=1, type=1, x=0, y=0, z=0, r=1, parent=-1),
            SWCSample(id=2, type=3, x=1, y=1, z=1, r=1, parent=99),
        ]
        with pytest.raises(ValidationError, match="parent"):
            NeuronMorphology(type="NeuronMorphology", tree=tree)

    def test_empty_tree_rejected(self):
        with pytest.raises(ValidationError):
            NeuronMorphology(type="NeuronMorphology", tree=[])

    def test_roundtrip_json(self):
        nm = NeuronMorphology(type="NeuronMorphology", tree=self._simple_tree())
        data = nm.model_dump()
        nm2 = NeuronMorphology.model_validate(data)
        assert len(nm2.tree) == 3
        assert nm2.tree[0].parent == -1


# ---------------------------------------------------------------------------
# SWC converter tests
# ---------------------------------------------------------------------------


class TestSWCConverters:
    def test_swc_to_microjson(self):
        from microjson.swc import swc_to_microjson

        feature = swc_to_microjson(str(SAMPLE_SWC))
        assert feature.type == "Feature"
        assert feature.geometry.type == "NeuronMorphology"
        assert len(feature.geometry.tree) == 8
        # soma is node 1
        assert feature.geometry.tree[0].type == 1
        assert feature.geometry.tree[0].parent == -1

    def test_microjson_to_swc(self):
        from microjson.swc import swc_to_microjson, microjson_to_swc

        feature = swc_to_microjson(str(SAMPLE_SWC))
        swc_text = microjson_to_swc(feature)
        lines = [l for l in swc_text.strip().split("\n") if not l.startswith("#")]
        assert len(lines) == 8
        # first data line: id=1, type=1, parent=-1
        parts = lines[0].split()
        assert parts[0] == "1"
        assert parts[1] == "1"
        assert parts[6] == "-1"

    def test_swc_roundtrip(self):
        from microjson.swc import swc_to_microjson, microjson_to_swc

        feature = swc_to_microjson(str(SAMPLE_SWC))
        swc_text = microjson_to_swc(feature)
        # Parse the generated SWC text back
        lines = [l for l in swc_text.strip().split("\n") if not l.startswith("#")]
        assert len(lines) == 8
        for line in lines:
            parts = line.split()
            assert len(parts) == 7

    def test_swc_to_linestring3d(self):
        from microjson.swc import swc_to_linestring3d

        mls = swc_to_linestring3d(str(SAMPLE_SWC))
        assert mls.type == "MultiLineString"
        # 7 edges in the tree (8 nodes, each non-root has one parent edge)
        assert len(mls.coordinates) == 7
        # Each edge is a 2-point line
        for line in mls.coordinates:
            assert len(line) == 2
            assert len(line[0]) == 3  # 3D coordinates
