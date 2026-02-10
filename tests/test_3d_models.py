"""Tests for 3D coordinate support on existing geojson_pydantic types
and new MicroJSON 3D geometry types."""

import pytest
from pydantic import ValidationError
from geojson_pydantic import (
    Point,
    MultiPoint,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon,
)
from microjson.model import (
    MicroJSON,
    MicroFeature,
    MicroFeatureCollection,
    PolyhedralSurface,
    TIN,
    Slice,
    SliceStack,
    NeuronMorphology,
)


# ---------------------------------------------------------------------------
# Step 1a: Existing geometry types accept [x, y, z] coordinates
# ---------------------------------------------------------------------------


class TestExistingTypes3DCoords:
    """Verify geojson_pydantic geometry types accept 3D coordinates."""

    def test_point_3d(self):
        p = Point(type="Point", coordinates=(1.0, 2.0, 3.0))
        assert p.coordinates[2] == 3.0

    def test_multipoint_3d(self):
        mp = MultiPoint(
            type="MultiPoint",
            coordinates=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        assert len(mp.coordinates) == 2
        assert mp.coordinates[0][2] == 3.0

    def test_linestring_3d(self):
        ls = LineString(
            type="LineString",
            coordinates=[(0.0, 0.0, 0.0), (10.0, 10.0, 5.0)],
        )
        assert ls.coordinates[1][2] == 5.0

    def test_multilinestring_3d(self):
        mls = MultiLineString(
            type="MultiLineString",
            coordinates=[
                [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
                [(2.0, 2.0, 2.0), (3.0, 3.0, 3.0)],
            ],
        )
        assert mls.coordinates[1][0][2] == 2.0

    def test_polygon_3d(self):
        ring = [
            (0.0, 0.0, 1.0),
            (10.0, 0.0, 1.0),
            (10.0, 10.0, 1.0),
            (0.0, 0.0, 1.0),
        ]
        poly = Polygon(type="Polygon", coordinates=[ring])
        assert poly.coordinates[0][0][2] == 1.0

    def test_multipolygon_3d(self):
        ring = [
            (0.0, 0.0, 5.0),
            (10.0, 0.0, 5.0),
            (10.0, 10.0, 5.0),
            (0.0, 0.0, 5.0),
        ]
        mp = MultiPolygon(type="MultiPolygon", coordinates=[[ring]])
        assert mp.coordinates[0][0][0][2] == 5.0


# ---------------------------------------------------------------------------
# Step 1b: 3D bounding box support
# ---------------------------------------------------------------------------


class TestBBox3D:
    """Verify 3D bbox [minx, miny, minz, maxx, maxy, maxz] works."""

    def test_point_3d_bbox(self):
        p = Point(
            type="Point",
            coordinates=(5.0, 10.0, 15.0),
            bbox=(0.0, 0.0, 0.0, 100.0, 100.0, 50.0),
        )
        assert len(p.bbox) == 6
        assert p.bbox == (0.0, 0.0, 0.0, 100.0, 100.0, 50.0)

    def test_polygon_3d_bbox(self):
        ring = [
            (0.0, 0.0, 1.0),
            (10.0, 0.0, 1.0),
            (10.0, 10.0, 1.0),
            (0.0, 0.0, 1.0),
        ]
        poly = Polygon(
            type="Polygon",
            coordinates=[ring],
            bbox=(0.0, 0.0, 1.0, 10.0, 10.0, 1.0),
        )
        assert len(poly.bbox) == 6

    def test_linestring_3d_bbox(self):
        ls = LineString(
            type="LineString",
            coordinates=[(0.0, 0.0, 0.0), (10.0, 10.0, 5.0)],
            bbox=(0.0, 0.0, 0.0, 10.0, 10.0, 5.0),
        )
        assert ls.bbox[5] == 5.0


# ---------------------------------------------------------------------------
# Step 1c: has_z property detection
# ---------------------------------------------------------------------------


class TestHasZ:
    """Verify has_z property detects 3D coordinates."""

    def test_point_2d_has_z_false(self):
        p = Point(type="Point", coordinates=(1.0, 2.0))
        assert p.has_z is False

    def test_point_3d_has_z_true(self):
        p = Point(type="Point", coordinates=(1.0, 2.0, 3.0))
        assert p.has_z is True

    def test_linestring_3d_has_z_true(self):
        ls = LineString(
            type="LineString",
            coordinates=[(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
        )
        assert ls.has_z is True

    def test_polygon_2d_has_z_false(self):
        ring = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
        poly = Polygon(type="Polygon", coordinates=[ring])
        assert poly.has_z is False

    def test_polygon_3d_has_z_true(self):
        ring = [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 1.0),
        ]
        poly = Polygon(type="Polygon", coordinates=[ring])
        assert poly.has_z is True

    def test_multipoint_3d_has_z_true(self):
        mp = MultiPoint(
            type="MultiPoint",
            coordinates=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
        )
        assert mp.has_z is True


# ---------------------------------------------------------------------------
# Step 1 (extra): Round-trip serialization with 3D coords
# ---------------------------------------------------------------------------


class TestRoundTrip3D:
    """Verify 3D geometries survive JSON round-trip."""

    def test_point_3d_roundtrip(self):
        data = {"type": "Point", "coordinates": [1.0, 2.0, 3.0]}
        p = Point.model_validate(data)
        dumped = p.model_dump()
        assert dumped["coordinates"][2] == 3.0

    def test_polygon_3d_roundtrip(self):
        ring = [[0, 0, 1], [10, 0, 1], [10, 10, 1], [0, 0, 1]]
        data = {"type": "Polygon", "coordinates": [ring]}
        poly = Polygon.model_validate(data)
        dumped = poly.model_dump()
        assert dumped["coordinates"][0][0][2] == 1.0

    def test_linestring_3d_json_roundtrip(self):
        data = {
            "type": "LineString",
            "coordinates": [[0, 0, 0], [10, 10, 5]],
        }
        ls = LineString.model_validate(data)
        json_str = ls.model_dump_json()
        ls2 = LineString.model_validate_json(json_str)
        assert ls2.coordinates[1][2] == 5.0


# ---------------------------------------------------------------------------
# Step 2: PolyhedralSurface geometry type
# ---------------------------------------------------------------------------

# A tetrahedron: 4 triangular faces, all 3D coordinates
TETRAHEDRON_COORDS = [
    # face 0 (base triangle)
    [[(0, 0, 0), (10, 0, 0), (5, 10, 0), (0, 0, 0)]],
    # face 1
    [[(0, 0, 0), (10, 0, 0), (5, 5, 10), (0, 0, 0)]],
    # face 2
    [[(10, 0, 0), (5, 10, 0), (5, 5, 10), (10, 0, 0)]],
    # face 3
    [[(5, 10, 0), (0, 0, 0), (5, 5, 10), (5, 10, 0)]],
]


class TestPolyhedralSurface:
    """Tests for the PolyhedralSurface 3D geometry type."""

    def test_create_tetrahedron(self):
        ps = PolyhedralSurface(
            type="PolyhedralSurface", coordinates=TETRAHEDRON_COORDS
        )
        assert ps.type == "PolyhedralSurface"
        assert len(ps.coordinates) == 4

    def test_bbox(self):
        ps = PolyhedralSurface(
            type="PolyhedralSurface", coordinates=TETRAHEDRON_COORDS
        )
        bb = ps.bbox3d()
        assert bb == (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)

    def test_centroid(self):
        ps = PolyhedralSurface(
            type="PolyhedralSurface", coordinates=TETRAHEDRON_COORDS
        )
        c = ps.centroid3d()
        assert len(c) == 3
        # centroid of unique vertices (0,0,0),(10,0,0),(5,10,0),(5,5,10)
        assert c[0] == pytest.approx(5.0)
        assert c[1] == pytest.approx(3.75)
        assert c[2] == pytest.approx(2.5)

    def test_roundtrip_json(self):
        ps = PolyhedralSurface(
            type="PolyhedralSurface", coordinates=TETRAHEDRON_COORDS
        )
        data = ps.model_dump()
        ps2 = PolyhedralSurface.model_validate(data)
        assert ps2.type == "PolyhedralSurface"
        assert len(ps2.coordinates) == 4

    def test_empty_coordinates_rejected(self):
        with pytest.raises(ValidationError):
            PolyhedralSurface(type="PolyhedralSurface", coordinates=[])

    def test_single_face(self):
        face = [[(0, 0, 0), (10, 0, 0), (5, 10, 0), (0, 0, 0)]]
        ps = PolyhedralSurface(
            type="PolyhedralSurface", coordinates=[face]
        )
        assert len(ps.coordinates) == 1


# ---------------------------------------------------------------------------
# Step 3: TIN (Triangulated Irregular Network) geometry type
# ---------------------------------------------------------------------------

# Two triangles sharing an edge
TIN_COORDS = [
    # triangle 0 (closed ring of 4 = 3 vertices + repeated first)
    [[(0, 0, 0), (10, 0, 0), (5, 10, 5), (0, 0, 0)]],
    # triangle 1
    [[(10, 0, 0), (20, 0, 0), (5, 10, 5), (10, 0, 0)]],
]


class TestTIN:
    """Tests for the TIN (Triangulated Irregular Network) geometry type."""

    def test_create_tin(self):
        tin = TIN(type="TIN", coordinates=TIN_COORDS)
        assert tin.type == "TIN"
        assert len(tin.coordinates) == 2

    def test_bbox(self):
        tin = TIN(type="TIN", coordinates=TIN_COORDS)
        bb = tin.bbox3d()
        assert bb == (0.0, 0.0, 0.0, 20.0, 10.0, 5.0)

    def test_centroid(self):
        tin = TIN(type="TIN", coordinates=TIN_COORDS)
        c = tin.centroid3d()
        assert len(c) == 3

    def test_roundtrip_json(self):
        tin = TIN(type="TIN", coordinates=TIN_COORDS)
        data = tin.model_dump()
        tin2 = TIN.model_validate(data)
        assert tin2.type == "TIN"
        assert len(tin2.coordinates) == 2

    def test_non_triangle_face_rejected(self):
        """Each face must be a triangle (ring of exactly 4 coords)."""
        bad_coords = [
            # a quad, not a triangle
            [[(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 0)]],
        ]
        with pytest.raises(ValidationError):
            TIN(type="TIN", coordinates=bad_coords)

    def test_empty_coordinates_rejected(self):
        with pytest.raises(ValidationError):
            TIN(type="TIN", coordinates=[])


# ---------------------------------------------------------------------------
# Step 4: SliceStack geometry type
# ---------------------------------------------------------------------------


class TestSlice:
    """Tests for the Slice model."""

    def test_create_slice(self):
        s = Slice(
            z=0.0,
            geometry=Polygon(
                type="Polygon",
                coordinates=[[(0, 0), (10, 0), (10, 10), (0, 0)]],
            ),
        )
        assert s.z == 0.0
        assert s.geometry.type == "Polygon"

    def test_slice_with_properties(self):
        s = Slice(
            z=1.5,
            geometry=Polygon(
                type="Polygon",
                coordinates=[[(0, 0), (10, 0), (10, 10), (0, 0)]],
            ),
            properties={"slice_index": 3, "image_ref": "section_003.tif"},
        )
        assert s.properties["slice_index"] == 3

    def test_slice_with_multipolygon(self):
        s = Slice(
            z=2.0,
            geometry=MultiPolygon(
                type="MultiPolygon",
                coordinates=[[[(0, 0), (5, 0), (5, 5), (0, 0)]]],
            ),
        )
        assert s.geometry.type == "MultiPolygon"


class TestSliceStack:
    """Tests for the SliceStack 2.5D geometry type."""

    def _make_ring(self):
        return [(0, 0), (100, 0), (100, 100), (0, 0)]

    def _make_stack(self, z_values=None):
        if z_values is None:
            z_values = [0.0, 0.05, 0.10]
        ring = self._make_ring()
        slices = [
            Slice(
                z=z,
                geometry=Polygon(type="Polygon", coordinates=[ring]),
            )
            for z in z_values
        ]
        return SliceStack(type="SliceStack", slices=slices)

    def test_create_basic(self):
        ss = self._make_stack()
        assert ss.type == "SliceStack"
        assert len(ss.slices) == 3
        assert ss.axis == "z"

    def test_default_axis(self):
        ss = self._make_stack()
        assert ss.axis == "z"

    def test_custom_axis(self):
        ring = self._make_ring()
        ss = SliceStack(
            type="SliceStack",
            slices=[
                Slice(z=0.0, geometry=Polygon(type="Polygon", coordinates=[ring])),
            ],
            axis="x",
        )
        assert ss.axis == "x"

    def test_units(self):
        ss = self._make_stack()
        assert ss.units is None
        ring = self._make_ring()
        ss2 = SliceStack(
            type="SliceStack",
            slices=[
                Slice(z=0.0, geometry=Polygon(type="Polygon", coordinates=[ring])),
            ],
            units="micrometers",
        )
        assert ss2.units == "micrometers"

    def test_interpolation(self):
        ring = self._make_ring()
        ss = SliceStack(
            type="SliceStack",
            slices=[
                Slice(z=0.0, geometry=Polygon(type="Polygon", coordinates=[ring])),
            ],
            interpolation="linear",
        )
        assert ss.interpolation == "linear"

    def test_unsorted_z_rejected(self):
        """Slices must be sorted by z value."""
        with pytest.raises(ValidationError):
            self._make_stack(z_values=[0.0, 0.10, 0.05])

    def test_duplicate_z_rejected(self):
        """No duplicate z values allowed."""
        with pytest.raises(ValidationError):
            self._make_stack(z_values=[0.0, 0.05, 0.05])

    def test_bbox3d(self):
        ss = self._make_stack(z_values=[0.0, 0.05, 0.10])
        bb = ss.bbox3d()
        # x: 0..100, y: 0..100, z: 0.0..0.10
        assert bb[0] == 0.0   # minx
        assert bb[1] == 0.0   # miny
        assert bb[2] == 0.0   # minz
        assert bb[3] == 100.0  # maxx
        assert bb[4] == 100.0  # maxy
        assert bb[5] == pytest.approx(0.10)  # maxz

    def test_roundtrip_json(self):
        ss = self._make_stack()
        data = ss.model_dump()
        ss2 = SliceStack.model_validate(data)
        assert ss2.type == "SliceStack"
        assert len(ss2.slices) == 3
        assert ss2.slices[0].z == 0.0

    def test_empty_slices_rejected(self):
        with pytest.raises(ValidationError):
            SliceStack(type="SliceStack", slices=[])


# ---------------------------------------------------------------------------
# Step 7: Integration — MicroJSON with 3D geometry types
# ---------------------------------------------------------------------------

# Full 3D MicroJSON document from spec Appendix A (adapted)
FULL_3D_DOCUMENT = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "NeuronMorphology",
                "tree": [
                    {"id": 1, "type": 1, "x": 500.0, "y": 300.0,
                     "z": 100.0, "r": 8.0, "parent": -1},
                    {"id": 2, "type": 3, "x": 510.0, "y": 305.0,
                     "z": 102.0, "r": 2.0, "parent": 1},
                    {"id": 3, "type": 3, "x": 520.0, "y": 310.0,
                     "z": 105.0, "r": 1.5, "parent": 2},
                ],
            },
            "properties": {
                "cell_type": "pyramidal",
                "species": "Mus musculus",
            },
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "SliceStack",
                "slices": [
                    {
                        "z": 100.0,
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[400, 200], [600, 200], [600, 400],
                                 [400, 400], [400, 200]]
                            ],
                        },
                        "properties": {"slice_index": 0},
                    },
                    {
                        "z": 100.05,
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[405, 205], [595, 205], [595, 395],
                                 [405, 395], [405, 205]]
                            ],
                        },
                        "properties": {"slice_index": 1},
                    },
                ],
                "axis": "z",
                "units": "micrometers",
            },
            "properties": {"region_name": "CA1_stratum_pyramidale"},
        },
    ],
}


class TestMicroJSON3DIntegration:
    """Test MicroJSON root model with 3D geometry types."""

    def test_feature_with_neuron_morphology(self):
        data = {
            "type": "Feature",
            "geometry": {
                "type": "NeuronMorphology",
                "tree": [
                    {"id": 1, "type": 1, "x": 0, "y": 0, "z": 0,
                     "r": 1.0, "parent": -1},
                ],
            },
            "properties": {},
        }
        mj = MicroJSON.model_validate(data)
        assert mj.root.geometry.type == "NeuronMorphology"

    def test_feature_with_slicestack(self):
        data = {
            "type": "Feature",
            "geometry": {
                "type": "SliceStack",
                "slices": [
                    {
                        "z": 0.0,
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[0, 0], [10, 0], [10, 10], [0, 0]]
                            ],
                        },
                    },
                ],
            },
            "properties": {},
        }
        mj = MicroJSON.model_validate(data)
        assert mj.root.geometry.type == "SliceStack"

    def test_feature_with_polyhedral_surface(self):
        face = [[(0, 0, 0), (10, 0, 0), (5, 10, 0), (0, 0, 0)]]
        data = {
            "type": "Feature",
            "geometry": {
                "type": "PolyhedralSurface",
                "coordinates": [face],
            },
            "properties": {},
        }
        mj = MicroJSON.model_validate(data)
        assert mj.root.geometry.type == "PolyhedralSurface"

    def test_feature_with_tin(self):
        face = [[(0, 0, 0), (10, 0, 0), (5, 10, 5), (0, 0, 0)]]
        data = {
            "type": "Feature",
            "geometry": {"type": "TIN", "coordinates": [face]},
            "properties": {},
        }
        mj = MicroJSON.model_validate(data)
        assert mj.root.geometry.type == "TIN"

    def test_full_3d_document_roundtrip(self):
        """Parse the full 3D MicroJSON example from the spec."""
        mfc = MicroFeatureCollection.model_validate(FULL_3D_DOCUMENT)
        assert len(mfc.features) == 2
        assert mfc.features[0].geometry.type == "NeuronMorphology"
        assert mfc.features[1].geometry.type == "SliceStack"

        # Round-trip through JSON
        dumped = mfc.model_dump()
        mfc2 = MicroFeatureCollection.model_validate(dumped)
        assert len(mfc2.features) == 2

    def test_standard_geojson_still_works(self):
        """Ensure standard 2D GeoJSON features still parse."""
        data = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [1.0, 2.0],
            },
            "properties": {"name": "test"},
        }
        mj = MicroJSON.model_validate(data)
        assert mj.root.geometry.type == "Point"
