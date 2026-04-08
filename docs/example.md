# muDM Examples

## Basic FeatureCollection

A FeatureCollection with polygon features and flat key-value properties.

```json
{
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[0.0, 0.0], [0.0, 50.0], [50.0, 50.0], [50.0, 0.0], [0.0, 0.0]]]
        },
        "properties": {
            "well": "A1",
            "cellCount": 5,
            "ratioInfectivity": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
      },
      {
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[50.0, 0.0], [50.0, 50.0], [100.0, 50.0], [100.0, 0.0], [50.0, 0.0]]]
        },
        "properties": {
            "well": "A2",
            "cellCount": 10,
            "ratioInfectivity": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
      }
    ]
}
```

## Point Feature with Radius

A point geometry representing a circular region, using the optional `radius` field.

```json
{
    "type": "Feature",
    "geometry": {
        "type": "Point",
        "coordinates": [256.0, 512.0],
        "radius": 15.0
    },
    "properties": {
        "featureClass": "nucleus",
        "confidence": 0.95
    }
}
```

## FeatureCollection with Coordinate System

A collection with a `multiscale` object defining the coordinate system and units.

```json
{
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]]
        },
        "properties": {
            "label": 1,
            "cellType": "pyramidal"
        }
      }
    ],
    "multiscale": {
        "axes": [
            {
                "name": "x",
                "unit": "micrometer",
                "description": "x-axis"
            },
            {
                "name": "y",
                "unit": "micrometer",
                "description": "y-axis"
            }
        ],
        "coordinateTransformations": [
            {
                "type": "scale",
                "scale": [0.5, 0.5]
            }
        ]
    }
}
```

## 3D Mesh (TIN Geometry)

A triangulated surface mesh using the `TIN` geometry type.

```json
{
    "type": "Feature",
    "geometry": {
        "type": "TIN",
        "coordinates": [
            [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]],
            [[[1, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]]],
            [[[0, 0, 0], [1, 0, 0], [0.5, 0.5, 1], [0, 0, 0]]],
            [[[1, 0, 0], [1, 1, 0], [0.5, 0.5, 1], [1, 0, 0]]]
        ]
    },
    "properties": {
        "featureClass": "neuron",
        "cellType": "pyramidal",
        "region": "CA1"
    }
}
```

## Image Feature

A special feature representing an image with a rectangular geometry and URI reference.

```json
{
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "subtype": "Rectangle",
        "coordinates": [[[0, 0], [0, 1024], [1024, 1024], [1024, 0], [0, 0]]]
    },
    "properties": {
        "type": "Image",
        "URI": "./image_001.tif"
    }
}
```

For more usage patterns including tiling pipelines, format converters, and GeoParquet I/O, see the [Usage](usage.md) page.
