# Metadata Usage in muDM Example

This guide demonstrates how to designate metadata in muDM using the `properties` field in the `Feature` class. The `properties` field is used to store metadata related to a feature. This guide provides examples of how to populate these fields in both JSON and Python.

Now, let's explore an example to understand how these fields can be populated in both JSON and Python.

## JSON Example

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [100.0, 0.0],
        [101.0, 0.0],
        [101.0, 1.0],
        [100.0, 1.0],
        [100.0, 0.0]
      ]
    ]
  },
  "properties": {
    "name": "Sample Polygon",
    "description": "This is a sample rectangular polygon.",
    "cellCount": 5000,
    "ratioInfectivity": [0.2, 0.5, 0.8]
  }
}
```

## Python Example

```python
from mudm.model import MuDMFeature

# Usage
example_properties = {
  "name": "Sample Polygon",
  "description": "This is a sample rectangular polygon.",
  "cellCount": 5000,
  "ratioInfectivity": [0.2, 0.5, 0.8]
}

example_feature = MuDMFeature(
    type="Feature",
    geometry={
        "type": "Polygon",
        "coordinates": [
            [
                [100.0, 0.0],
                [101.0, 0.0],
                [101.0, 1.0],
                [100.0, 1.0],
                [100.0, 0.0]
            ]
        ]
    },
    properties=example_properties
)
# print json
print(example_feature.model_dump_json(indent=2, exclude_unset=True))

```

---

In this example, a muDM feature is defined with a rectangular polygon geometry, with metadata stored as flat key-value pairs in the `properties` dictionary. The JSON representation offers a clear formatting of the data, while the Python script showcases how to instantiate a `MuDMFeature` with the same data.
