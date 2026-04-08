# Validating muDM and GeoJSON

For a comprehensive guide to validation and all other usage patterns, see the [Usage](usage.md) page.

The core validation workflow uses Pydantic models from `mudm.model`:

```python
import mudm.model as mj
import json

# Validate a muDM file
with open("annotations.json") as f:
    data = json.load(f)
mudm_obj = mj.MuDM.model_validate(data)

# Any GeoJSON is also valid muDM
with open("features.geojson") as f:
    data = json.load(f)
geojson_obj = mj.GeoJSON.model_validate(data)
```

If validation fails, Pydantic raises a `ValidationError` with details about which fields are invalid. This ensures your JSON adheres to the muDM specification before further processing.
