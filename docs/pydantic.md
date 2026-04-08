# Pydantic Models for muDM and GeoJSON

## Introduction

This document describes the Pydantic models used for GeoJSON and muDM objects. These models leverage Python's type hinting and Pydantic's validation mechanisms, making it robust and efficient to work with complex GeoJSON and muDM objects.

## Models

::: mudm.model

### Base Objects

### Geometry Types

Uses geojson-pydantic models for GeoJSON geometry types, included here for reference. Please refer to the [geojson-pydantic documentation](https://developmentseed.org/geojson-pydantic/) for more information.

#### Point

Represents a GeoJSON Point object.

#### MultiPoint

Represents a GeoJSON MultiPoint object.

#### LineString

Represents a GeoJSON LineString object.

#### MultiLineString

Represents a GeoJSON MultiLineString object.

#### Polygon

Represents a GeoJSON Polygon object.

#### MultiPolygon

Represents a GeoJSON MultiPolygon object.

### Compound Objects

#### GeometryCollection

A collection of multiple geometries. From [geojson-pydantic](https://developmentseed.org/geojson-pydantic/), included here for reference.

#### Feature

Represents a GeoJSON feature object, from [geojson-pydantic](https://developmentseed.org/geojson-pydantic/), included here for reference.

#### FeatureCollection

Represents a GeoJSON feature collection, from [geojson-pydantic](https://developmentseed.org/geojson-pydantic/), included here for reference.

#### GeoJSON

The root object of a GeoJSON file.

::: mudm.model.GeoJSON

### muDM Extended Models

#### MuDMFeature

A muDM feature, which is an extension of a GeoJSON feature.

::: mudm.model.MuDMFeature

#### MuDMFeatureCollection

A muDM feature collection, which is an extension of a GeoJSON feature collection.

::: mudm.model.MuDMFeatureCollection

#### MuDM

The root object of a muDM file.

::: mudm.model.MuDM
