/// Streaming 2D tile generator — processes GeoJSON features through a quadtree.
///
/// Architecture:
///   For each feature: project → clip through quadtree → write Fragment2D to disk
///   For Parquet output: read fragments → encode to world-coord f32 LE → return to Python
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use rayon::prelude::*;
use ahash::AHashMap;

use crate::types2d::BBox2D;
use crate::clip2d::{self, ClipFeature2D};
use crate::fragment2d::{Fragment2D, Fragment2DWriter, Fragment2DReader};
use crate::simplify2d;
use crate::streaming::TagValue;
use crate::encoder_mvt;

// Geometry type constants
const POINT: u8 = 1;
const LINESTRING: u8 = 2;
const POLYGON: u8 = 3;

/// Unique generator ID counter for temp file naming.
static GENERATOR_2D_ID: AtomicU32 = AtomicU32::new(0);

// ---------------------------------------------------------------------------
// Parquet row for 2D data
// ---------------------------------------------------------------------------

struct ParquetRow2D {
    zoom: u8,
    tile_x: u16,
    tile_y: u16,
    feature_id: u32,
    geom_type: u8,
    positions: Vec<u8>,    // raw LE float32 bytes [x,y,x,y,...]
    indices: Vec<u8>,      // raw LE uint32 bytes (line segment indices)
    ring_lengths: Vec<u32>, // ring vertex counts for polygons
}

// ---------------------------------------------------------------------------
// Parquet row collection (pure Rust, no GIL)
// ---------------------------------------------------------------------------

fn collect_parquet_rows_2d(
    tiles: &[((u32, u32, u32), Vec<Fragment2D>)],
    world_bounds: &(f64, f64, f64, f64),
    max_zoom: u32,
    simplify: bool,
) -> Vec<ParquetRow2D> {
    let (xmin, ymin, xmax, ymax) = *world_bounds;
    let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
    let dy = if ymax != ymin { ymax - ymin } else { 1.0 };

    tiles.par_iter()
        .flat_map(|((tz, tx, ty), frags)| {
            let tz = *tz;
            let do_simplify = simplify && tz < max_zoom;

            let mut rows: Vec<ParquetRow2D> = Vec::new();

            for frag in frags {
                let n_verts = frag.xy.len() / 2;
                if n_verts == 0 {
                    continue;
                }

                // Optionally simplify at coarse zoom levels
                let (work_xy, work_rl) = if do_simplify && frag.geom_type == POLYGON && !frag.ring_lengths.is_empty() {
                    let eps = simplify2d::compute_tolerance(tz, max_zoom);
                    if eps > 0.0 {
                        simplify2d::simplify_polygon_rings(&frag.xy, &frag.ring_lengths, eps)
                    } else {
                        (frag.xy.clone(), frag.ring_lengths.clone())
                    }
                } else if do_simplify && frag.geom_type == LINESTRING {
                    let eps = simplify2d::compute_tolerance(tz, max_zoom);
                    if eps > 0.0 {
                        let simplified = simplify2d::douglas_peucker(&frag.xy, eps);
                        (simplified, vec![])
                    } else {
                        (frag.xy.clone(), vec![])
                    }
                } else {
                    (frag.xy.clone(), frag.ring_lengths.clone())
                };

                let work_n = work_xy.len() / 2;
                if work_n == 0 {
                    continue;
                }

                // Convert to world coordinates as f32 LE bytes (x,y pairs)
                let mut pos_f32: Vec<f32> = Vec::with_capacity(work_n * 2);
                for i in 0..work_n {
                    pos_f32.push((xmin + work_xy[i * 2] * dx) as f32);
                    pos_f32.push((ymin + work_xy[i * 2 + 1] * dy) as f32);
                }

                // Build indices for LineString (line segment pairs)
                let idx_u32: Vec<u32> = if frag.geom_type == LINESTRING {
                    let mut indices = Vec::with_capacity(work_n.saturating_sub(1) * 2);
                    for i in 0..work_n.saturating_sub(1) {
                        indices.push(i as u32);
                        indices.push((i + 1) as u32);
                    }
                    indices
                } else {
                    Vec::new()
                };

                let pos_bytes: Vec<u8> = pos_f32.iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let idx_bytes: Vec<u8> = idx_u32.iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect();

                rows.push(ParquetRow2D {
                    zoom: tz as u8,
                    tile_x: *tx as u16,
                    tile_y: *ty as u16,
                    feature_id: frag.feature_id,
                    geom_type: frag.geom_type,
                    positions: pos_bytes,
                    indices: idx_bytes,
                    ring_lengths: work_rl,
                });
            }

            rows
        })
        .collect()
}

// ---------------------------------------------------------------------------
// PyO3 helpers: extract tags
// ---------------------------------------------------------------------------

fn extract_tags_2d(feat: &Bound<'_, PyDict>) -> PyResult<Vec<(String, TagValue)>> {
    let tags_obj = feat.get_item("tags")?;
    let mut result = Vec::new();

    if let Some(tags) = tags_obj {
        if let Ok(tags_dict) = tags.downcast::<PyDict>() {
            for (k, v) in tags_dict.iter() {
                if v.is_none() { continue; }
                let key: String = k.extract()?;
                if v.is_instance_of::<PyBool>() {
                    result.push((key, TagValue::Bool(v.extract()?)));
                } else if v.is_instance_of::<PyString>() {
                    result.push((key, TagValue::Str(v.extract()?)));
                } else if v.is_instance_of::<PyFloat>() {
                    result.push((key, TagValue::Float(v.extract()?)));
                } else if v.is_instance_of::<PyInt>() {
                    result.push((key, TagValue::Int(v.extract()?)));
                }
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// GeoJSON parsing helpers
// ---------------------------------------------------------------------------

/// Parse a single GeoJSON geometry into ClipFeature2D(s).
fn parse_geojson_geometry(
    geom_type: &str,
    coordinates: &serde_json::Value,
    projector_xmin: f64, projector_ymin: f64,
    proj_dx: f64, proj_dy: f64,
) -> Vec<(ClipFeature2D, u8)> {
    let mut results = Vec::new();

    match geom_type {
        "Point" => {
            if let Some(coords) = coordinates.as_array() {
                if coords.len() >= 2 {
                    let x = coords[0].as_f64().unwrap_or(0.0);
                    let y = coords[1].as_f64().unwrap_or(0.0);
                    let nx = (x - projector_xmin) / proj_dx;
                    let ny = (y - projector_ymin) / proj_dy;
                    results.push((ClipFeature2D {
                        xy: vec![nx, ny],
                        ring_lengths: vec![],
                        geom_type: POINT,
                        bbox: BBox2D { min_x: nx, min_y: ny, max_x: nx, max_y: ny },
                    }, POINT));
                }
            }
        }
        "MultiPoint" => {
            if let Some(points) = coordinates.as_array() {
                for pt in points {
                    if let Some(coords) = pt.as_array() {
                        if coords.len() >= 2 {
                            let x = coords[0].as_f64().unwrap_or(0.0);
                            let y = coords[1].as_f64().unwrap_or(0.0);
                            let nx = (x - projector_xmin) / proj_dx;
                            let ny = (y - projector_ymin) / proj_dy;
                            results.push((ClipFeature2D {
                                xy: vec![nx, ny],
                                ring_lengths: vec![],
                                geom_type: POINT,
                                bbox: BBox2D { min_x: nx, min_y: ny, max_x: nx, max_y: ny },
                            }, POINT));
                        }
                    }
                }
            }
        }
        "LineString" => {
            if let Some(coords) = coordinates.as_array() {
                let mut xy = Vec::with_capacity(coords.len() * 2);
                let mut bb = BBox2D::empty();
                for pt in coords {
                    if let Some(c) = pt.as_array() {
                        let x = c[0].as_f64().unwrap_or(0.0);
                        let y = c[1].as_f64().unwrap_or(0.0);
                        let nx = (x - projector_xmin) / proj_dx;
                        let ny = (y - projector_ymin) / proj_dy;
                        xy.push(nx);
                        xy.push(ny);
                        bb.expand(nx, ny);
                    }
                }
                if xy.len() >= 4 {
                    results.push((ClipFeature2D {
                        xy, ring_lengths: vec![], geom_type: LINESTRING, bbox: bb,
                    }, LINESTRING));
                }
            }
        }
        "MultiLineString" => {
            if let Some(lines) = coordinates.as_array() {
                for line in lines {
                    let sub = parse_geojson_geometry("LineString", line, projector_xmin, projector_ymin, proj_dx, proj_dy);
                    results.extend(sub);
                }
            }
        }
        "Polygon" => {
            if let Some(rings) = coordinates.as_array() {
                let mut xy = Vec::new();
                let mut ring_lengths = Vec::new();
                let mut bb = BBox2D::empty();
                for ring in rings {
                    if let Some(coords) = ring.as_array() {
                        let start = xy.len() / 2;
                        for pt in coords {
                            if let Some(c) = pt.as_array() {
                                let x = c[0].as_f64().unwrap_or(0.0);
                                let y = c[1].as_f64().unwrap_or(0.0);
                                let nx = (x - projector_xmin) / proj_dx;
                                let ny = (y - projector_ymin) / proj_dy;
                                xy.push(nx);
                                xy.push(ny);
                                bb.expand(nx, ny);
                            }
                        }
                        let count = xy.len() / 2 - start;
                        if count >= 3 {
                            ring_lengths.push(count as u32);
                        }
                    }
                }
                if !ring_lengths.is_empty() {
                    results.push((ClipFeature2D {
                        xy, ring_lengths, geom_type: POLYGON, bbox: bb,
                    }, POLYGON));
                }
            }
        }
        "MultiPolygon" => {
            if let Some(polygons) = coordinates.as_array() {
                // Flatten all polygons into a single feature with all rings
                let mut xy = Vec::new();
                let mut ring_lengths = Vec::new();
                let mut bb = BBox2D::empty();
                for polygon in polygons {
                    if let Some(rings) = polygon.as_array() {
                        for ring in rings {
                            if let Some(coords) = ring.as_array() {
                                let start = xy.len() / 2;
                                for pt in coords {
                                    if let Some(c) = pt.as_array() {
                                        let x = c[0].as_f64().unwrap_or(0.0);
                                        let y = c[1].as_f64().unwrap_or(0.0);
                                        let nx = (x - projector_xmin) / proj_dx;
                                        let ny = (y - projector_ymin) / proj_dy;
                                        xy.push(nx);
                                        xy.push(ny);
                                        bb.expand(nx, ny);
                                    }
                                }
                                let count = xy.len() / 2 - start;
                                if count >= 3 {
                                    ring_lengths.push(count as u32);
                                }
                            }
                        }
                    }
                }
                if !ring_lengths.is_empty() {
                    results.push((ClipFeature2D {
                        xy, ring_lengths, geom_type: POLYGON, bbox: bb,
                    }, POLYGON));
                }
            }
        }
        "GeometryCollection" => {
            if let Some(geometries) = coordinates.as_array() {
                for geom in geometries {
                    if let (Some(gt), Some(c)) = (geom.get("type").and_then(|v| v.as_str()), geom.get("coordinates")) {
                        let sub = parse_geojson_geometry(gt, c, projector_xmin, projector_ymin, proj_dx, proj_dy);
                        results.extend(sub);
                    }
                }
            }
        }
        _ => {}
    }

    results
}

// ---------------------------------------------------------------------------
// StreamingTileGenerator2D — PyO3 class
// ---------------------------------------------------------------------------

#[pyclass]
pub struct StreamingTileGenerator2D {
    min_zoom: u32,
    max_zoom: u32,
    buffer: f64,
    feature_count: u32,
    tags_registry: HashMap<u32, Vec<(String, TagValue)>>,
    fragment_writer: Option<Fragment2DWriter>,
    frag_dir: PathBuf,
    shard_counter: std::sync::atomic::AtomicUsize,
    fragment_reader: Option<Fragment2DReader>,
    parquet_stream_active: bool,
}

#[pymethods]
impl StreamingTileGenerator2D {
    /// Create a new streaming 2D tile generator.
    ///
    /// Args:
    ///   min_zoom: Minimum zoom level (default 0).
    ///   max_zoom: Maximum zoom level (default 4).
    ///   buffer: Tile buffer in normalized space (fraction of tile size).
    #[new]
    #[pyo3(signature = (min_zoom=0, max_zoom=4, buffer=0.0))]
    fn new(
        min_zoom: u32,
        max_zoom: u32,
        buffer: f64,
    ) -> PyResult<Self> {
        let gen_id = GENERATOR_2D_ID.fetch_add(1, Ordering::Relaxed);
        let frag_dir = std::env::temp_dir()
            .join(format!("microjson_frags2d_{}_{}", std::process::id(), gen_id));
        std::fs::create_dir_all(&frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let shard_path = frag_dir.join("shard_000.mf2d");
        let writer = Fragment2DWriter::new(&shard_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(Self {
            min_zoom,
            max_zoom,
            buffer,
            feature_count: 0,
            tags_registry: HashMap::new(),
            fragment_writer: Some(writer),
            frag_dir,
            shard_counter: std::sync::atomic::AtomicUsize::new(1),
            fragment_reader: None,
            parquet_stream_active: false,
        })
    }

    /// Add a single 2D feature (already projected to [0,1]²).
    ///
    /// The feature dict must have keys: xy (flat [x,y,...]), geom_type (1/2/3),
    /// min_x/min_y/max_x/max_y, and optionally tags, ring_lengths.
    ///
    /// Returns the assigned feature ID.
    fn add_feature(&mut self, feat: &Bound<'_, PyDict>) -> PyResult<u32> {
        let fid = self.feature_count;
        self.feature_count += 1;

        let tags = extract_tags_2d(feat)?;
        self.tags_registry.insert(fid, tags);

        let xy: Vec<f64> = feat.get_item("xy")?.unwrap().extract()?;
        let geom_type: u8 = feat.get_item("geom_type")?.unwrap().extract()?;
        let ring_lengths: Vec<u32> = if let Some(rl) = feat.get_item("ring_lengths")? {
            if rl.is_none() { vec![] } else { rl.extract()? }
        } else {
            vec![]
        };

        let bbox = BBox2D {
            min_x: feat.get_item("min_x")?.unwrap().extract()?,
            min_y: feat.get_item("min_y")?.unwrap().extract()?,
            max_x: feat.get_item("max_x")?.unwrap().extract()?,
            max_y: feat.get_item("max_y")?.unwrap().extract()?,
        };

        let clip_feat = ClipFeature2D { xy, ring_lengths, geom_type, bbox };

        let fragments = clip2d::quadtree_clip(&clip_feat, self.min_zoom, self.max_zoom, self.buffer);

        let writer = self.fragment_writer.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot add features after generate"))?;

        for ((tz, tx, ty), cf) in fragments {
            let frag = Fragment2D {
                feature_id: fid,
                tile_z: tz,
                tile_x: tx,
                tile_y: ty,
                geom_type: cf.geom_type,
                xy: cf.xy,
                ring_lengths: cf.ring_lengths,
            };
            writer.write(&frag)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        Ok(fid)
    }

    /// Add a GeoJSON string — parse, project, clip, and write fragments.
    ///
    /// Args:
    ///   json_str: GeoJSON string (Feature, FeatureCollection, or Geometry).
    ///   bounds: World bounding box (xmin, ymin, xmax, ymax).
    ///
    /// Returns list of assigned feature IDs.
    fn add_geojson(
        &mut self,
        json_str: &str,
        bounds: (f64, f64, f64, f64),
    ) -> PyResult<Vec<u32>> {
        let (xmin, ymin, xmax, ymax) = bounds;
        let proj_dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let proj_dy = if ymax != ymin { ymax - ymin } else { 1.0 };

        let parsed: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        let features = extract_geojson_features(&parsed);

        let writer = self.fragment_writer.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot add features after generate"))?;

        let mut fids = Vec::new();

        for (geom_type, coordinates, properties) in features {
            let clip_feats = parse_geojson_geometry(
                &geom_type, &coordinates,
                xmin, ymin, proj_dx, proj_dy,
            );

            if clip_feats.is_empty() {
                continue;
            }

            let fid = self.feature_count;
            self.feature_count += 1;

            // Extract tags from properties
            let mut tag_vec: Vec<(String, TagValue)> = Vec::new();
            if let Some(obj) = properties.as_object() {
                for (k, v) in obj {
                    match v {
                        serde_json::Value::String(s) => tag_vec.push((k.clone(), TagValue::Str(s.clone()))),
                        serde_json::Value::Number(n) => {
                            if let Some(i) = n.as_i64() {
                                tag_vec.push((k.clone(), TagValue::Int(i)));
                            } else if let Some(f) = n.as_f64() {
                                tag_vec.push((k.clone(), TagValue::Float(f)));
                            }
                        }
                        serde_json::Value::Bool(b) => tag_vec.push((k.clone(), TagValue::Bool(*b))),
                        _ => {}
                    }
                }
            }
            self.tags_registry.insert(fid, tag_vec);

            for (cf, _gt) in clip_feats {
                let fragments = clip2d::quadtree_clip(&cf, self.min_zoom, self.max_zoom, self.buffer);
                for ((tz, tx, ty), clipped) in fragments {
                    let frag = Fragment2D {
                        feature_id: fid,
                        tile_z: tz,
                        tile_x: tx,
                        tile_y: ty,
                        geom_type: clipped.geom_type,
                        xy: clipped.xy,
                        ring_lengths: clipped.ring_lengths,
                    };
                    writer.write(&frag)
                        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                }
            }

            fids.push(fid);
        }

        Ok(fids)
    }

    /// Add multiple GeoJSON files in parallel using rayon.
    ///
    /// Args:
    ///   paths: List of file paths.
    ///   bounds: World bounding box (xmin, ymin, xmax, ymax).
    ///
    /// Returns list of assigned feature IDs.
    fn add_geojson_files(
        &mut self,
        py: Python<'_>,
        paths: Vec<String>,
        bounds: (f64, f64, f64, f64),
    ) -> PyResult<Vec<u32>> {
        let (xmin, ymin, xmax, ymax) = bounds;
        let proj_dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let proj_dy = if ymax != ymin { ymax - ymin } else { 1.0 };

        // Close the single-feature writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        let min_zoom = self.min_zoom;
        let max_zoom = self.max_zoom;
        let buffer = self.buffer;

        // Pre-create per-thread writers
        let n_writers = rayon::current_num_threads() + 1;
        let writers: Vec<std::sync::Mutex<Fragment2DWriter>> = (0..n_writers)
            .map(|_| {
                let shard_id = self.shard_counter.fetch_add(1, Ordering::Relaxed);
                let shard_path = self.frag_dir.join(format!("shard_{:03}.mf2d", shard_id));
                std::sync::Mutex::new(
                    Fragment2DWriter::new(&shard_path)
                        .expect("Failed to create fragment2d shard file")
                )
            })
            .collect();
        let writers_ref = &writers;

        // Atomic feature ID counter
        let fid_counter = AtomicU32::new(self.feature_count);
        let fid_counter_ref = &fid_counter;

        // Collect tags from all files in parallel
        let all_results: Vec<Vec<(u32, Vec<(String, TagValue)>)>> = py.allow_threads(|| {
            paths.par_iter().map(|path| {
                let json_str = match std::fs::read_to_string(path) {
                    Ok(s) => s,
                    Err(_) => return vec![],
                };
                let parsed: serde_json::Value = match serde_json::from_str(&json_str) {
                    Ok(v) => v,
                    Err(_) => return vec![],
                };

                let features = extract_geojson_features(&parsed);
                let mut file_results = Vec::new();

                for (geom_type, coordinates, properties) in features {
                    let clip_feats = parse_geojson_geometry(
                        &geom_type, &coordinates,
                        xmin, ymin, proj_dx, proj_dy,
                    );
                    if clip_feats.is_empty() { continue; }

                    let fid = fid_counter_ref.fetch_add(1, Ordering::Relaxed);

                    let mut tag_vec: Vec<(String, TagValue)> = Vec::new();
                    if let Some(obj) = properties.as_object() {
                        for (k, v) in obj {
                            match v {
                                serde_json::Value::String(s) => tag_vec.push((k.clone(), TagValue::Str(s.clone()))),
                                serde_json::Value::Number(n) => {
                                    if let Some(i) = n.as_i64() { tag_vec.push((k.clone(), TagValue::Int(i))); }
                                    else if let Some(f) = n.as_f64() { tag_vec.push((k.clone(), TagValue::Float(f))); }
                                }
                                serde_json::Value::Bool(b) => tag_vec.push((k.clone(), TagValue::Bool(*b))),
                                _ => {}
                            }
                        }
                    }

                    // Clip and write
                    for (cf, _gt) in clip_feats {
                        let fragments = clip2d::quadtree_clip(&cf, min_zoom, max_zoom, buffer);
                        let writer_idx = rayon::current_thread_index()
                            .unwrap_or(n_writers - 1);
                        let mut w = writers_ref[writer_idx].lock().unwrap();
                        for ((tz, tx, ty), clipped) in fragments {
                            let frag_out = Fragment2D {
                                feature_id: fid,
                                tile_z: tz, tile_x: tx, tile_y: ty,
                                geom_type: clipped.geom_type,
                                xy: clipped.xy,
                                ring_lengths: clipped.ring_lengths,
                            };
                            w.write(&frag_out).ok();
                        }
                    }

                    file_results.push((fid, tag_vec));
                }

                file_results
            }).collect()
        });

        // Flush all shard writers
        for w in &writers {
            w.lock().unwrap().flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Collect all fids and tags back into the registry
        let mut fids = Vec::new();
        for file_results in all_results {
            for (fid, tags) in file_results {
                self.tags_registry.insert(fid, tags);
                fids.push(fid);
            }
        }

        self.feature_count = fid_counter.load(Ordering::Relaxed);
        fids.sort();
        Ok(fids)
    }

    /// Number of features added so far.
    fn feature_count_val(&self) -> u32 {
        self.feature_count
    }

    // ------------------------------------------------------------------
    // Parquet data collection (in-memory)
    // ------------------------------------------------------------------

    /// Collect tile-centric Parquet data from all fragments.
    #[pyo3(signature = (world_bounds, simplify=true))]
    fn _collect_parquet_data(
        &mut self,
        py: Python<'_>,
        world_bounds: (f64, f64, f64, f64),
        simplify: bool,
    ) -> PyResult<PyObject> {
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        let mut reader = Fragment2DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tiles: Vec<((u32, u32, u32), Vec<Fragment2D>)> = groups.into_iter().collect();
        let wb = world_bounds;
        let max_zoom = self.max_zoom;

        let rows = py.allow_threads(|| {
            collect_parquet_rows_2d(&tiles, &wb, max_zoom, simplify)
        });

        build_python_dict(py, &rows, &self.tags_registry)
    }

    // ------------------------------------------------------------------
    // Streaming Parquet batch API
    // ------------------------------------------------------------------

    fn _init_parquet_stream(&mut self) -> PyResult<()> {
        if self.parquet_stream_active {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Parquet stream already active"));
        }
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }
        let reader = Fragment2DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        self.fragment_reader = Some(reader);
        self.parquet_stream_active = true;
        Ok(())
    }

    #[pyo3(signature = (batch_size, world_bounds, max_batch_bytes=2_000_000_000, simplify=true))]
    fn _next_parquet_batch(
        &mut self,
        py: Python<'_>,
        batch_size: usize,
        world_bounds: (f64, f64, f64, f64),
        max_batch_bytes: usize,
        simplify: bool,
    ) -> PyResult<Option<PyObject>> {
        if !self.parquet_stream_active {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Parquet stream not active"));
        }

        let reader = self.fragment_reader.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Fragment reader is None")
        })?;

        let mut groups: AHashMap<(u32, u32, u32), Vec<Fragment2D>> = AHashMap::new();
        let mut count = 0usize;
        let mut batch_bytes = 0usize;
        while count < batch_size && (count == 0 || batch_bytes < max_batch_bytes) {
            match reader.read_next() {
                Ok(Some(frag)) => {
                    batch_bytes += frag.estimate_bytes();
                    let key = (frag.tile_z, frag.tile_x, frag.tile_y);
                    groups.entry(key).or_default().push(frag);
                    count += 1;
                }
                Ok(None) => break,
                Err(e) => return Err(pyo3::exceptions::PyIOError::new_err(e.to_string())),
            }
        }

        if count == 0 {
            return Ok(None);
        }

        let tiles: Vec<((u32, u32, u32), Vec<Fragment2D>)> = groups.into_iter().collect();
        let wb = world_bounds;
        let max_zoom = self.max_zoom;

        let rows = py.allow_threads(|| {
            collect_parquet_rows_2d(&tiles, &wb, max_zoom, simplify)
        });

        let dict = build_python_dict(py, &rows, &self.tags_registry)?;
        Ok(Some(dict))
    }

    fn _close_parquet_stream(&mut self) -> PyResult<()> {
        self.fragment_reader.take();
        self.parquet_stream_active = false;
        Ok(())
    }

    // ------------------------------------------------------------------
    // PBF (Mapbox Vector Tile) output
    // ------------------------------------------------------------------

    /// Generate PBF vector tiles from all fragments.
    ///
    /// Writes tiles to `{output_dir}/{z}/{x}/{y}.pbf` and a `metadata.json`.
    /// Returns the number of tiles written.
    #[pyo3(signature = (output_dir, world_bounds, extent=4096, simplify=true, layer_name="geojsonLayer"))]
    fn generate_pbf(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64),
        extent: u32,
        simplify: bool,
        layer_name: &str,
    ) -> PyResult<u32> {
        // Flush fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by tile
        let mut reader = Fragment2DReader::open_dir(&self.frag_dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let tiles: Vec<((u32, u32, u32), Vec<Fragment2D>)> = groups.into_iter().collect();
        let max_zoom = self.max_zoom;
        let min_zoom = self.min_zoom;
        let out_dir = output_dir.to_string();
        let ln = layer_name.to_string();
        let tags_reg = &self.tags_registry;

        // Build the tag vectors for each feature outside of the parallel section
        // (tags_registry is not Send, so we need to snapshot it)
        let tags_snapshot: HashMap<u32, Vec<(String, TagValue)>> = tags_reg.clone();

        let tile_count = py.allow_threads(|| {
            generate_pbf_tiles(
                &tiles, &out_dir, &world_bounds, max_zoom, extent,
                simplify, &ln, &tags_snapshot,
            )
        }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        // Write metadata.json in TileJSON 3.0.0 format
        let meta_path = std::path::Path::new(output_dir).join("metadata.json");
        let center_x = (world_bounds.0 + world_bounds.2) / 2.0;
        let center_y = (world_bounds.1 + world_bounds.3) / 2.0;
        let meta = serde_json::json!({
            "tilejson": "3.0.0",
            "tiles": ["{z}/{x}/{y}.pbf"],
            "name": "MicroJSON Vector Tiles",
            "description": "Vector tiles generated by microjson Rust pipeline",
            "version": "1.0.0",
            "minzoom": min_zoom,
            "maxzoom": max_zoom,
            "bounds": [world_bounds.0, world_bounds.1, world_bounds.2, world_bounds.3],
            "center": [0.0, center_x, center_y],
            "vector_layers": [{
                "id": ln,
                "fields": {},
                "minzoom": min_zoom,
                "maxzoom": max_zoom,
            }],
            "tile_count": tile_count,
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(tile_count)
    }
}

// ---------------------------------------------------------------------------
// PBF generation (pure Rust, parallel)
// ---------------------------------------------------------------------------

fn generate_pbf_tiles(
    tiles: &[((u32, u32, u32), Vec<Fragment2D>)],
    output_dir: &str,
    _world_bounds: &(f64, f64, f64, f64),
    max_zoom: u32,
    extent: u32,
    simplify: bool,
    layer_name: &str,
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
) -> Result<u32, String> {
    let tile_count = std::sync::atomic::AtomicU32::new(0);

    tiles.par_iter().try_for_each(|((tz, tx, ty), frags)| -> Result<(), String> {
        let tz = *tz;
        let tx = *tx;
        let ty = *ty;
        let do_simplify = simplify && tz < max_zoom;

        let mut mvt_features: Vec<encoder_mvt::MvtFeature> = Vec::new();

        for frag in frags {
            let n_verts = frag.xy.len() / 2;
            if n_verts == 0 {
                continue;
            }

            // Optionally simplify at coarse zoom levels
            let (work_xy, work_rl) = if do_simplify && frag.geom_type == POLYGON && !frag.ring_lengths.is_empty() {
                let eps = simplify2d::compute_tolerance(tz, max_zoom);
                if eps > 0.0 {
                    simplify2d::simplify_polygon_rings(&frag.xy, &frag.ring_lengths, eps)
                } else {
                    (frag.xy.clone(), frag.ring_lengths.clone())
                }
            } else if do_simplify && frag.geom_type == LINESTRING {
                let eps = simplify2d::compute_tolerance(tz, max_zoom);
                if eps > 0.0 {
                    let simplified = simplify2d::douglas_peucker(&frag.xy, eps);
                    (simplified, vec![])
                } else {
                    (frag.xy.clone(), vec![])
                }
            } else {
                (frag.xy.clone(), frag.ring_lengths.clone())
            };

            let work_n = work_xy.len() / 2;
            if work_n == 0 {
                continue;
            }

            // Build MVT geometry commands from normalized coords
            let cmds = encoder_mvt::encode_geometry_commands(
                frag.geom_type, &work_xy, &work_rl,
                tz, tx, ty, extent,
            );

            if cmds.is_empty() {
                continue;
            }

            // Collect tags for this feature
            let tags = tags_registry.get(&frag.feature_id)
                .cloned()
                .unwrap_or_default();

            mvt_features.push(encoder_mvt::MvtFeature {
                id: frag.feature_id as u64,
                geom_type: frag.geom_type,
                geometry_commands: cmds,
                tags,
            });
        }

        if mvt_features.is_empty() {
            return Ok(());
        }

        // Encode the tile
        let tile_bytes = encoder_mvt::encode_mvt_tile(&mvt_features, layer_name, extent);

        // Write to file: {output_dir}/{z}/{x}/{y}.pbf
        let tile_dir = std::path::Path::new(output_dir)
            .join(tz.to_string())
            .join(tx.to_string());
        std::fs::create_dir_all(&tile_dir)
            .map_err(|e| format!("Failed to create dir {}: {}", tile_dir.display(), e))?;

        let tile_path = tile_dir.join(format!("{}.pbf", ty));
        std::fs::write(&tile_path, &tile_bytes)
            .map_err(|e| format!("Failed to write {}: {}", tile_path.display(), e))?;

        tile_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    })?;

    Ok(tile_count.load(std::sync::atomic::Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract features from a GeoJSON value (supports Feature, FeatureCollection, and bare Geometry).
fn extract_geojson_features(parsed: &serde_json::Value) -> Vec<(String, serde_json::Value, serde_json::Value)> {
    let mut features = Vec::new();

    let geojson_type = parsed.get("type").and_then(|v| v.as_str()).unwrap_or("");

    match geojson_type {
        "FeatureCollection" => {
            if let Some(feats) = parsed.get("features").and_then(|v| v.as_array()) {
                for feat in feats {
                    let props = feat.get("properties").cloned().unwrap_or(serde_json::Value::Null);
                    if let Some(geom) = feat.get("geometry") {
                        let gt = geom.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        let coords = geom.get("coordinates").cloned().unwrap_or(serde_json::Value::Null);
                        if !gt.is_empty() {
                            features.push((gt.to_string(), coords, props));
                        }
                    }
                }
            }
        }
        "Feature" => {
            let props = parsed.get("properties").cloned().unwrap_or(serde_json::Value::Null);
            if let Some(geom) = parsed.get("geometry") {
                let gt = geom.get("type").and_then(|v| v.as_str()).unwrap_or("");
                let coords = geom.get("coordinates").cloned().unwrap_or(serde_json::Value::Null);
                if !gt.is_empty() {
                    features.push((gt.to_string(), coords, props));
                }
            }
        }
        _ => {
            // Bare geometry
            let coords = parsed.get("coordinates").cloned().unwrap_or(serde_json::Value::Null);
            if !geojson_type.is_empty() && !coords.is_null() {
                features.push((geojson_type.to_string(), coords, serde_json::Value::Null));
            }
        }
    }

    features
}

/// Build the Python dict from ParquetRow2D slices.
fn build_python_dict(
    py: Python<'_>,
    rows: &[ParquetRow2D],
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
) -> PyResult<PyObject> {
    let n = rows.len();

    let zoom_list = PyList::empty(py);
    let tx_list = PyList::empty(py);
    let ty_list = PyList::empty(py);
    let fid_list = PyList::empty(py);
    let gt_list = PyList::empty(py);
    let pos_list = PyList::empty(py);
    let idx_list = PyList::empty(py);
    let rl_list = PyList::empty(py);
    let tags_list = PyList::empty(py);

    for row in rows {
        zoom_list.append(row.zoom)?;
        tx_list.append(row.tile_x)?;
        ty_list.append(row.tile_y)?;
        fid_list.append(row.feature_id)?;
        gt_list.append(row.geom_type)?;
        pos_list.append(pyo3::types::PyBytes::new(py, &row.positions))?;
        idx_list.append(pyo3::types::PyBytes::new(py, &row.indices))?;

        // Ring lengths as Python list
        let rl_py = PyList::empty(py);
        for &rl in &row.ring_lengths {
            rl_py.append(rl)?;
        }
        rl_list.append(rl_py)?;

        // Tags
        let tag_pairs = PyList::empty(py);
        if let Some(tags) = tags_registry.get(&row.feature_id) {
            for (k, v) in tags {
                let vs = match v {
                    TagValue::Str(s) => s.clone(),
                    TagValue::Int(i) => i.to_string(),
                    TagValue::Float(f) => f.to_string(),
                    TagValue::Bool(b) => b.to_string(),
                };
                tag_pairs.append((k.as_str(), vs.as_str()))?;
            }
        }
        tags_list.append(tag_pairs)?;
    }

    let dict = PyDict::new(py);
    dict.set_item("zoom", zoom_list)?;
    dict.set_item("tile_x", tx_list)?;
    dict.set_item("tile_y", ty_list)?;
    dict.set_item("feature_id", fid_list)?;
    dict.set_item("geom_type", gt_list)?;
    dict.set_item("positions", pos_list)?;
    dict.set_item("indices", idx_list)?;
    dict.set_item("ring_lengths", rl_list)?;
    dict.set_item("tags", tags_list)?;
    dict.set_item("row_count", n)?;

    Ok(dict.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_geojson_point() {
        let coords = serde_json::json!([10.0, 20.0]);
        let results = parse_geojson_geometry("Point", &coords, 0.0, 0.0, 100.0, 100.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, POINT);
        assert!((results[0].0.xy[0] - 0.1).abs() < 1e-10);
        assert!((results[0].0.xy[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_parse_geojson_linestring() {
        let coords = serde_json::json!([[0.0, 0.0], [50.0, 50.0], [100.0, 0.0]]);
        let results = parse_geojson_geometry("LineString", &coords, 0.0, 0.0, 100.0, 100.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, LINESTRING);
        assert_eq!(results[0].0.xy.len(), 6); // 3 vertices * 2
    }

    #[test]
    fn test_parse_geojson_polygon() {
        let coords = serde_json::json!([[[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]]);
        let results = parse_geojson_geometry("Polygon", &coords, 0.0, 0.0, 100.0, 100.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, POLYGON);
        assert_eq!(results[0].0.ring_lengths.len(), 1);
        assert_eq!(results[0].0.ring_lengths[0], 4);
    }

    #[test]
    fn test_extract_geojson_feature_collection() {
        let json = serde_json::json!({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
                    "properties": {"name": "test"}
                }
            ]
        });
        let features = extract_geojson_features(&json);
        assert_eq!(features.len(), 1);
        assert_eq!(features[0].0, "Point");
    }
}
