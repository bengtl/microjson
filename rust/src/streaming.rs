/// Streaming 3D tile generator — processes features one at a time.
///
/// Memory model:
/// - During ingestion (add_feature): O(1 feature) — clip through octree, write fragments
/// - During encoding (generate_mjb): O(all fragments) — read from disk, transform, encode
///
/// Architecture:
///   For each feature: clip through octree → write binary fragments to temp file
///   For each tile (rayon parallel): read fragments → transform → encode → write .mjb
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use rayon::prelude::*;
use ahash::AHashMap;

use crate::types::BBox3D;
use crate::fragment::{Fragment, FragmentWriter, FragmentReader};
use crate::encoder_mjb;
use crate::encoder_glb::{self, GlbFeature};
use crate::tileset_json;
use crate::tile_transform;
use crate::obj_parser;

// Geometry type constants matching protobuf GeomType.
const POINT3D: u8 = 1;
const LINESTRING3D: u8 = 2;
const POLYGON3D: u8 = 3;
const POLYHEDRALSURFACE: u8 = 4;
const TIN: u8 = 5;

/// Small epsilon for boundary points (matches Python octree._EPS).
const EPS: f64 = 1e-10;

/// Unique generator ID counter for temp file naming.
static GENERATOR_ID: AtomicU32 = AtomicU32::new(0);

// ---------------------------------------------------------------------------
// Tag value type for dictionary-encoded properties
// ---------------------------------------------------------------------------

/// Tag value matching protobuf Value variants.
#[derive(Debug, Clone)]
pub(crate) enum TagValue {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

// ---------------------------------------------------------------------------
// Intermediate feature for pure-Rust octree clipping
// ---------------------------------------------------------------------------

/// Feature in normalized [0,1]³ space with bounding box for clip acceleration.
#[derive(Debug, Clone)]
struct ClipFeature {
    xy: Vec<f64>,
    z: Vec<f64>,
    ring_lengths: Vec<u32>,
    geom_type: u8,
    bbox: BBox3D,
}

fn compute_bbox_rs(xy: &[f64], z: &[f64]) -> BBox3D {
    let mut bb = BBox3D::empty();
    let n = z.len();
    for j in 0..n {
        bb.expand(xy[j * 2], xy[j * 2 + 1], z[j]);
    }
    bb
}

// ---------------------------------------------------------------------------
// Pure-Rust clip functions (matching clip3d.py / clip.rs logic)
// ---------------------------------------------------------------------------

/// Clip a list of features to [k1, k2) along axis (0=X, 1=Y, 2=Z).
fn clip_features(features: &[ClipFeature], k1: f64, k2: f64, axis: u8) -> Vec<ClipFeature> {
    let mut result = Vec::new();
    for feat in features {
        let (a_min, a_max) = match axis {
            0 => (feat.bbox.min_x, feat.bbox.max_x),
            1 => (feat.bbox.min_y, feat.bbox.max_y),
            _ => (feat.bbox.min_z, feat.bbox.max_z),
        };

        // Trivial reject — half-open interval [k1, k2)
        if a_min >= k2 || a_max < k1 {
            continue;
        }

        // Trivial accept — fully within [k1, k2)
        if a_min >= k1 && a_max < k2 {
            result.push(feat.clone());
            continue;
        }

        // Type-specific clipping
        match feat.geom_type {
            TIN | POLYHEDRALSURFACE => {
                if let Some(clipped) = clip_surface_rs(feat, k1, k2, axis) {
                    result.push(clipped);
                }
            }
            POINT3D => {
                result.extend(clip_points_rs(feat, k1, k2, axis));
            }
            LINESTRING3D => {
                result.extend(clip_line_rs(feat, k1, k2, axis));
            }
            POLYGON3D => {
                // Trivial accept (same as Python _clip_polygon)
                result.push(feat.clone());
            }
            _ => {}
        }
    }
    result
}

/// Clip TIN/PolyhedralSurface per-face — keep faces overlapping [k1, k2).
fn clip_surface_rs(feat: &ClipFeature, k1: f64, k2: f64, axis: u8) -> Option<ClipFeature> {
    let ring_lengths = if feat.ring_lengths.is_empty() {
        vec![feat.z.len() as u32]
    } else {
        feat.ring_lengths.clone()
    };

    let mut out_xy = Vec::new();
    let mut out_z = Vec::new();
    let mut out_rl = Vec::new();

    let mut offset = 0usize;
    for &rl in &ring_lengths {
        let rl = rl as usize;
        let mut f_min = f64::INFINITY;
        let mut f_max = f64::NEG_INFINITY;

        for j in 0..rl {
            let val = match axis {
                0 => feat.xy[(offset + j) * 2],
                1 => feat.xy[(offset + j) * 2 + 1],
                _ => feat.z[offset + j],
            };
            if val < f_min { f_min = val; }
            if val > f_max { f_max = val; }
        }

        if f_min >= k2 || f_max < k1 {
            offset += rl;
            continue;
        }

        out_xy.extend_from_slice(&feat.xy[offset * 2..(offset + rl) * 2]);
        out_z.extend_from_slice(&feat.z[offset..offset + rl]);
        out_rl.push(rl as u32);
        offset += rl;
    }

    if out_rl.is_empty() {
        return None;
    }

    let bbox = compute_bbox_rs(&out_xy, &out_z);
    Some(ClipFeature {
        xy: out_xy,
        z: out_z,
        ring_lengths: out_rl,
        geom_type: feat.geom_type,
        bbox,
    })
}

/// Clip LineString3D to [k1, k2) — returns zero or more line segments.
fn clip_line_rs(feat: &ClipFeature, k1: f64, k2: f64, axis: u8) -> Vec<ClipFeature> {
    let xy = &feat.xy;
    let z = &feat.z;
    let n = z.len();

    if n < 2 {
        return vec![];
    }

    let axis_val = |idx: usize| -> f64 {
        match axis {
            0 => xy[idx * 2],
            1 => xy[idx * 2 + 1],
            _ => z[idx],
        }
    };

    let intersect_at = |i: usize, j: usize, k: f64| -> (f64, f64, f64) {
        let ax = xy[i * 2]; let ay = xy[i * 2 + 1]; let az = z[i];
        let bx = xy[j * 2]; let by = xy[j * 2 + 1]; let bz = z[j];
        let t = match axis {
            0 => { let d = bx - ax; if d != 0.0 { (k - ax) / d } else { 0.0 } }
            1 => { let d = by - ay; if d != 0.0 { (k - ay) / d } else { 0.0 } }
            _ => { let d = bz - az; if d != 0.0 { (k - az) / d } else { 0.0 } }
        };
        (ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t)
    };

    let mut segments: Vec<ClipFeature> = Vec::new();
    let mut out_xy: Vec<f64> = Vec::new();
    let mut out_z: Vec<f64> = Vec::new();

    let flush = |out_xy: &mut Vec<f64>, out_z: &mut Vec<f64>, segments: &mut Vec<ClipFeature>| {
        if out_z.len() >= 2 {
            let bbox = compute_bbox_rs(out_xy, out_z);
            segments.push(ClipFeature {
                xy: std::mem::take(out_xy),
                z: std::mem::take(out_z),
                ring_lengths: vec![],
                geom_type: LINESTRING3D,
                bbox,
            });
        } else {
            out_xy.clear();
            out_z.clear();
        }
    };

    for i in 0..n - 1 {
        let a_val = axis_val(i);
        let b_val = axis_val(i + 1);
        let a_in = k1 <= a_val && a_val < k2;

        if a_in {
            out_xy.push(xy[i * 2]);
            out_xy.push(xy[i * 2 + 1]);
            out_z.push(z[i]);
        }

        let cross_k1 = (a_val < k1 && b_val > k1) || (b_val < k1 && a_val > k1);
        let cross_k2 = (a_val < k2 && b_val >= k2) || (b_val < k2 && a_val >= k2);

        let mut crossings: Vec<(f64, f64, f64, f64, bool)> = Vec::new();
        if cross_k1 {
            let (ix, iy, iz) = intersect_at(i, i + 1, k1);
            let d = b_val - a_val;
            let t = if d != 0.0 { (k1 - a_val) / d } else { 0.0 };
            let entering = b_val > a_val;
            crossings.push((t, ix, iy, iz, entering));
        }
        if cross_k2 {
            let (ix, iy, iz) = intersect_at(i, i + 1, k2);
            let d = b_val - a_val;
            let t = if d != 0.0 { (k2 - a_val) / d } else { 0.0 };
            let entering = b_val < a_val;
            crossings.push((t, ix, iy, iz, entering));
        }

        crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        for (_, ix, iy, iz, entering) in crossings {
            if entering {
                flush(&mut out_xy, &mut out_z, &mut segments);
                out_xy.push(ix);
                out_xy.push(iy);
                out_z.push(iz);
            } else {
                out_xy.push(ix);
                out_xy.push(iy);
                out_z.push(iz);
                flush(&mut out_xy, &mut out_z, &mut segments);
            }
        }
    }

    // Last point
    if n > 0 {
        let last_val = axis_val(n - 1);
        if k1 <= last_val && last_val < k2 {
            out_xy.push(xy[(n - 1) * 2]);
            out_xy.push(xy[(n - 1) * 2 + 1]);
            out_z.push(z[n - 1]);
        }
    }

    flush(&mut out_xy, &mut out_z, &mut segments);
    segments
}

/// Clip points to [k1, k2) — filter by axis value.
fn clip_points_rs(feat: &ClipFeature, k1: f64, k2: f64, axis: u8) -> Vec<ClipFeature> {
    let mut results = Vec::new();
    let n = feat.z.len();

    for i in 0..n {
        let val = match axis {
            0 => feat.xy[i * 2],
            1 => feat.xy[i * 2 + 1],
            _ => feat.z[i],
        };
        if k1 <= val && val < k2 {
            let px = feat.xy[i * 2];
            let py = feat.xy[i * 2 + 1];
            let pz = feat.z[i];
            results.push(ClipFeature {
                xy: vec![px, py],
                z: vec![pz],
                ring_lengths: vec![],
                geom_type: POINT3D,
                bbox: BBox3D {
                    min_x: px, min_y: py, min_z: pz,
                    max_x: px, max_y: py, max_z: pz,
                },
            });
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Octree traversal — clip one feature across all zoom levels
// ---------------------------------------------------------------------------

/// Clip a feature through the octree, producing (tile_key, clipped_feature) pairs.
///
/// Matches the stack-based octree in `octree.py`:
/// - Split X → Y → Z at each level
/// - Store tile at each zoom in [min_zoom, max_zoom]
/// - Pad upper boundaries with EPS for boundary inclusion
fn octree_clip(
    feat: &ClipFeature,
    min_zoom: u32,
    max_zoom: u32,
    buffer: f64,
) -> Vec<((u32, u32, u32, u32), ClipFeature)> {
    let mut result = Vec::new();

    // Stack: (zoom, x, y, d, features_at_this_node)
    let mut stack: Vec<(u32, u32, u32, u32, Vec<ClipFeature>)> = vec![
        (0, 0, 0, 0, vec![feat.clone()])
    ];

    while let Some((z, x, y, d, feats)) = stack.pop() {
        if feats.is_empty() {
            continue;
        }

        // Store at this level if within zoom range
        if z >= min_zoom {
            for f in &feats {
                result.push(((z, x, y, d), f.clone()));
            }
        }

        if z >= max_zoom {
            continue;
        }

        // Split into 8 octants
        let nz = z + 1;
        let n = 1u32 << nz;

        let x0 = (x * 2) as f64 / n as f64;
        let x1 = (x * 2 + 1) as f64 / n as f64;
        let x2 = (x * 2 + 2) as f64 / n as f64;
        let y0 = (y * 2) as f64 / n as f64;
        let y1 = (y * 2 + 1) as f64 / n as f64;
        let y2 = (y * 2 + 2) as f64 / n as f64;
        let d0 = (d * 2) as f64 / n as f64;
        let d1 = (d * 2 + 1) as f64 / n as f64;
        let d2 = (d * 2 + 2) as f64 / n as f64;

        let buf = if buffer > 0.0 { buffer / n as f64 } else { 0.0 };

        let x2_pad = if x * 2 + 2 == n { x2 + EPS } else { x2 };
        let y2_pad = if y * 2 + 2 == n { y2 + EPS } else { y2 };
        let d2_pad = if d * 2 + 2 == n { d2 + EPS } else { d2 };

        // Clip X → two halves
        let left = clip_features(&feats, x0 - buf, x1 + buf, 0);
        let right = clip_features(&feats, x1 - buf, x2_pad + buf, 0);

        // Clip Y → four quadrants
        for (x_feats, nx) in [(left, x * 2), (right, x * 2 + 1)] {
            if x_feats.is_empty() {
                continue;
            }
            let bottom = clip_features(&x_feats, y0 - buf, y1 + buf, 1);
            let top = clip_features(&x_feats, y1 - buf, y2_pad + buf, 1);

            // Clip Z → eight octants
            for (y_feats, ny) in [(bottom, y * 2), (top, y * 2 + 1)] {
                if y_feats.is_empty() {
                    continue;
                }
                let front = clip_features(&y_feats, d0 - buf, d1 + buf, 2);
                let back = clip_features(&y_feats, d1 - buf, d2_pad + buf, 2);

                for (z_feats, nd) in [(front, d * 2), (back, d * 2 + 1)] {
                    if !z_feats.is_empty() {
                        stack.push((nz, nx, ny, nd, z_feats));
                    }
                }
            }
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Tile encoding from fragments (pure Rust, no GIL)
// ---------------------------------------------------------------------------

/// Encode a single tag value to protobuf bytes.
fn encode_tag_value(val: &TagValue) -> Vec<u8> {
    let mut buf = Vec::new();
    match val {
        TagValue::Bool(b) => {
            encoder_mjb::write_varint_field(&mut buf, encoder_mjb::VALUE_BOOL, if *b { 1 } else { 0 });
        }
        TagValue::Str(s) => {
            encoder_mjb::write_bytes_field(&mut buf, encoder_mjb::VALUE_STRING, s.as_bytes());
        }
        TagValue::Float(d) => {
            encoder_mjb::write_tag(&mut buf, encoder_mjb::VALUE_DOUBLE, 1); // wire type 1 = 64-bit
            buf.extend_from_slice(&d.to_le_bytes());
        }
        TagValue::Int(i) => {
            if *i < 0 {
                encoder_mjb::write_tag(&mut buf, encoder_mjb::VALUE_SINT, 0);
                encoder_mjb::encode_varint(&mut buf, encoder_mjb::zigzag(*i));
            } else {
                encoder_mjb::write_varint_field(&mut buf, encoder_mjb::VALUE_UINT, *i as u64);
            }
        }
    }
    buf
}

/// Transform fragment geometry to tile-local integers and encode as protobuf tile.
fn encode_tile_from_fragments(
    frags: &[Fragment],
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
    tz: u32, tx: u32, ty: u32, td: u32,
    extent: u32, extent_z: u32,
    layer_name: &str,
) -> Vec<u8> {
    let n = 1u32 << tz;
    let x0 = tx as f64 / n as f64;
    let y0 = ty as f64 / n as f64;
    let z0 = td as f64 / n as f64;
    let scale_x = n as f64;
    let scale_y = n as f64;
    let scale_z = n as f64;

    // Dictionary-encoded tags (shared across features in this tile)
    let mut key_indices: AHashMap<String, u32> = AHashMap::new();
    let mut value_indices: AHashMap<String, u32> = AHashMap::new();
    let mut keys_list: Vec<String> = Vec::new();
    let mut values_encoded: Vec<Vec<u8>> = Vec::new();

    let mut features_encoded: Vec<Vec<u8>> = Vec::new();

    for (feat_idx, frag) in frags.iter().enumerate() {
        let mut feat_buf = Vec::new();

        // id
        encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_ID, feat_idx as u64);

        // type
        encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_TYPE, frag.geom_type as u64);

        // tags (dictionary-encoded from registry)
        if let Some(tags) = tags_registry.get(&frag.feature_id) {
            let mut tag_indices: Vec<u32> = Vec::new();
            for (key, val) in tags {
                // Key index
                let ki = match key_indices.get(key) {
                    Some(&idx) => idx,
                    None => {
                        let idx = key_indices.len() as u32;
                        key_indices.insert(key.clone(), idx);
                        keys_list.push(key.clone());
                        idx
                    }
                };
                tag_indices.push(ki);

                // Value index (use repr for dedup)
                let v_repr = match val {
                    TagValue::Str(s) => format!("'{}'", s),
                    TagValue::Int(i) => i.to_string(),
                    TagValue::Float(f) => format!("{:?}", f),
                    TagValue::Bool(b) => b.to_string(),
                };
                let vi = match value_indices.get(&v_repr) {
                    Some(&idx) => idx,
                    None => {
                        let idx = value_indices.len() as u32;
                        value_indices.insert(v_repr, idx);
                        values_encoded.push(encode_tag_value(val));
                        idx
                    }
                };
                tag_indices.push(vi);
            }
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_TAGS, &tag_indices);
        }

        // Transform geometry to tile-local integers
        let nv = frag.z.len();
        let mut new_xy: Vec<i64> = Vec::with_capacity(nv * 2);
        let mut new_z: Vec<i64> = Vec::with_capacity(nv);

        for i in 0..nv {
            let lx = (frag.xy[i * 2] - x0) * scale_x;
            let ly = (frag.xy[i * 2 + 1] - y0) * scale_y;
            let lz = (frag.z[i] - z0) * scale_z;
            new_xy.push(tile_transform::round_half_to_even(lx * extent as f64));
            new_xy.push(tile_transform::round_half_to_even(ly * extent as f64));
            new_z.push(tile_transform::round_half_to_even(lz * extent_z as f64));
        }

        // Encode geometry by type
        let gt = frag.geom_type;
        if gt == 1 {
            // POINT3D
            let geom = encoder_mjb::encode_point_geometry(&new_xy);
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_mjb::encode_z(&new_z);
            encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 2 {
            // LINESTRING3D
            let geom = encoder_mjb::encode_line_geometry(&new_xy);
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_mjb::encode_z(&new_z);
            encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 4 || gt == 5 {
            // TIN / POLYHEDRALSURFACE — indexed mesh
            let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                vec![new_xy.len() / 2]
            } else {
                frag.ring_lengths.iter().map(|&r| r as usize).collect()
            };

            let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
            let mut positions: Vec<f32> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();
            let mut offset = 0usize;

            for &rl in &rls {
                let nv = if rl >= 4 { 3 } else { rl };
                if nv == 3 {
                    let mut tri = [0u32; 3];
                    for vi_off in 0..3 {
                        let vi = offset + vi_off;
                        let x = new_xy[vi * 2];
                        let y = new_xy[vi * 2 + 1];
                        let zc = if vi < new_z.len() { new_z[vi] } else { 0 };
                        let key = (x, y, zc);
                        let idx = match vertex_map.get(&key) {
                            Some(&idx) => idx,
                            None => {
                                let idx = vertex_map.len() as u32;
                                vertex_map.insert(key, idx);
                                positions.push(x as f32);
                                positions.push(y as f32);
                                positions.push(zc as f32);
                                idx
                            }
                        };
                        tri[vi_off] = idx;
                    }
                    indices.extend_from_slice(&tri);
                }
                offset += rl;
            }

            let pos_bytes: Vec<u8> = positions.iter().flat_map(|f| f.to_le_bytes()).collect();
            let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
            encoder_mjb::write_bytes_field(&mut feat_buf, encoder_mjb::FEAT_MESH_POSITIONS, &pos_bytes);
            encoder_mjb::write_bytes_field(&mut feat_buf, encoder_mjb::FEAT_MESH_INDICES, &idx_bytes);
        } else if gt == 3 {
            // POLYGON3D
            let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                vec![new_xy.len() / 2]
            } else {
                frag.ring_lengths.iter().map(|&r| r as usize).collect()
            };
            let geom = encoder_mjb::encode_polygon_geometry(&new_xy, &rls);
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_mjb::encode_z(&new_z);
            encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
        } else {
            // Default: line encoding
            let geom = encoder_mjb::encode_line_geometry(&new_xy);
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_mjb::encode_z(&new_z);
            encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
        }

        features_encoded.push(feat_buf);
    }

    // --- Encode layer ---
    let mut layer_buf = Vec::new();
    encoder_mjb::write_varint_field(&mut layer_buf, encoder_mjb::LAYER_VERSION, 3);
    encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_NAME, layer_name.as_bytes());
    for feat_buf in &features_encoded {
        encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_FEATURES, feat_buf);
    }
    for key in &keys_list {
        encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_KEYS, key.as_bytes());
    }
    for val_buf in &values_encoded {
        encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_VALUES, val_buf);
    }
    encoder_mjb::write_varint_field(&mut layer_buf, encoder_mjb::LAYER_EXTENT, extent as u64);
    encoder_mjb::write_varint_field(&mut layer_buf, encoder_mjb::LAYER_EXTENT_Z, extent_z as u64);

    // --- Encode tile ---
    let mut tile_buf = Vec::new();
    encoder_mjb::write_bytes_field(&mut tile_buf, encoder_mjb::TILE_LAYERS, &layer_buf);
    tile_buf
}

// ---------------------------------------------------------------------------
// Per-feature MJB encoding (world coordinates, single feature per tile)
// ---------------------------------------------------------------------------

/// Encode a single feature's max-zoom fragments as one MJB tile in world coordinates.
///
/// Unlike tile-centric MJB (which uses tile-local integer coordinates), this
/// produces a feature-centric MJB where mesh positions are float32 world
/// coordinates. Non-mesh geometry (point, line, polygon) is encoded with
/// MVT commands in a bbox-local integer space.
///
/// Returns `(mjb_bytes, feature_bbox)` where feature_bbox is [xmin,ymin,zmin,xmax,ymax,zmax].
fn encode_feature_mjb(
    frags: &[&Fragment],
    tags: Option<&Vec<(String, TagValue)>>,
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    layer_name: &str,
    extent: u32,
    extent_z: u32,
) -> (Vec<u8>, [f64; 6]) {
    let (xmin, ymin, zmin, xmax, ymax, zmax) = *world_bounds;
    let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
    let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
    let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

    // Track feature bounding box in world coords
    let mut bb_min = [f64::INFINITY; 3];
    let mut bb_max = [f64::NEG_INFINITY; 3];

    // Dictionary-encoded tags
    let mut key_indices: AHashMap<String, u32> = AHashMap::new();
    let mut value_indices: AHashMap<String, u32> = AHashMap::new();
    let mut keys_list: Vec<String> = Vec::new();
    let mut values_encoded: Vec<Vec<u8>> = Vec::new();

    // Separate fragments by geometry type
    let mut mesh_frags: Vec<&Fragment> = Vec::new();
    let mut other_frags: Vec<&Fragment> = Vec::new();

    for frag in frags {
        match frag.geom_type {
            4 | 5 => mesh_frags.push(frag),
            _ => other_frags.push(frag),
        }
    }

    let mut features_encoded: Vec<Vec<u8>> = Vec::new();
    let mut feat_idx = 0u64;

    // --- Encode TIN/PolyhedralSurface as a single merged indexed mesh ---
    if !mesh_frags.is_empty() {
        let mut feat_buf = Vec::new();
        encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_ID, feat_idx);
        feat_idx += 1;

        // Use geom_type from first mesh fragment
        let geom_type = mesh_frags[0].geom_type;
        encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_TYPE, geom_type as u64);

        // Tags
        if let Some(tag_list) = tags {
            let mut tag_indices_vec: Vec<u32> = Vec::new();
            for (key, val) in tag_list {
                let ki = match key_indices.get(key) {
                    Some(&idx) => idx,
                    None => {
                        let idx = key_indices.len() as u32;
                        key_indices.insert(key.clone(), idx);
                        keys_list.push(key.clone());
                        idx
                    }
                };
                tag_indices_vec.push(ki);

                let v_repr = match val {
                    TagValue::Str(s) => format!("'{}'", s),
                    TagValue::Int(i) => i.to_string(),
                    TagValue::Float(f) => format!("{:?}", f),
                    TagValue::Bool(b) => b.to_string(),
                };
                let vi = match value_indices.get(&v_repr) {
                    Some(&idx) => idx,
                    None => {
                        let idx = value_indices.len() as u32;
                        value_indices.insert(v_repr, idx);
                        values_encoded.push(encode_tag_value(val));
                        idx
                    }
                };
                tag_indices_vec.push(vi);
            }
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_TAGS, &tag_indices_vec);
        }

        // Merge geometry with vertex dedup (float32 bit-level, like Neuroglancer)
        let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
        let mut positions: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for frag in &mesh_frags {
            let n_verts = frag.z.len();
            let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                vec![n_verts]
            } else {
                frag.ring_lengths.iter().map(|&r| r as usize).collect()
            };

            let mut offset = 0usize;
            for &rl in &ring_lengths {
                let nv = if rl >= 4 { 3 } else { rl };
                if nv < 3 || offset + 2 >= n_verts {
                    offset += rl;
                    continue;
                }

                let mut tri = [0u32; 3];
                for vi_off in 0..3 {
                    let vi = offset + vi_off;
                    let wx = xmin + frag.xy[vi * 2] * dx;
                    let wy = ymin + frag.xy[vi * 2 + 1] * dy;
                    let wz = zmin + frag.z[vi] * dz;

                    // Update bbox
                    if wx < bb_min[0] { bb_min[0] = wx; }
                    if wy < bb_min[1] { bb_min[1] = wy; }
                    if wz < bb_min[2] { bb_min[2] = wz; }
                    if wx > bb_max[0] { bb_max[0] = wx; }
                    if wy > bb_max[1] { bb_max[1] = wy; }
                    if wz > bb_max[2] { bb_max[2] = wz; }

                    let wx_f32 = wx as f32;
                    let wy_f32 = wy as f32;
                    let wz_f32 = wz as f32;

                    let key = (
                        wx_f32.to_bits() as i64,
                        wy_f32.to_bits() as i64,
                        wz_f32.to_bits() as i64,
                    );

                    let idx = match vertex_map.get(&key) {
                        Some(&idx) => idx,
                        None => {
                            let idx = vertex_map.len() as u32;
                            vertex_map.insert(key, idx);
                            positions.push(wx_f32);
                            positions.push(wy_f32);
                            positions.push(wz_f32);
                            idx
                        }
                    };
                    tri[vi_off] = idx;
                }

                // Skip degenerate triangles
                if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                    indices.extend_from_slice(&tri);
                }

                offset += rl;
            }
        }

        if !positions.is_empty() && !indices.is_empty() {
            let pos_bytes: Vec<u8> = positions.iter().flat_map(|f| f.to_le_bytes()).collect();
            let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
            encoder_mjb::write_bytes_field(&mut feat_buf, encoder_mjb::FEAT_MESH_POSITIONS, &pos_bytes);
            encoder_mjb::write_bytes_field(&mut feat_buf, encoder_mjb::FEAT_MESH_INDICES, &idx_bytes);
        }

        features_encoded.push(feat_buf);
    }

    // --- Encode non-mesh geometry (point, line, polygon) ---
    for frag in &other_frags {
        let n_verts = frag.z.len();
        if n_verts == 0 {
            continue;
        }

        // Compute world coordinates and update bbox
        for i in 0..n_verts {
            let wx = xmin + frag.xy[i * 2] * dx;
            let wy = ymin + frag.xy[i * 2 + 1] * dy;
            let wz = zmin + frag.z[i] * dz;
            if wx < bb_min[0] { bb_min[0] = wx; }
            if wy < bb_min[1] { bb_min[1] = wy; }
            if wz < bb_min[2] { bb_min[2] = wz; }
            if wx > bb_max[0] { bb_max[0] = wx; }
            if wy > bb_max[1] { bb_max[1] = wy; }
            if wz > bb_max[2] { bb_max[2] = wz; }
        }

        // Encode in bbox-local integer space using extent
        let feat_xmin = bb_min[0];
        let feat_ymin = bb_min[1];
        let feat_zmin = bb_min[2];
        let feat_dx = if bb_max[0] != bb_min[0] { bb_max[0] - bb_min[0] } else { 1.0 };
        let feat_dy = if bb_max[1] != bb_min[1] { bb_max[1] - bb_min[1] } else { 1.0 };
        let feat_dz = if bb_max[2] != bb_min[2] { bb_max[2] - bb_min[2] } else { 1.0 };

        let mut new_xy: Vec<i64> = Vec::with_capacity(n_verts * 2);
        let mut new_z: Vec<i64> = Vec::with_capacity(n_verts);

        for i in 0..n_verts {
            let wx = xmin + frag.xy[i * 2] * dx;
            let wy = ymin + frag.xy[i * 2 + 1] * dy;
            let wz = zmin + frag.z[i] * dz;
            let lx = (wx - feat_xmin) / feat_dx;
            let ly = (wy - feat_ymin) / feat_dy;
            let lz = (wz - feat_zmin) / feat_dz;
            new_xy.push(tile_transform::round_half_to_even(lx * extent as f64));
            new_xy.push(tile_transform::round_half_to_even(ly * extent as f64));
            new_z.push(tile_transform::round_half_to_even(lz * extent_z as f64));
        }

        let mut feat_buf = Vec::new();
        encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_ID, feat_idx);
        feat_idx += 1;
        encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_TYPE, frag.geom_type as u64);

        // Tags (only on first non-mesh feature to avoid duplication)
        if features_encoded.is_empty() {
            if let Some(tag_list) = tags {
                let mut tag_indices_vec: Vec<u32> = Vec::new();
                for (key, val) in tag_list {
                    let ki = match key_indices.get(key) {
                        Some(&idx) => idx,
                        None => {
                            let idx = key_indices.len() as u32;
                            key_indices.insert(key.clone(), idx);
                            keys_list.push(key.clone());
                            idx
                        }
                    };
                    tag_indices_vec.push(ki);

                    let v_repr = match val {
                        TagValue::Str(s) => format!("'{}'", s),
                        TagValue::Int(i) => i.to_string(),
                        TagValue::Float(f) => format!("{:?}", f),
                        TagValue::Bool(b) => b.to_string(),
                    };
                    let vi = match value_indices.get(&v_repr) {
                        Some(&idx) => idx,
                        None => {
                            let idx = value_indices.len() as u32;
                            value_indices.insert(v_repr, idx);
                            values_encoded.push(encode_tag_value(val));
                            idx
                        }
                    };
                    tag_indices_vec.push(vi);
                }
                encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_TAGS, &tag_indices_vec);
            }
        }

        // Encode geometry by type
        let gt = frag.geom_type;
        if gt == 1 {
            let geom = encoder_mjb::encode_point_geometry(&new_xy);
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_mjb::encode_z(&new_z);
            encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 2 {
            let geom = encoder_mjb::encode_line_geometry(&new_xy);
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_mjb::encode_z(&new_z);
            encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
        } else if gt == 3 {
            let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                vec![new_xy.len() / 2]
            } else {
                frag.ring_lengths.iter().map(|&r| r as usize).collect()
            };
            let geom = encoder_mjb::encode_polygon_geometry(&new_xy, &rls);
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_mjb::encode_z(&new_z);
            encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
        } else {
            let geom = encoder_mjb::encode_line_geometry(&new_xy);
            encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
            let z_enc = encoder_mjb::encode_z(&new_z);
            encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
        }

        features_encoded.push(feat_buf);
    }

    // --- Encode layer ---
    let mut layer_buf = Vec::new();
    encoder_mjb::write_varint_field(&mut layer_buf, encoder_mjb::LAYER_VERSION, 3);
    encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_NAME, layer_name.as_bytes());
    for feat_buf in &features_encoded {
        encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_FEATURES, feat_buf);
    }
    for key in &keys_list {
        encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_KEYS, key.as_bytes());
    }
    for val_buf in &values_encoded {
        encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_VALUES, val_buf);
    }
    encoder_mjb::write_varint_field(&mut layer_buf, encoder_mjb::LAYER_EXTENT, extent as u64);
    encoder_mjb::write_varint_field(&mut layer_buf, encoder_mjb::LAYER_EXTENT_Z, extent_z as u64);

    // --- Encode tile ---
    let mut tile_buf = Vec::new();
    encoder_mjb::write_bytes_field(&mut tile_buf, encoder_mjb::TILE_LAYERS, &layer_buf);

    let bbox = [bb_min[0], bb_min[1], bb_min[2], bb_max[0], bb_max[1], bb_max[2]];
    (tile_buf, bbox)
}

// ---------------------------------------------------------------------------
// Multi-LOD per-feature MJB encoder
// ---------------------------------------------------------------------------

/// Encode a feature's fragments across all zoom levels into a multi-LOD MJB tile.
///
/// Each zoom level becomes a separate Layer named "lod_0" (finest) through "lod_N" (coarsest).
/// - At max_zoom: exact float32 vertex dedup (same as single-LOD encode_feature_mjb)
/// - At coarser levels: grid-based vertex clustering (cell_size = world_extent / (base_cells * 2^zoom))
///
/// Returns (mjb_bytes, finest_lod_bbox).
fn encode_feature_mjb_multilod(
    all_frags: &[&Fragment],
    tags: Option<&Vec<(String, TagValue)>>,
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    max_zoom: u32,
    extent: u32,
    extent_z: u32,
    base_cells: u32,
) -> (Vec<u8>, [f64; 6]) {
    let (xmin, ymin, zmin, xmax, ymax, zmax) = *world_bounds;
    let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
    let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
    let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

    // Group fragments by tile_z (zoom level)
    let mut by_zoom: std::collections::BTreeMap<u32, Vec<&Fragment>> = std::collections::BTreeMap::new();
    for frag in all_frags {
        by_zoom.entry(frag.tile_z).or_default().push(frag);
    }

    // Track finest-LOD bounding box (from max_zoom fragments)
    let mut bb_min = [f64::INFINITY; 3];
    let mut bb_max = [f64::NEG_INFINITY; 3];

    let mut layer_bufs: Vec<Vec<u8>> = Vec::new();

    // Process each zoom level: max_zoom → 0 produces lod_0 → lod_N
    for (&zoom, zoom_frags) in by_zoom.iter().rev() {
        let lod_index = max_zoom.saturating_sub(zoom);
        let layer_name = format!("lod_{}", lod_index);
        let is_finest = zoom == max_zoom;
        let do_simplify = zoom < max_zoom;

        // Simplification grid parameters
        let cell_size = if do_simplify {
            let cells = base_cells * (1u32 << zoom);
            [dx / cells as f64, dy / cells as f64, dz / cells as f64]
        } else {
            [0.0; 3]
        };

        // Separate mesh vs non-mesh fragments
        let mut mesh_frags: Vec<&Fragment> = Vec::new();
        let mut other_frags: Vec<&Fragment> = Vec::new();
        for frag in zoom_frags {
            match frag.geom_type {
                4 | 5 => mesh_frags.push(frag),
                _ => other_frags.push(frag),
            }
        }

        // Dictionary-encoded tags (per-layer)
        let mut key_indices: AHashMap<String, u32> = AHashMap::new();
        let mut value_indices: AHashMap<String, u32> = AHashMap::new();
        let mut keys_list: Vec<String> = Vec::new();
        let mut values_encoded: Vec<Vec<u8>> = Vec::new();
        let mut features_encoded: Vec<Vec<u8>> = Vec::new();
        let mut feat_idx = 0u64;

        // --- Encode TIN/PolyhedralSurface ---
        if !mesh_frags.is_empty() {
            let mut feat_buf = Vec::new();
            encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_ID, feat_idx);
            feat_idx += 1;
            let geom_type = mesh_frags[0].geom_type;
            encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_TYPE, geom_type as u64);

            // Tags (only on first feature)
            if let Some(tag_list) = tags {
                let mut tag_indices_vec: Vec<u32> = Vec::new();
                for (key, val) in tag_list {
                    let ki = *key_indices.entry(key.clone()).or_insert_with(|| {
                        let idx = keys_list.len() as u32;
                        keys_list.push(key.clone());
                        idx
                    });
                    tag_indices_vec.push(ki);

                    let v_repr = match val {
                        TagValue::Str(s) => format!("'{}'", s),
                        TagValue::Int(i) => i.to_string(),
                        TagValue::Float(f) => format!("{:?}", f),
                        TagValue::Bool(b) => b.to_string(),
                    };
                    let vi = *value_indices.entry(v_repr).or_insert_with(|| {
                        let idx = values_encoded.len() as u32;
                        values_encoded.push(encode_tag_value(val));
                        idx
                    });
                    tag_indices_vec.push(vi);
                }
                encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_TAGS, &tag_indices_vec);
            }

            // Merge geometry with vertex dedup or clustering
            let mut positions: Vec<f32> = Vec::new();
            let mut indices: Vec<u32> = Vec::new();

            if do_simplify {
                // --- Vertex clustering for coarser LODs ---
                let mut cell_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
                let mut accum: Vec<[f64; 4]> = Vec::new(); // [sum_x, sum_y, sum_z, count]
                let mut vert_remap: Vec<u32> = Vec::new();

                for frag in &mesh_frags {
                    let n_verts = frag.z.len();
                    let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                        vec![n_verts]
                    } else {
                        frag.ring_lengths.iter().map(|&r| r as usize).collect()
                    };

                    let mut offset = 0usize;
                    for &rl in &ring_lengths {
                        let nv = if rl >= 4 { 3 } else { rl };
                        for vi_off in 0..nv {
                            let vi = offset + vi_off;
                            if vi >= n_verts { break; }
                            let wx = xmin + frag.xy[vi * 2] * dx;
                            let wy = ymin + frag.xy[vi * 2 + 1] * dy;
                            let wz = zmin + frag.z[vi] * dz;

                            let qx = ((wx - xmin) / cell_size[0]).floor() as i64;
                            let qy = ((wy - ymin) / cell_size[1]).floor() as i64;
                            let qz = ((wz - zmin) / cell_size[2]).floor() as i64;
                            let key = (qx, qy, qz);

                            let idx = match cell_map.get(&key) {
                                Some(&idx) => {
                                    accum[idx as usize][0] += wx;
                                    accum[idx as usize][1] += wy;
                                    accum[idx as usize][2] += wz;
                                    accum[idx as usize][3] += 1.0;
                                    idx
                                }
                                None => {
                                    let idx = cell_map.len() as u32;
                                    cell_map.insert(key, idx);
                                    accum.push([wx, wy, wz, 1.0]);
                                    idx
                                }
                            };
                            vert_remap.push(idx);
                        }
                        offset += rl;
                    }
                }

                // Compute centroids
                let centroids: Vec<[f32; 3]> = accum.iter().map(|a| [
                    (a[0] / a[3]) as f32,
                    (a[1] / a[3]) as f32,
                    (a[2] / a[3]) as f32,
                ]).collect();

                // Build deduplicated indexed mesh from centroids
                let mut vertex_map: AHashMap<u32, u32> = AHashMap::new();
                let mut remap_offset = 0usize;

                for frag in &mesh_frags {
                    let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                        vec![frag.z.len()]
                    } else {
                        frag.ring_lengths.iter().map(|&r| r as usize).collect()
                    };
                    for &rl in &ring_lengths {
                        let nv = if rl >= 4 { 3 } else { rl };
                        if nv == 3 && remap_offset + 2 < vert_remap.len() {
                            let c0 = vert_remap[remap_offset];
                            let c1 = vert_remap[remap_offset + 1];
                            let c2 = vert_remap[remap_offset + 2];

                            if c0 != c1 && c1 != c2 && c0 != c2 {
                                let mut tri = [0u32; 3];
                                for (ti, &ci) in [c0, c1, c2].iter().enumerate() {
                                    let idx = match vertex_map.get(&ci) {
                                        Some(&idx) => idx,
                                        None => {
                                            let idx = (positions.len() / 3) as u32;
                                            vertex_map.insert(ci, idx);
                                            let c = &centroids[ci as usize];
                                            positions.extend_from_slice(c);
                                            // Update bbox from centroids
                                            if c[0] as f64 > bb_max[0] || is_finest { /* updated below */ }
                                            idx
                                        }
                                    };
                                    tri[ti] = idx;
                                }
                                if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                                    indices.extend_from_slice(&tri);
                                }
                            }
                        }
                        remap_offset += nv;
                    }
                }
            } else {
                // --- Exact float32 vertex dedup (finest LOD) ---
                let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();

                for frag in &mesh_frags {
                    let n_verts = frag.z.len();
                    let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                        vec![n_verts]
                    } else {
                        frag.ring_lengths.iter().map(|&r| r as usize).collect()
                    };

                    let mut offset = 0usize;
                    for &rl in &ring_lengths {
                        let nv = if rl >= 4 { 3 } else { rl };
                        if nv < 3 || offset + 2 >= n_verts {
                            offset += rl;
                            continue;
                        }

                        let mut tri = [0u32; 3];
                        for vi_off in 0..3 {
                            let vi = offset + vi_off;
                            let wx = xmin + frag.xy[vi * 2] * dx;
                            let wy = ymin + frag.xy[vi * 2 + 1] * dy;
                            let wz = zmin + frag.z[vi] * dz;

                            let wx_f32 = wx as f32;
                            let wy_f32 = wy as f32;
                            let wz_f32 = wz as f32;

                            let key = (
                                wx_f32.to_bits() as i64,
                                wy_f32.to_bits() as i64,
                                wz_f32.to_bits() as i64,
                            );

                            let idx = match vertex_map.get(&key) {
                                Some(&idx) => idx,
                                None => {
                                    let idx = vertex_map.len() as u32;
                                    vertex_map.insert(key, idx);
                                    positions.push(wx_f32);
                                    positions.push(wy_f32);
                                    positions.push(wz_f32);
                                    idx
                                }
                            };
                            tri[vi_off] = idx;
                        }

                        if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                            indices.extend_from_slice(&tri);
                        }
                        offset += rl;
                    }
                }
            }

            // Update finest-LOD bbox from positions
            if is_finest {
                let n_verts = positions.len() / 3;
                for i in 0..n_verts {
                    let wx = positions[i * 3] as f64;
                    let wy = positions[i * 3 + 1] as f64;
                    let wz = positions[i * 3 + 2] as f64;
                    if wx < bb_min[0] { bb_min[0] = wx; }
                    if wy < bb_min[1] { bb_min[1] = wy; }
                    if wz < bb_min[2] { bb_min[2] = wz; }
                    if wx > bb_max[0] { bb_max[0] = wx; }
                    if wy > bb_max[1] { bb_max[1] = wy; }
                    if wz > bb_max[2] { bb_max[2] = wz; }
                }
            }

            if !positions.is_empty() && !indices.is_empty() {
                let pos_bytes: Vec<u8> = positions.iter().flat_map(|f| f.to_le_bytes()).collect();
                let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
                encoder_mjb::write_bytes_field(&mut feat_buf, encoder_mjb::FEAT_MESH_POSITIONS, &pos_bytes);
                encoder_mjb::write_bytes_field(&mut feat_buf, encoder_mjb::FEAT_MESH_INDICES, &idx_bytes);
            }

            features_encoded.push(feat_buf);
        }

        // --- Encode non-mesh geometry (point, line, polygon) ---
        for frag in &other_frags {
            let n_verts = frag.z.len();
            if n_verts == 0 { continue; }

            // Compute world coordinates for bbox
            let mut local_bb_min = [f64::INFINITY; 3];
            let mut local_bb_max = [f64::NEG_INFINITY; 3];
            for i in 0..n_verts {
                let wx = xmin + frag.xy[i * 2] * dx;
                let wy = ymin + frag.xy[i * 2 + 1] * dy;
                let wz = zmin + frag.z[i] * dz;
                if wx < local_bb_min[0] { local_bb_min[0] = wx; }
                if wy < local_bb_min[1] { local_bb_min[1] = wy; }
                if wz < local_bb_min[2] { local_bb_min[2] = wz; }
                if wx > local_bb_max[0] { local_bb_max[0] = wx; }
                if wy > local_bb_max[1] { local_bb_max[1] = wy; }
                if wz > local_bb_max[2] { local_bb_max[2] = wz; }
            }

            if is_finest {
                for d in 0..3 {
                    if local_bb_min[d] < bb_min[d] { bb_min[d] = local_bb_min[d]; }
                    if local_bb_max[d] > bb_max[d] { bb_max[d] = local_bb_max[d]; }
                }
            }

            // Encode in bbox-local integer space
            let feat_dx = if local_bb_max[0] != local_bb_min[0] { local_bb_max[0] - local_bb_min[0] } else { 1.0 };
            let feat_dy = if local_bb_max[1] != local_bb_min[1] { local_bb_max[1] - local_bb_min[1] } else { 1.0 };
            let feat_dz = if local_bb_max[2] != local_bb_min[2] { local_bb_max[2] - local_bb_min[2] } else { 1.0 };

            let mut new_xy: Vec<i64> = Vec::with_capacity(n_verts * 2);
            let mut new_z: Vec<i64> = Vec::with_capacity(n_verts);
            for i in 0..n_verts {
                let wx = xmin + frag.xy[i * 2] * dx;
                let wy = ymin + frag.xy[i * 2 + 1] * dy;
                let wz = zmin + frag.z[i] * dz;
                let lx = (wx - local_bb_min[0]) / feat_dx;
                let ly = (wy - local_bb_min[1]) / feat_dy;
                let lz = (wz - local_bb_min[2]) / feat_dz;
                new_xy.push(tile_transform::round_half_to_even(lx * extent as f64));
                new_xy.push(tile_transform::round_half_to_even(ly * extent as f64));
                new_z.push(tile_transform::round_half_to_even(lz * extent_z as f64));
            }

            let mut feat_buf = Vec::new();
            encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_ID, feat_idx);
            feat_idx += 1;
            encoder_mjb::write_varint_field(&mut feat_buf, encoder_mjb::FEAT_TYPE, frag.geom_type as u64);

            // Tags only on first non-mesh feature if no mesh features exist
            if features_encoded.is_empty() {
                if let Some(tag_list) = tags {
                    let mut tag_indices_vec: Vec<u32> = Vec::new();
                    for (key, val) in tag_list {
                        let ki = *key_indices.entry(key.clone()).or_insert_with(|| {
                            let idx = keys_list.len() as u32;
                            keys_list.push(key.clone());
                            idx
                        });
                        tag_indices_vec.push(ki);

                        let v_repr = match val {
                            TagValue::Str(s) => format!("'{}'", s),
                            TagValue::Int(i) => i.to_string(),
                            TagValue::Float(f) => format!("{:?}", f),
                            TagValue::Bool(b) => b.to_string(),
                        };
                        let vi = *value_indices.entry(v_repr).or_insert_with(|| {
                            let idx = values_encoded.len() as u32;
                            values_encoded.push(encode_tag_value(val));
                            idx
                        });
                        tag_indices_vec.push(vi);
                    }
                    encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_TAGS, &tag_indices_vec);
                }
            }

            let gt = frag.geom_type;
            if gt == 1 {
                let geom = encoder_mjb::encode_point_geometry(&new_xy);
                encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
                let z_enc = encoder_mjb::encode_z(&new_z);
                encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
            } else if gt == 2 {
                let geom = encoder_mjb::encode_line_geometry(&new_xy);
                encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
                let z_enc = encoder_mjb::encode_z(&new_z);
                encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
            } else if gt == 3 {
                let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                    vec![new_xy.len() / 2]
                } else {
                    frag.ring_lengths.iter().map(|&r| r as usize).collect()
                };
                let geom = encoder_mjb::encode_polygon_geometry(&new_xy, &rls);
                encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
                let z_enc = encoder_mjb::encode_z(&new_z);
                encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
            } else {
                let geom = encoder_mjb::encode_line_geometry(&new_xy);
                encoder_mjb::encode_packed_uint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY, &geom);
                let z_enc = encoder_mjb::encode_z(&new_z);
                encoder_mjb::encode_packed_sint32(&mut feat_buf, encoder_mjb::FEAT_GEOMETRY_Z, &z_enc);
            }

            features_encoded.push(feat_buf);
        }

        // --- Encode layer ---
        let mut layer_buf = Vec::new();
        encoder_mjb::write_varint_field(&mut layer_buf, encoder_mjb::LAYER_VERSION, 3);
        encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_NAME, layer_name.as_bytes());
        for feat_buf in &features_encoded {
            encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_FEATURES, feat_buf);
        }
        for key in &keys_list {
            encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_KEYS, key.as_bytes());
        }
        for val_buf in &values_encoded {
            encoder_mjb::write_bytes_field(&mut layer_buf, encoder_mjb::LAYER_VALUES, val_buf);
        }
        encoder_mjb::write_varint_field(&mut layer_buf, encoder_mjb::LAYER_EXTENT, extent as u64);
        encoder_mjb::write_varint_field(&mut layer_buf, encoder_mjb::LAYER_EXTENT_Z, extent_z as u64);

        layer_bufs.push(layer_buf);
    }

    // --- Assemble tile with all layers ---
    let mut tile_buf = Vec::new();
    for layer_buf in &layer_bufs {
        encoder_mjb::write_bytes_field(&mut tile_buf, encoder_mjb::TILE_LAYERS, layer_buf);
    }

    let bbox = [bb_min[0], bb_min[1], bb_min[2], bb_max[0], bb_max[1], bb_max[2]];
    (tile_buf, bbox)
}

// ---------------------------------------------------------------------------
// GLB tile encoding from fragments (pure Rust, no GIL)
// ---------------------------------------------------------------------------

/// Default base grid resolution at zoom 0 (matches simplify.rs).
const DEFAULT_BASE_CELLS: u32 = 10;

/// Encode a single tile's fragments as GLB binary with vertex-clustering simplification.
///
/// For TIN/PolyhedralSurface at non-leaf zoom levels, vertices are snapped to a
/// world-aligned grid (BASE_CELLS * 2^zoom cells per axis) to produce LOD.
/// At max_zoom, exact vertex dedup is used instead.
fn encode_glb_tile_from_fragments(
    frags: &[Fragment],
    tags_registry: &HashMap<u32, Vec<(String, TagValue)>>,
    world_bounds: &(f64, f64, f64, f64, f64, f64),
    tz: u32,
    max_zoom: u32,
    base_cells: u32,
) -> Vec<u8> {
    let (xmin, ymin, zmin, xmax, ymax, zmax) = *world_bounds;
    let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
    let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
    let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

    // Simplification grid parameters
    let do_simplify = tz < max_zoom;
    let cell_size = if do_simplify {
        let cells = base_cells * (1u32 << tz);
        [
            dx / cells as f64,
            dy / cells as f64,
            dz / cells as f64,
        ]
    } else {
        [0.0; 3]
    };

    let mut features: Vec<GlbFeature> = Vec::new();

    for frag in frags {
        let n_verts = frag.z.len();
        if n_verts == 0 {
            continue;
        }

        // Build tags as JSON extras
        let extras = tags_registry.get(&frag.feature_id).map(|tags| {
            let obj: serde_json::Map<String, serde_json::Value> = tags
                .iter()
                .map(|(k, v)| {
                    let jv = match v {
                        TagValue::Str(s) => serde_json::json!(s),
                        TagValue::Int(i) => serde_json::json!(i),
                        TagValue::Float(f) => serde_json::json!(f),
                        TagValue::Bool(b) => serde_json::json!(b),
                    };
                    (k.clone(), jv)
                })
                .collect();
            serde_json::Value::Object(obj)
        });

        match frag.geom_type {
            5 | 4 => {
                // TIN or PolyhedralSurface — per-face vertices (matching Python pipeline)
                //
                // Python pipeline: decimate_tin() → _convert_tin() writes 3 separate
                // vertices per face with sequential indices [0,1,2, 3,4,5, ...].
                // No vertex deduplication at any zoom level.
                let rls: Vec<usize> = if frag.ring_lengths.is_empty() {
                    vec![n_verts]
                } else {
                    frag.ring_lengths.iter().map(|&r| r as usize).collect()
                };
                let n_faces = rls.len();

                // Python: `if target_ratio >= 1.0 or len(coordinates) <= 4: return`
                // Skip simplification at max_zoom or for tiny features
                let do_simplify_this = do_simplify && n_faces > 4;

                if do_simplify_this {
                    // --- Vertex clustering (matches Python decimate_tin) ---

                    // Phase 1: quantize all vertices and accumulate centroids
                    let mut cell_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
                    let mut accum: Vec<[f64; 4]> = Vec::new(); // [sum_x, sum_y, sum_z, count]
                    let mut vert_remap: Vec<u32> = Vec::with_capacity(n_verts);
                    let mut offset = 0usize;

                    for &rl in &rls {
                        // Only process first 3 vertices per ring — skip closing
                        // duplicate vertex. Matches Python: `ring[:3]`
                        let nv_clust = if rl >= 4 { 3 } else { rl };
                        for vi_off in 0..nv_clust {
                            let vi = offset + vi_off;
                            if vi >= n_verts {
                                break;
                            }
                            let wx = xmin + frag.xy[vi * 2] * dx;
                            let wy = ymin + frag.xy[vi * 2 + 1] * dy;
                            let wz = zmin + frag.z[vi] * dz;

                            let qx = ((wx - xmin) / cell_size[0]).floor() as i64;
                            let qy = ((wy - ymin) / cell_size[1]).floor() as i64;
                            let qz = ((wz - zmin) / cell_size[2]).floor() as i64;
                            let key = (qx, qy, qz);

                            let idx = match cell_map.get(&key) {
                                Some(&idx) => {
                                    accum[idx as usize][0] += wx;
                                    accum[idx as usize][1] += wy;
                                    accum[idx as usize][2] += wz;
                                    accum[idx as usize][3] += 1.0;
                                    idx
                                }
                                None => {
                                    let idx = cell_map.len() as u32;
                                    cell_map.insert(key, idx);
                                    accum.push([wx, wy, wz, 1.0]);
                                    idx
                                }
                            };
                            vert_remap.push(idx);
                        }
                        offset += rl; // advance by full ring length in fragment data
                    }

                    // Phase 2: compute centroids
                    let centroids: Vec<[f32; 3]> = accum
                        .iter()
                        .map(|a| {
                            [
                                (a[0] / a[3]) as f32,
                                (a[1] / a[3]) as f32,
                                (a[2] / a[3]) as f32,
                            ]
                        })
                        .collect();

                    // Phase 3: build per-face vertices (matching Python _convert_tin)
                    // Each non-degenerate face writes 3 separate vertex positions.
                    let mut positions: Vec<f32> = Vec::new();
                    let mut indices: Vec<u32> = Vec::new();
                    let mut remap_offset = 0usize;
                    let mut face_idx = 0u32;

                    for &rl in &rls {
                        let nv = if rl >= 4 { 3 } else { rl };
                        if nv == 3 && remap_offset + 2 < vert_remap.len() {
                            let i0 = vert_remap[remap_offset];
                            let i1 = vert_remap[remap_offset + 1];
                            let i2 = vert_remap[remap_offset + 2];

                            if i0 != i1 && i1 != i2 && i0 != i2 {
                                // Write 3 separate centroid positions per face
                                positions.extend_from_slice(&centroids[i0 as usize]);
                                positions.extend_from_slice(&centroids[i1 as usize]);
                                positions.extend_from_slice(&centroids[i2 as usize]);
                                let base = face_idx * 3;
                                indices.extend_from_slice(&[base, base + 1, base + 2]);
                                face_idx += 1;
                            }
                        }
                        remap_offset += nv; // advance by entries pushed (3), not ring length (4)
                    }

                    if positions.is_empty() {
                        // All faces degenerate — keep first 4 original faces
                        // (matches Python: `return coordinates[:4]`)
                        let max_faces = n_faces.min(4);
                        let mut off = 0usize;
                        let mut fi = 0u32;
                        for (face_i, &rl) in rls.iter().enumerate() {
                            if face_i >= max_faces { break; }
                            let nv = if rl >= 4 { 3 } else { rl };
                            if nv == 3 && off + 2 < n_verts {
                                for vi_off in 0..3 {
                                    let vi = off + vi_off;
                                    let wx = xmin + frag.xy[vi * 2] * dx;
                                    let wy = ymin + frag.xy[vi * 2 + 1] * dy;
                                    let wz = zmin + frag.z[vi] * dz;
                                    positions.push(wx as f32);
                                    positions.push(wy as f32);
                                    positions.push(wz as f32);
                                }
                                let base = fi * 3;
                                indices.extend_from_slice(&[base, base + 1, base + 2]);
                                fi += 1;
                            }
                            off += rl;
                        }
                    }

                    if !positions.is_empty() && !indices.is_empty() {
                        features.push(GlbFeature {
                            positions,
                            indices,
                            mode: encoder_glb::MODE_TRIANGLES,
                            extras,
                        });
                    }
                } else {
                    // No simplification — write per-face vertices directly
                    // (matches Python _convert_tin: 3 verts per face, sequential indices)
                    let mut positions: Vec<f32> = Vec::new();
                    let mut indices: Vec<u32> = Vec::new();
                    let mut offset = 0usize;
                    let mut face_idx = 0u32;

                    for &rl in &rls {
                        let nv = if rl >= 4 { 3 } else { rl };
                        if nv == 3 && offset + 2 < n_verts {
                            for vi_off in 0..3 {
                                let vi = offset + vi_off;
                                let wx = xmin + frag.xy[vi * 2] * dx;
                                let wy = ymin + frag.xy[vi * 2 + 1] * dy;
                                let wz = zmin + frag.z[vi] * dz;
                                positions.push(wx as f32);
                                positions.push(wy as f32);
                                positions.push(wz as f32);
                            }
                            let base = face_idx * 3;
                            indices.extend_from_slice(&[base, base + 1, base + 2]);
                            face_idx += 1;
                        }
                        offset += rl;
                    }

                    if !positions.is_empty() && !indices.is_empty() {
                        features.push(GlbFeature {
                            positions,
                            indices,
                            mode: encoder_glb::MODE_TRIANGLES,
                            extras,
                        });
                    }
                }
            }
            2 => {
                // LineString — line segments
                let mut positions: Vec<f32> = Vec::new();
                let mut indices: Vec<u32> = Vec::new();

                for i in 0..n_verts {
                    let wx = xmin + frag.xy[i * 2] * dx;
                    let wy = ymin + frag.xy[i * 2 + 1] * dy;
                    let wz = zmin + frag.z[i] * dz;
                    positions.push(wx as f32);
                    positions.push(wy as f32);
                    positions.push(wz as f32);
                }

                for i in 0..n_verts.saturating_sub(1) {
                    indices.push(i as u32);
                    indices.push((i + 1) as u32);
                }

                if !positions.is_empty() && !indices.is_empty() {
                    features.push(GlbFeature {
                        positions,
                        indices,
                        mode: encoder_glb::MODE_LINES,
                        extras,
                    });
                }
            }
            1 => {
                // Point
                let mut positions: Vec<f32> = Vec::new();
                for i in 0..n_verts {
                    let wx = xmin + frag.xy[i * 2] * dx;
                    let wy = ymin + frag.xy[i * 2 + 1] * dy;
                    let wz = zmin + frag.z[i] * dz;
                    positions.push(wx as f32);
                    positions.push(wy as f32);
                    positions.push(wz as f32);
                }

                if !positions.is_empty() {
                    features.push(GlbFeature {
                        positions,
                        indices: vec![],
                        mode: encoder_glb::MODE_POINTS,
                        extras,
                    });
                }
            }
            _ => {}
        }
    }

    encoder_glb::encode_glb(&features)
}

// ---------------------------------------------------------------------------
// PyO3 helpers: extract features and tags from Python dicts
// ---------------------------------------------------------------------------

fn extract_tags(feat: &Bound<'_, PyDict>) -> PyResult<Vec<(String, TagValue)>> {
    let tags_obj = feat.get_item("tags")?;
    let mut result = Vec::new();

    if let Some(tags) = tags_obj {
        if let Ok(tags_dict) = tags.downcast::<PyDict>() {
            for (k, v) in tags_dict.iter() {
                if v.is_none() {
                    continue;
                }
                let key: String = k.extract()?;

                // Check bool before int (bool is subclass of int in Python)
                if v.is_instance_of::<PyBool>() {
                    let b: bool = v.extract()?;
                    result.push((key, TagValue::Bool(b)));
                } else if v.is_instance_of::<PyString>() {
                    let s: String = v.extract()?;
                    result.push((key, TagValue::Str(s)));
                } else if v.is_instance_of::<PyFloat>() {
                    let f: f64 = v.extract()?;
                    result.push((key, TagValue::Float(f)));
                } else if v.is_instance_of::<PyInt>() {
                    let i: i64 = v.extract()?;
                    result.push((key, TagValue::Int(i)));
                }
            }
        }
    }

    Ok(result)
}

fn extract_clip_feature(feat: &Bound<'_, PyDict>) -> PyResult<ClipFeature> {
    let xy: Vec<f64> = feat.get_item("geometry")?.unwrap().extract()?;
    let z: Vec<f64> = feat.get_item("geometry_z")?.unwrap().extract()?;
    let geom_type: u8 = feat.get_item("type")?.unwrap().extract()?;

    let ring_lengths: Vec<u32> = if let Some(rl) = feat.get_item("ring_lengths")? {
        if rl.is_none() {
            vec![]
        } else {
            rl.extract()?
        }
    } else {
        vec![]
    };

    let bbox = BBox3D {
        min_x: feat.get_item("minX")?.unwrap().extract()?,
        min_y: feat.get_item("minY")?.unwrap().extract()?,
        min_z: feat.get_item("minZ")?.unwrap().extract()?,
        max_x: feat.get_item("maxX")?.unwrap().extract()?,
        max_y: feat.get_item("maxY")?.unwrap().extract()?,
        max_z: feat.get_item("maxZ")?.unwrap().extract()?,
    };

    Ok(ClipFeature { xy, z, ring_lengths, geom_type, bbox })
}

// ---------------------------------------------------------------------------
// StreamingTileGenerator — PyO3 class
// ---------------------------------------------------------------------------

/// Streaming 3D tile generator.
///
/// Processes features one at a time via ``add_feature()``, clipping through
/// an octree and writing binary fragments to a temp file. Call
/// ``generate_mjb()`` to read fragments back, transform to tile-local
/// integers, encode to protobuf, and write ``.mjb`` tiles in parallel.
///
/// Memory: O(1 feature) during ingestion, O(fragments) during encoding.
#[pyclass]
pub struct StreamingTileGenerator {
    min_zoom: u32,
    max_zoom: u32,
    extent: u32,
    extent_z: u32,
    buffer: f64,
    base_cells: u32,
    feature_count: u32,
    tags_registry: HashMap<u32, Vec<(String, TagValue)>>,
    fragment_writer: Option<FragmentWriter>,
    fragment_path: PathBuf,
    tiles_written: u32,
}

#[pymethods]
impl StreamingTileGenerator {
    /// Create a new streaming tile generator.
    ///
    /// Args:
    ///   min_zoom: Minimum zoom level to generate tiles for.
    ///   max_zoom: Maximum zoom level to generate tiles for.
    ///   extent: XY tile extent in integer coordinates (default 4096).
    ///   extent_z: Z tile extent in integer coordinates (default 4096).
    ///   buffer: Tile buffer in normalized space (fraction of tile size).
    ///   base_cells: Grid resolution at zoom 0 for vertex clustering (default 10).
    ///     Higher values preserve finer detail at lower zoom levels.
    ///     Solid regions: 10. Thin branching structures (neurons): 50-200.
    #[new]
    #[pyo3(signature = (min_zoom=0, max_zoom=4, extent=4096, extent_z=4096, buffer=0.0, base_cells=10))]
    fn new(
        min_zoom: u32,
        max_zoom: u32,
        extent: u32,
        extent_z: u32,
        buffer: f64,
        base_cells: u32,
    ) -> PyResult<Self> {
        let gen_id = GENERATOR_ID.fetch_add(1, Ordering::Relaxed);
        let frag_path = std::env::temp_dir()
            .join(format!("microjson_frags_{}_{}.mjf", std::process::id(), gen_id));

        let writer = FragmentWriter::new(&frag_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(Self {
            min_zoom,
            max_zoom,
            extent,
            extent_z,
            buffer,
            base_cells: if base_cells == 0 { DEFAULT_BASE_CELLS } else { base_cells },
            feature_count: 0,
            tags_registry: HashMap::new(),
            fragment_writer: Some(writer),
            fragment_path: frag_path,
            tiles_written: 0,
        })
    }

    /// Add a single intermediate feature (already projected to [0,1]³).
    ///
    /// The feature dict must have keys: geometry, geometry_z, type,
    /// minX/minY/minZ/maxX/maxY/maxZ, and optionally tags, ring_lengths.
    ///
    /// Returns the assigned feature ID.
    fn add_feature(&mut self, feat: &Bound<'_, PyDict>) -> PyResult<u32> {
        let fid = self.feature_count;
        self.feature_count += 1;

        // Store tags in registry
        let tags = extract_tags(feat)?;
        self.tags_registry.insert(fid, tags);

        // Extract geometry to ClipFeature
        let clip_feat = extract_clip_feature(feat)?;

        // Clip through octree at all zoom levels
        let fragments = octree_clip(&clip_feat, self.min_zoom, self.max_zoom, self.buffer);

        // Write fragments to disk
        let writer = self.fragment_writer.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot add features after generate_mjb()"))?;

        for ((tz, tx, ty, td), cf) in fragments {
            let frag = Fragment {
                feature_id: fid,
                tile_z: tz,
                tile_x: tx,
                tile_y: ty,
                tile_d: td,
                geom_type: cf.geom_type,
                xy: cf.xy,
                z: cf.z,
                ring_lengths: cf.ring_lengths,
            };
            writer.write(&frag)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        Ok(fid)
    }

    /// Generate .mjb tiles from accumulated fragments.
    ///
    /// Reads all fragments from the temp file, groups by tile key,
    /// transforms geometry to tile-local integers, encodes to protobuf,
    /// and writes tiles to ``output_dir/{z}/{x}/{y}/{d}.mjb``.
    ///
    /// Uses rayon for parallel tile encoding (GIL released).
    ///
    /// Returns the number of tiles written.
    #[pyo3(signature = (output_dir, layer_name="default"))]
    fn generate_mjb(&mut self, py: Python<'_>, output_dir: &str, layer_name: &str) -> PyResult<u32> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by tile key
        let mut reader = FragmentReader::new(&self.fragment_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Collect for rayon
        let tiles: Vec<((u32, u32, u32, u32), Vec<Fragment>)> = groups.into_iter().collect();
        let tags_ref = &self.tags_registry;
        let extent = self.extent;
        let extent_z = self.extent_z;
        let out_dir = PathBuf::from(output_dir);
        let layer = layer_name.to_string();

        // Parallel tile encoding (GIL released)
        let count = py.allow_threads(|| {
            tiles.par_iter()
                .map(|((tz, tx, ty, td), frags)| {
                    let data = encode_tile_from_fragments(
                        frags, tags_ref,
                        *tz, *tx, *ty, *td,
                        extent, extent_z, &layer,
                    );
                    let tile_path = out_dir
                        .join(tz.to_string())
                        .join(tx.to_string())
                        .join(ty.to_string())
                        .join(format!("{}.mjb", td));
                    if let Some(parent) = tile_path.parent() {
                        std::fs::create_dir_all(parent).ok();
                    }
                    std::fs::write(&tile_path, data).ok();
                    1u32
                })
                .sum::<u32>()
        });

        self.tiles_written = count;
        Ok(count)
    }

    /// Number of tiles written by the last generate_mjb() call.
    fn tile_count(&self) -> u32 {
        self.tiles_written
    }

    /// Number of features added so far.
    fn feature_count_val(&self) -> u32 {
        self.feature_count
    }

    /// Generate .glb tiles for OGC 3D Tiles from accumulated fragments.
    ///
    /// Reads all fragments from the temp file, groups by tile key,
    /// unprojections to world coordinates, applies vertex-clustering
    /// simplification for non-leaf tiles, encodes to GLB binary,
    /// and writes tiles to ``output_dir/{z}/{x}/{y}/{d}.glb``.
    ///
    /// Also writes ``tileset.json`` to ``output_dir/tileset.json``.
    ///
    /// Uses rayon for parallel tile encoding (GIL released).
    ///
    /// Returns the number of tiles written.
    #[pyo3(signature = (output_dir, world_bounds, layer_name="default"))]
    fn generate_3dtiles(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64, f64, f64),
        #[allow(unused)]
        layer_name: &str,
    ) -> PyResult<u32> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by tile key
        let mut reader = FragmentReader::new(&self.fragment_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let groups = reader.read_all_grouped()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        // Collect for rayon
        let tiles: Vec<((u32, u32, u32, u32), Vec<Fragment>)> = groups.into_iter().collect();
        let tags_ref = &self.tags_registry;
        let wb = world_bounds;
        let max_zoom = self.max_zoom;
        let base_cells = self.base_cells;
        let out_dir = PathBuf::from(output_dir);

        // Parallel GLB encoding (GIL released)
        let count = py.allow_threads(|| {
            tiles.par_iter()
                .map(|((tz, tx, ty, td), frags)| {
                    let data = encode_glb_tile_from_fragments(
                        frags, tags_ref, &wb, *tz, max_zoom, base_cells,
                    );
                    let tile_path = out_dir
                        .join(tz.to_string())
                        .join(tx.to_string())
                        .join(ty.to_string())
                        .join(format!("{}.glb", td));
                    if let Some(parent) = tile_path.parent() {
                        std::fs::create_dir_all(parent).ok();
                    }
                    std::fs::write(&tile_path, data).ok();
                    1u32
                })
                .sum::<u32>()
        });

        // Write tileset.json
        let tile_keys: Vec<(u32, u32, u32, u32)> =
            tiles.iter().map(|(k, _)| *k).collect();
        let tileset = tileset_json::generate_tileset_json(
            &tile_keys, &wb, self.min_zoom, self.max_zoom,
        );
        let tileset_path = out_dir.join("tileset.json");
        let tileset_str = serde_json::to_string_pretty(&tileset)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        std::fs::write(&tileset_path, tileset_str)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        self.tiles_written = count;
        Ok(count)
    }

    /// Add an OBJ file directly — parse, project, clip, and write fragments in Rust.
    ///
    /// This avoids creating Python intermediate objects entirely.
    /// Memory: O(1 mesh) — the OBJ vertices/faces are freed after clipping.
    ///
    /// Args:
    ///   path: Path to the OBJ file.
    ///   bounds: World bounding box (xmin, ymin, zmin, xmax, ymax, zmax).
    ///   tags: Property dict to attach to the feature.
    ///
    /// Returns the assigned feature ID.
    fn add_obj_file(
        &mut self,
        path: &str,
        bounds: (f64, f64, f64, f64, f64, f64),
        tags: &Bound<'_, PyDict>,
    ) -> PyResult<u32> {
        let (xmin, ymin, zmin, xmax, ymax, zmax) = bounds;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

        // Parse OBJ in Rust
        let (vertices, faces) = obj_parser::parse_obj(path)?;

        // Assign feature ID and store tags
        let fid = self.feature_count;
        self.feature_count += 1;

        let mut tag_vec: Vec<(String, TagValue)> = Vec::new();
        for (k, v) in tags.iter() {
            if v.is_none() {
                continue;
            }
            let key: String = k.extract()?;
            if v.is_instance_of::<PyBool>() {
                tag_vec.push((key, TagValue::Bool(v.extract()?)));
            } else if v.is_instance_of::<PyString>() {
                tag_vec.push((key, TagValue::Str(v.extract()?)));
            } else if v.is_instance_of::<PyFloat>() {
                tag_vec.push((key, TagValue::Float(v.extract()?)));
            } else if v.is_instance_of::<PyInt>() {
                tag_vec.push((key, TagValue::Int(v.extract()?)));
            }
        }
        self.tags_registry.insert(fid, tag_vec);

        // Build TIN geometry: each triangle → 4 vertices (3 + closing)
        // Project to [0,1]³ in the same pass
        let n_faces = faces.len();
        let n_verts_total = n_faces * 4;
        let mut xy = Vec::with_capacity(n_verts_total * 2);
        let mut z = Vec::with_capacity(n_verts_total);
        let ring_lengths: Vec<u32> = vec![4; n_faces];

        let mut bb = BBox3D::empty();

        for face in &faces {
            for &vi in &[face[0], face[1], face[2], face[0]] {
                let v = &vertices[vi as usize];
                let px = (v[0] - xmin) / dx;
                let py = (v[1] - ymin) / dy;
                let pz = (v[2] - zmin) / dz;
                xy.push(px);
                xy.push(py);
                z.push(pz);
                bb.expand(px, py, pz);
            }
        }

        // vertices and faces are freed here (Rust ownership)
        drop(vertices);
        drop(faces);

        let clip_feat = ClipFeature {
            xy,
            z,
            ring_lengths,
            geom_type: TIN,
            bbox: bb,
        };

        // Clip through octree
        let fragments = octree_clip(&clip_feat, self.min_zoom, self.max_zoom, self.buffer);
        drop(clip_feat);

        // Write fragments to disk
        let writer = self.fragment_writer.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot add features after generate"))?;

        for ((tz, tx, ty, td), cf) in fragments {
            let frag = Fragment {
                feature_id: fid,
                tile_z: tz,
                tile_x: tx,
                tile_y: ty,
                tile_d: td,
                geom_type: cf.geom_type,
                xy: cf.xy,
                z: cf.z,
                ring_lengths: cf.ring_lengths,
            };
            writer.write(&frag)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        Ok(fid)
    }

    /// Add multiple OBJ files in parallel using rayon.
    ///
    /// Tags are extracted from Python dicts while holding the GIL,
    /// then the GIL is released for parallel Rust processing.
    /// Each rayon thread parses an OBJ, projects to [0,1]³, clips through
    /// the octree, and writes fragments to disk under a mutex.
    ///
    /// Memory: O(N_cores * 1 mesh) — only in-flight meshes are in memory.
    ///
    /// Args:
    ///   paths: List of OBJ file paths.
    ///   bounds: World bounding box (xmin, ymin, zmin, xmax, ymax, zmax).
    ///   tags_list: List of property dicts (one per file).
    ///
    /// Returns the list of assigned feature IDs.
    fn add_obj_files(
        &mut self,
        py: Python<'_>,
        paths: Vec<String>,
        bounds: (f64, f64, f64, f64, f64, f64),
        tags_list: &Bound<'_, PyList>,
    ) -> PyResult<Vec<u32>> {
        let n_files = paths.len();
        if n_files != tags_list.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "paths and tags_list must have the same length"));
        }

        let (xmin, ymin, zmin, xmax, ymax, zmax) = bounds;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

        // Pre-assign sequential feature IDs
        let base_fid = self.feature_count;
        self.feature_count += n_files as u32;
        let fids: Vec<u32> = (base_fid..base_fid + n_files as u32).collect();

        // Extract all tags from Python dicts (GIL held)
        for (i, item) in tags_list.iter().enumerate() {
            let tags_dict: &Bound<'_, PyDict> = item.downcast()?;
            let mut tag_vec: Vec<(String, TagValue)> = Vec::new();
            for (k, v) in tags_dict.iter() {
                if v.is_none() { continue; }
                let key: String = k.extract()?;
                if v.is_instance_of::<PyBool>() {
                    tag_vec.push((key, TagValue::Bool(v.extract()?)));
                } else if v.is_instance_of::<PyString>() {
                    tag_vec.push((key, TagValue::Str(v.extract()?)));
                } else if v.is_instance_of::<PyFloat>() {
                    tag_vec.push((key, TagValue::Float(v.extract()?)));
                } else if v.is_instance_of::<PyInt>() {
                    tag_vec.push((key, TagValue::Int(v.extract()?)));
                }
            }
            self.tags_registry.insert(fids[i], tag_vec);
        }

        // Take the writer, wrap in Mutex for thread-safe writing
        let writer = self.fragment_writer.take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot add features after generate"))?;
        let writer_mutex = Mutex::new(writer);

        let min_zoom = self.min_zoom;
        let max_zoom = self.max_zoom;
        let buffer = self.buffer;

        // Release GIL — parallel parse + project + clip + write
        py.allow_threads(|| {
            paths.par_iter().enumerate().for_each(|(i, path)| {
                let fid = base_fid + i as u32;

                // Parse OBJ in Rust
                let (vertices, faces) = match obj_parser::parse_obj(path) {
                    Ok(vf) => vf,
                    Err(_) => return, // skip broken files
                };

                // Build TIN geometry: each triangle → 4 vertices (3 + closing)
                // Project to [0,1]³ in the same pass
                let n_faces = faces.len();
                let mut xy = Vec::with_capacity(n_faces * 4 * 2);
                let mut z_vec = Vec::with_capacity(n_faces * 4);
                let ring_lengths: Vec<u32> = vec![4; n_faces];
                let mut bb = BBox3D::empty();

                for face in &faces {
                    for &vi in &[face[0], face[1], face[2], face[0]] {
                        let v = &vertices[vi as usize];
                        let px = (v[0] - xmin) / dx;
                        let py_val = (v[1] - ymin) / dy;
                        let pz = (v[2] - zmin) / dz;
                        xy.push(px);
                        xy.push(py_val);
                        z_vec.push(pz);
                        bb.expand(px, py_val, pz);
                    }
                }
                drop(vertices);
                drop(faces);

                let clip_feat = ClipFeature {
                    xy, z: z_vec, ring_lengths,
                    geom_type: TIN,
                    bbox: bb,
                };

                // Clip through octree — produces all (tile_key, clipped) pairs
                let clip_results = octree_clip(&clip_feat, min_zoom, max_zoom, buffer);
                drop(clip_feat);

                // Write fragments to disk under mutex (serial per file)
                let mut w = writer_mutex.lock().unwrap();
                for ((tz, tx, ty, td), cf) in clip_results {
                    let frag = Fragment {
                        feature_id: fid,
                        tile_z: tz, tile_x: tx, tile_y: ty, tile_d: td,
                        geom_type: cf.geom_type,
                        xy: cf.xy, z: cf.z,
                        ring_lengths: cf.ring_lengths,
                    };
                    w.write(&frag).ok();
                }
                // mutex released here
            });
        });

        // Put writer back
        self.fragment_writer = Some(writer_mutex.into_inner().unwrap());

        Ok(fids)
    }

    /// Generate Neuroglancer precomputed legacy meshes from accumulated fragments.
    ///
    /// Unlike MJB/3D Tiles which are tile-centric (one tile → many features),
    /// Neuroglancer meshes are segment-centric (one mesh per feature ID).
    ///
    /// For each feature:
    /// 1. Collect all fragments at max_zoom (highest resolution)
    /// 2. Merge geometry, unproject [0,1]³ → world coordinates
    /// 3. Vertex-dedup via hash map
    /// 4. Write binary mesh + JSON manifest + info file
    ///
    /// Directory structure:
    ///   {output_dir}/info                    — JSON {"@type":"neuroglancer_legacy_mesh"}
    ///   {output_dir}/{segment_id}            — binary mesh data
    ///   {output_dir}/{segment_id}:0          — JSON fragment manifest
    ///   {output_dir}/segment_properties/info — segment properties from tags
    ///
    /// Returns the number of segments written.
    #[pyo3(signature = (output_dir, world_bounds))]
    fn generate_neuroglancer(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64, f64, f64),
    ) -> PyResult<u32> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by feature_id
        let mut reader = FragmentReader::new(&self.fragment_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let feature_groups = reader.read_all_grouped_by_feature()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let (xmin, ymin, zmin, xmax, ymax, zmax) = world_bounds;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

        let max_zoom = self.max_zoom;
        let out_dir = PathBuf::from(output_dir);
        let tags_ref = &self.tags_registry;

        // Process features in parallel
        let features: Vec<(u32, Vec<Fragment>)> = feature_groups.into_iter().collect();

        let count = py.allow_threads(|| {
            // Create output directory
            std::fs::create_dir_all(&out_dir).ok();

            // Write info file
            let info = serde_json::json!({
                "@type": "neuroglancer_legacy_mesh",
                "segment_properties": "segment_properties"
            });
            let info_str = serde_json::to_string_pretty(&info).unwrap_or_default();
            std::fs::write(out_dir.join("info"), info_str).ok();

            // Process each feature in parallel
            let segment_count: u32 = features.par_iter()
                .map(|(feature_id, frags)| {
                    // Filter to max_zoom fragments only (highest resolution)
                    let max_zoom_frags: Vec<&Fragment> = frags.iter()
                        .filter(|f| f.tile_z == max_zoom)
                        .collect();

                    if max_zoom_frags.is_empty() {
                        return 0u32;
                    }

                    // Merge geometry with vertex deduplication
                    let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
                    let mut positions: Vec<f32> = Vec::new();
                    let mut indices: Vec<u32> = Vec::new();

                    for frag in &max_zoom_frags {
                        let n_verts = frag.z.len();
                        let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                            vec![n_verts]
                        } else {
                            frag.ring_lengths.iter().map(|&r| r as usize).collect()
                        };

                        match frag.geom_type {
                            5 | 4 => {
                                // TIN / PolyhedralSurface
                                let mut offset = 0usize;
                                for &rl in &ring_lengths {
                                    let nv = if rl >= 4 { 3 } else { rl };
                                    if nv < 3 || offset + 2 >= n_verts {
                                        offset += rl;
                                        continue;
                                    }

                                    let mut tri = [0u32; 3];
                                    for vi_off in 0..3 {
                                        let vi = offset + vi_off;
                                        // Unproject to world coordinates
                                        let wx = xmin + frag.xy[vi * 2] * dx;
                                        let wy = ymin + frag.xy[vi * 2 + 1] * dy;
                                        let wz = zmin + frag.z[vi] * dz;

                                        // Quantize to float32 precision for dedup
                                        let wx_f32 = wx as f32;
                                        let wy_f32 = wy as f32;
                                        let wz_f32 = wz as f32;

                                        let key = (
                                            wx_f32.to_bits() as i64,
                                            wy_f32.to_bits() as i64,
                                            wz_f32.to_bits() as i64,
                                        );

                                        let idx = match vertex_map.get(&key) {
                                            Some(&idx) => idx,
                                            None => {
                                                let idx = vertex_map.len() as u32;
                                                vertex_map.insert(key, idx);
                                                positions.push(wx_f32);
                                                positions.push(wy_f32);
                                                positions.push(wz_f32);
                                                idx
                                            }
                                        };
                                        tri[vi_off] = idx;
                                    }

                                    // Skip degenerate triangles
                                    if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                                        indices.extend_from_slice(&tri);
                                    }

                                    offset += rl;
                                }
                            }
                            _ => {
                                // Skip non-triangle geometry for mesh output
                            }
                        }
                    }

                    if positions.is_empty() || indices.is_empty() {
                        return 0u32;
                    }

                    let num_vertices = positions.len() / 3;

                    // Encode as Neuroglancer legacy mesh binary
                    // Layout: uint32 num_vertices | float32[N*3] xyz | uint32[M] indices
                    let mut buf: Vec<u8> = Vec::with_capacity(
                        4 + positions.len() * 4 + indices.len() * 4
                    );
                    buf.extend_from_slice(&(num_vertices as u32).to_le_bytes());
                    for &p in &positions {
                        buf.extend_from_slice(&p.to_le_bytes());
                    }
                    for &i in &indices {
                        buf.extend_from_slice(&i.to_le_bytes());
                    }

                    // Write binary mesh file
                    let seg_path = out_dir.join(feature_id.to_string());
                    std::fs::write(&seg_path, &buf).ok();

                    // Write fragment manifest JSON
                    let manifest = format!(
                        "{{\"fragments\":[\"{}\"]}}",
                        feature_id
                    );
                    let manifest_path = out_dir.join(format!("{}:0", feature_id));
                    std::fs::write(&manifest_path, manifest).ok();

                    1u32
                })
                .sum::<u32>();

            // Write segment_properties/info from tags_registry
            let sp_dir = out_dir.join("segment_properties");
            std::fs::create_dir_all(&sp_dir).ok();

            let mut ids: Vec<String> = Vec::new();
            let mut all_keys: Vec<String> = Vec::new();
            let mut seen_keys: AHashMap<String, usize> = AHashMap::new();
            let mut columns: Vec<Vec<serde_json::Value>> = Vec::new();

            // Collect all feature IDs and property keys
            let mut sorted_features: Vec<&(u32, Vec<Fragment>)> = features.iter().collect();
            sorted_features.sort_by_key(|(fid, _)| *fid);

            for (fid, _) in &sorted_features {
                ids.push(fid.to_string());
                if let Some(tags) = tags_ref.get(fid) {
                    for (key, _) in tags {
                        if !seen_keys.contains_key(key) {
                            let col_idx = all_keys.len();
                            seen_keys.insert(key.clone(), col_idx);
                            all_keys.push(key.clone());
                            columns.push(Vec::new());
                        }
                    }
                }
            }

            // Fill column values
            for (fid, _) in &sorted_features {
                let tags = tags_ref.get(fid);
                for (ki, key) in all_keys.iter().enumerate() {
                    let val = tags.and_then(|t| {
                        t.iter().find(|(k, _)| k == key).map(|(_, v)| match v {
                            TagValue::Str(s) => serde_json::json!(s),
                            TagValue::Int(i) => serde_json::json!(i),
                            TagValue::Float(f) => serde_json::json!(f),
                            TagValue::Bool(b) => serde_json::json!(b),
                        })
                    }).unwrap_or(serde_json::json!(""));
                    columns[ki].push(val);
                }
            }

            // Build properties array
            let mut properties = Vec::new();
            for (ki, key) in all_keys.iter().enumerate() {
                let values = &columns[ki];
                // Determine type
                let all_numeric = values.iter().all(|v| v.is_number() || v == "");
                if all_numeric {
                    properties.push(serde_json::json!({
                        "id": key,
                        "type": "label",
                        "values": values
                    }));
                } else {
                    properties.push(serde_json::json!({
                        "id": key,
                        "type": "label",
                        "values": values
                    }));
                }
            }

            let sp_info = serde_json::json!({
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": ids,
                    "properties": properties
                }
            });
            let sp_str = serde_json::to_string_pretty(&sp_info).unwrap_or_default();
            std::fs::write(sp_dir.join("info"), sp_str).ok();

            segment_count
        });

        self.tiles_written = count;
        Ok(count)
    }

    /// Generate per-feature .mjb files from accumulated fragments.
    ///
    /// Unlike tile-centric MJB (one tile → many features), this produces
    /// feature-centric MJB (one .mjb file per feature) for O(1) segment retrieval.
    ///
    /// When `multilod=True` (default), each .mjb contains one Layer per zoom level
    /// with genuine vertex reduction at coarser levels via grid-based clustering.
    /// When `multilod=False`, only max_zoom fragments are used (single-LOD, backward compat).
    ///
    /// Also writes `manifest.json` with per-feature bboxes, tags, and LOD count.
    ///
    /// Returns the number of feature files written.
    #[pyo3(signature = (output_dir, world_bounds, layer_name="default", multilod=true))]
    fn generate_feature_mjb(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64, f64, f64),
        layer_name: &str,
        multilod: bool,
    ) -> PyResult<u32> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by feature_id
        let mut reader = FragmentReader::new(&self.fragment_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let feature_groups = reader.read_all_grouped_by_feature()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let max_zoom = self.max_zoom;
        let extent = self.extent;
        let extent_z = self.extent_z;
        let base_cells = self.base_cells;
        let out_dir = PathBuf::from(output_dir);
        let tags_ref = &self.tags_registry;
        let wb = world_bounds;
        let layer = layer_name.to_string();

        let features: Vec<(u32, Vec<Fragment>)> = feature_groups.into_iter().collect();

        // Results include lod_count for manifest
        let count = py.allow_threads(|| {
            std::fs::create_dir_all(&out_dir).ok();

            // Process each feature in parallel → collect results for manifest
            let results: Vec<(u32, [f64; 6], usize, u32)> = features.par_iter()
                .filter_map(|(feature_id, frags)| {
                    if multilod {
                        // Multi-LOD: use all zoom levels
                        let all_refs: Vec<&Fragment> = frags.iter().collect();
                        if all_refs.is_empty() {
                            return None;
                        }

                        let tags = tags_ref.get(feature_id);
                        let (mjb_bytes, bbox) = encode_feature_mjb_multilod(
                            &all_refs, tags, &wb, max_zoom, extent, extent_z, base_cells,
                        );

                        if mjb_bytes.is_empty() {
                            return None;
                        }

                        // Count distinct zoom levels
                        let mut zoom_set: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
                        for f in frags { zoom_set.insert(f.tile_z); }
                        let lod_count = zoom_set.len() as u32;
                        let byte_count = mjb_bytes.len();

                        let feat_path = out_dir.join(format!("{}.mjb", feature_id));
                        std::fs::write(&feat_path, mjb_bytes).ok();

                        Some((*feature_id, bbox, byte_count, lod_count))
                    } else {
                        // Single-LOD: only max_zoom fragments (backward compat)
                        let max_zoom_frags: Vec<&Fragment> = frags.iter()
                            .filter(|f| f.tile_z == max_zoom)
                            .collect();

                        if max_zoom_frags.is_empty() {
                            return None;
                        }

                        let tags = tags_ref.get(feature_id);
                        let (mjb_bytes, bbox) = encode_feature_mjb(
                            &max_zoom_frags, tags, &wb, &layer, extent, extent_z,
                        );

                        if mjb_bytes.is_empty() {
                            return None;
                        }

                        let byte_count = mjb_bytes.len();
                        let feat_path = out_dir.join(format!("{}.mjb", feature_id));
                        std::fs::write(&feat_path, mjb_bytes).ok();

                        Some((*feature_id, bbox, byte_count, 1u32))
                    }
                })
                .collect();

            // Write manifest.json
            let (wb_xmin, wb_ymin, wb_zmin, wb_xmax, wb_ymax, wb_zmax) = wb;
            let mut features_json = serde_json::Map::new();
            for (fid, bbox, _byte_count, lod_count) in &results {
                let mut entry = serde_json::Map::new();
                entry.insert("bbox".to_string(), serde_json::json!([
                    bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]
                ]));
                if multilod {
                    entry.insert("lod_count".to_string(), serde_json::json!(lod_count));
                }
                // Include tags
                if let Some(tag_list) = tags_ref.get(fid) {
                    let mut tags_obj = serde_json::Map::new();
                    for (k, v) in tag_list {
                        let jv = match v {
                            TagValue::Str(s) => serde_json::json!(s),
                            TagValue::Int(i) => serde_json::json!(i),
                            TagValue::Float(f) => serde_json::json!(f),
                            TagValue::Bool(b) => serde_json::json!(b),
                        };
                        tags_obj.insert(k.clone(), jv);
                    }
                    entry.insert("tags".to_string(), serde_json::Value::Object(tags_obj));
                }
                features_json.insert(fid.to_string(), serde_json::Value::Object(entry));
            }

            let version = if multilod { 2 } else { 1 };
            let mut manifest_map = serde_json::Map::new();
            manifest_map.insert("format".to_string(), serde_json::json!("microjson_feature_mjb"));
            manifest_map.insert("version".to_string(), serde_json::json!(version));
            manifest_map.insert("feature_count".to_string(), serde_json::json!(results.len()));
            manifest_map.insert("world_bounds".to_string(), serde_json::json!([wb_xmin, wb_ymin, wb_zmin, wb_xmax, wb_ymax, wb_zmax]));
            if multilod {
                manifest_map.insert("multilod".to_string(), serde_json::json!(true));
            }
            manifest_map.insert("features".to_string(), serde_json::Value::Object(features_json));

            let manifest = serde_json::Value::Object(manifest_map);
            let manifest_str = serde_json::to_string_pretty(&manifest).unwrap_or_default();
            std::fs::write(out_dir.join("manifest.json"), manifest_str).ok();

            results.len() as u32
        });

        self.tiles_written = count;
        Ok(count)
    }

    /// Generate Neuroglancer multilod_draco output from accumulated fragments.
    ///
    /// Produces one `.index` manifest and one Draco fragment data file per segment.
    /// All zoom levels are used — lod 0 = max_zoom (finest), lod N = zoom 0 (coarsest).
    /// Fragment positions within each LOD are sorted in Z-curve (Morton) order.
    ///
    /// Output structure:
    ///   {output_dir}/info                    (@type: neuroglancer_multilod_draco)
    ///   {output_dir}/{seg_id}.index          (binary manifest)
    ///   {output_dir}/{seg_id}                (concatenated Draco fragments)
    ///   {output_dir}/segment_properties/info (metadata from tags)
    ///
    /// Returns the number of segments written.
    #[pyo3(signature = (output_dir, world_bounds, vertex_quantization_bits=10))]
    fn generate_neuroglancer_multilod(
        &mut self,
        py: Python<'_>,
        output_dir: &str,
        world_bounds: (f64, f64, f64, f64, f64, f64),
        vertex_quantization_bits: u8,
    ) -> PyResult<u32> {
        // Flush and close the fragment writer
        if let Some(mut writer) = self.fragment_writer.take() {
            writer.flush()
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        }

        // Read all fragments grouped by feature_id
        let mut reader = FragmentReader::new(&self.fragment_path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let feature_groups = reader.read_all_grouped_by_feature()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let (xmin, ymin, zmin, xmax, ymax, zmax) = world_bounds;
        let dx = if xmax != xmin { xmax - xmin } else { 1.0 };
        let dy = if ymax != ymin { ymax - ymin } else { 1.0 };
        let dz = if zmax != zmin { zmax - zmin } else { 1.0 };

        let max_zoom = self.max_zoom;
        let qbits = vertex_quantization_bits;
        let qmax = ((1u32 << qbits) - 1) as f64;
        let out_dir = PathBuf::from(output_dir);
        let tags_ref = &self.tags_registry;

        let features: Vec<(u32, Vec<Fragment>)> = feature_groups.into_iter().collect();

        // chunk_shape = world extent / 2^max_zoom (finest tile size in world coords)
        let n_tiles = (1u32 << max_zoom) as f64;
        let chunk_shape = [dx / n_tiles, dy / n_tiles, dz / n_tiles];
        let grid_origin = [xmin as f32, ymin as f32, zmin as f32];

        let count = py.allow_threads(|| {
            std::fs::create_dir_all(&out_dir).ok();

            // Write info file
            // Build the 4x3 transform matrix: identity scaling from stored model to model coords
            // stored_model_coord = grid_origin + vertex_offset + chunk_shape * 2^lod * (frag_pos + v/(2^bits-1))
            // The info transform maps stored model → model (identity for us since we use world coords)
            let transform = [
                1.0f64, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ];
            let info = serde_json::json!({
                "@type": "neuroglancer_multilod_draco",
                "vertex_quantization_bits": qbits,
                "transform": transform,
                "lod_scale_multiplier": 1.0,
                "segment_properties": "segment_properties"
            });
            let info_str = serde_json::to_string_pretty(&info).unwrap_or_default();
            std::fs::write(out_dir.join("info"), info_str).ok();

            // Process each feature (segment) in parallel
            let segment_count: u32 = features.par_iter()
                .map(|(feature_id, frags)| {
                    // Group fragments by tile_z → BTreeMap for ordered iteration
                    let mut by_zoom: std::collections::BTreeMap<u32, Vec<&Fragment>> = std::collections::BTreeMap::new();
                    for frag in frags {
                        if frag.geom_type == 4 || frag.geom_type == 5 {
                            by_zoom.entry(frag.tile_z).or_default().push(frag);
                        }
                    }

                    if by_zoom.is_empty() {
                        return 0u32;
                    }

                    let num_lods = by_zoom.len() as u32;
                    let mut all_draco_data: Vec<u8> = Vec::new();
                    let mut lod_scales: Vec<f32> = Vec::new();
                    let mut vertex_offsets: Vec<f32> = Vec::new();
                    let mut num_fragments_per_lod: Vec<u32> = Vec::new();
                    // Per-LOD fragment positions and sizes
                    let mut all_frag_positions: Vec<Vec<[u32; 3]>> = Vec::new();
                    let mut all_frag_offsets: Vec<Vec<u32>> = Vec::new();

                    // Process each zoom level: max_zoom → 0 = lod 0 → lod N
                    for (&zoom, zoom_frags) in by_zoom.iter().rev() {
                        let lod = max_zoom.saturating_sub(zoom);
                        let lod_scale = (1u32 << lod) as f32;
                        lod_scales.push(lod_scale);

                        // Vertex offsets (zero for now since we quantize per-fragment)
                        vertex_offsets.push(0.0);
                        vertex_offsets.push(0.0);
                        vertex_offsets.push(0.0);

                        // Group fragments by tile position (tile_x, tile_y, tile_d)
                        let mut by_tile: std::collections::BTreeMap<(u32, u32, u32), Vec<&Fragment>> = std::collections::BTreeMap::new();
                        for frag in zoom_frags {
                            by_tile.entry((frag.tile_x, frag.tile_y, frag.tile_d)).or_default().push(frag);
                        }

                        // Sort by Morton code
                        let mut tile_keys: Vec<(u32, u32, u32)> = by_tile.keys().cloned().collect();
                        tile_keys.sort_by(|a, b| crate::morton::morton_cmp(*a, *b));

                        let mut frag_positions: Vec<[u32; 3]> = Vec::new();
                        let mut frag_offsets: Vec<u32> = Vec::new();

                        for (tx, ty, td) in &tile_keys {
                            let tile_frags = &by_tile[&(*tx, *ty, *td)];

                            // Merge all TIN fragments in this tile into one mesh
                            let mut vertex_map: AHashMap<(i64, i64, i64), u32> = AHashMap::new();
                            let mut positions: Vec<f32> = Vec::new();
                            let mut indices: Vec<u32> = Vec::new();

                            for frag in tile_frags {
                                let n_verts = frag.z.len();
                                let ring_lengths: Vec<usize> = if frag.ring_lengths.is_empty() {
                                    vec![n_verts]
                                } else {
                                    frag.ring_lengths.iter().map(|&r| r as usize).collect()
                                };

                                let mut offset = 0usize;
                                for &rl in &ring_lengths {
                                    let nv = if rl >= 4 { 3 } else { rl };
                                    if nv < 3 || offset + 2 >= n_verts {
                                        offset += rl;
                                        continue;
                                    }

                                    let mut tri = [0u32; 3];
                                    for vi_off in 0..3 {
                                        let vi = offset + vi_off;
                                        let wx = xmin + frag.xy[vi * 2] * dx;
                                        let wy = ymin + frag.xy[vi * 2 + 1] * dy;
                                        let wz = zmin + frag.z[vi] * dz;

                                        let wx_f32 = wx as f32;
                                        let wy_f32 = wy as f32;
                                        let wz_f32 = wz as f32;

                                        let key = (
                                            wx_f32.to_bits() as i64,
                                            wy_f32.to_bits() as i64,
                                            wz_f32.to_bits() as i64,
                                        );

                                        let idx = match vertex_map.get(&key) {
                                            Some(&idx) => idx,
                                            None => {
                                                let idx = vertex_map.len() as u32;
                                                vertex_map.insert(key, idx);
                                                positions.push(wx_f32);
                                                positions.push(wy_f32);
                                                positions.push(wz_f32);
                                                idx
                                            }
                                        };
                                        tri[vi_off] = idx;
                                    }

                                    if tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2] {
                                        indices.extend_from_slice(&tri);
                                    }
                                    offset += rl;
                                }
                            }

                            if positions.is_empty() || indices.is_empty() {
                                continue;
                            }

                            // Quantize world-coord positions to [0, 2^bits) within this tile's bounding box
                            let tile_scale = (1u32 << zoom) as f64;
                            let tile_xmin = xmin + (*tx as f64 / tile_scale) * dx;
                            let tile_ymin = ymin + (*ty as f64 / tile_scale) * dy;
                            let tile_zmin = zmin + (*td as f64 / tile_scale) * dz;
                            let tile_dx = dx / tile_scale;
                            let tile_dy = dy / tile_scale;
                            let tile_dz = dz / tile_scale;

                            let n_mesh_verts = positions.len() / 3;
                            let mut quant_positions: Vec<u32> = Vec::with_capacity(n_mesh_verts * 3);
                            for i in 0..n_mesh_verts {
                                let wx = positions[i * 3] as f64;
                                let wy = positions[i * 3 + 1] as f64;
                                let wz = positions[i * 3 + 2] as f64;
                                let qx = ((wx - tile_xmin) / tile_dx * qmax).round().max(0.0).min(qmax) as u32;
                                let qy = ((wy - tile_ymin) / tile_dy * qmax).round().max(0.0).min(qmax) as u32;
                                let qz = ((wz - tile_zmin) / tile_dz * qmax).round().max(0.0).min(qmax) as u32;
                                quant_positions.push(qx);
                                quant_positions.push(qy);
                                quant_positions.push(qz);
                            }

                            // Encode with Draco
                            match crate::encoder_draco::encode_draco_mesh(&quant_positions, &indices) {
                                Ok(draco_bytes) => {
                                    frag_positions.push([*tx, *ty, *td]);
                                    frag_offsets.push(draco_bytes.len() as u32);
                                    all_draco_data.extend_from_slice(&draco_bytes);
                                }
                                Err(_) => {
                                    // Skip this fragment on encoding error
                                }
                            }
                        }

                        num_fragments_per_lod.push(frag_positions.len() as u32);
                        all_frag_positions.push(frag_positions);
                        all_frag_offsets.push(frag_offsets);
                    }

                    if all_draco_data.is_empty() {
                        return 0u32;
                    }

                    // Write .index binary manifest
                    let mut index_buf: Vec<u8> = Vec::new();
                    // chunk_shape: 3 x float32le
                    for &cs in &chunk_shape {
                        index_buf.extend_from_slice(&(cs as f32).to_le_bytes());
                    }
                    // grid_origin: 3 x float32le
                    for &go in &grid_origin {
                        index_buf.extend_from_slice(&go.to_le_bytes());
                    }
                    // num_lods: uint32le
                    index_buf.extend_from_slice(&num_lods.to_le_bytes());
                    // lod_scales: num_lods x float32le
                    for &ls in &lod_scales {
                        index_buf.extend_from_slice(&ls.to_le_bytes());
                    }
                    // vertex_offsets: num_lods * 3 x float32le (C-order [num_lods, 3])
                    for &vo in &vertex_offsets {
                        index_buf.extend_from_slice(&vo.to_le_bytes());
                    }
                    // num_fragments_per_lod: num_lods x uint32le
                    for &nf in &num_fragments_per_lod {
                        index_buf.extend_from_slice(&nf.to_le_bytes());
                    }
                    // Per-LOD: fragment_positions [3, N] uint32le then fragment_offsets [N] uint32le
                    for lod_idx in 0..num_lods as usize {
                        let fp = &all_frag_positions[lod_idx];
                        let fo = &all_frag_offsets[lod_idx];
                        // fragment_positions: [3, num_fragments] uint32le — column-major per spec
                        // x values
                        for pos in fp { index_buf.extend_from_slice(&pos[0].to_le_bytes()); }
                        // y values
                        for pos in fp { index_buf.extend_from_slice(&pos[1].to_le_bytes()); }
                        // z values
                        for pos in fp { index_buf.extend_from_slice(&pos[2].to_le_bytes()); }
                        // fragment_offsets: [num_fragments] uint32le
                        for &off in fo { index_buf.extend_from_slice(&off.to_le_bytes()); }
                    }

                    let index_path = out_dir.join(format!("{}.index", feature_id));
                    std::fs::write(&index_path, &index_buf).ok();

                    // Write concatenated Draco data
                    let data_path = out_dir.join(feature_id.to_string());
                    std::fs::write(&data_path, &all_draco_data).ok();

                    1u32
                })
                .sum::<u32>();

            // Write segment_properties/info
            let sp_dir = out_dir.join("segment_properties");
            std::fs::create_dir_all(&sp_dir).ok();

            let mut ids: Vec<String> = Vec::new();
            let mut all_keys: Vec<String> = Vec::new();
            let mut seen_keys: AHashMap<String, usize> = AHashMap::new();
            let mut columns: Vec<Vec<serde_json::Value>> = Vec::new();

            let mut sorted_features: Vec<&(u32, Vec<Fragment>)> = features.iter().collect();
            sorted_features.sort_by_key(|(fid, _)| *fid);

            for (fid, _) in &sorted_features {
                ids.push(fid.to_string());
                if let Some(tags) = tags_ref.get(fid) {
                    for (key, _) in tags {
                        if !seen_keys.contains_key(key) {
                            let col_idx = all_keys.len();
                            seen_keys.insert(key.clone(), col_idx);
                            all_keys.push(key.clone());
                            columns.push(Vec::new());
                        }
                    }
                }
            }

            for (fid, _) in &sorted_features {
                let tags = tags_ref.get(fid);
                for (ki, key) in all_keys.iter().enumerate() {
                    let val = tags.and_then(|t| {
                        t.iter().find(|(k, _)| k == key).map(|(_, v)| match v {
                            TagValue::Str(s) => serde_json::json!(s),
                            TagValue::Int(i) => serde_json::json!(i),
                            TagValue::Float(f) => serde_json::json!(f),
                            TagValue::Bool(b) => serde_json::json!(b),
                        })
                    }).unwrap_or(serde_json::json!(""));
                    columns[ki].push(val);
                }
            }

            let mut properties = Vec::new();
            for (ki, key) in all_keys.iter().enumerate() {
                properties.push(serde_json::json!({
                    "id": key,
                    "type": "label",
                    "values": columns[ki]
                }));
            }

            let sp_info = serde_json::json!({
                "@type": "neuroglancer_segment_properties",
                "inline": {
                    "ids": ids,
                    "properties": properties
                }
            });
            let sp_str = serde_json::to_string_pretty(&sp_info).unwrap_or_default();
            std::fs::write(sp_dir.join("info"), sp_str).ok();

            segment_count
        });

        self.tiles_written = count;
        Ok(count)
    }

    /// Write tilejson3d.json metadata file.
    ///
    /// Args:
    ///   path: Output file path.
    ///   bounds: World bounds (xmin, ymin, zmin, xmax, ymax, zmax).
    ///   layer_name: Layer name (default "default").
    #[pyo3(signature = (path, bounds, layer_name="default"))]
    fn write_tilejson3d(
        &self,
        path: &str,
        bounds: (f64, f64, f64, f64, f64, f64),
    layer_name: &str,
    ) -> PyResult<()> {
        let (xmin, ymin, zmin, xmax, ymax, zmax) = bounds;
        let center = [
            (xmin + xmax) / 2.0,
            (ymin + ymax) / 2.0,
            (zmin + zmax) / 2.0,
            self.min_zoom as f64,
        ];

        let json = serde_json::json!({
            "tilejson": "3.0.0",
            "tiles": ["{z}/{x}/{y}/{d}.mjb"],
            "name": layer_name,
            "minzoom": self.min_zoom,
            "maxzoom": self.max_zoom,
            "bounds3d": [xmin, ymin, zmin, xmax, ymax, zmax],
            "center3d": center,
            "depthsize": self.extent_z,
            "vector_layers": [{
                "id": layer_name,
                "fields": {},
                "minzoom": self.min_zoom,
                "maxzoom": self.max_zoom,
            }],
        });

        let json_str = serde_json::to_string_pretty(&json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        std::fs::write(path, json_str)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(())
    }
}

/// Clean up temp fragment file on drop.
impl Drop for StreamingTileGenerator {
    fn drop(&mut self) {
        // Drop the writer first to release the file handle
        self.fragment_writer.take();
        std::fs::remove_file(&self.fragment_path).ok();
    }
}

/// Scan a list of OBJ files and return the combined world bounding box.
///
/// Uses rayon to scan files in parallel across CPU cores.
/// Only reads vertex data — skips face parsing for speed.
/// Returns (xmin, ymin, zmin, xmax, ymax, zmax).
#[pyfunction]
pub fn scan_obj_bounds(py: Python<'_>, paths: Vec<String>) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
    py.allow_threads(|| {
        scan_bounds_parallel(&paths)
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e))
}

/// Inner parallel bounds scan (no GIL).
fn scan_bounds_parallel(paths: &[String]) -> Result<(f64, f64, f64, f64, f64, f64), String> {
    use std::io::BufRead;

    // Each file scanned independently in parallel
    let per_file: Result<Vec<([f64; 3], [f64; 3])>, String> = paths.par_iter().map(|path| {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("Cannot open {}: {}", path, e))?;
        let reader = std::io::BufReader::new(file);

        let mut fmin = [f64::INFINITY; 3];
        let mut fmax = [f64::NEG_INFINITY; 3];

        for line in reader.lines() {
            let line = line.map_err(|e| e.to_string())?;
            let line = line.trim_start();
            if !line.starts_with("v ") {
                continue;
            }
            let mut parts = line.split_whitespace();
            parts.next(); // skip "v"
            if let (Some(xs), Some(ys), Some(zs)) = (parts.next(), parts.next(), parts.next()) {
                if let (Ok(x), Ok(y), Ok(z)) = (xs.parse::<f64>(), ys.parse::<f64>(), zs.parse::<f64>()) {
                    if x < fmin[0] { fmin[0] = x; }
                    if y < fmin[1] { fmin[1] = y; }
                    if z < fmin[2] { fmin[2] = z; }
                    if x > fmax[0] { fmax[0] = x; }
                    if y > fmax[1] { fmax[1] = y; }
                    if z > fmax[2] { fmax[2] = z; }
                }
            }
        }

        Ok((fmin, fmax))
    }).collect();

    let per_file = per_file?;

    // Merge per-file bounds
    let mut gmin = [f64::INFINITY; 3];
    let mut gmax = [f64::NEG_INFINITY; 3];

    for (fmin, fmax) in per_file {
        for i in 0..3 {
            if fmin[i] < gmin[i] { gmin[i] = fmin[i]; }
            if fmax[i] > gmax[i] { gmax[i] = fmax[i]; }
        }
    }

    Ok((gmin[0], gmin[1], gmin[2], gmax[0], gmax[1], gmax[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_surface_basic() {
        // A single triangle face at x=[0.1, 0.9], should be kept when clipping [0, 0.5)
        let feat = ClipFeature {
            xy: vec![0.1, 0.2, 0.9, 0.3, 0.5, 0.8],
            z: vec![0.1, 0.2, 0.3],
            ring_lengths: vec![3],
            geom_type: TIN,
            bbox: BBox3D {
                min_x: 0.1, min_y: 0.2, min_z: 0.1,
                max_x: 0.9, max_y: 0.8, max_z: 0.3,
            },
        };

        // Should keep the face (it straddles [0, 0.5))
        let result = clip_surface_rs(&feat, 0.0, 0.5, 0);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.z.len(), 3);
    }

    #[test]
    fn test_clip_surface_reject() {
        // Triangle entirely at x=[0.6, 0.9], clipping [0, 0.5) should reject
        let feat = ClipFeature {
            xy: vec![0.6, 0.2, 0.9, 0.3, 0.7, 0.8],
            z: vec![0.1, 0.2, 0.3],
            ring_lengths: vec![3],
            geom_type: TIN,
            bbox: BBox3D {
                min_x: 0.6, min_y: 0.2, min_z: 0.1,
                max_x: 0.9, max_y: 0.8, max_z: 0.3,
            },
        };

        let result = clip_surface_rs(&feat, 0.0, 0.5, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_octree_clip_single_point() {
        let feat = ClipFeature {
            xy: vec![0.3, 0.4],
            z: vec![0.5],
            ring_lengths: vec![],
            geom_type: POINT3D,
            bbox: BBox3D {
                min_x: 0.3, min_y: 0.4, min_z: 0.5,
                max_x: 0.3, max_y: 0.4, max_z: 0.5,
            },
        };

        let result = octree_clip(&feat, 0, 1, 0.0);
        // Should appear at zoom 0 (one tile) and zoom 1 (one tile)
        assert!(result.len() >= 2);

        // At zoom 0, should be (0,0,0,0)
        let zoom0: Vec<_> = result.iter().filter(|((z, _, _, _), _)| *z == 0).collect();
        assert_eq!(zoom0.len(), 1);
        assert_eq!(zoom0[0].0, (0, 0, 0, 0));

        // At zoom 1, point at (0.3, 0.4, 0.5) should be in tile (1, 0, 0, 1)
        // x=0.3 → left half (x=0), y=0.4 → bottom half (y=0), z=0.5 → back half (d=1)
        let zoom1: Vec<_> = result.iter().filter(|((z, _, _, _), _)| *z == 1).collect();
        assert_eq!(zoom1.len(), 1);
        assert_eq!(zoom1[0].0, (1, 0, 0, 1));
    }

    #[test]
    fn test_clip_line_basic() {
        // Line from x=0.1 to x=0.9
        let feat = ClipFeature {
            xy: vec![0.1, 0.5, 0.9, 0.5],
            z: vec![0.5, 0.5],
            ring_lengths: vec![],
            geom_type: LINESTRING3D,
            bbox: BBox3D {
                min_x: 0.1, min_y: 0.5, min_z: 0.5,
                max_x: 0.9, max_y: 0.5, max_z: 0.5,
            },
        };

        // Clip to [0, 0.5) along X
        let result = clip_line_rs(&feat, 0.0, 0.5, 0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].z.len(), 2); // start + intersection at x=0.5
    }

    #[test]
    fn test_encode_tag_value_roundtrip() {
        let tag = TagValue::Int(42);
        let encoded = encode_tag_value(&tag);
        assert!(!encoded.is_empty());

        let tag = TagValue::Str("hello".to_string());
        let encoded = encode_tag_value(&tag);
        assert!(!encoded.is_empty());
    }
}
