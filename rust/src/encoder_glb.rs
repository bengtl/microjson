/// Minimal glTF 2.0 binary (GLB) encoder for OGC 3D Tiles output.
///
/// Produces a single GLB file containing one or more nodes, each with
/// a mesh primitive (triangles, lines, or points). Feature properties
/// are stored in node `extras`.
///
/// No external dependencies beyond serde_json + byteorder.

use serde_json::{json, Value};

// glTF component types
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;

// Buffer view targets
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;

// Primitive modes
pub(crate) const MODE_POINTS: u32 = 0;
pub(crate) const MODE_LINES: u32 = 1;
pub(crate) const MODE_TRIANGLES: u32 = 4;

// GLB header
const GLB_MAGIC: u32 = 0x46546C67; // "glTF"
const GLB_VERSION: u32 = 2;
const JSON_CHUNK_TYPE: u32 = 0x4E4F534A; // "JSON"
const BIN_CHUNK_TYPE: u32 = 0x004E4942; // "BIN\0"

/// A feature to be encoded into GLB.
pub(crate) struct GlbFeature {
    /// Flat f32 positions [x,y,z, x,y,z, ...] in world coordinates.
    pub positions: Vec<f32>,
    /// Triangle/line indices (empty for points).
    pub indices: Vec<u32>,
    /// GL primitive mode (0=POINTS, 1=LINES, 4=TRIANGLES).
    pub mode: u32,
    /// Feature metadata stored in node.extras.
    pub extras: Option<Value>,
}

/// Encode a set of features as a single GLB binary.
pub(crate) fn encode_glb(features: &[GlbFeature]) -> Vec<u8> {
    let mut bin_data: Vec<u8> = Vec::new();
    let mut buffer_views: Vec<Value> = Vec::new();
    let mut accessors: Vec<Value> = Vec::new();
    let mut meshes: Vec<Value> = Vec::new();
    let mut nodes: Vec<Value> = Vec::new();
    let mut scene_nodes: Vec<Value> = Vec::new();

    for feat in features {
        if feat.positions.is_empty() {
            continue;
        }

        let n_verts = feat.positions.len() / 3;

        // Compute min/max for position accessor
        let mut pos_min = [f32::INFINITY; 3];
        let mut pos_max = [f32::NEG_INFINITY; 3];
        for v in 0..n_verts {
            for d in 0..3 {
                let val = feat.positions[v * 3 + d];
                if val < pos_min[d] {
                    pos_min[d] = val;
                }
                if val > pos_max[d] {
                    pos_max[d] = val;
                }
            }
        }

        // --- Positions buffer view ---
        let pos_offset = bin_data.len();
        let pos_bytes: Vec<u8> = feat
            .positions
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let pos_byte_len = pos_bytes.len();
        bin_data.extend_from_slice(&pos_bytes);
        // Pad to 4 bytes
        let pad = (4 - (bin_data.len() % 4)) % 4;
        bin_data.extend(std::iter::repeat(0u8).take(pad));

        let pos_bv_idx = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": pos_offset,
            "byteLength": pos_byte_len,
            "target": ARRAY_BUFFER,
        }));

        let pos_acc_idx = accessors.len();
        accessors.push(json!({
            "bufferView": pos_bv_idx,
            "byteOffset": 0,
            "componentType": FLOAT,
            "count": n_verts,
            "type": "VEC3",
            "min": [pos_min[0], pos_min[1], pos_min[2]],
            "max": [pos_max[0], pos_max[1], pos_max[2]],
        }));

        // --- Indices buffer view (if any) ---
        let primitive = if !feat.indices.is_empty() {
            let idx_offset = bin_data.len();
            let idx_bytes: Vec<u8> = feat
                .indices
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            let idx_byte_len = idx_bytes.len();
            bin_data.extend_from_slice(&idx_bytes);
            let pad = (4 - (bin_data.len() % 4)) % 4;
            bin_data.extend(std::iter::repeat(0u8).take(pad));

            let idx_bv_idx = buffer_views.len();
            buffer_views.push(json!({
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": idx_byte_len,
                "target": ELEMENT_ARRAY_BUFFER,
            }));

            let idx_acc_idx = accessors.len();
            accessors.push(json!({
                "bufferView": idx_bv_idx,
                "byteOffset": 0,
                "componentType": UNSIGNED_INT,
                "count": feat.indices.len(),
                "type": "SCALAR",
            }));

            json!({
                "attributes": {"POSITION": pos_acc_idx},
                "indices": idx_acc_idx,
                "material": 0,
                "mode": feat.mode,
            })
        } else {
            json!({
                "attributes": {"POSITION": pos_acc_idx},
                "material": 0,
                "mode": feat.mode,
            })
        };

        let mesh_idx = meshes.len();
        meshes.push(json!({"primitives": [primitive]}));

        let mut node = json!({"mesh": mesh_idx, "name": format!("feature_{}", nodes.len())});
        if let Some(extras) = &feat.extras {
            node.as_object_mut()
                .unwrap()
                .insert("extras".to_string(), extras.clone());
        }

        let node_idx = nodes.len();
        nodes.push(node);
        scene_nodes.push(json!(node_idx));
    }

    // If no features, return minimal valid GLB
    if meshes.is_empty() {
        return encode_empty_glb();
    }

    // Default material — matches Python pipeline (metallicFactor=0.1, roughnessFactor=0.8)
    let material = json!({
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.8, 0.8, 1.0],
            "metallicFactor": 0.1,
            "roughnessFactor": 0.8,
        },
        "doubleSided": true,
    });

    // Assemble JSON descriptor
    let gltf_json = json!({
        "asset": {"version": "2.0", "generator": "microjson-rs"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": bin_data.len()}],
        "materials": [material],
    });

    assemble_glb(&gltf_json, &bin_data)
}

fn encode_empty_glb() -> Vec<u8> {
    let gltf_json = json!({
        "asset": {"version": "2.0", "generator": "microjson-rs"},
        "scene": 0,
        "scenes": [{"nodes": []}],
    });
    assemble_glb(&gltf_json, &[])
}

/// Pack a JSON descriptor and binary buffer into a GLB byte stream.
fn assemble_glb(gltf_json: &Value, bin_data: &[u8]) -> Vec<u8> {
    let json_str = serde_json::to_string(gltf_json).unwrap();
    let json_bytes = json_str.as_bytes();
    let json_pad = (4 - (json_bytes.len() % 4)) % 4;
    let json_chunk_len = json_bytes.len() + json_pad;

    let has_bin = !bin_data.is_empty();
    let bin_pad = if has_bin {
        (4 - (bin_data.len() % 4)) % 4
    } else {
        0
    };
    let bin_chunk_len = bin_data.len() + bin_pad;

    let total_length = 12  // GLB header
        + 8 + json_chunk_len  // JSON chunk header + data
        + if has_bin { 8 + bin_chunk_len } else { 0 }; // BIN chunk

    let mut out = Vec::with_capacity(total_length);

    // GLB header (12 bytes)
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&GLB_VERSION.to_le_bytes());
    out.extend_from_slice(&(total_length as u32).to_le_bytes());

    // JSON chunk
    out.extend_from_slice(&(json_chunk_len as u32).to_le_bytes());
    out.extend_from_slice(&JSON_CHUNK_TYPE.to_le_bytes());
    out.extend_from_slice(json_bytes);
    out.extend(std::iter::repeat(b' ').take(json_pad));

    // BIN chunk
    if has_bin {
        out.extend_from_slice(&(bin_chunk_len as u32).to_le_bytes());
        out.extend_from_slice(&BIN_CHUNK_TYPE.to_le_bytes());
        out.extend_from_slice(bin_data);
        out.extend(std::iter::repeat(0u8).take(bin_pad));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_empty() {
        let glb = encode_glb(&[]);
        // GLB header: magic + version + length
        assert!(glb.len() >= 12);
        assert_eq!(&glb[0..4], &GLB_MAGIC.to_le_bytes());
        assert_eq!(
            u32::from_le_bytes([glb[4], glb[5], glb[6], glb[7]]),
            GLB_VERSION
        );
        assert_eq!(
            u32::from_le_bytes([glb[8], glb[9], glb[10], glb[11]]),
            glb.len() as u32
        );
    }

    #[test]
    fn test_encode_triangle() {
        let feat = GlbFeature {
            positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0],
            indices: vec![0, 1, 2],
            mode: MODE_TRIANGLES,
            extras: Some(json!({"name": "triangle"})),
        };
        let glb = encode_glb(&[feat]);

        // Verify header
        assert_eq!(&glb[0..4], &GLB_MAGIC.to_le_bytes());

        // Parse JSON chunk
        let json_len =
            u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
        let json_type =
            u32::from_le_bytes([glb[16], glb[17], glb[18], glb[19]]);
        assert_eq!(json_type, JSON_CHUNK_TYPE);

        let json_str = std::str::from_utf8(&glb[20..20 + json_len])
            .unwrap()
            .trim();
        let json: Value = serde_json::from_str(json_str).unwrap();

        // Verify structure
        assert_eq!(json["asset"]["version"], "2.0");
        assert_eq!(json["nodes"][0]["extras"]["name"], "triangle");
        assert_eq!(json["meshes"].as_array().unwrap().len(), 1);
        assert_eq!(
            json["meshes"][0]["primitives"][0]["mode"],
            MODE_TRIANGLES
        );
    }

    #[test]
    fn test_encode_multiple_features() {
        let feats = vec![
            GlbFeature {
                positions: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0],
                indices: vec![0, 1, 2],
                mode: MODE_TRIANGLES,
                extras: Some(json!({"id": 1})),
            },
            GlbFeature {
                positions: vec![2.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                indices: vec![0, 1],
                mode: MODE_LINES,
                extras: Some(json!({"id": 2})),
            },
            GlbFeature {
                positions: vec![5.0, 5.0, 5.0],
                indices: vec![],
                mode: MODE_POINTS,
                extras: None,
            },
        ];
        let glb = encode_glb(&feats);

        // Parse JSON
        let json_len =
            u32::from_le_bytes([glb[12], glb[13], glb[14], glb[15]]) as usize;
        let json_str = std::str::from_utf8(&glb[20..20 + json_len])
            .unwrap()
            .trim();
        let json: Value = serde_json::from_str(json_str).unwrap();

        assert_eq!(json["nodes"].as_array().unwrap().len(), 3);
        assert_eq!(json["meshes"].as_array().unwrap().len(), 3);
        assert_eq!(json["scenes"][0]["nodes"].as_array().unwrap().len(), 3);
    }
}
