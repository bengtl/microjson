/// Vertex-clustering mesh decimation for LOD tile generation.
///
/// Port of simplify_mesh.py — reduces triangle count by snapping vertices
/// to a uniform 3D grid, merging coincident vertices, and removing
/// degenerate faces. The grid is world-aligned so shared boundary vertices
/// in adjacent tiles snap to the same cell (no cracks).
use pyo3::prelude::*;
use pyo3::types::PyList;
use ahash::AHashMap;

/// Base grid resolution at zoom 0.
const BASE_CELLS: u32 = 10;

/// Decimate TIN coordinates using vertex clustering.
///
/// Args:
///   coordinates: TIN coords as list of [[[v0, v1, v2, v0]]] (closed triangle rings)
///   target_ratio: fraction of faces to keep (0.0–1.0), >= 1.0 returns input unchanged
///   world_bounds: optional (xmin, ymin, zmin, xmax, ymax, zmax) for world-aligned grid
///   zoom: current zoom level (used with world_bounds)
///
/// Returns: simplified TIN coordinates in same format
#[pyfunction]
#[pyo3(signature = (coordinates, target_ratio, world_bounds=None, zoom=0))]
pub fn decimate_tin<'py>(
    py: Python<'py>,
    coordinates: &Bound<'py, PyList>,
    target_ratio: f64,
    world_bounds: Option<(f64, f64, f64, f64, f64, f64)>,
    zoom: u32,
) -> PyResult<PyObject> {
    let n_faces = coordinates.len();

    if target_ratio >= 1.0 || n_faces <= 4 {
        return Ok(coordinates.into_py(py));
    }

    // Extract all triangle vertices (3 per face, skip closing vertex)
    let mut all_verts: Vec<[f64; 3]> = Vec::with_capacity(n_faces * 3);

    for i in 0..n_faces {
        let face: &Bound<'_, PyList> = &coordinates.get_item(i)?.downcast_into()?;
        let ring: &Bound<'_, PyList> = &face.get_item(0)?.downcast_into()?;
        for j in 0..3.min(ring.len()) {
            let v: Vec<f64> = ring.get_item(j)?.extract()?;
            all_verts.push([
                v[0],
                if v.len() > 1 { v[1] } else { 0.0 },
                if v.len() > 2 { v[2] } else { 0.0 },
            ]);
        }
    }

    // Compute grid parameters
    let (grid_origin, cell_size) = if let Some((xmin, ymin, zmin, xmax, ymax, zmax)) = world_bounds {
        let origin = [xmin, ymin, zmin];
        let world_size = [
            (xmax - xmin).max(1e-10),
            (ymax - ymin).max(1e-10),
            (zmax - zmin).max(1e-10),
        ];
        let cells_per_axis = BASE_CELLS * (1u32 << zoom);
        let cs = [
            world_size[0] / cells_per_axis as f64,
            world_size[1] / cells_per_axis as f64,
            world_size[2] / cells_per_axis as f64,
        ];
        (origin, cs)
    } else {
        // Fallback: mesh-local grid
        let mut vmin = [f64::INFINITY; 3];
        let mut vmax = [f64::NEG_INFINITY; 3];
        for v in &all_verts {
            for d in 0..3 {
                if v[d] < vmin[d] { vmin[d] = v[d]; }
                if v[d] > vmax[d] { vmax[d] = v[d]; }
            }
        }
        let bbox_size = [
            (vmax[0] - vmin[0]).max(1e-10),
            (vmax[1] - vmin[1]).max(1e-10),
            (vmax[2] - vmin[2]).max(1e-10),
        ];
        let current_res = (all_verts.len() as f64).cbrt().max(2.0);
        let target_res = (current_res * target_ratio.sqrt()).max(2.0);
        let cs = [
            bbox_size[0] / target_res,
            bbox_size[1] / target_res,
            bbox_size[2] / target_res,
        ];
        (vmin, cs)
    };

    // Quantize vertices to grid cells
    let mut cell_map: AHashMap<(i32, i32, i32), u32> = AHashMap::new();
    let mut new_verts_accum: Vec<[f64; 4]> = Vec::new(); // [sum_x, sum_y, sum_z, count]
    let mut vert_remap: Vec<u32> = Vec::with_capacity(all_verts.len());

    for v in &all_verts {
        let qx = ((v[0] - grid_origin[0]) / cell_size[0]).floor() as i32;
        let qy = ((v[1] - grid_origin[1]) / cell_size[1]).floor() as i32;
        let qz = ((v[2] - grid_origin[2]) / cell_size[2]).floor() as i32;
        let key = (qx, qy, qz);

        let idx = match cell_map.get(&key) {
            Some(&idx) => {
                new_verts_accum[idx as usize][0] += v[0];
                new_verts_accum[idx as usize][1] += v[1];
                new_verts_accum[idx as usize][2] += v[2];
                new_verts_accum[idx as usize][3] += 1.0;
                idx
            }
            None => {
                let idx = new_verts_accum.len() as u32;
                cell_map.insert(key, idx);
                new_verts_accum.push([v[0], v[1], v[2], 1.0]);
                idx
            }
        };
        vert_remap.push(idx);
    }

    // Compute centroids
    let centroids: Vec<[f64; 3]> = new_verts_accum
        .iter()
        .map(|a| [a[0] / a[3], a[1] / a[3], a[2] / a[3]])
        .collect();

    // Remap faces and remove degenerates
    let result = PyList::empty(py);
    let mut kept = 0usize;

    for f in 0..n_faces {
        let i0 = vert_remap[f * 3];
        let i1 = vert_remap[f * 3 + 1];
        let i2 = vert_remap[f * 3 + 2];

        // Skip degenerate
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }

        let v0 = &centroids[i0 as usize];
        let v1 = &centroids[i1 as usize];
        let v2 = &centroids[i2 as usize];

        // Build [[v0, v1, v2, v0]] closed ring
        let ring = PyList::new(py, &[
            PyList::new(py, v0.as_slice())?,
            PyList::new(py, v1.as_slice())?,
            PyList::new(py, v2.as_slice())?,
            PyList::new(py, v0.as_slice())?,
        ])?;
        let face = PyList::new(py, &[ring])?;
        result.append(face)?;
        kept += 1;
    }

    if kept == 0 {
        // Degenerated completely — return minimal subset
        let subset = PyList::empty(py);
        let limit = n_faces.min(4);
        for i in 0..limit {
            subset.append(coordinates.get_item(i)?)?;
        }
        return Ok(subset.into());
    }

    Ok(result.into())
}
