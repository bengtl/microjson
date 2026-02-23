use pyo3::prelude::*;

mod types;
mod obj_parser;
mod projector;
mod clip;
mod simplify;
mod tile_transform;
mod encoder_mjb;
mod fragment;
mod encoder_glb;
mod tileset_json;
mod streaming;
mod morton;
mod encoder_draco;
mod encoder_meshopt;

/// The main Python module implemented in Rust.
#[pymodule]
fn _rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Phase 2: OBJ parser
    m.add_function(wrap_pyfunction!(obj_parser::parse_obj, m)?)?;

    // Phase 3: Clipping
    m.add_function(wrap_pyfunction!(clip::clip_surface, m)?)?;
    m.add_function(wrap_pyfunction!(clip::clip_line, m)?)?;
    m.add_function(wrap_pyfunction!(clip::clip_points, m)?)?;

    // Phase 4: Tile transform + MJB encoder
    m.add_function(wrap_pyfunction!(tile_transform::transform_tile_3d, m)?)?;
    m.add_function(wrap_pyfunction!(encoder_mjb::encode_tile_3d, m)?)?;
    m.add_function(wrap_pyfunction!(encoder_mjb::build_indexed_mesh, m)?)?;

    // Phase 5: Projector + Simplification
    m.add_class::<projector::CartesianProjector3D>()?;
    m.add_function(wrap_pyfunction!(simplify::decimate_tin, m)?)?;

    // Phase 6: Streaming tile generator
    m.add_class::<streaming::StreamingTileGenerator>()?;
    m.add_function(wrap_pyfunction!(streaming::scan_obj_bounds, m)?)?;

    // Draco encoder
    m.add_function(wrap_pyfunction!(encoder_draco::draco_encode_mesh, m)?)?;

    Ok(())
}
