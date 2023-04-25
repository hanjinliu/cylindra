use pyo3::{pymodule, types::PyModule, PyResult, Python};
pub mod viterbi;
pub mod coordinates;
pub mod cylindric;
pub mod alleviate;
pub mod annealing;

/// A Python module implemented in Rust.
#[pymodule]
fn _cylindra_ext(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add __version__
    let mut version = env!("CARGO_PKG_VERSION").to_string();
    version = version.replace("-alpha", "a").replace("-beta", "b");
    m.add("__version__", version)?;

    m.add_class::<viterbi::ViterbiGrid>()?;
    m.add_class::<cylindric::CylinderGeometry>()?;
    m.add_class::<cylindric::Sources>()?;
    m.add_class::<cylindric::Index>()?;
    m.add_function(pyo3::wrap_pyfunction!(alleviate::alleviate, m)?)?;
    Ok(())
}
