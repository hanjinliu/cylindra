use pyo3::{prelude::*, Python};
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray2,
    ndarray::Array2,
};
use crate::value_error;


#[pyfunction]
/// Map integer coordinates to oblique coordinates.
pub fn oblique_coordinates<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<i32>,  // N x 2
    tilts: (f32, f32),
    intervals: (f32, f32),
    offsets: (f32, f32),
) -> PyResult<Py<PyArray2<f32>>> {
    if coords.shape()[1] != 2 {
        return value_error!("ints must be N x 2");
    }
    let ints = coords.as_array();
    let (tan0, tan1) = tilts;
    let (d0, d1) = intervals;
    let (c0, c1) = offsets;

    let mut out = Array2::<f32>::zeros((ints.shape()[0], 2));
    for i in 0..ints.shape()[0] {
        let nth = ints[[i, 0]] as f32;
        let npf = ints[[i, 1]] as f32;
        out[[i, 0]] = (nth + npf * tan1) * d0 + c0;
        out[[i, 1]] = (nth * tan0 + npf) * d1 + c1;
    }
    Ok(out.into_pyarray(py).to_owned())

}
