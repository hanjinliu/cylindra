use pyo3::{prelude::*, Python};
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Array2, ArrayView1},
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

pub fn indices_to_array(
    nth: ArrayView1<i32>,
    npf: ArrayView1<i32>,
    values: ArrayView1<f32>,
) -> PyResult<Array2<f32>> {
    let nsize = nth.shape()[0];
    if npf.shape()[0] != nsize || values.shape()[0] != nsize {
        return value_error!("nth, npf, and values must have the same length.");
    }

    let (nth_min, nth_max) = extrema(&nth);
    let (npf_min, npf_max) = extrema(&npf);
    let nth_size = (nth_max - nth_min + 1) as usize;
    let npf_size = (npf_max - npf_min + 1) as usize;
    let mut out = Array2::<f32>::from_elem((nth_size, npf_size), f32::NAN);
    for i in 0..nsize {
        out[[nth[[i]] as usize, npf[[i]] as usize]] = values[[i]];
    }
    Ok(out)
}

#[pyfunction]
pub fn indices_to_pyarray<'py>(
    py: Python<'py>,
    nth: PyReadonlyArray1<i32>,
    npf: PyReadonlyArray1<i32>,
    values: PyReadonlyArray1<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let nth = nth.as_array();
    let npf = npf.as_array();
    let values = values.as_array();
    let out = indices_to_array(nth, npf, values)?;
    Ok(out.into_pyarray(py).to_owned())
}

pub fn extrema<_D>(ar: &ArrayView1<_D>) -> (_D, _D) where _D: std::cmp::PartialOrd + Copy {
    let mut min = ar[[0]];
    let mut max = ar[[0]];
    for i in 1..ar.shape()[0] {
        if ar[[i]] < min {
            min = ar[[i]];
        }
        if ar[[i]] > max {
            max = ar[[i]];
        }
    }
    (min, max)
}
