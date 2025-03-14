use pyo3::{prelude::*, Python};
use numpy::{
    ndarray::{Array2, s}, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use crate::value_error;


#[pyfunction]
/// Map integer coordinates to oblique coordinates.
pub fn oblique_coordinates<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<f32>,  // N x 2
    tilts: (f32, f32),
    intervals: (f32, f32),
    offsets: (f32, f32),
) -> PyResult<Py<PyArray2<f32>>> {
    if coords.shape()[1] != 2 {
        return value_error!("ints must be N x 2");
    }
    let coords = coords.as_array();
    let (tan0, tan1) = tilts;
    let (d0, d1) = intervals;
    let (c0, c1) = offsets;

    let mut out = Array2::<f32>::zeros((coords.shape()[0], 2));
    for i in 0..coords.shape()[0] {
        let nth = coords[[i, 0]];
        let npf = coords[[i, 1]];
        out[[i, 0]] = (nth + npf * tan1) * d0 + c0;
        out[[i, 1]] = (nth * tan0 + npf) * d1 + c1;
    }
    Ok(out.into_pyarray(py).into())

}

#[pyfunction]
/// build vertices and the corresponding coordinate for a cylinder
pub fn cylinder_faces<'py>(
    py: Python<'py>,
    ny: i32,
    npf: i32,
) -> PyResult<Py<PyArray2<i32>>> {
    let mut vert: Vec<(i32, i32, i32)> = Vec::new();
    for y in 0..ny {
        for a in 0..npf {
            let idx = y * npf + a;
            if y > 0 {
                //   y   o-O o
                //        \|
                // y-1   o o o
                if a > 0 {
                    vert.push((idx, idx - 1, idx - npf));
                } else {
                    vert.push((idx, idx + npf - 1, idx - npf));
                }
            }
            if y < ny - 1 {
                // y+1   o o o
                //         |\
                //   y   o O-o
                if a < npf - 1 {
                    vert.push((idx, idx + 1, idx + npf));
                } else {
                    vert.push((idx, idx - npf + 1, idx + npf));
                }
            }
        }
    }
    // convert vertices to ndarray
    let mut out_vert = Array2::<i32>::zeros((vert.len(), 3));
    for (i, v) in vert.iter().enumerate() {
        out_vert[[i, 0]] = v.0;
        out_vert[[i, 1]] = v.1;
        out_vert[[i, 2]] = v.2;
    }

    Ok(out_vert.into_pyarray(py).into())
}

#[pyfunction]
/// Find the index of the changing point by minimizing the sum of squares of two parts.
/// This is not an efficient implementation, but a easily understandable one.
/// For an efficient implementation, see https://github.com/hanjinliu/scikit-step.
pub fn find_changing_point(
    arr: PyReadonlyArray1<f32>,
) -> usize {
    let arr = arr.as_array();
    let mut idx = 0;
    let mut s2_min = f32::MAX;
    for i in 1..arr.len() {
        let arr_former = arr.slice(s![..i]);
        let arr_latter = arr.slice(s![i..]);
        let mean_former = arr_former.mean().unwrap();
        let mean_latter = arr_latter.mean().unwrap();
        let s2_former = arr_former.mapv(|x| (x - mean_former).powi(2)).sum();
        let s2_latter = arr_latter.mapv(|x| (x - mean_latter).powi(2)).sum();
        let s2 = s2_former + s2_latter;
        if s2 < s2_min {
            s2_min = s2;
            idx = i;
        }
    }
    idx
}
