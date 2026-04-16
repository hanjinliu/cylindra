use pyo3::{prelude::*, Python};
use numpy::{
    IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods, ndarray::{Array2, Array3, s}
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
        return value_error!("`coords` must be N x 2");
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
pub fn displacement_array<'py>(
    py: Python<'py>,
    mesh_shape: (usize, usize),  // (ny, npf)
    dilate: PyReadonlyArray1<f32>,  // N
    expand: PyReadonlyArray1<f32>,  // N
    twist: PyReadonlyArray1<f32>,  // N
) -> PyResult<Py<PyArray3<f32>>> {
    let (ny, npf) = mesh_shape;
    let dilate = reshape(&dilate, (ny, npf));
    let expand = reshape(&expand, (ny, npf));
    let twist = reshape(&twist, (ny, npf));

    let mut out = Array3::<f32>::zeros((ny, npf, 3));
    for iy in 0..ny {
        for ipf in 0..npf {
            out[[iy, ipf, 0]] += dilate[[iy, ipf]];
            let exp0 = expand[[iy, ipf]];
            let twist0 = twist[[iy, ipf]];
            for jy in iy..ny {
                out[[jy, ipf, 1]] += exp0;
                out[[jy, ipf, 2]] += twist0;
            }
        }
    }
    Ok(out.into_pyarray(py).into())
}

fn reshape(arr: &PyReadonlyArray1<f32>, shape: (usize, usize)) -> Array2<f32> {
    let arr = arr.as_array();
    arr.as_standard_layout().into_shape(shape).unwrap().to_owned()
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

#[pyfunction]
/// Convert label array to segments.
/// Returns an array of shape (M, 3), where each row is (start_index, end_index, value).
pub fn labels_to_segments(
    py: Python,
    labels: PyReadonlyArray1<i32>,
    background_label: i32,
    min_length: usize,
) -> PyResult<Py<PyArray2<i32>>> {
    let labels = labels.as_array();
    let n = labels.len();
    let mut segments: Vec<(i32, i32, i32)> = Vec::new();
    if n == 0 {
        return Ok(Array2::<i32>::zeros((0, 3)).into_pyarray(py).into());
    }
    let mut start = 0;
    let mut current_label = labels[0];
    for i in 1..n {
        let this_label = labels[i];
        if this_label != current_label {
            if current_label != background_label && (i - start) >= min_length {
                segments.push((start as i32, i as i32, current_label));
            }
            start = i;
            current_label = this_label;
        }
    }
    if current_label != background_label && (n - start) >= min_length {
        segments.push((start as i32, n as i32, current_label));
    }
    segments.push((start as i32, n as i32, current_label));

    // Vec to ndarray
    let mut out_segments = Array2::<i32>::zeros((segments.len(), 3));
    for (i, seg) in segments.iter().enumerate() {
        out_segments[[i, 0]] = seg.0;
        out_segments[[i, 1]] = seg.1;
        out_segments[[i, 2]] = seg.2;
    }

    Ok(out_segments.into_pyarray(py).into())
}
