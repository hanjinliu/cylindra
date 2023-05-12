use std::collections::HashSet;

use pyo3::{prelude::*, Python};
use numpy::{
    IntoPyArray, PyArray3, PyReadonlyArray2, PyReadonlyArray3,
    ndarray::{ArcArray2, s},
};
use super::cylindric::{Index, CylinderGeometry};
use crate::value_error;

/// Convert (N, 2) pyarray in to a vector of Index objects
/// As a pseudo-code example, arrayToIndices(np.array([[2, 3], [5, 4]])) will return
/// {Index(2, 3), Index(5, 4)}.
pub fn array_to_indices(array: &ArcArray2<isize>) -> Vec<Index> {
    let ndix = array.shape()[0];
    let mut indices: Vec<Index> = Vec::new();
    for i in 0..ndix {
        indices.push(
            Index::new(array[[i, 0]], array[[i, 1]])
        );
    }
    indices
}

/// Alleviate molecule displacements by iterative local-averaging algorithm.
/// Molecule positions labeled by the argument `label` will not be moved. The other
/// molecules will be averaged by the surroudning molecules. This procedure will be
/// repeated `iterations` times.
#[pyfunction]
pub fn alleviate<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<f32>,
    label: PyReadonlyArray2<isize>,
    nrise: isize,
) -> PyResult<Py<PyArray3<f32>>> {
    let arr = arr.as_array();
    let label = label.as_array();
    let ny = arr.shape()[0] as isize;
    let na = arr.shape()[1] as isize;
    let ndim = arr.shape()[2] as isize;
    if ndim != 3 {
        return value_error!("The last dimension of `arr` must be 3.");
    }
    let nsize = (ny * na) as usize;

    let mut indices = array_to_indices(&label.to_shared());
    let mut arr = arr.to_owned();
    let geometry = CylinderGeometry::new(ny, na, nrise);
    let mut processed = HashSet::new();
    for index in indices.iter() {
        processed.insert(index.clone());
    }

    // propagate the average-and-update step.
    let mut nrepeat = 0;  // just in case of infinite recursion.
    while processed.len() < nsize {
        let neighbors = geometry.get_neighbors(&indices)?;
        let mut arr_updated = arr.clone();
        for neighbor in neighbors.iter() {
            if processed.contains(&neighbor) {
                continue;
            }
            processed.insert(neighbor.clone());

            let cur_neighbor = geometry.get_neighbor(neighbor.y, neighbor.a)?;
            let mut n_cur_neighbor = 1.0;

            let (y, a) = (neighbor.y as usize, neighbor.a as usize);
            let mut sum_r = arr[[y, a, 0]];
            let mut sum_y = arr[[y, a, 1]];
            let a = arr[[y, a, 2]];
            let mut sum_a_cos = a.cos();
            let mut sum_a_sin = a.sin();

            for nbr in cur_neighbor.y_pair().iter().chain(cur_neighbor.a_pair().iter()) {
                let (y, a) = (nbr.y as usize, nbr.a as usize);
                sum_r += arr[[y, a, 0]];
                sum_y += arr[[y, a, 1]];
                let a = arr[[y, a, 2]];
                sum_a_cos += a.cos();
                sum_a_sin += a.sin();
                n_cur_neighbor += 1.0;
            }

            let (y, a) = (neighbor.y as usize, neighbor.a as usize);
            let theta = sum_a_sin.atan2(sum_a_cos);
            arr_updated[[y, a, 0]] = sum_r / n_cur_neighbor;
            arr_updated[[y, a, 1]] = sum_y / n_cur_neighbor;
            arr_updated[[y, a, 2]] = if theta < 0.0 { theta + 2.0 * std::f32::consts::PI } else { theta };
        }
        // update target indices since affected molecules propagates during iteration.
        indices = neighbors;
        // update array for next iteration
        arr.slice_mut(s![.., .., ..]).assign(&arr_updated);
        nrepeat += 1;
        if nrepeat > nsize {
            return value_error!("Infinite recursion occurred.");
        }
    }
    Ok(arr.into_pyarray(py).to_owned())
}
