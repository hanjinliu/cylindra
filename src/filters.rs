use std::{hash::Hash, collections::HashMap};

use pyo3::{prelude::*, Python};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::{Array1, Array2, ArrayView1},
};
use crate::value_error;

/// A kernel array which is indexed by integer coordinates from the center.
/// The center of a kernel
/// [[0, 1, 0],
///  [1, 1, 1],
///  [0, 1, 0]]
/// will be indexed by [0, 0]. To get the first value, [-1, -1] should be used.
pub struct Kernel<_D> {
    kernel: Array2<_D>,
    center: (usize, usize),
}

impl<_D> Kernel<_D> where _D: Copy {
    pub fn new(kernel: Array2<_D>) -> Self {
        let center = ((kernel.shape()[0] - 1) / 2, (kernel.shape()[1] - 1) / 2);
        Self { kernel, center }
    }

    fn get_range(&self) -> ((isize, isize), (isize, isize)) {
        let (ny, na) = (self.kernel.shape()[0], self.kernel.shape()[1]);
        let (i0, i1) = (self.center.0 as isize, self.center.1 as isize);
        ((-i0, ny as isize - i0 - 1), (-i1, na as isize - i1 - 1))
    }

    fn iter_indices(&self) -> impl Iterator<Item = [isize; 2]> {
        let (i0_range, i1_range) = self.get_range();
        let (i0_min, i0_max) = i0_range;
        let (i1_min, i1_max) = i1_range;
        (i0_min..=i0_max).flat_map(
            move |i0| (i1_min..=i1_max).map(move |i1| [i0, i1])
        )
    }
}

// implement get item
impl<_D> core::ops::Index<[isize; 2]> for Kernel<_D> {
    type Output = _D;

    fn index(&self, index: [isize; 2]) -> &Self::Output {
        let i0 = (index[0] + self.center.0 as isize) as usize;
        let i1 = (index[1] + self.center.1 as isize) as usize;
        &self.kernel[[i0, i1]]
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CylindricArray {
    array: Array2<f32>,
    nrise: isize,
    ycoords: Array1<usize>,
    acoords: Array1<usize>,
}

#[pymethods]
impl CylindricArray {
    #[new]
    /// Create a new CylindricArray from 1D arrays.
    /// Input arrays are supposed to be the molecules features "nth", "pf-id" and any column.
    pub fn from_indices(
        nth: PyReadonlyArray1<i32>,
        npf: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        nrise: isize,
    ) -> PyResult<Self> {
        let nth = nth.as_array();
        let npf = npf.as_array();
        let values = values.as_array();
        Self::from_indices_(&nth, &npf, &values, nrise)
    }

    pub fn nrise(&self) -> isize {
        self.nrise
    }

    /// Convert the CylindricArray to a 2D numpy array.
    pub fn asarray(&self, py: Python) -> Py<PyArray2<f32>> {
        self.array.clone().into_pyarray_bound(py).unbind()
    }

    pub fn as1d(&self, py: Python) -> Py<PyArray1<f32>> {
        let mut out = Array1::<f32>::zeros(self.ycoords.shape()[0]);
        for i in 0..self.ycoords.len() {
            out[[i]] = self.array[[self.ycoords[[i]], self.acoords[[i]]]];
        }
        out.into_pyarray_bound(py).unbind()
    }

    /// Convolution on the cylinder surface.
    pub fn convolve(
        &self,
        weight: PyReadonlyArray2<f32>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(weight.as_array().to_owned());
        let out = self.convolve_(&kernel);
        Ok(self.new_like(out))
    }

    pub fn count_neighbors(
        &self,
        footprint: PyReadonlyArray2<bool>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(footprint.as_array().to_owned());
        if kernel[[0, 0]] {
            return value_error!("The center of the kernel must be 0.");
        }
        let out = self.count_neighbors_(&kernel);
        Ok(self.new_like(out))
    }

    /// Mean filter on the cylinder surface.
    pub fn mean_filter(
        &self,
        footprint: PyReadonlyArray2<bool>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(footprint.as_array().to_owned());
        let out = self.mean_filter_(&kernel);
        Ok(self.new_like(out))
    }

    /// Maximum filter on the cylinder surface.
    pub fn max_filter(
        &self,
        footprint: PyReadonlyArray2<bool>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(footprint.as_array().to_owned());
        let out = self.max_filter_(&kernel);
        Ok(self.new_like(out))
    }

    /// Minimum filter on the cylinder surface.
    pub fn min_filter(
        &self,
        footprint: PyReadonlyArray2<bool>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(footprint.as_array().to_owned());
        let out = self.min_filter_(&kernel);
        Ok(self.new_like(out))
    }

    /// Median filter on the cylinder surface.
    pub fn median_filter(
        &self,
        footprint: PyReadonlyArray2<bool>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(footprint.as_array().to_owned());
        let out = self.median_filter_(&kernel);
        Ok(self.new_like(out))
    }

    /// Label boolean patterns on the cylinder surface.
    pub fn label(&self) -> Self {
        let shape = self.array.shape();
        let mut labels = self.zeros_like();
        let mut current_label = 1.0;
        for y in 0..shape[0] as isize {
            for x in 0..shape[1] as isize {
                if labels[[y, x]] != 0.0 || self[[y as isize, x as isize]] == 0.0 {
                    continue;
                }
                let mut stack = vec![(y, x)];
                while let Some((i, j)) = stack.pop() {
                    labels[[i, j]] = current_label;
                    let neighbors = [(i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j)];
                    for &(ni, nj) in &neighbors {
                        let value = self[[ni, nj]];
                        if value.is_nan() {
                            continue;
                        }
                        if value > 0.5 && labels[[ni, nj]] < 0.5 {
                            stack.push(self.norm_indices(&[ni, nj]));
                        }
                    }
                }
                current_label += 1.0;
            }
        }
        labels
    }

    pub fn with_values(&self, value: PyReadonlyArray1<f32>) -> PyResult<Self> {
        self.with_values_(&value.as_array())
    }
}

impl CylindricArray {
    fn new(array: Array2<f32>, nrise: isize, ycoords: Array1<usize>, acoords: Array1<usize>) -> Self {
        Self { array, nrise, ycoords, acoords }
    }

    fn zeros(shape: &[usize], nrize: isize, ycoords: Array1<usize>, acoords: Array1<usize>) -> Self {
        Self::new(Array2::<f32>::zeros((shape[0], shape[1])), nrize, ycoords, acoords)
    }

    fn new_like(&self, array: Array2<f32>) -> Self {
        Self::new(array, self.nrise, self.ycoords.clone(), self.acoords.clone())
    }

    fn zeros_like(&self) -> Self {
        Self::zeros(self.array.shape(), self.nrise, self.ycoords.clone(), self.acoords.clone())
    }

    pub fn from_indices_(
        nth: &ArrayView1<i32>,
        npf: &ArrayView1<i32>,
        values: &ArrayView1<f32>,
        nrise: isize,
    ) -> PyResult<Self> {
        let nsize = nth.len();
        if npf.len() != nsize || values.len() != nsize {
            return value_error!("nth, npf, and values must have the same length.");
        }

        let nth_map = unique_map(&nth);
        let npf_map = unique_map(&npf);

        let mut arr = Array2::<f32>::from_elem((nth_map.len(), npf_map.len()), f32::NAN);
        let nth_relabeled = nth.mapv(|x| nth_map[&x]);
        let npf_relabeled = npf.mapv(|x| npf_map[&x]);
        for i in 0..nsize {
            arr[[nth_relabeled[i], npf_relabeled[i]]] = values[[i]];
        }
        Ok(Self { array: arr.to_owned(), nrise, ycoords: nth_relabeled, acoords: npf_relabeled })
    }

    fn with_values_(&self, values: &ArrayView1<f32>) -> PyResult<Self> {
        let nsize = self.ycoords.len();
        if values.len() != nsize {
            return value_error!("values must have the same length as the array.");
        }
        let mut arr = self.array.clone();
        for i in 0..nsize {
            arr[[self.ycoords[i], self.acoords[i]]] = values[[i]];
        }
        Ok(Self { array: arr.to_owned(), nrise: self.nrise, ycoords: self.ycoords.clone(), acoords: self.acoords.clone() })
    }

    fn convolve_(&self, kernel: &Kernel<f32>) -> Array2<f32> {
        let (ny, na) = (self.array.shape()[0], self.array.shape()[1]);
        let mut out = Array2::<f32>::zeros((ny, na));
        for i in 0..ny {
            for j in 0..na {
                if self.array[[i, j]].is_nan() {
                    out[[i, j]] = f32::NAN;
                    continue;
                }
                let mut sum = 0.0;
                for ind in kernel.iter_indices() {
                    if kernel[ind] == 0.0 {
                        continue;
                    }
                    let value = self[[i as isize + ind[0], j as isize + ind[1]]];
                    if value.is_nan() {
                        continue;
                    }
                    sum += value * kernel[ind];
                }
                out[[i, j]] = sum;
            }
        }
        out
    }

    fn mean_filter_(&self, kernel: &Kernel<bool>) -> Array2<f32> {
        let (ny, na) = (self.array.shape()[0], self.array.shape()[1]);
        let mut out = Array2::<f32>::zeros((ny, na));
        for i in 0..ny {
            for j in 0..na {
                if self.array[[i, j]].is_nan() {
                    out[[i, j]] = f32::NAN;
                    continue;
                }
                let mut sum = 0.0;
                let mut count = 0.0;
                for ind in kernel.iter_indices() {
                    if !kernel[ind] {
                        continue;
                    }
                    let value = self[[i as isize + ind[0], j as isize + ind[1]]];
                    if value.is_nan() {
                        continue;
                    }
                    sum += value;
                    count += 1.0;
                }
                out[[i, j]] = if count > 0.0 {
                    sum / count
                } else {
                    f32::NAN
                };
            }
        }
        out
    }

    fn count_neighbors_(&self, kernel: &Kernel<bool>) -> Array2<f32> {
        let (ny, na) = (self.array.shape()[0], self.array.shape()[1]);
        let mut out = Array2::<f32>::zeros((ny, na));
        for i in 0..ny {
            for j in 0..na {
                if self[[i as isize, j as isize]].is_nan() {
                    continue;
                }
                let mut n_neighbors = 0.0;
                for ind in kernel.iter_indices() {
                    if !kernel[ind] {
                        continue
                    }
                    let value = self[[i as isize + ind[0], j as isize + ind[1]]];
                    if !value.is_nan() {
                        n_neighbors += 1.0;
                    }

                }
                out[[i, j]] = n_neighbors;
            }
        }
        out
    }

    fn max_filter_(&self, kernel: &Kernel<bool>) -> Array2<f32> {
        let (ny, na) = (self.array.shape()[0], self.array.shape()[1]);
        let mut out = Array2::<f32>::zeros((ny, na));
        for i in 0..ny {
            for j in 0..na {
                if self[[i as isize, j as isize]].is_nan() {
                    continue;
                }
                let mut max = f32::NEG_INFINITY;
                for ind in kernel.iter_indices() {
                    if !kernel[ind] {
                        continue
                    }
                    let value = self[[i as isize + ind[0], j as isize + ind[1]]];
                    if value.is_nan() {
                        continue;
                    }
                    if value > max {
                        max = value;
                    }
                }
                out[[i, j]] = max;
            }
        }
        out
    }

    fn min_filter_(&self, kernel: &Kernel<bool>) -> Array2<f32> {
        let (ny, na) = (self.array.shape()[0], self.array.shape()[1]);
        let mut out = Array2::<f32>::zeros((ny, na));
        for i in 0..ny {
            for j in 0..na {
                if self[[i as isize, j as isize]].is_nan() {
                    continue;
                }
                let mut min = f32::INFINITY;
                for ind in kernel.iter_indices() {
                    if !kernel[ind] {
                        continue
                    }
                    let value = self[[i as isize + ind[0], j as isize + ind[1]]];
                    if value.is_nan() {
                        continue;
                    }
                    if value < min {
                        min = value;
                    }
                }
                out[[i, j]] = min;
            }
        }
        out
    }

    fn median_filter_(&self, kernel: &Kernel<bool>) -> Array2<f32> {
        let (ny, na) = (self.array.shape()[0], self.array.shape()[1]);
        let mut out = Array2::<f32>::zeros((ny, na));
        for i in 0..ny {
            for j in 0..na {
                if self[[i as isize, j as isize]].is_nan() {
                    continue;
                }
                let mut values = Vec::new();
                for ind in kernel.iter_indices() {
                    if !kernel[ind] {
                        continue
                    }
                    let value = self[[i as isize + ind[0], j as isize + ind[1]]];
                    if value.is_nan() {
                        continue;
                    }
                    values.push(value);
                }
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = values.len();
                if n % 2 == 0 {
                    out[[i, j]] = (values[n / 2 - 1] + values[n / 2]) / 2.0;
                } else {
                    out[[i, j]] = values[n / 2];
                }
            }
        }
        out
    }

    fn norm_indices(&self, index: &[isize; 2]) -> (isize, isize) {
        let shape = self.array.shape();
        let nrise = self.nrise;
        let ny = shape[0] as isize;
        let na = shape[1] as isize;
        let (incr, i1n) = if index[1] < 0 {
            (-nrise, index[1] + na)
        } else if index[1] >= na {
            (nrise, index[1] - na)
        } else {
            (0, index[1])
        };
        if i1n < 0 || i1n >= na {
            panic!("index ({}, {}) out of bounds", index[0], index[1])
        }
        let i0n = index[0] + incr;
        if i0n >= ny {
            (-1, i1n)
        } else {
            (i0n, i1n)
        }
    }
}

impl core::ops::Index<[isize; 2]> for CylindricArray {
    type Output = f32;

    fn index(&self, index: [isize; 2]) -> &Self::Output {
        let (i0n, i1n) = self.norm_indices(&index);
        if i0n < 0 {
            return &f32::NAN;
        }
        &self.array[[i0n as usize, i1n as usize]]
    }
}

// impl set item
impl core::ops::IndexMut<[isize; 2]> for CylindricArray {
    fn index_mut(&mut self, index: [isize; 2]) -> &mut Self::Output {
        let (i0n, i1n) = self.norm_indices(&index);
        if i0n < 0 {
            panic!("index ({}, {}) out of bounds", index[0], index[1])
        }
        &mut self.array[[i0n as usize, i1n as usize]]
    }
}

/// Construct a hashmap from an array.
/// For example, unique_map of [1, 3, 6, 3] will return {1: 0, 3: 1, 6: 2}
pub fn unique_map<_D>(ar: &ArrayView1<_D>) -> HashMap<_D, usize> where _D: std::cmp::Eq + Copy + Hash + Ord {
    let mut uniques = HashMap::new();
    let mut count = 0;
    for i in 0..ar.shape()[0] {
        let val = ar[[i]];
        if uniques.contains_key(&val) {
            continue;
        }
        uniques.insert(val, count);
        count += 1;
    }
    uniques
}
