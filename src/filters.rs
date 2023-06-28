use pyo3::{prelude::*, Python};
use numpy::{
    IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    ndarray::Array2,
};
use crate::array::indices_to_array;

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
        (i0_min..=i0_max).flat_map(move |i0| (i1_min..=i1_max).map(move |i1| [i0, i1]))
    }
}

// implement get item
impl<_D> core::ops::Index<[isize; 2]> for Kernel<_D> {
    type Output = _D;

    fn index(&self, index: [isize; 2]) -> &Self::Output {
        let i0 = index[0] as usize + self.center.0;
        let i1 = index[1] as usize + self.center.1;
        &self.kernel[[i0, i1]]
    }
}

#[pyclass]
pub struct CylindricArray {
    array: Array2<f32>,
    nrise: isize,
}

#[pymethods]
impl CylindricArray {
    #[new]
    pub fn from_indices(
        nth: PyReadonlyArray1<i32>,
        npf: PyReadonlyArray1<i32>,
        values: PyReadonlyArray1<f32>,
        nrise: isize,
    ) -> PyResult<Self> {
        let arr = indices_to_array(nth.as_array(), npf.as_array(), values.as_array())?;
        Ok(Self { array: arr.to_owned(), nrise })
    }

    pub fn asarray(&self, py: Python) -> Py<PyArray2<f32>> {
        self.array.clone().into_pyarray(py).to_owned()
    }

    pub fn convolve<'py>(
        &self,
        weight: PyReadonlyArray2<f32>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(weight.as_array().to_owned());
        let out = self.convolve_(&kernel);
        Ok(Self::new(out, self.nrise))
    }

    pub fn max_filter<'py>(
        &self,
        footprint: PyReadonlyArray2<bool>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(footprint.as_array().to_owned());
        let out = self.max_filter_(&kernel);
        Ok(Self::new(out, self.nrise))
    }

    pub fn min_filter<'py>(
        &self,
        footprint: PyReadonlyArray2<bool>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(footprint.as_array().to_owned());
        let out = self.min_filter_(&kernel);
        Ok(Self::new(out, self.nrise))
    }

    pub fn median_filter<'py>(
        &self,
        footprint: PyReadonlyArray2<bool>,
    ) -> PyResult<Self> {
        let kernel = Kernel::new(footprint.as_array().to_owned());
        let out = self.median_filter_(&kernel);
        Ok(Self::new(out, self.nrise))
    }

}

impl CylindricArray {
    fn new(array: Array2<f32>, nrise: isize) -> Self {
        Self { array, nrise }
    }

    fn convolve_(&self, kernel: &Kernel<f32>) -> Array2<f32> {
        let (ny, na) = (self.array.shape()[0], self.array.shape()[1]);
        let mut out = Array2::<f32>::zeros((ny, na));
        for i in 0..ny {
            for j in 0..na {
                if self[[i as isize, j as isize]].is_nan() {
                    out[[i, j]] = f32::NAN;
                    continue;
                }
                let mut sum = 0.0;
                for ind in kernel.iter_indices() {
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
}

impl core::ops::Index<[isize; 2]> for CylindricArray {
    type Output = f32;

    fn index(&self, index: [isize; 2]) -> &Self::Output {
        let ny = self.array.shape()[0] as isize;
        let na = self.array.shape()[1] as isize;
        let (incr, i1n) = if index[1] < 0 {
            (-1, index[1] + na)
        } else if index[1] >= na {
            (1, index[1] - na)
        } else {
            (0, index[1])
        };
        let i0n = index[0] + incr * self.nrise;
        if i0n < 0 || i0n >= ny {
            return &f32::NAN;
        }
        &self.array[[i0n as usize, i1n as usize]]
    }
}
