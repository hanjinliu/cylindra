use std::collections::{HashMap, HashSet};

use pyo3::{prelude::*, Python, Py};
use numpy::{PyArray1, PyReadonlyArray2, ndarray::{Array1, Array2}, IntoPyArray};
use crate::value_error;

struct Region {
    indices: Vec<(usize, usize)>,
    per: usize,
    nrise: isize,
}

impl Region {
    fn area(&self, _: &Array2<f32>) -> usize {
        self.indices.len()
    }

    fn length(&self, _: &Array2<f32>) -> usize {
        let mut xs: HashMap<usize, HashSet<usize>> = HashMap::new();
        for &(y, x) in self.indices.iter() {
            if xs.contains_key(&x) {
                xs.get_mut(&x).unwrap().insert(y);
            } else {
                xs.insert(x, HashSet::from([y]));
            }
        }

        let mut xv = Vec::new();
        for &x in xs.keys() {
            xv.push(x);
        }
        xv.sort();

        if xv[0] > 0 || xv[xv.len() - 1] < self.per - 1 || xv.len() == self.per {
            let mut ymin = usize::MAX;
            let mut ymax = usize::MIN;
            for value in xs.values() {
                let cur_ymin = *value.iter().min().unwrap();
                let cur_ymax = *value.iter().max().unwrap();
                if cur_ymin < ymin {
                    ymin = cur_ymin;
                }
                if cur_ymax > ymax {
                    ymax = cur_ymax;
                }
            }
            ymax - ymin + 1
        } else {
            // label region is continuous at the border
            // oo......
            // oo....oo
            // ......oo

            // find the position of this
            //        v
            // ooo....ooo
            let mut start = 0;
            for idx in 0..self.per {
                if xs.contains_key(&idx) {
                    continue;
                }
                for _idx in (idx + 1)..self.per {
                    if xs.contains_key(&_idx) {
                        start = _idx;
                        break;
                    }
                }
            }
            let mut ymin = isize::MAX;
            let mut ymax = isize::MIN;
            for (key, value) in xs {
                let mut cur_ymin = *value.iter().min().unwrap() as isize;
                let mut cur_ymax = *value.iter().max().unwrap() as isize;
                if key < start {
                    cur_ymin += self.nrise;
                    cur_ymax += self.nrise;
                }
                if cur_ymin < ymin {
                    ymin = cur_ymin;
                }
                if cur_ymax > ymax {
                    ymax = cur_ymax;
                }
            }
            (ymax - ymin + 1) as usize
        }

    }

    fn width(&self, _: &Array2<f32>) -> usize {
        let mut appeared = HashSet::new();
        for &(_, x) in self.indices.iter() {
            appeared.insert(x);
        }
        appeared.len()
    }

    fn intensity_sum(&self, image: &Array2<f32>) -> f32 {
        let mut sum = 0.0;
        for &(y, x) in self.indices.iter() {
            sum += image[[y, x]];
        }
        sum
    }

    fn intensity_mean(&self, image: &Array2<f32>) -> f32 {
        self.intensity_sum(image) / self.area(image) as f32
    }

    fn intensity_median(&self, image: &Array2<f32>) -> f32 {
        let mut values = Vec::new();
        for &(y, x) in self.indices.iter() {
            values.push(image[[y, x]]);
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = values.len();
        if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        }
    }

    fn intensity_max(&self, image: &Array2<f32>) -> f32 {
        let mut max = f32::MIN;
        for &(y, x) in self.indices.iter() {
            let value = image[[y, x]];
            if value > max {
                max = value;
            }
        }
        max
    }

    fn intensity_min(&self, image: &Array2<f32>) -> f32 {
        let mut min = f32::MAX;
        for &(y, x) in self.indices.iter() {
            let value = image[[y, x]];
            if value < min {
                min = value;
            }
        }
        min
    }

    fn intensity_std(&self, image: &Array2<f32>) -> f32 {
        let mean = self.intensity_mean(image);
        let mut sum = 0.0;
        for &(y, x) in self.indices.iter() {
            let value = image[[y, x]];
            sum += (value - mean).powi(2);
        }
        (sum / self.area(image) as f32).sqrt()
    }
}

#[pyclass]
pub struct RegionProfiler {
    image: Array2<f32>,
    labels: Vec<Region>
}

impl RegionProfiler {
    fn area(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for region in self.labels.iter() {
            out.push(region.area(&self.image));
        }
        out
    }

    fn intensity_sum(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for region in self.labels.iter() {
            out.push(region.intensity_sum(&self.image));
        }
        out
    }

    fn intensity_mean(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for region in self.labels.iter() {
            out.push(region.intensity_mean(&self.image));
        }
        out
    }

    fn intensity_median(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for region in self.labels.iter() {
            out.push(region.intensity_median(&self.image));
        }
        out
    }

    fn intensity_max(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for region in self.labels.iter() {
            out.push(region.intensity_max(&self.image));
        }
        out
    }

    fn intensity_min(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for region in self.labels.iter() {
            out.push(region.intensity_min(&self.image));
        }
        out
    }

    fn intensity_std(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for region in self.labels.iter() {
            out.push(region.intensity_std(&self.image));
        }
        out
    }
}

#[pymethods]
impl RegionProfiler {
    #[new]
    fn new(
        image: PyReadonlyArray2<f32>,
        label_image: PyReadonlyArray2<u32>,
        nrise: isize,
    ) -> Self {
        let image = image.as_array().to_owned();
        let label_image = label_image.as_array();
        let mut indices_map: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
        let per = image.shape()[1];
        let (ny, nx) = label_image.dim();
        for y in 0..ny {
            for x in 0..nx {
                let v = label_image[[y, x]];
                if v == 0 {
                    // zero is background
                    continue;
                }
                if !indices_map.contains_key(&v) {
                    indices_map.insert(v, Vec::new());
                }
                indices_map.get_mut(&v).unwrap().push((y as u32, x as u32));
            }
        }
        let mut keys_sorted: Vec<&u32> = indices_map.keys().collect();
        keys_sorted.sort();
        let mut labels = Vec::new();
        for key in keys_sorted {
            let mut indices = Vec::new();
            for val in indices_map.get(key).unwrap().iter() {
                indices.push((val.0 as usize, val.1 as usize));
            }
            labels.push(Region { indices, per, nrise});
        }
        Self { image, labels }
    }

    fn calculate<'py>(&self, py: Python<'py>, props: Vec<String>) -> PyResult<HashMap<String, Py<PyArray1<f32>>>> {
        let mut out = HashMap::new();
        for prop in props {
            let vec = match prop.as_str() {
                "area" => self.area().iter().map(|&x| x as f32).collect(),
                "length" => self.labels.iter().map(|x| x.length(&self.image) as f32).collect(),
                "width" => self.labels.iter().map(|x| x.width(&self.image) as f32).collect(),
                "sum" => self.intensity_sum(),
                "mean" => self.intensity_mean(),
                "median" => self.intensity_median(),
                "max" => self.intensity_max(),
                "min" => self.intensity_min(),
                "std" => self.intensity_std(),
                _ => return value_error!(format!("Unknown property: {}", prop)),
            };
            out.insert(prop, Array1::from(vec).into_pyarray(py).to_owned());
        }
        Ok(out)
    }
}
