use std::ops;
use numpy::ndarray::Array1;

pub struct HashMap1D<V> {
    arrays: Array1<Option<V>>,
    len: usize,
}

impl<V> HashMap1D<V> {
    pub fn new() -> Self {
        Self {
            arrays: Array1::default(0),
            len: 0,
        }
    }

    pub fn from_shape(n: usize) -> Self {
        Self {
            arrays: Array1::default(n),
            len: 0,
        }
    }

    pub fn insert(&mut self, index: usize, value: V) {
        self.arrays[index] = Some(value);
        self.len += 1;
    }

    pub fn get(&self, index: isize) -> &Option<V> {
        let n = self.arrays.shape()[0] as isize;
        if index < 0 || index >= n {
            return &None;
        }
        &self.arrays[index as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item=(usize, &V)> {
        self.arrays.indexed_iter().filter_map(|(index, value)| {
            value.as_ref().map(|v| (index, v))
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.arrays.shape()[0], self.arrays.shape()[1])
    }
}

impl<V> ops::Index<isize> for HashMap1D<V> {
    type Output = V;

    fn index(&self, index: isize) -> &Self::Output {
        self.get(index).as_ref().unwrap()
    }
}

impl<V> ops::IndexMut<isize> for HashMap1D<V> {
    fn index_mut(&mut self, index: isize) -> &mut Self::Output {
        self.arrays[index as usize].as_mut().unwrap()
    }
}
