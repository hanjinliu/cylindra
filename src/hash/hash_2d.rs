use std::ops;
use numpy::ndarray::Array2;

pub struct HashMap2D<V> {
    arrays: Array2<Option<V>>,
    len: usize,
}

impl<V> HashMap2D<V> {
    pub fn new() -> Self {
        Self {
            arrays: Array2::default((0, 0)),
            len: 0,
        }
    }

    pub fn from_shape(n0: usize, n1: usize) -> Self {
        Self {
            arrays: Array2::default((n0, n1)),
            len: 0,
        }
    }

    pub fn insert(&mut self, index: (usize, usize), value: V) {
        self.arrays[index] = Some(value);
        self.len += 1;
    }

    pub fn get(&self, index: (isize, isize)) -> &Option<V> {
        let n0 = self.arrays.shape()[0] as isize;
        let n1 = self.arrays.shape()[1] as isize;
        if index.0 < 0 || index.1 < 0 || index.0 >= n0 || index.1 >= n1 {
            return &None;
        }
        &self.arrays[[index.0 as usize, index.1 as usize]]
    }

    pub fn iter(&self) -> impl Iterator<Item=((usize, usize), &V)> {
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

impl<V> ops::Index<(isize, isize)> for HashMap2D<V> {
    type Output = V;

    fn index(&self, index: (isize, isize)) -> &Self::Output {
        self.get(index).as_ref().unwrap()
    }
}

impl<V> ops::IndexMut<(isize, isize)> for HashMap2D<V> {
    fn index_mut(&mut self, index: (isize, isize)) -> &mut Self::Output {
        self.arrays[[index.0 as usize, index.1 as usize]].as_mut().unwrap()
    }
}
