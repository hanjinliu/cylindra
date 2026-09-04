use std::ops;

pub struct HashMap1D<V> {
    arrays: Vec<Option<V>>,
    len: usize,
}

impl<V> HashMap1D<V> {
    pub fn new() -> Self {
        Self {
            arrays: Vec::new(),
            len: 0,
        }
    }

    pub fn from_shape(n: usize) -> Self {
        let mut arrays = Vec::with_capacity(n);
        for _ in 0..n {
            arrays.push(None);
        }
        Self {
            arrays,
            len: 0,
        }
    }

    pub fn insert(&mut self, index: usize, value: V) {
        self.arrays[index] = Some(value);
        self.len += 1;
    }

    pub fn get(&self, index: isize) -> &Option<V> {
        let n = self.arrays.len() as isize;
        if index < 0 || index >= n {
            return &None;
        }
        &self.arrays[index as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item=(usize, &V)> {
        self.arrays.iter().enumerate().filter_map(|(index, value)| {
            value.as_ref().map(|v| (index, v))
        })
    }

    pub fn len(&self) -> usize {
        self.len
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
