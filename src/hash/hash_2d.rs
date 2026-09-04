use std::ops;

#[derive(Clone)]
pub struct HashMap2D<V> {
    arrays: Vec<Option<V>>,
    shape0: usize,
    shape1: usize,
    len: usize,
}

impl<V> HashMap2D<V> {
    pub fn new() -> Self {
        Self {
            arrays: Vec::new(),
            shape0: 0,
            shape1: 0,
            len: 0,
        }
    }

    pub fn from_shape(n0: usize, n1: usize) -> Self {
        let mut arrays = Vec::with_capacity(n0 * n1);
        for _ in 0..(n0 * n1) {
            arrays.push(None);
        }
        Self {
            arrays,
            shape0: n0,
            shape1: n1,
            len: 0,
        }
    }

    pub fn insert(&mut self, index: (usize, usize), value: V) {
        let ith = index.0 * self.shape1 + index.1;
        self.arrays[ith] = Some(value);
        self.len += 1;
    }

    pub fn get(&self, index: (isize, isize)) -> &Option<V> {
        let n0 = self.shape0 as isize;
        let n1 = self.shape1 as isize;
        if index.0 < 0 || index.1 < 0 || index.0 >= n0 || index.1 >= n1 {
            return &None;
        }
        &self.arrays[index.0 as usize * self.shape1 + index.1 as usize]
    }

    pub fn iter(&self) -> impl Iterator<Item=((usize, usize), &V)> {
        self.arrays.iter().enumerate().filter_map(|(ith, value)| {
            value.as_ref().map(|v| {
                ((ith / self.shape1, ith % self.shape1), v)
            })
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.shape0, self.shape1)
    }
}

impl<V> ops::Index<(isize, isize)> for HashMap2D<V> {
    type Output = V;

    fn index(&self, index: (isize, isize)) -> &Self::Output {
        let ith = index.0 as usize * self.shape1 + index.1 as usize;
        self.arrays[ith].as_ref().unwrap()
    }
}

impl<V> ops::Index<(usize, usize)> for HashMap2D<V> {
    type Output = V;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let ith = index.0 * self.shape1 + index.1;
        self.arrays[ith].as_ref().unwrap()
    }
}

impl<V> ops::IndexMut<(isize, isize)> for HashMap2D<V> {
    fn index_mut(&mut self, index: (isize, isize)) -> &mut Self::Output {
        let ith = index.0 as usize * self.shape1 + index.1 as usize;
        self.arrays[ith].as_mut().unwrap()
    }
}
