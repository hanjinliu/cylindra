use std::collections::HashSet;

use pyo3::prelude::*;
use crate::index_error;

#[pyclass]
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
/// Indices of a molecule on a cylinder lattice.
pub struct Index {
    pub y: isize,
    pub a: isize,
}

#[pymethods]
impl Index {
    #[new]
    pub fn new(y: isize, a: isize) -> Self {
        Self { y, a }
    }

    /// Check if the index is valid for the given geometry.
    pub fn is_valid(&self, ny: isize, na: isize) -> bool {
        self.y >= 0 && self.y < ny && self.a >= 0 && self.a < na
    }

    /// Python __eq__ operator (not implemented in PyO3 yet).
    pub fn __eq__(&self, other: (isize, isize)) -> bool {
        self.y == other.0 && self.a == other.1
    }

    /// Python __repr__ method.
    pub fn __repr__(&self) -> String {
        format!("Index(y={}, a={})", self.y, self.a)
    }
}

impl Index {
    pub fn as_tuple_usize(&self) -> (usize, usize) {
        (self.y as usize, self.a as usize)
    }

    pub fn get_neighbors(&self, na: isize, nrise: isize) -> Neighbors {
        let y_fw = Index::new(self.y + 1, self.a);
        let y_bw = Index::new(self.y - 1, self.a);
        let a_fw = if self.a < na - 1 {
            Index::new(self.y, self.a + 1)
        } else {
            Index::new(self.y + nrise, 0)
        };
        let a_bw = if self.a > 0 {
            Index::new(self.y, self.a - 1)
        } else {
            Index::new(self.y - nrise, na - 1)
        };
        Neighbors::new(y_fw, y_bw, a_fw, a_bw)
    }
}

pub struct Neighbors {
    pub y_fw: Option<Index>,
    pub y_bw: Option<Index>,
    pub a_fw: Option<Index>,
    pub a_bw: Option<Index>,
}

impl Neighbors {
    pub fn new(y_fw: Index, y_bw: Index, a_fw: Index, a_bw: Index) -> Self {
        Self {
            y_fw: Some(y_fw),
            y_bw: Some(y_bw),
            a_fw: Some(a_fw),
            a_bw: Some(a_bw),
        }
    }

    pub fn new_none() -> Self {
        Self {
            y_fw: None,
            y_bw: None,
            a_fw: None,
            a_bw: None,
        }
    }

    pub fn y_iter(&self) -> impl Iterator<Item=Index> {
        let mut vec = Vec::new();
        if self.y_fw.is_some() {
            vec.push(self.y_fw.clone().unwrap());
        }
        if self.y_bw.is_some() {
            vec.push(self.y_bw.clone().unwrap());
        }
        vec.into_iter()
    }

    pub fn a_iter(&self) -> impl Iterator<Item=Index> {
        let mut vec = Vec::new();
        if self.a_fw.is_some() {
            vec.push(self.a_fw.clone().unwrap());
        }
        if self.a_bw.is_some() {
            vec.push(self.a_bw.clone().unwrap());
        }
        vec.into_iter()
    }

    pub fn iter(&self) -> impl Iterator<Item=Index> {
        let mut vec = Vec::new();
        if self.y_fw.is_some() {
            vec.push(self.y_fw.clone().unwrap());
        }
        if self.y_bw.is_some() {
            vec.push(self.y_bw.clone().unwrap());
        }
        if self.a_fw.is_some() {
            vec.push(self.a_fw.clone().unwrap());
        }
        if self.a_bw.is_some() {
            vec.push(self.a_bw.clone().unwrap());
        }
        vec.into_iter()
    }

    pub fn y_pair(&self) -> Vec<Index> {
        match (&self.y_fw, &self.y_bw) {
            (Some(y_fw), Some(y_bw)) => vec![y_fw.clone(), y_bw.clone()],
            _ => Vec::new(),
        }
    }

    pub fn a_pair(&self) -> Vec<Index> {
        match (&self.a_fw, &self.a_bw) {
            (Some(a_fw), Some(a_bw)) => vec![a_fw.clone(), a_bw.clone()],
            _ => Vec::new(),
        }
    }
}

#[pyclass]
#[derive(Clone, PartialEq, Eq)]
/// A struct represents cylinder geometry with rise.
/// nrise is the number of increase in `y` when `a` increases by `na`.
pub struct CylinderGeometry {
    pub ny: isize,
    pub na: isize,
    pub nrise: isize,
}

#[pymethods]
impl CylinderGeometry {
    #[new]
    #[pyo3(signature = (ny, na, nrise))]
    pub fn new(ny: isize, na: isize, nrise: isize) -> Self {
        Self { ny, na, nrise }
    }

    pub fn __repr__(&self) -> String {
        format!("CylinderGeometry(ny={}, na={}, nrise={})", self.ny, self.na, self.nrise)
    }

    pub fn count(&self) -> isize {
        self.ny * self.na
    }

    #[pyo3(signature = (y, a))]
    /// Get an Index struct at the given position.
    /// `a` can be negative or greater than `na`. In this case, `y` is adjusted
    /// accordingly to keep the index valid.
    pub fn get_index(&self, y: isize, a: isize) -> PyResult<Index> {
        let mut y = y;
        let mut a = a;
        while a >= self.na {
            a -= self.na;
            y += self.nrise;
        }
        while a < 0 {
            a += self.na;
            y -= self.nrise;
        }

        if y < 0 || self.ny <= y {
            return index_error!(
                format!(
                    "Index(y={}, a={}) out of bounds for {}.",
                    y, a, self.__repr__()
                )
            );
        }
        Ok(Index::new(y, a))
    }
}

impl CylinderGeometry {
    /// Get the index of the neighbor at the given position.
    pub fn get_neighbor(&self, y: isize, a: isize) -> PyResult<Neighbors> {
        let mut neighbors = Neighbors::new_none();

        if y < 0 || self.ny <= y || a < 0 || self.na <= a {
            return index_error!(
                format!(
                    "Index(y={}, a={}) out of bounds for {}.",
                    y, a, self.__repr__()
                )
            );
        }

        if y > 0 {
            match self.get_index(y - 1, a) {
                Ok(index) => {
                    neighbors.y_bw = Some(index);
                }
                Err(_) => (),
            }
        }

        if y < self.ny - 1 {
            match self.get_index(y + 1, a) {
                Ok(index) => {
                    neighbors.y_fw = Some(index);
                }
                Err(_) => (),
            }
        }

        let index_l = if a > 0 {
            self.get_index(y, a - 1)
        } else {
            self.get_index(y - self.nrise, self.na - 1)
        };
        match index_l {
            Ok(index) => {
                neighbors.a_bw = Some(index);
            }
            Err(_) => (),
        }

        let index_r = if a < self.na - 1 {
            self.get_index(y, a + 1)
        } else {
            self.get_index(y + self.nrise, 0)
        };
        match index_r {
            Ok(index) => {
                neighbors.a_fw = Some(index);
            }
            Err(_) => (),
        }
        Ok(neighbors)
    }

    pub fn get_neighbors(&self, indices: &HashSet<Index>) -> PyResult<HashSet<Index>> {
        let mut unique_neighbors: HashSet<Index> = HashSet::new();
        // add all the neighbor candidates
        for index in indices.iter() {
            for neighbor in self.get_neighbor(index.y, index.a)?.iter() {
                unique_neighbors.insert(neighbor);
            }
        }

        // remove inputs
        for index in indices.iter() {
            unique_neighbors.remove(index);
        }

        Ok(unique_neighbors)
    }
}
