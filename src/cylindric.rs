use std::collections::HashSet;

use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
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

    pub fn is_valid(&self, ny: isize, na: isize) -> bool {
        self.y >= 0 && self.y < ny && self.a >= 0 && self.a < na
    }

    pub fn __eq__(&self, other: (isize, isize)) -> bool {
        self.y == other.0 && self.a == other.1
    }

    pub fn __repr__(&self) -> String {
        format!("Index(y={}, a={})", self.y, self.a)
    }
}

#[pyclass]
pub struct CylinderGeometry {
    ny: isize,
    na: isize,
    nrise: isize,
}

#[pymethods]
impl CylinderGeometry {
    #[new]
    pub fn new(ny: isize, na: isize, nrise: isize) -> Self {
        Self { ny, na, nrise }
    }

    pub fn count(&self) -> isize {
        self.ny * self.na
    }

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

        if y < 0 || y >= self.ny {
            return Err(
                pyo3::exceptions::PyValueError::new_err(
                    format!("Index ({}, {}) out of bounds.", y, a)
                )
            );
        }
        Ok(Index{ y, a })
    }

    pub fn get_neighbor(&self, y: isize, a: isize) -> PyResult<Vec<Index>> {
        let mut neighbors: Vec<Index> = Vec::new();

        if y > 0 {
            let index = self.get_index(y - 1, a)?;
            if index.is_valid(self.ny, self.na) {
                neighbors.push(index);
            }
        }

        if y < self.ny - 1 {
            let index = self.get_index(y + 1, a)?;
            if index.is_valid(self.ny, self.na) {
                neighbors.push(index);
            }
        }

        let index_l = if a > 0 {
            self.get_index(y, a - 1)
        } else {
            self.get_index(y - self.nrise, self.na - 1)
        }?;

        if index_l.is_valid(self.ny, self.na) {
            neighbors.push(index_l);
        }

        let index_r = if a < self.na - 1 {
            self.get_index(y, a + 1)
        } else {
            self.get_index(y + self.nrise, 0)
        }?;

        if index_r.is_valid(self.ny, self.na) {
            neighbors.push(index_r);
        }
        Ok(neighbors)
    }

    pub fn get_neighbors(&self, indices: Vec<Index>) -> PyResult<Vec<Index>> {
        self.get_neighbors_(&indices)
    }

    /// Return all the pairs of indices that are connected longitudinally.
    pub fn all_longitudinal_pairs(&self) -> PyResult<Vec<(Index, Index)>> {
        let mut pairs: Vec<(Index, Index)> = Vec::new();
        for y in 0..self.ny {
            for a in 0..self.na {
                let idx1 = Index{ y, a };
                let sources = self.source_forward(y, a);
                match sources.lon {
                    Some(idx0) => pairs.push((idx0, idx1)),
                    None => (),
                }
            }
        }
        Ok(pairs)
    }

    /// Return all the pairs of indices that are connected laterally.
    pub fn all_lateral_pairs(&self) -> PyResult<Vec<(Index, Index)>> {
        let mut pairs: Vec<(Index, Index)> = Vec::new();
        for y in 0..self.ny {
            for a in 0..self.na {
                let idx1 = Index{ y, a };
                let sources = self.source_forward(y, a);
                match sources.lat {
                    Some(idx0) => pairs.push((idx0, idx1)),
                    None => (),
                }
            }
        }
        Ok(pairs)
    }
}

struct Sources {
    lon: Option<Index>,
    lat: Option<Index>,
}

impl Sources {
    fn new(lon: Index, lat: Index) -> Self {
        Self { lon: Some(lon), lat: Some(lat)}
    }
}

impl CylinderGeometry {
    pub fn get_neighbors_(&self, indices: &Vec<Index>) -> PyResult<Vec<Index>> {
        let mut unique_neighbors: HashSet<Index> = HashSet::new();
        // add all the neighbor candidates
        for index in indices.iter() {
            let new_neighbors = self.get_neighbor(index.y, index.a)?;
            for neighbor in new_neighbors {
                unique_neighbors.insert(neighbor);
            }
        }

        // remove inputs
        for index in indices.iter() {
            unique_neighbors.remove(index);
        }

        // convert to a vector
        let mut neighbors: Vec<Index> = Vec::new();
        for neighbor in unique_neighbors {
            neighbors.push(neighbor);
        }
        Ok(neighbors)
    }

    fn source_forward(&self, y: isize, a: isize) -> Sources {
        if self.nrise >= 0 {
            if a > 0 {
                Sources::new(Index { y: y - 1, a }, Index { y, a: a - 1 })
            } else {
                let y0 = y - self.nrise;
                if y0 >= 0 {
                    Sources::new(Index { y: y - 1, a }, Index { y: y0, a: self.na - 1 })
                } else {
                    Sources {
                        lon: Some(Index { y: y - 1, a }),
                        lat: None,
                    }
                }
            }
        } else {
            if a < self.na - 1 {
                Sources::new(Index { y: y - 1, a }, Index { y, a: a + 1 })
            } else {
                let y0 = y + self.nrise;
                if y0 >= 0 {
                    Sources::new(Index { y: y - 1, a }, Index { y: y0, a: 0 })
                } else {
                    Sources {
                        lon: Some(Index { y: y - 1, a }),
                        lat: None,
                    }
                }
            }
        }
    }
}
