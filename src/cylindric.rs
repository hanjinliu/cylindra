use std::collections::HashSet;

use pyo3::prelude::*;
use crate::index_error;

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
#[derive(Clone, PartialEq, Eq)]
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

    #[pyo3(signature = (y, a))]
    pub fn get_neighbor(&self, y: isize, a: isize) -> PyResult<Vec<Index>> {
        let mut neighbors: Vec<Index> = Vec::new();

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
                Ok(index) => neighbors.push(index),
                Err(_) => (),
            }
        }

        if y < self.ny - 1 {
            match self.get_index(y + 1, a) {
                Ok(index) => neighbors.push(index),
                Err(_) => (),
            }
        }

        let index_l = if a > 0 {
            self.get_index(y, a - 1)
        } else {
            self.get_index(y - self.nrise, self.na - 1)
        };
        match index_l {
            Ok(index) => neighbors.push(index),
            Err(_) => (),
        }

        let index_r = if a < self.na - 1 {
            self.get_index(y, a + 1)
        } else {
            self.get_index(y + self.nrise, 0)
        };
        match index_r {
            Ok(index) => neighbors.push(index),
            Err(_) => (),
        }
        Ok(neighbors)
    }

    #[pyo3(signature = (indices))]
    /// Return the neighbors of the given indices.
    pub fn get_neighbors(&self, indices: Vec<(isize, isize)>) -> PyResult<Vec<Index>> {
        let mut inds = Vec::new();
        for (y, a) in indices {
            let index = self.get_index(y, a)?;
            inds.push(index);
        }
        self._get_neighbors(&inds)
    }

    /// Return all the pairs of indices that are connected longitudinally.
    pub fn all_longitudinal_pairs(&self) -> Vec<(Index, Index)> {
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
        pairs
    }

    /// Return all the pairs of indices that are connected laterally.
    pub fn all_lateral_pairs(&self) -> Vec<(Index, Index)> {
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
        pairs
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
                    Sources::from_lon(Index { y: y - 1, a })
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
                    Sources::from_lon(Index { y: y - 1, a })
                }
            }
        }
    }
}

#[pyclass]
#[derive(PartialEq, Eq)]
pub struct Sources {
    lon: Option<Index>,
    lat: Option<Index>,
}

impl Sources {
    fn new(lon: Index, lat: Index) -> Self {
        let lon = if lon.y < 0 { None } else { Some(lon) };
        let lat = if lat.y < 0 { None } else { Some(lat) };
        Self { lon, lat }
    }

    fn from_lon(lon: Index) -> Self {
        let lon = if lon.y < 0 { None } else { Some(lon) };
        Self { lon, lat: None }
    }
}

#[pymethods]
impl Sources {
    pub fn __repr__(&self) -> String {
        match (&self.lon, &self.lat) {
            (Some(lon), Some(lat)) => format!("Sources(lon={}, lat={})", lon.__repr__(), lat.__repr__()),
            (Some(lon), None) => format!("Sources(lon={}, lat=None)", lon.__repr__()),
            (None, Some(lat)) => format!("Sources(lon=None, lat={})", lat.__repr__()),
            (None, None) => format!("Sources(lon=None, lat=None)"),
        }
    }

    pub fn equals(&self, lon: Option<(isize, isize)>, lat: Option<(isize, isize)>) -> bool {
        let _lon = match lon {
            Some(lon) => Index::new(lon.0, lon.1),
            None => Index::new(-1, -1),
        };
        let _lat = match lat {
            Some(lat) => Index::new(lat.0, lat.1),
            None => Index::new(-1, -1),
        };
        let other = Self::new(_lon, _lat);
        self == &other
    }
}

impl CylinderGeometry {
    pub fn _get_neighbors(&self, indices: &Vec<Index>) -> PyResult<Vec<Index>> {
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
}
