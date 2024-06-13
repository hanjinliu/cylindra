use pyo3::{prelude::*, Python};
use numpy::{
    ndarray::{s, ArcArray, Array, Array2, Ix4}, IntoPyArray, Ix3, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4
};
use crate::{
    coordinates::{Vector3D, CoordinateSystem},
    value_error,
    index_error
};
use super::constraint::{Constraint, AngleConstraint, CheckResult};

#[pyclass]
pub struct ViterbiGrid {
    pub score: ArcArray<f32, Ix4>,
    pub coords: Vec<CoordinateSystem<f32>>,
    pub nmole: usize,
    pub nz: usize,
    pub ny: usize,
    pub nx: usize,
}

#[pymethods]
impl ViterbiGrid {
    #[new]
    #[pyo3(signature = (score_array, origin, zvec, yvec, xvec))]
    pub fn new(
        score_array: PyReadonlyArray4<f32>,
        origin: PyReadonlyArray2<f32>,
        zvec: PyReadonlyArray2<f32>,
        yvec: PyReadonlyArray2<f32>,
        xvec: PyReadonlyArray2<f32>,
    ) -> PyResult<Self> {
        let score = score_array.as_array();
        let origin = origin.as_array();
        let zvec = zvec.as_array();
        let yvec = yvec.as_array();
        let xvec = xvec.as_array();

        let score_shape = score.shape();
        let nmole = score_shape[0];
        let nz = score_shape[1];
        let ny = score_shape[2];
        let nx = score_shape[3];

        if origin.shape() != [nmole, 3] {
            return value_error!(
                format!("Shape of 'origin' must be (N, 3) but got {:?}.", origin.shape())
            );
        } else if zvec.shape() != [nmole, 3] {
            return value_error!(
                format!("Shape of 'zvec' must be (N, 3). but got {:?}", zvec.shape())
            );
        } else if yvec.shape() != [nmole, 3] {
            return value_error!(
                format!("Shape of 'yvec' must be (N, 3). but got {:?}", yvec.shape())
            );
        } else if xvec.shape() != [nmole, 3] {
            return value_error!(
                format!("Shape of 'xvec' must be (N, 3). but got {:?}", xvec.shape())
            );
        }

        let mut coords: Vec<CoordinateSystem<f32>> = Vec::new();

        for t in 0..nmole {
            let _ori = Vector3D::new(origin[[t, 0]], origin[[t, 1]], origin[[t, 2]]);
            let _ez = Vector3D::new(zvec[[t, 0]], zvec[[t, 1]], zvec[[t, 2]]);
            let _ey = Vector3D::new(yvec[[t, 0]], yvec[[t, 1]], yvec[[t, 2]]);
            let _ex = Vector3D::new(xvec[[t, 0]], xvec[[t, 1]], xvec[[t, 2]]);
            coords.push(CoordinateSystem::new(_ori, _ez, _ey, _ex));
        }

        Ok(ViterbiGrid { score: score.to_shared(), coords, nmole, nz, ny, nx })
    }

    pub fn __repr__(&self) -> String {
        format!("ViterbiGrid(nmole={}, nz={}, ny={}, nx={})", self.nmole, self.nz, self.ny, self.nx)
    }

    #[pyo3(signature = (n, z, y, x))]
    pub fn world_pos<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        z: usize,
        y: usize,
        x: usize,
    ) -> PyResult<Py<PyArray1<f32>>> {
        if n >= self.nmole {
            return index_error!(
                format!("Index out of range: n={}, nmole={}", n, self.nmole)
            );
        } else if z >= self.nz {
            return index_error!(
                format!("Index out of range: z={}, nz={}", z, self.nz)
            );
        } else if y >= self.ny {
            return index_error!(
                format!("Index out of range: y={}, ny={}", y, self.ny)
            );
        } else if x >= self.nx {
            return index_error!(
                format!("Index out of range: x={}, nx={}", x, self.nx)
            );
        }

        let pos = self.coords[n].at(z as f32, y as f32, x as f32);
        let mut out = Array::zeros((3,));
        out[0] = pos.z;
        out[1] = pos.y;
        out[2] = pos.x;
        Ok(out.into_pyarray_bound(py).into())
    }

    #[pyo3(signature = (dist_min, dist_max, angle_max = None))]
    /// Find the optimal alignment on the Viterbi grid.
    /// The values of `dist_min` and `dist_max` must be normalized so that the unit of
    /// the grid is 1.
    pub fn viterbi<'py>(
        &self,
        py: Python<'py>,
        dist_min: f32,
        dist_max: f32,
        angle_max: Option<f32>,
    ) -> PyResult<(Py<PyArray2<isize>>, f32)> {
        let (states, score) = py.allow_threads(
            move || {
                match angle_max {
                    Some(s) => self.viterbi_with_angle(dist_min, dist_max, s),
                    None => self.viterbi_simple(dist_min, dist_max)
                }
            }
        )?;
        Ok((states.into_pyarray_bound(py).into(), score))
    }

    #[pyo3(signature = (dist_min, dist_max, coords, origin, angle_max = None))]
    /// Find the optimal alignment on the Viterbi grid with the first local coordinate
    /// system connected with a fixed point `coords`.
    pub fn viterbi_fixed_start<'py>(
        &self,
        py: Python<'py>,
        dist_min: f32,
        dist_max: f32,
        coords: PyReadonlyArray1<f32>,
        origin: PyReadonlyArray1<f32>,
        angle_max: Option<f32>,
    ) -> PyResult<(Py<PyArray2<isize>>, f32)> {
        let coords = Vector3D::from(coords.as_array());
        let origin = Vector3D::from(origin.as_array());
        let (states, score) = py.allow_threads(
            move || {
                match angle_max {
                    Some(s) => self.viterbi_with_angle_fixed_start(
                        dist_min, dist_max, s, coords, origin
                    ),
                    None => self.viterbi_with_angle_fixed_start(
                        dist_min, dist_max, 90.0, coords, origin
                    ),
                }
            }
        )?;
        Ok((states.into_pyarray_bound(py).into(), score))
    }
}

impl ViterbiGrid {
    pub fn viterbi_simple(&self, dist_min: f32, dist_max: f32) -> PyResult<(Array2<isize>, f32)> {
        if dist_min >= dist_max {
            return value_error!(
                format!(
                    "dist_min must be smaller than dist_max, but got dist_min={}, dist_max={}",
                    dist_min, dist_max,
                )
            );
        }
        let dist_min2 = dist_min.powi(2);
        let dist_max2 = dist_max.powi(2);

        let mut viterbi_lattice = self.init_viterbi_lattice();
        viterbi_lattice.slice_mut(s![0, .., .., ..]).assign(&self.score.slice(s![0, .., .., ..]));
        let mut state_sequence = Array::zeros((self.nmole, 3));
        let constraint = Constraint::new(
            self.nz as f32,
            self.ny as f32,
            self.nx as f32,
            dist_min2,
            dist_max2,
        );

        for t in 1..self.nmole {
            let coord_prev = &self.coords[t - 1];
            let coord = &self.coords[t];
            for z1 in 0..self.nz {
                for y1 in 0..self.ny {
                    for x1 in 0..self.nx {
                        let mut max = f32::NEG_INFINITY;
                        let end_point = coord.at(z1 as f32, y1 as f32, x1 as f32);
                        for y0 in 0..self.ny {
                            if constraint.fast_check_longitudinal(coord_prev, &end_point, y0 as f32) != CheckResult::OK {
                                continue;
                            }
                            for z0 in 0..self.nz {
                                for x0 in 0..self.nx {
                                    if constraint.check_constraint(&coord_prev.at(z0 as f32, y0 as f32, x0 as f32), &end_point) {
                                        continue;
                                    }
                                    max = f32::max(max, viterbi_lattice[[t - 1, z0, y0, x0]]);
                                }
                            }
                        }
                        let next_score = self.score[[t, z1, y1, x1]];
                        viterbi_lattice[[t, z1, y1, x1]] = max + next_score;
                    }
                }
            }
        }

        let (argmax, max_score) = argmax3d(&viterbi_lattice);
        state_sequence[[self.nmole - 1, 0]] = argmax.z;
        state_sequence[[self.nmole - 1, 1]] = argmax.y;
        state_sequence[[self.nmole - 1, 2]] = argmax.x;

        // backward tracking
        let mut prev = argmax.clone();

        for t in (0..self.nmole - 1).rev() {
            let point_prev = self.coords[t + 1].at(prev.z as f32, prev.y as f32, prev.x as f32);
            let coord = &self.coords[t];
            let mut max = f32::NEG_INFINITY;
            let mut argmax: Vector3D<isize> = Vector3D::new(-1, -1, -1);
            for z0 in 0..self.nz {
                for y0 in 0..self.ny {
                    for x0 in 0..self.nx {
                        if constraint.check_constraint(&coord.at(z0 as f32, y0 as f32, x0 as f32), &point_prev) {
                            continue;
                        }
                        let value = viterbi_lattice[[t, z0, y0, x0]];
                        if value > max {
                            max = value;
                            argmax.z = z0 as isize;
                            argmax.y = y0 as isize;
                            argmax.x = x0 as isize;
                        }
                    }
                }
            }
            prev = argmax;
            state_sequence[[t, 0]] = prev.z;
            state_sequence[[t, 1]] = prev.y;
            state_sequence[[t, 2]] = prev.x;
        }

        Ok((state_sequence, max_score))
    }

    fn viterbi_with_angle(
        &self,
        dist_min: f32,
        dist_max: f32,
        angle_max: f32,
    ) -> PyResult<(Array2<isize>, f32)> {
        self.viterbi_with_angle_given_start_score(
            dist_min,
            dist_max,
            angle_max,
            self.score.slice(s![0, .., .., ..]).to_owned()
        )
    }

    fn viterbi_with_angle_fixed_start(
        &self,
        dist_min: f32,
        dist_max: f32,
        angle_max: f32,
        fixed_point: Vector3D<f32>,
        origin: Vector3D<f32>,
    ) -> PyResult<(Array2<isize>, f32)> {
        let dist_min2 = dist_min.powi(2);
        let dist_max2 = dist_max.powi(2);
        let cos_angle_max = angle_max.cos();
        let constraint = AngleConstraint::new(
            self.nz as f32,
            self.ny as f32,
            self.nx as f32,
            dist_min2,
            dist_max2,
            cos_angle_max,
        );

        let coords_first = &self.coords[0];
        let origin_vector = origin - coords_first.origin;
        let origin_dist2 = origin_vector.length2();
        let mut start = Array::zeros((self.nz, self.ny, self.nx));
        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    if constraint.check_constraint(
                        &fixed_point,
                        &coords_first.at(z as f32, y as f32, x as f32),
                        &origin_vector,
                        origin_dist2,
                    ) {
                        continue;
                    }
                    start[[z, y, x]] = self.score[[0, z, y, x]];
                }
            }
        }
        self.viterbi_with_angle_given_start_score(dist_min, dist_max, angle_max, start)
    }

    /// Run viterbi with given start landscape (the start probability in HMM)
    fn viterbi_with_angle_given_start_score(
        &self,
        dist_min: f32,
        dist_max: f32,
        angle_max: f32,
        start: Array<f32, Ix3>,
    ) -> PyResult<(Array2<isize>, f32)> {
        let mut viterbi_lattice = self.init_viterbi_lattice();
        viterbi_lattice.slice_mut(s![0, .., .., ..]).assign(&start);
        self.run_viterbi(&mut viterbi_lattice, dist_min, dist_max, angle_max)
    }

    fn run_viterbi(
        &self,
        viterbi_lattice: &mut Array<f32, Ix4>,
        dist_min: f32,
        dist_max: f32,
        angle_max: f32,
    ) -> PyResult<(Array2<isize>, f32)> {
        if dist_min >= dist_max {
            return value_error!(
                format!(
                    "dist_min must be smaller than dist_max, but got dist_min={}, dist_max={}",
                    dist_min, dist_max,
                )
            );
        } else if angle_max <= 0.0 || std::f32::consts::FRAC_PI_2 < angle_max {
            return value_error!(
                format!(
                    "angle_max must be between 0 and pi/2, but got angle_max={}",
                    angle_max,
                )
            );
        }
        let dist_min2 = dist_min.powi(2);
        let dist_max2 = dist_max.powi(2);
        let cos_angle_max = angle_max.cos();

        let mut state_sequence = Array::zeros((self.nmole, 3));
        let constraint = AngleConstraint::new(
            self.nz as f32,
            self.ny as f32,
            self.nx as f32,
            dist_min2,
            dist_max2,
            cos_angle_max,
        );

        for t in 1..self.nmole {
            let coord_prev = &self.coords[t - 1];
            let coord = &self.coords[t];
            let origin_vector = coord_prev.origin - coord.origin;
            let origin_dist2 = origin_vector.length2();
            for z1 in 0..self.nz {
                for y1 in 0..self.ny {
                    for x1 in 0..self.nx {
                        let mut max = f32::NEG_INFINITY;
                        let end_point = coord.at(z1 as f32, y1 as f32, x1 as f32);
                        for y0 in 0..self.ny {
                            if constraint.fast_check_longitudinal(&coord_prev, &end_point, y0 as f32) != CheckResult::OK {
                                continue;
                            }
                            for z0 in 0..self.nz {
                                for x0 in 0..self.nx {
                                    if constraint.check_constraint(
                                        &coord_prev.at(z0 as f32, y0 as f32, x0 as f32),
                                        &end_point,
                                        &origin_vector,
                                        origin_dist2,
                                    ) {
                                        continue;
                                    }
                                    max = f32::max(max, viterbi_lattice[[t - 1, z0, y0, x0]]);
                                }
                            }
                        }
                        let next_score = self.score[[t, z1, y1, x1]];
                        viterbi_lattice[[t, z1, y1, x1]] = max + next_score;
                    }
                }
            }
        }
        let (argmax, max_score) = argmax3d(&viterbi_lattice);

        state_sequence[[self.nmole - 1, 0]] = argmax.z;
        state_sequence[[self.nmole - 1, 1]] = argmax.y;
        state_sequence[[self.nmole - 1, 2]] = argmax.x;

        let mut prev = argmax.clone();

        // backward tracking
        for t in (0..self.nmole - 1).rev() {
            let coord_prev = &self.coords[t + 1];
            let coord = &self.coords[t];
            let origin_vector = coord.origin - coord_prev.origin;
            let origin_dist2 = origin_vector.length2();
            let point_prev = coord_prev.at(prev.z as f32, prev.y as f32, prev.x as f32);
            let mut max = f32::NEG_INFINITY;
            let mut argmax: Vector3D<isize> = Vector3D::new(-1, -1, -1);
            for z0 in 0..self.nz {
                for y0 in 0..self.ny {
                    for x0 in 0..self.nx {
                        if constraint.check_constraint(
                            &coord.at(z0 as f32, y0 as f32, x0 as f32),
                            &point_prev,
                            &origin_vector,
                            origin_dist2,
                        ) {
                            continue;
                        }
                        let value = viterbi_lattice[[t, z0, y0, x0]];
                        if value > max {
                            max = value;
                            argmax.z = z0 as isize;
                            argmax.y = y0 as isize;
                            argmax.x = x0 as isize;
                        }
                    }
                }
            }
            prev = argmax;
            state_sequence[[t, 0]] = prev.z;
            state_sequence[[t, 1]] = prev.y;
            state_sequence[[t, 2]] = prev.x;
        }

        Ok((state_sequence, max_score))
    }

    fn init_viterbi_lattice(&self) -> Array<f32, Ix4> {
        Array::from_elem((self.nmole, self.nz, self.ny, self.nx), f32::NEG_INFINITY)
    }
}

/// Find the maximum value and its index in a 3D array
fn argmax3d(viterbi_lattice: &Array<f32, Ix4>) -> (Vector3D<isize>, f32) {
    let mut max_score = f32::NEG_INFINITY;
    let mut argmax = (0, 0, 0);
    let (nmole, nz, ny, nx) = viterbi_lattice.dim();

    // find maximum score
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let s = viterbi_lattice[[nmole - 1, z, y, x]];
                if s > max_score {
                    max_score = s;
                    argmax = (z, y, x);
                }
            }
        }
    }
    let vec = Vector3D::new(
        argmax.0 as isize,
        argmax.1 as isize,
        argmax.2 as isize
    );
    (vec, max_score)
}
