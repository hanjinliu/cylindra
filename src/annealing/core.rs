use pyo3::{
    prelude::{PyResult, pyclass, pymethods},
    Python, Py, PyRefMut
};
use numpy::{
    IntoPyArray, PyReadonlyArray2, PyReadonlyArray4, PyArray1, PyArray2,
    ndarray::Array2, PyUntypedArrayMethods
};

use super::{
    random::RandomNumberGenerator,
    graph::CylindricGraph,
    reservoir::Reservoir,
};
use crate::{value_error, cylindric::Index};

#[derive(Clone, PartialEq, Eq)]
/// Current state of the annealing model
enum OptimizationState {
    NotConverged,  // Optimization is not converged yet
    Converged,  // Optimization converged
    Failed,  // Optimization failed due to wrong parameters
}

#[pyclass]
/// A class to perform simulated annealing on a cylindric lattice.
pub struct CylindricAnnealingModel {
    rng: RandomNumberGenerator,
    optimization_state: OptimizationState,
    graph: CylindricGraph,
    reservoir: Reservoir,
    iteration: usize,
    reject_limit: usize,
}

#[pymethods]
impl CylindricAnnealingModel {
    #[new]
    #[pyo3(signature = (seed=0))]
    pub fn new(seed: u64) -> Self {
        let rng = RandomNumberGenerator::new(seed);
        let optimization_state = OptimizationState::NotConverged;
        Self {
            rng,
            optimization_state,
            graph: CylindricGraph::empty(),
            reservoir: Reservoir::new(1.0, 1.0, 0.0),
            iteration: 0,
            reject_limit: 1000,
        }
    }

    #[pyo3(signature = (seed))]
    /// Return a new instance with different random seed.
    pub fn with_seed<'py>(&self, py: Python<'py>, seed: u64) -> Py<Self> {
        let mut out = Self {
            rng: self.rng.with_seed(seed),
            optimization_state: self.optimization_state.clone(),
            graph: self.graph.clone(),
            reservoir: self.reservoir.clone(),
            iteration: self.iteration,
            reject_limit: self.reject_limit,
        };
        out.reservoir.initialize();
        Py::new(py, out).unwrap()
    }

    #[pyo3(signature = (reject_limit))]
    /// Return a new instance with different reject limit.
    pub fn with_reject_limit<'py>(&self, py: Python<'py>, reject_limit: usize) -> Py<Self> {
        let mut out = Self {
            rng: self.rng.clone(),
            optimization_state: self.optimization_state.clone(),
            graph: self.graph.clone(),
            reservoir: self.reservoir.clone(),
            iteration: self.iteration,
            reject_limit,
        };
        out.reservoir.initialize();
        Py::new(py, out).unwrap()
    }

    #[pyo3(signature = (temperature, time_constant, min_temperature=0.0))]
    /// Set a standard reservoir.
    pub fn set_reservoir(
        mut slf: PyRefMut<Self>,
        temperature: f32,
        time_constant: f32,
        min_temperature: f32,
    ) -> PyRefMut<Self> {
        slf.reservoir = Reservoir::new(temperature, time_constant, min_temperature);
        slf
    }

    /// Initialize the node states randomly.
    pub fn init_shift_random(&mut self) -> PyResult<()> {
        let shape = self.local_shape();
        let mut shift_array = Array2::zeros((self.node_count(), 3));
        for i in 0..self.node_count() {
            let z = self.rng.uniform_int(shape.0 as usize);
            let y = self.rng.uniform_int(shape.1 as usize);
            let x = self.rng.uniform_int(shape.2 as usize);
            shift_array[[i, 0]] = z as isize;
            shift_array[[i, 1]] = y as isize;
            shift_array[[i, 2]] = x as isize;
        }
        self.graph.set_shifts(&shift_array)?;
        Ok(())
    }

    /// Get all the existing distances of longitudinal connections as a numpy array.
    pub fn longitudinal_distances<'py>(&self, py: Python<'py>) -> Py<PyArray1<f32>> {
        self.graph.get_longitudinal_distances().into_pyarray_bound(py).into()
    }

    /// Get all the existing distances of lateral connections as a numpy array.
    pub fn lateral_distances<'py>(&self, py: Python<'py>) -> Py<PyArray1<f32>> {
        self.graph.get_lateral_distances().into_pyarray_bound(py).into()
    }

    pub fn longitudinal_angles<'py>(&self, py: Python<'py>) -> Py<PyArray1<f32>> {
        self.graph.get_longitudinal_angles().into_pyarray_bound(py).into()
    }

    pub fn lateral_angles<'py>(&self, py: Python<'py>) -> Py<PyArray1<f32>> {
        self.graph.get_lateral_angles().into_pyarray_bound(py).into()
    }

    pub fn get_edge_info<'py>(&self, py: Python<'py>) -> (Py<PyArray2<f32>>, Py<PyArray2<f32>>, Py<PyArray1<i32>>) {
        let (out0, out1, out2) = self.graph.get_edge_states();
        (out0.into_pyarray_bound(py).into(), out1.into_pyarray_bound(py).into(), out2.into_pyarray_bound(py).into())
    }

    #[pyo3(signature = (indices, npf, nrise))]
    pub fn construct_graph<'py>(
        mut slf: PyRefMut<'py, Self>,
        indices: PyReadonlyArray2<i32>,
        npf: isize,
        nrise: isize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // indices into Vec<Index>
        let indices = indices.as_array().to_shared();
        if indices.shape()[1] != 2 {
            return value_error!("indices must be a Nx2 array");
        }
        let indices = (0..indices.shape()[0])
            .map(|i| Index::new(indices[[i, 0]] as isize, indices[[i, 1]] as isize))
            .collect::<Vec<_>>();
        slf.graph.construct(indices, npf, nrise)?;
        Ok(slf)
    }

    #[pyo3(signature = (origin, zvec, yvec, xvec))]
    pub fn set_graph_coordinates<'py>(
        mut slf: PyRefMut<'py, Self>,
        origin: PyReadonlyArray2<f32>,
        zvec: PyReadonlyArray2<f32>,
        yvec: PyReadonlyArray2<f32>,
        xvec: PyReadonlyArray2<f32>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let origin = origin.as_array().to_shared();
        let zvec = zvec.as_array().to_shared();
        let yvec = yvec.as_array().to_shared();
        let xvec = xvec.as_array().to_shared();
        slf.graph.set_coordinates(origin, zvec, yvec, xvec)?;
        Ok(slf)
    }

    pub fn node_count(&self) -> usize {
        self.graph.components().node_count()
    }

    #[pyo3(signature = (energy))]
    pub fn set_energy_landscape<'py>(
        mut slf: PyRefMut<'py, Self>,
        energy: PyReadonlyArray4<f32>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let shape = energy.shape();
        slf.rng.set_shape((shape[1], shape[2], shape[3]));
        let energy = energy.as_array().to_shared();
        slf.graph.set_energy_landscape(energy)?;
        Ok(slf)
    }

    #[pyo3(signature =
        (lon_dist_min, lon_dist_max, lat_dist_min, lat_dist_max, lon_ang_max=-1.0, cooling_rate=1e-3))
    ]
    /// Set a box potential with given borders.
    pub fn set_box_potential(
        mut slf: PyRefMut<Self>,
        lon_dist_min: f32,
        lon_dist_max: f32,
        lat_dist_min: f32,
        lat_dist_max: f32,
        lon_ang_max: f32,
        cooling_rate: f32,
    ) -> PyResult<PyRefMut<Self>>{
        let model = slf.graph.binding_potential
            .with_lon_dist(lon_dist_min, lon_dist_max)?
            .with_lat_dist(lat_dist_min, lat_dist_max)?
            .with_lon_ang(lon_ang_max)?
            .with_cooling_rate(cooling_rate);
        slf.graph.set_potential_model(model);
        Ok(slf)
    }

    /// Get integer shift in each local coordinates as a numpy array.
    pub fn shifts<'py>(&self, py: Python<'py>) -> Py<PyArray2<isize>> {
        self.graph.get_shifts().into_pyarray_bound(py).into()
    }

    pub fn set_shifts<'py>(
        mut slf: PyRefMut<'py, Self>,
        shifts: PyReadonlyArray2<isize>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let shifts = shifts.as_array().to_shared();
        slf.graph.set_shifts_arc(&shifts)?;
        Ok(slf)
    }

    pub fn local_shape(&self) -> (isize, isize, isize) {
        let shape = self.graph.local_shape;
        (shape.z, shape.y, shape.x)
    }

    /// Calculate the current energy of the graph.
    pub fn energy(&self) -> f32 {
        self.graph.energy()
    }

    pub fn binding_energies<'py>(&self, py: Python<'py>) -> (Py<PyArray1<f32>>, Py<PyArray1<f32>>) {
        let (lon, lat) = self.graph.binding_energies();
        (lon.into_pyarray_bound(py).into(), lat.into_pyarray_bound(py).into())
    }

    /// Get current optimization state as a string.
    pub fn optimization_state(&self) -> String {
        match self.optimization_state {
            OptimizationState::NotConverged => "not_converged".to_string(),
            OptimizationState::Converged => "converged".to_string(),
            OptimizationState::Failed => "failed".to_string(),
        }
    }

    /// Time constant of reservoir
    pub fn time_constant(&self) -> f32 {
        self.reservoir.time_constant()
    }

    /// Get the temperature of the reservoir.
    pub fn temperature(&self) -> f32 {
        self.reservoir.temperature()
    }

    /// Get the current iteration count.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    #[pyo3(signature = (nsteps=10000))]
    /// Run simulation for given number of steps.
    /// If simulation failed or converged, it will stop.
    pub fn simulate<'py>(&mut self, py: Python<'py>, nsteps: usize) -> PyResult<()> {
        if nsteps <= 0 {
            return value_error!("nsteps must be positive");
        }
        if self.temperature() <= 0.0 {
            return value_error!("temperature must be positive");
        }
        py.allow_threads(
            move || {
                // Simulate while cooling.
                let mut reject_count = 0;
                for _ in 0..nsteps {
                    if self.proceed() {
                        reject_count = 0;
                    } else {
                        reject_count += 1;
                        if reject_count > self.reject_limit {
                            if self.graph.energy() == std::f32::INFINITY {
                                self.optimization_state = OptimizationState::Failed;
                            } else {
                                self.optimization_state = OptimizationState::Converged;
                            }
                            break;
                        }
                    }
                    self.iteration += 1;
                    self.reservoir.cool(self.iteration);
                    self.graph.cool(self.iteration);
                }
                Ok(())
            }
        )
    }

    /// Cool the system until the energy is not decreased. This method is deterministic.
    pub fn cool_completely<'py>(&mut self, py: Python<'py>) {
        // NOTE: This is not efficient, because shifting a local state does not alter
        // the energy difference of the other local states that are not directly
        // connected to it. However, since this method is called only in the end,
        // and usually it is very fast (<10 ms). just leave it as it is for better
        // readability.
        py.allow_threads(
            move || {
                loop {
                    let shift = self.graph.try_all_shifts();
                    if shift.energy_diff < 0.0 {
                        self.graph.apply_shift(&shift);
                        self.iteration += 1;
                    } else {
                        break;
                    }
                }
            }
        )
    }
}

impl CylindricAnnealingModel {
    /// Proceed one step of simulation. Return true if the shift is accepted.
    fn proceed(&mut self) -> bool {
        // Randomly shift a node.
        let result = self.graph.try_random_shift(&mut self.rng);

        // If the shift causes energy change from Inf to Inf, energy difference is NaN.
        if result.energy_diff.is_nan() {
            return false;
        }

        // Decide whether to accept the shift.
        let prob = self.reservoir.prob(result.energy_diff);
        if self.rng.bernoulli(prob) {
            // accept shift
            self.graph.apply_shift(&result);
            true
        } else {
            false
        }
    }
}
