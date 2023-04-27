use pyo3::{
    prelude::{PyResult, pyclass, pymethods},
    Python, Py, PyRefMut
};
use numpy::{
    IntoPyArray, PyReadonlyArray3, PyReadonlyArray5, PyArray1, PyArray3,
};

use super::{
    random::RandomNumberGenerator,
    graph::CylindricGraph,
    reservoir::Reservoir,
    potential::BoxPotential2D,
};
use crate::value_error;

/// Current state of the annealing model
#[derive(Clone, PartialEq, Eq)]
enum OptimizationState {
    NotConverged,  // Optimization is not converged yet
    Converged,  // Optimization converged
    Failed,  // Optimization failed due to wrong parameters
}

#[pyclass]
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
            reject_limit: 200,
        }
    }

    pub fn with_seed<'py>(&self, py: Python<'py>, seed: u64) -> Py<Self> {
        let rng = RandomNumberGenerator::new(seed);
        let mut out = Self {
            rng,
            optimization_state: self.optimization_state.clone(),
            graph: self.graph.clone(),
            reservoir: self.reservoir.clone(),
            iteration: 0,
            reject_limit: self.reject_limit,
        };
        out.reservoir.initialize();
        Py::new(py, out).unwrap()
    }

    pub fn with_reject_limit<'py>(&self, py: Python<'py>, reject_limit: usize) -> Py<Self> {
        let mut out = Self {
            rng: self.rng.clone(),
            optimization_state: self.optimization_state.clone(),
            graph: self.graph.clone(),
            reservoir: self.reservoir.clone(),
            iteration: 0,
            reject_limit,
        };
        out.reservoir.initialize();
        Py::new(py, out).unwrap()
    }

    #[pyo3(signature = (temperature, time_constant, min_temperature=0.0))]
    pub fn set_reservoir(
        mut slf: PyRefMut<Self>,
        temperature: f32,
        time_constant: f32,
        min_temperature: f32,
    ) -> PyRefMut<Self> {
        slf.reservoir = Reservoir::new(temperature, time_constant, min_temperature);
        slf
    }

    pub fn temperature(&self) -> f32 {
        self.reservoir.temperature()
    }

    pub fn longitudinal_distances<'py>(&self, py: Python<'py>) -> Py<PyArray1<f32>> {
        self.graph.get_longitudinal_distances().into_pyarray(py).into()
    }

    pub fn lateral_distances<'py>(&self, py: Python<'py>) -> Py<PyArray1<f32>> {
        self.graph.get_lateral_distances().into_pyarray(py).into()
    }

    #[pyo3(signature = (score, origin, zvec, yvec, xvec, nrise))]
    pub fn set_graph<'py>(
        mut slf: PyRefMut<'py, Self>,
        score: PyReadonlyArray5<f32>,
        origin: PyReadonlyArray3<f32>,
        zvec: PyReadonlyArray3<f32>,
        yvec: PyReadonlyArray3<f32>,
        xvec: PyReadonlyArray3<f32>,
        nrise: isize,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let score = score.as_array().to_shared();
        let origin = origin.as_array().to_shared();
        let zvec = zvec.as_array().to_shared();
        let yvec = yvec.as_array().to_shared();
        let xvec = xvec.as_array().to_shared();
        slf.graph.update(score, origin, zvec, yvec, xvec, nrise)?;
        Ok(slf)
    }

    #[pyo3(signature = (lon_dist_min, lon_dist_max, lat_dist_min, lat_dist_max))]
    pub fn set_box_potential(
        mut slf: PyRefMut<Self>,
        lon_dist_min: f32,
        lon_dist_max: f32,
        lat_dist_min: f32,
        lat_dist_max: f32,
    ) -> PyResult<PyRefMut<Self>>{
        slf.graph.set_potential_model(
            BoxPotential2D::new(lon_dist_min, lon_dist_max, lat_dist_min, lat_dist_max)?
        );
        Ok(slf)
    }

    pub fn shifts<'py>(&self, py: Python<'py>) -> Py<PyArray3<isize>> {
        self.graph.get_shifts().into_pyarray(py).into()
    }

    pub fn energy(&self) -> f32 {
        self.graph.energy()
    }

    #[pyo3(signature = (nsteps=10000))]
    pub fn simulate(&mut self, nsteps: usize) -> PyResult<()>{
        self.graph.check_graph()?;
        if nsteps <= 0 {
            return value_error!("nsteps must be positive");
        }
        if self.temperature() <= 0.0 {
            return value_error!("temperature must be positive");
        }
        let mut reject_count = 0;
        for _ in 0..nsteps {
            if self.proceed() {
                reject_count = 0;
            } else {
                reject_count += 1;
            }
            if reject_count > self.reject_limit {
                if self.graph.energy() == std::f32::INFINITY {
                    self.optimization_state = OptimizationState::Failed;
                } else {
                    self.optimization_state = OptimizationState::Converged;
                }
                break;
            }
            self.iteration += 1;
            self.reservoir.cool(self.iteration);
        }
        for _ in 0..nsteps {
            if !self.proceed() {
                break;
            }
        }
        Ok(())
    }

}

impl CylindricAnnealingModel {
    fn proceed(&mut self) -> bool {
        let result = self.graph.try_random_shift(&mut self.rng);
        if result.energy_diff.is_nan() {
            return false;
        }
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
