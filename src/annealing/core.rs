use pyo3::{
    prelude::{PyResult, pyclass, pymethods},
    Python, Py, AsPyPointer
};
use numpy::{
    IntoPyArray, PyReadonlyArray3, PyReadonlyArray5, PyArray3,
};

use super::{
    random::RandomNumberGenerator,
    graph::CylindricGraph,
    reservoir::Reservoir,
    potential::BoxPotential2D,
};

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

    pub fn with_seed(&self, seed: u64) -> Self {
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
        return out
    }

    pub fn with_reject_limit(&self, reject_limit: usize) -> Self {
        let mut out = Self {
            rng: self.rng.clone(),
            optimization_state: self.optimization_state.clone(),
            graph: self.graph.clone(),
            reservoir: self.reservoir.clone(),
            iteration: 0,
            reject_limit,
        };
        out.reservoir.initialize();
        return out
    }

    pub fn get_reservoir(&self) -> &Reservoir {
        &self.reservoir
    }

    pub fn set_reservoir(&mut self, temperature: f32, time_constant: f32, min_temperature: f32) {
        self.reservoir = Reservoir::new(temperature, time_constant, min_temperature);
    }

    pub fn get_graph(&self) -> &CylindricGraph {
        &self.graph
    }

    pub fn set_graph(
        &mut self,
        score: PyReadonlyArray5<f32>,
        origin: PyReadonlyArray3<f32>,
        zvec: PyReadonlyArray3<f32>,
        yvec: PyReadonlyArray3<f32>,
        xvec: PyReadonlyArray3<f32>,
        nrise: isize,
    ) -> PyResult<()> {
        let score = score.as_array().to_owned();
        let origin = origin.as_array().to_owned();
        let zvec = zvec.as_array().to_owned();
        let yvec = yvec.as_array().to_owned();
        let xvec = xvec.as_array().to_owned();
        self.graph.update(score, origin, zvec, yvec, xvec, nrise);
        Ok(())
    }

    pub fn set_box_potential(
        &mut self,
        lon_dist_min: f32,
        lon_dist_max: f32,
        lat_dist_min: f32,
        lat_dist_max: f32,
    ) -> PyResult<()>{
        self.graph.set_potential_model(
            BoxPotential2D::new(lon_dist_min, lon_dist_max, lat_dist_min, lat_dist_max)?
        );
        Ok(())
    }

    pub fn get_shifts<'py>(&self, py: Python<'py>) -> Py<PyArray3<isize>> {
        self.graph.get_shifts().into_pyarray(py).into()
    }

    pub fn energy(&self) -> f32 {
        self.graph.energy()
    }

    pub fn simulate(&mut self, nsteps: usize) {
        // TODO:: check graph
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
    }

}

impl CylindricAnnealingModel {
    fn proceed(&mut self) -> bool {
        let result = self.graph.try_random_shift(&mut self.rng);
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


impl AsPyPointer for CylindricAnnealingModel {
    fn as_ptr(&self) -> *mut pyo3::ffi::PyObject {
        let ptr = self as *const CylindricAnnealingModel as *mut CylindricAnnealingModel as *mut std::ffi::c_void;
        let pyobj = unsafe { pyo3::ffi::PyCapsule_New(ptr, std::ptr::null_mut(), None) };
        pyobj
    }
}
