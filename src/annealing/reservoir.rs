use pyo3::prelude::{pyclass, pymethods};
use pyo3::AsPyPointer;

#[pyclass]
#[derive(Clone)]
pub struct Reservoir {
    temperature_diff: f32,
    temperature: f32,
    time_constant: f32,
    min_temperature: f32,
}

impl Reservoir {
    pub fn new(temperature: f32, time_constant: f32, min_temperature: f32) -> Self {
        if min_temperature < 0.0 {
            panic!("Minimum temperature must be positive");
        } else if temperature < min_temperature {
            panic!("Initial temperature must be greater than minimum temperature");
        } else if time_constant <= 0.0 {
            panic!("Time constant must be positive.");
        }
        let initial_temperature = temperature;
        let temperature0 = initial_temperature - min_temperature;
        Self {
            temperature_diff: temperature0,
            temperature,
            time_constant,
            min_temperature,
        }
    }

    pub fn cool(&mut self, n: usize) {
        self.temperature =
            self.temperature_diff * (-(n as f32) / self.time_constant).exp()
            + self.min_temperature;
    }

    pub fn prob(&self, de: f32) -> f32 {
        if de < 0.0 {
            1.0
        } else {
            (-de / self.temperature).exp()
        }
    }

}

#[pymethods]
impl Reservoir {
    #[getter]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn initialize(&mut self) {
        self.temperature = self.temperature_diff + self.min_temperature;
    }
}

impl AsPyPointer for Reservoir {
    fn as_ptr(&self) -> *mut pyo3::ffi::PyObject {
        let ptr = self as *const Reservoir as *mut Reservoir as *mut std::ffi::c_void;
        let pyobj = unsafe { pyo3::ffi::PyCapsule_New(ptr, std::ptr::null_mut(), None) };
        pyobj
    }
}
