use numpy::{ndarray::ArcArray, Ix4, PyReadonlyArray4};
use pyo3::{prelude::*, PyRef};

#[pyclass(frozen)]
/// An energy landscape array that can be shared by annealing models.
///
/// The input array is copied only once on construction. Passing the same instance
/// to `set_energy_landscape` of multiple models does not copy the array again.
pub struct EnergyLandscape {
    array: ArcArray<f32, Ix4>,
}

#[pymethods]
impl EnergyLandscape {
    #[new]
    #[pyo3(signature = (energy))]
    pub fn new(energy: PyReadonlyArray4<f32>) -> Self {
        Self { array: energy.as_array().to_shared() }
    }

    #[getter]
    /// Shape of the energy landscape array.
    pub fn shape(&self) -> (usize, usize, usize, usize) {
        let shape = self.array.shape();
        (shape[0], shape[1], shape[2], shape[3])
    }
}

#[derive(FromPyObject)]
/// Input of `set_energy_landscape`, either a shared landscape or a numpy array.
pub enum EnergyLandscapeInput<'py> {
    Shared(PyRef<'py, EnergyLandscape>),
    Array(PyReadonlyArray4<'py, f32>),
}

impl EnergyLandscapeInput<'_> {
    /// Convert the input into a shared array. Numpy arrays are copied but shared
    /// landscapes are not.
    pub fn into_shared(self) -> ArcArray<f32, Ix4> {
        match self {
            Self::Shared(landscape) => landscape.array.clone(),
            Self::Array(array) => array.as_array().to_shared(),
        }
    }
}
