use pyo3::prelude::PyResult;

pub trait BindingPotential<Se, T> {
    fn calculate(&self, dist2: T, typ: Se) -> T;
}

pub trait BindingPotential2D {
    fn longitudinal(&self, dist2: f32) -> f32;
    fn lateral(&self, dist2: f32) -> f32;
    fn calculate(&self, dist2: f32, typ: &EdgeType) -> f32 {
        match typ {
            EdgeType::Longitudinal => self.longitudinal(dist2),
            EdgeType::Lateral => self.lateral(dist2),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum EdgeType {
    Longitudinal,
    Lateral,
}

pub struct EmptyPotential2D {}

impl BindingPotential2D for EmptyPotential2D {
    fn longitudinal(&self, _dist2: f32) -> f32 { 0.0 }
    fn lateral(&self, _dist2: f32) -> f32 { 0.0 }
}

#[derive(Clone)]
pub struct BoxPotential2D {
    lon_dist_min2: f32,
    lon_dist_max2: f32,
    lat_dist_min2: f32,
    lat_dist_max2: f32,
}

impl BoxPotential2D {
    pub fn new(
        lon_dist_min: f32,
        lon_dist_max: f32,
        lat_dist_min: f32,
        lat_dist_max: f32,
    ) -> PyResult<Self> {
        if lon_dist_min < 0.0 || lon_dist_max < 0.0 || lat_dist_min < 0.0 || lat_dist_max < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All distances must be positive",
            ));
        } else if lon_dist_min >= lon_dist_max || lat_dist_min >= lat_dist_max {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Minimum distance must be smaller than maximum distance",
            ));
        }
        Ok(
            Self {
                lon_dist_min2: lon_dist_min * lon_dist_min,
                lon_dist_max2: lon_dist_max * lon_dist_max,
                lat_dist_min2: lat_dist_min * lat_dist_min,
                lat_dist_max2: lat_dist_max * lat_dist_max,
            }
        )
    }

    pub fn unbounded() -> Self {
        Self {
            lon_dist_min2: 0.0,
            lon_dist_max2: std::f32::INFINITY,
            lat_dist_min2: 0.0,
            lat_dist_max2: std::f32::INFINITY,
        }
    }
}

impl BindingPotential2D for BoxPotential2D {
    fn longitudinal(&self, dist2: f32) -> f32 {
        if dist2 < self.lon_dist_min2 || self.lon_dist_max2 < dist2 {
            std::f32::INFINITY
        } else {
            0.0
        }
    }

    fn lateral(&self, dist2: f32) -> f32 {
        if dist2 < self.lat_dist_min2 || self.lat_dist_max2 < dist2 {
            std::f32::INFINITY
        } else {
            0.0
        }
    }
}

pub struct HarmonicPotential2D {
    halfk0: f32,
    halfk1: f32,
    r0: f32,
    r1: f32,
}

impl HarmonicPotential2D {
    pub fn new(
        halfk0: f32,
        halfk1: f32,
        r0: f32,
        r1: f32,
    ) -> PyResult<Self> {
        if halfk0 < 0.0 || halfk1 < 0.0 || r0 < 0.0 || r1 < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All parameters must be positive",
            ));
        }
        Ok(Self { halfk0, halfk1, r0, r1 })
    }
}

impl BindingPotential2D for HarmonicPotential2D {
    fn longitudinal(&self, dist2: f32) -> f32 {
        let x = dist2.sqrt() - self.r0;
        self.halfk0 * x * x
    }

    fn lateral(&self, dist2: f32) -> f32 {
        let x = dist2.sqrt() - self.r1;
        self.halfk1 * x * x
    }
}
