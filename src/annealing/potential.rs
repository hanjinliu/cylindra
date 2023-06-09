use pyo3::prelude::PyResult;
use crate::value_error;

pub trait BindingPotential<Se, T> {
    fn calculate(&self, dist2: T, typ: Se) -> T;
    fn cool(&mut self, _n: usize) {
        // Do nothing by default.
    }
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
    fn cool(&mut self, _n: usize) {
        // Do nothing by default.
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
/// A potential model with shape:
///    ~~    ~~
///     │    │
///     └────┘
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
            return value_error!("All distances must be positive");
        } else if lon_dist_min >= lon_dist_max || lat_dist_min >= lat_dist_max {
            return value_error!("Minimum distance must be smaller than maximum distance");
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

#[derive(Clone)]
/// A potential model with shape:
///    ~~      ~~
///     \      /
///      \____/
pub struct TrapezoidalPotential2D {
    lon_dist_min2: f32,
    lon_dist_max2: f32,
    lat_dist_min2: f32,
    lat_dist_max2: f32,
    lon_slope: f32,
    lat_slope: f32,
    cooling_rate: f32,
}

impl TrapezoidalPotential2D {
    pub fn new(
        lon_dist_min: f32,
        lon_dist_max: f32,
        lat_dist_min: f32,
        lat_dist_max: f32,
        cooling_rate: f32,
    ) -> PyResult<Self> {
        if lon_dist_min < 0.0 || lon_dist_max < 0.0 || lat_dist_min < 0.0 || lat_dist_max < 0.0 {
            return value_error!("All distances must be positive");
        } else if lon_dist_min >= lon_dist_max || lat_dist_min >= lat_dist_max {
            return value_error!("Minimum distance must be smaller than maximum distance");
        } else if cooling_rate < 0.0 {
            return value_error!("Cooling rate must be non-negative");
        }

        Ok(
            Self {
                lon_dist_min2: lon_dist_min * lon_dist_min,
                lon_dist_max2: lon_dist_max * lon_dist_max,
                lat_dist_min2: lat_dist_min * lat_dist_min,
                lat_dist_max2: lat_dist_max * lat_dist_max,
                lon_slope: 0.0,
                lat_slope: 0.0,
                cooling_rate,
            }
        )
    }


    pub fn unbounded() -> Self {
        Self {
            lon_dist_min2: 0.0,
            lon_dist_max2: std::f32::INFINITY,
            lat_dist_min2: 0.0,
            lat_dist_max2: std::f32::INFINITY,
            lon_slope: 0.0,
            lat_slope: 0.0,
            cooling_rate: 0.0,
        }
    }

    pub fn slopes(&self)-> (f32, f32) {
        (self.lon_slope, self.lat_slope)
    }
}

impl BindingPotential2D for TrapezoidalPotential2D {
    fn longitudinal(&self, dist2: f32) -> f32 {
        if dist2 < self.lon_dist_min2 {
            self.lon_slope * (self.lon_dist_min2 - dist2).sqrt()
        } else if self.lon_dist_max2 < dist2 {
            self.lon_slope * (dist2 - self.lon_dist_max2).sqrt()
        } else {
            0.0
        }
    }

    fn lateral(&self, dist2: f32) -> f32 {
        if dist2 < self.lat_dist_min2 {
            self.lat_slope * (self.lat_dist_min2 - dist2).sqrt()
        } else if self.lat_dist_max2 < dist2 {
            self.lat_slope * (dist2 - self.lat_dist_max2).sqrt()
        } else {
            0.0
        }
    }

    fn cool(&mut self, n: usize) {
        self.lon_slope = self.cooling_rate * n as f32;
        self.lat_slope = self.cooling_rate * n as f32;
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
            return value_error!("All parameters must be positive");
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
