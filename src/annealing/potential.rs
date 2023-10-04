use pyo3::prelude::PyResult;
use crate::{value_error, coordinates::Vector3D};

pub trait BindingPotential2D {
    fn longitudinal(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>) -> f32;
    fn lateral(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>) -> f32;

    /// Calculate the binding energy of the given conditions.
    /// # Arguments
    /// * `dr` - The vector in the world coordinate between the two molecule centers.
    /// * `vec` - The vector in the world coordinate between the origin of the local coordinate
    ///   systems.
    /// * `typ` - The type of the edge.
    fn calculate(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>, typ: &EdgeType) -> f32 {
        match typ {
            EdgeType::Longitudinal => self.longitudinal(dr, vec),
            EdgeType::Lateral => self.lateral(dr, vec),
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

#[derive(Clone)]
/// A 1D potential model with shape:
///    ~~      ~~
///     \      /
///      \____/
/// With this boundary, distances will be softly restricted to the range
/// [dist_min2.sqrt(), dist_max2.sqrt()].
struct TrapezoidalBoundary {
    dist_min: f32,
    dist_max: f32,
    slope: f32,
}

impl TrapezoidalBoundary {
    pub fn new(dist_min: f32, dist_max: f32, slope: f32) -> PyResult<Self> {
        if dist_min < 0.0 || dist_max < 0.0 {
            return value_error!("All distances must be positive");
        } else if dist_min >= dist_max {
            return value_error!("Minimum distance must be smaller than maximum distance");
        }
        Ok(Self { dist_min, dist_max, slope })
    }

    /// An unbounded version of the model.
    pub fn unbounded() -> Self {
        Self {
            dist_min: 0.0,
            dist_max: std::f32::INFINITY,
            slope: 0.0,
        }
    }

    /// Calculated energy of given square of distance.
    pub fn energy(&self, dr: &Vector3D<f32>) -> f32 {
        let dist = dr.length();
        if dist < self.dist_min {
            self.slope * (self.dist_min - dist)
        } else if self.dist_max < dist {
            self.slope * (dist - self.dist_max)
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
/// A 1D symmetric potential model for an angle.
/// With this boundary, angles will be softly restricted to the range
/// [-ang_max, ang_max].
struct TrapezoidalCosineBoundary {
    ang_max: f32,
    slope: f32,
}

impl TrapezoidalCosineBoundary {
    pub fn new(ang_max: f32, slope: f32) -> PyResult<Self> {
        if ang_max <= 0.0 {
            return value_error!("Maximum angle must be positive");
        }
        Ok(
            Self { ang_max, slope }
        )
    }

    pub fn unbounded() -> Self {
        Self { ang_max: std::f32::INFINITY, slope: 0.0, }
    }

    ///           o         Cosine is calculated as the angle between the
    ///    o     i+1        y axis and the vector from i to i+1. The y axis
    ///    i                of local coordinates is always parallel to the
    /// ---------------> y  y axis.
    pub fn energy(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>) -> f32 {
        let ang = dr.cos_angle(vec).abs().acos();
        if ang > self.ang_max {
            self.slope * (ang - self.ang_max)
        } else {
            0.0
        }
    }
}

#[derive(Clone)]
pub struct TrapezoidalPotential2D {
    lon: TrapezoidalBoundary,
    lat: TrapezoidalBoundary,
    angle: TrapezoidalCosineBoundary,
    cooling_rate: f32,
}

impl TrapezoidalPotential2D {
    pub fn new(
        lon_dist_min: f32,
        lon_dist_max: f32,
        lat_dist_min: f32,
        lat_dist_max: f32,
        lon_ang_max: f32,
        cooling_rate: f32,
    ) -> PyResult<Self> {
        if cooling_rate < 0.0 {
            return value_error!("Cooling rate must be non-negative");
        }

        Ok(
            Self {
                lon: TrapezoidalBoundary::new(lon_dist_min, lon_dist_max, 0.0)?,
                lat: TrapezoidalBoundary::new(lat_dist_min, lat_dist_max, 0.0)?,
                angle: TrapezoidalCosineBoundary::new(lon_ang_max, 0.0)?,
                cooling_rate,
            }
        )
    }

    pub fn unbounded() -> Self {
        Self {
            lon: TrapezoidalBoundary::unbounded(),
            lat: TrapezoidalBoundary::unbounded(),
            angle: TrapezoidalCosineBoundary::unbounded(),
            cooling_rate: 0.0,
        }
    }

    pub fn with_lon_dist(&self, min: f32, max: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.lon = TrapezoidalBoundary::new(min, max, self.lon.slope)?;
        Ok(new)
    }

    pub fn with_lat_dist(&self, min: f32, max: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.lat = TrapezoidalBoundary::new(min, max, self.lat.slope)?;
        Ok(new)
    }

    pub fn with_lon_ang(&self, max: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.angle = TrapezoidalCosineBoundary::new(max, self.angle.slope)?;
        Ok(new)
    }

    pub fn with_cooling_rate(&self, cooling_rate: f32) -> Self {
        let mut new = self.clone();
        new.cooling_rate = cooling_rate;
        new
    }

}

impl BindingPotential2D for TrapezoidalPotential2D {
    fn longitudinal(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>) -> f32 {
        // Energy coming from longitudinal distance
        let eng_dist = self.lon.energy(dr);
        let eng_ang = self.angle.energy(dr, vec);
        eng_dist + eng_ang
    }

    fn lateral(&self, dr: &Vector3D<f32>, _vec: &Vector3D<f32>) -> f32 {
        self.lat.energy(&dr)
    }

    /// Cool the potential by increasing the slope of the trapezoid.
    fn cool(&mut self, n: usize) {
        let slope = self.cooling_rate * n as f32;
        self.lon.slope = slope;
        self.lat.slope = slope;
        self.angle.slope = slope;
    }
}
