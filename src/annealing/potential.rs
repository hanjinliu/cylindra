use pyo3::prelude::PyResult;
use crate::{value_error, coordinates::Vector3D};

pub trait BindingPotential {
    fn cool(&mut self, _n: usize) {
        // Do nothing by default.
    }

}
#[derive(Clone, Copy, Default)]
/// The scalars that fully determine the binding energy of an edge.
/// Because these values do not depend on the potential parameters (which change every
/// iteration by cooling), they can be cached for the current state of the graph and be
/// reused as long as none of the two nodes moves.
pub struct EdgeScalars {
    /// Distance between the two molecule centers.
    pub dist: f32,
    /// |cos| of the angle between `dr` and the vector connecting the two local
    /// coordinate origins. Not used (and not calculated) for lateral edges.
    pub cos_abs: f32,
}

/// Calculate the scalars that describe the geometry of an edge.
/// # Arguments
/// * `dr` - The vector in the world coordinate between the two molecule centers.
/// * `dr_len` - The length of `dr`.
/// * `vec` - The vector in the world coordinate between the origin of the local coordinate
///   systems. Only used for longitudinal edges.
/// * `vec_len` - The length of `vec`.
/// * `typ` - The type of the edge.
pub fn edge_scalars(
    dr: &Vector3D<f32>,
    dr_len: f32,
    vec: &Vector3D<f32>,
    vec_len: f32,
    typ: &EdgeType,
) -> EdgeScalars {
    match typ {
        EdgeType::Longitudinal => EdgeScalars {
            dist: dr_len,
            cos_abs: dr.cos_angle_with_lengths(vec, dr_len, vec_len).abs(),
        },
        EdgeType::Lateral => EdgeScalars { dist: dr_len, cos_abs: 0.0 },
    }
}

pub trait BindingPotential2D : BindingPotential {
    /// Binding energy calculated from the cached scalars of an edge.
    fn energy_of(&self, scalars: &EdgeScalars, typ: &EdgeType) -> f32;

    /// Calculate the binding energy of the given conditions, with the vector lengths
    /// already known. See `edge_scalars` for the arguments.
    fn calculate_with_lengths(
        &self,
        dr: &Vector3D<f32>,
        dr_len: f32,
        vec: &Vector3D<f32>,
        vec_len: f32,
        typ: &EdgeType,
    ) -> f32 {
        self.energy_of(&edge_scalars(dr, dr_len, vec, vec_len, typ), typ)
    }

    /// Same as `calculate_with_lengths` but the vector lengths are calculated here.
    /// This method is for the code paths that are not performance critical.
    fn calculate(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>, typ: &EdgeType) -> f32 {
        let dr_len = dr.length();
        // `vec` is not used for lateral edges, so its length is not calculated.
        let vec_len = match typ {
            EdgeType::Longitudinal => vec.length(),
            EdgeType::Lateral => 0.0,
        };
        self.calculate_with_lengths(dr, dr_len, vec, vec_len, typ)
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
            dist_max: f32::INFINITY,
            slope: 0.0,
        }
    }

    /// Calculated energy of the given distance.
    pub fn energy_at(&self, dist: f32) -> f32 {
        if dist < self.dist_min {
            self.slope * (self.dist_min - dist)
        } else if self.dist_max < dist {
            self.slope * (dist - self.dist_max)
        } else {
            0.0
        }
    }

    /// Calculated energy of given distance vector.
    pub fn energy(&self, dr: &Vector3D<f32>) -> f32 {
        self.energy_at(dr.length())
    }
}

#[derive(Clone)]
/// A 1D symmetric potential model for an angle.
/// With this boundary, angles will be softly restricted to the range
/// [-ang_max, ang_max].
struct TrapezoidalCosineBoundary {
    ang_max: f32,
    slope: f32,
    /// |cos| above which the angle is surely within `ang_max`, so that `acos` does not
    /// have to be called. See `cos_threshold`.
    cos_thresh: f32,
}

/// Return the |cos| above which `acos(|cos|) <= ang_max` surely holds.
///
/// `acos` is monotonically decreasing, so the exact threshold is `cos(ang_max)`. The
/// margin covers the rounding errors of `cos` and `acos` (a few ulps of the angle,
/// which is orders of magnitude smaller than the margin); inputs that fall inside the
/// margin take the `acos` path, so the result is bit-wise identical to always calling
/// `acos`. `ang_max` greater than pi/2 can never be exceeded because `acos` of a
/// non-negative value is at most pi/2.
fn cos_threshold(ang_max: f32) -> f32 {
    const MARGIN: f32 = 1e-6;
    if ang_max > std::f32::consts::FRAC_PI_2 {
        -1.0  // |cos| >= -1 always holds, i.e. the energy is always zero.
    } else {
        ang_max.cos() + MARGIN
    }
}

impl TrapezoidalCosineBoundary {
    pub fn new(ang_max: f32, slope: f32) -> PyResult<Self> {
        if ang_max <= 0.0 {
            return value_error!("Maximum angle must be positive");
        }
        Ok(
            Self { ang_max, slope, cos_thresh: cos_threshold(ang_max) }
        )
    }

    pub fn unbounded() -> Self {
        Self {
            ang_max: f32::INFINITY,
            slope: 0.0,
            cos_thresh: cos_threshold(f32::INFINITY),
        }
    }

    ///           o         Cosine is calculated as the angle between the
    ///    o     i+1        y axis and the vector from i to i+1. The y axis
    ///    i                of local coordinates is always parallel to the
    /// ---------------> y  y axis.
    /// Energy calculated from |cos| of the angle between the two vectors.
    /// Note that the energy does not depend on the sign of the cosine.
    pub fn energy_of_cos(&self, cos_abs: f32) -> f32 {
        if cos_abs >= self.cos_thresh {
            // The angle is surely within the allowed range; `acos` is not needed.
            return 0.0;
        }
        let ang = cos_abs.acos();
        if ang > self.ang_max {
            self.slope * (ang - self.ang_max)
        } else {
            0.0
        }
    }

    pub fn energy(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>) -> f32 {
        self.energy_of_cos(dr.cos_angle(vec).abs())
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

impl BindingPotential for TrapezoidalPotential2D {
    /// Cool the potential by increasing the slope of the trapezoid.
    fn cool(&mut self, n: usize) {
        let slope = self.cooling_rate * n as f32;
        self.lon.slope = slope;
        self.lat.slope = slope;
        self.angle.slope = slope;
    }
}

impl BindingPotential2D for TrapezoidalPotential2D {
    fn energy_of(&self, scalars: &EdgeScalars, typ: &EdgeType) -> f32 {
        match typ {
            EdgeType::Longitudinal => {
                // Energy coming from the longitudinal distance and the angle
                let eng_dist = self.lon.energy_at(scalars.dist);
                let eng_ang = self.angle.energy_of_cos(scalars.cos_abs);
                eng_dist + eng_ang
            }
            EdgeType::Lateral => self.lat.energy_at(scalars.dist),
        }
    }
}

#[derive(Clone)]
pub struct StiffFilamentPotential {
    lon: TrapezoidalBoundary,
    angle: TrapezoidalCosineBoundary,
    cooling_rate: f32,
}

impl StiffFilamentPotential {
    pub fn new(
        lon_dist_min: f32,
        lon_dist_max: f32,
        lon_ang_max: f32,
        cooling_rate: f32,
    ) -> PyResult<Self> {
        if cooling_rate < 0.0 {
            return value_error!("Cooling rate must be non-negative");
        }

        Ok(
            Self {
                lon: TrapezoidalBoundary::new(lon_dist_min, lon_dist_max, 0.0)?,
                angle: TrapezoidalCosineBoundary::new(lon_ang_max, 0.0)?,
                cooling_rate,
            }
        )
    }

    pub fn unbounded() -> Self {
        Self {
            lon: TrapezoidalBoundary::unbounded(),
            angle: TrapezoidalCosineBoundary::unbounded(),
            cooling_rate: 0.0,
        }
    }

    pub fn with_dist(&self, min: f32, max: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.lon = TrapezoidalBoundary::new(min, max, self.lon.slope)?;
        Ok(new)
    }

    pub fn with_ang(&self, max: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.angle = TrapezoidalCosineBoundary::new(max, self.angle.slope)?;
        Ok(new)
    }

    pub fn with_cooling_rate(&self, cooling_rate: f32) -> Self {
        let mut new = self.clone();
        new.cooling_rate = cooling_rate;
        new
    }

    /// Energy coming from longitudinal distance
    pub fn calculate_bind(&self, dr: &Vector3D<f32>) -> f32 {
        self.lon.energy(dr)
    }

    /// Energy coming from deformation (curvature of the filament)
    pub fn calculate_deform(&self, dr1: &Vector3D<f32>, dr2: &Vector3D<f32>) -> f32 {
        let eng_ang = self.angle.energy(dr1, dr2);
        eng_ang
    }
}

impl BindingPotential for StiffFilamentPotential {
    /// Cool the potential by increasing the slope of the trapezoid.
    fn cool(&mut self, n: usize) {
        let slope = self.cooling_rate * n as f32;
        self.lon.slope = slope;
        self.angle.slope = slope;
    }
}

#[derive(Clone)]
struct LennardJonesLikeBoundary {
    dist_min: f32,
    dist_max: f32,
    slope: f32,
    energy_inf: f32,  // The energy when the distance is infinity.
}

impl LennardJonesLikeBoundary {
    pub fn new(dist_min: f32, dist_max: f32, slope: f32, energy_inf: f32) -> PyResult<Self> {
        if dist_min < 0.0 || dist_max < 0.0 {
            return value_error!("All distances must be positive");
        } else if dist_min >= dist_max {
            return value_error!("Minimum distance must be smaller than maximum distance");
        } else if energy_inf < 0.0 {
            return value_error!("Energy at infinity must be non-negative");
        }
        Ok(Self { dist_min, dist_max, slope, energy_inf })
    }

    /// An unbounded version of the model.
    pub fn unbounded() -> Self {
        Self {
            dist_min: 0.0,
            dist_max: f32::INFINITY,
            slope: 0.0,
            energy_inf: 0.0,
        }
    }

    /// Calculated energy of the given distance.
    pub fn energy_at(&self, dist: f32) -> f32 {
        if dist < self.dist_min {
            self.slope * (self.dist_min - dist)
        } else if self.dist_max < dist {
            self.energy_inf * (1.0 - (-self.slope * (dist - self.dist_max)).exp())
        } else {
            0.0
        }
    }
}


#[derive(Clone)]
pub struct LennardJonesLikePotential2D {
    lon: LennardJonesLikeBoundary,
    lat: LennardJonesLikeBoundary,
    angle: TrapezoidalCosineBoundary,
    cooling_rate: f32,
}

impl LennardJonesLikePotential2D {
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
                lon: LennardJonesLikeBoundary::new(lon_dist_min, lon_dist_max, 0.0, 0.0)?,
                lat: LennardJonesLikeBoundary::new(lat_dist_min, lat_dist_max, 0.0, 0.0)?,
                angle: TrapezoidalCosineBoundary::new(lon_ang_max, 0.0)?,
                cooling_rate,
            }
        )
    }

    pub fn unbounded() -> Self {
        Self {
            lon: LennardJonesLikeBoundary::unbounded(),
            lat: LennardJonesLikeBoundary::unbounded(),
            angle: TrapezoidalCosineBoundary::unbounded(),
            cooling_rate: 0.0,
        }
    }

    pub fn with_lon_dist(&self, min: f32, max: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.lon = LennardJonesLikeBoundary::new(min, max, self.lon.slope, self.lon.energy_inf)?;
        Ok(new)
    }

    pub fn with_lat_dist(&self, min: f32, max: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.lat = LennardJonesLikeBoundary::new(min, max, self.lat.slope, self.lat.energy_inf)?;
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

    pub fn with_energy_inf(&self, lon_energy_inf: f32, lat_energy_inf: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.lon = LennardJonesLikeBoundary::new(new.lon.dist_min, new.lon.dist_max, new.lon.slope, lon_energy_inf)?;
        new.lat = LennardJonesLikeBoundary::new(new.lat.dist_min, new.lat.dist_max, new.lat.slope, lat_energy_inf)?;
        Ok(new)
    }
}

impl BindingPotential for LennardJonesLikePotential2D {
    /// Cool the potential by increasing the slope of the trapezoid.
    fn cool(&mut self, n: usize) {
        let slope = self.cooling_rate * n as f32;
        self.lon.slope = slope;
        self.lat.slope = slope;
        self.angle.slope = slope;
    }
}

impl BindingPotential2D for LennardJonesLikePotential2D {
    fn energy_of(&self, scalars: &EdgeScalars, typ: &EdgeType) -> f32 {
        match typ {
            EdgeType::Longitudinal => {
                // Energy coming from the longitudinal distance and the angle
                let eng_dist = self.lon.energy_at(scalars.dist);
                let eng_ang = self.angle.energy_of_cos(scalars.cos_abs);
                eng_dist + eng_ang
            }
            EdgeType::Lateral => self.lat.energy_at(scalars.dist),
        }
    }
}
