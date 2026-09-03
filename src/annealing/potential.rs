use pyo3::prelude::PyResult;
use crate::{value_error, coordinates::Vector3D};

pub trait BindingPotential {
    fn cool(&mut self, _n: usize) {
        // Do nothing by default.
    }

}
pub trait BindingPotential2D : BindingPotential {
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
}

/// Binding potential used by a microtubule lattice graph. Unlike `BindingPotential2D`,
/// longitudinal binding does not carry an angle constraint: the angle (curvature)
/// constraint is asymmetric and is instead applied separately, between a molecule and
/// its two longitudinal neighbors, via `calculate_deform`.
pub trait MicrotubuleBindingPotential : BindingPotential {
    /// Energy coming from longitudinal distance.
    fn calculate_bind(&self, dr: &Vector3D<f32>) -> f32;
    /// Energy coming from lateral distance.
    fn calculate_lat_bind(&self, dr: &Vector3D<f32>) -> f32;
    /// Energy coming from deformation (curvature of the protofilament). `dz` is the
    /// local outward-pointing normal of the coordinate system at the center molecule,
    /// used to distinguish curling outward (away from the microtubule axis) from
    /// curling inward (toward the axis).
    fn calculate_deform(&self, dr1: &Vector3D<f32>, dr2: &Vector3D<f32>, dz: &Vector3D<f32>) -> f32;
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
    _dist_min2: f32,
    _dist_max2: f32,
}

impl TrapezoidalBoundary {
    pub fn new(dist_min: f32, dist_max: f32, slope: f32) -> PyResult<Self> {
        if dist_min < 0.0 || dist_max < 0.0 {
            return value_error!("All distances must be positive");
        } else if dist_min >= dist_max {
            return value_error!("Minimum distance must be smaller than maximum distance");
        }
        Ok(Self { dist_min, dist_max, slope, _dist_min2: dist_min * dist_min, _dist_max2: dist_max * dist_max })
    }

    /// An unbounded version of the model.
    pub fn unbounded() -> Self {
        Self {
            dist_min: 0.0,
            dist_max: f32::INFINITY,
            slope: 0.0,
            _dist_min2: 0.0,
            _dist_max2: f32::INFINITY,
        }
    }

    /// Calculated energy of given vector.
    pub fn energy(&self, dr: &Vector3D<f32>) -> f32 {
        let dist2 = dr.length2();
        if self._dist_min2 <= dist2 {
            if dist2 <= self._dist_max2 {
                0.0
            } else {
                self.slope * (dist2.sqrt() - self.dist_max)
            }
        } else {
            self.slope * (self.dist_min - dist2.sqrt())
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
    _cos_ang_min: f32,
}

impl TrapezoidalCosineBoundary {
    pub fn new(ang_max: f32, slope: f32) -> PyResult<Self> {
        if ang_max <= 0.0 {
            return value_error!("Maximum angle must be positive");
        }
        Ok(
            Self { ang_max, slope, _cos_ang_min: (ang_max).cos(), }
        )
    }

    pub fn unbounded() -> Self {
        Self { ang_max: f32::INFINITY, slope: 0.0, _cos_ang_min: 0.0 }
    }

    ///           o         Cosine is calculated as the angle between the
    ///    o     i+1        y axis and the vector from i to i+1. The y axis
    ///    i                of local coordinates is always parallel to the
    /// ---------------> y  y axis.
    pub fn energy(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>) -> f32 {
        let cos_angle = dr.cos_angle(vec).abs();
        if cos_angle >= self._cos_ang_min {
            0.0
        } else {
            let ang = cos_angle.acos();
            self.slope * (ang - self.ang_max)
        }
    }
}

#[derive(Clone)]
struct AsymmetricCosineBoundary {
    ang_min: f32,
    ang_max: f32,
    slope: f32,
}

impl AsymmetricCosineBoundary {
    pub fn new(ang_min: f32, ang_max: f32, slope: f32) -> PyResult<Self> {
        if ang_min >= ang_max {
            return value_error!("Minimum angle must be smaller than maximum angle");
        }
        Ok(Self { ang_min, ang_max, slope })
    }

    pub fn unbounded() -> Self {
        Self { ang_min: 0.0, ang_max: f32::INFINITY, slope: 0.0, }
    }

    pub fn energy(
        &self,
        dr: &Vector3D<f32>,
        vec: &Vector3D<f32>,
        dz: &Vector3D<f32>,
    ) -> f32 {
        // NOTE: vec _|_ dz
        let ang = dr.cos_angle(vec).abs().acos();
        let sign = if dr.dot(dz) >= 0.0 { 1.0 } else { -1.0 };
        let signed_ang = sign * ang;
        if signed_ang < self.ang_min {
            self.slope * (self.ang_min - signed_ang)
        } else if self.ang_max < signed_ang {
            self.slope * (signed_ang - self.ang_max)
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
    fn longitudinal(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>) -> f32 {
        // Energy coming from longitudinal distance
        let eng_dist = self.lon.energy(dr);
        let eng_ang = self.angle.energy(dr, vec);
        eng_dist + eng_ang
    }

    fn lateral(&self, dr: &Vector3D<f32>, _vec: &Vector3D<f32>) -> f32 {
        self.lat.energy(&dr)
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
    _dist_min2: f32,
    _dist_max2: f32,
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
        Ok(Self { dist_min, dist_max, slope, energy_inf, _dist_min2: dist_min * dist_min, _dist_max2: dist_max * dist_max })
    }

    /// An unbounded version of the model.
    pub fn unbounded() -> Self {
        Self {
            dist_min: 0.0,
            dist_max: f32::INFINITY,
            slope: 0.0,
            energy_inf: 0.0,
            _dist_min2: 0.0,
            _dist_max2: f32::INFINITY,
        }
    }

    /// Calculated energy of given square of distance.
    pub fn energy(&self, dr: &Vector3D<f32>) -> f32 {
        let dist2 = dr.length2();
        if self._dist_min2 <= dist2 {
            if dist2 <= self._dist_max2 {
                0.0
            } else {
                self.energy_inf * (1.0 - (-self.slope * (dist2.sqrt() - self.dist_max)).exp())
            }
        } else {
            self.slope * (self.dist_min - dist2.sqrt())
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
    fn longitudinal(&self, dr: &Vector3D<f32>, vec: &Vector3D<f32>) -> f32 {
        // Energy coming from longitudinal distance
        let eng_dist = self.lon.energy(dr);
        let eng_ang = self.angle.energy(dr, vec);
        eng_dist + eng_ang
    }

    fn lateral(&self, dr: &Vector3D<f32>, _vec: &Vector3D<f32>) -> f32 {
        self.lat.energy(&dr)
    }
}

#[derive(Clone)]
pub struct MicrotubulePotential {
    lon: TrapezoidalBoundary,
    lat: TrapezoidalBoundary,
    angle: AsymmetricCosineBoundary,
    cooling_rate: f32,
}

impl MicrotubulePotential {
    pub fn new(
        lon_dist_min: f32,
        lon_dist_max: f32,
        lat_dist_min: f32,
        lat_dist_max: f32,
        lon_ang_min: f32,
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
                angle: AsymmetricCosineBoundary::new(lon_ang_min, lon_ang_max, 0.0)?,
                cooling_rate,
            }
        )
    }

    pub fn unbounded() -> Self {
        Self {
            lon: TrapezoidalBoundary::unbounded(),
            lat: TrapezoidalBoundary::unbounded(),
            angle: AsymmetricCosineBoundary::unbounded(),
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

    pub fn with_lon_ang(&self, min: f32, max: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.angle = AsymmetricCosineBoundary::new(min, max, self.angle.slope)?;
        Ok(new)
    }

    pub fn with_cooling_rate(&self, cooling_rate: f32) -> Self {
        let mut new = self.clone();
        new.cooling_rate = cooling_rate;
        new
    }
}

impl BindingPotential for MicrotubulePotential {
    /// Cool the potential by increasing the slope of the trapezoid.
    fn cool(&mut self, n: usize) {
        let slope = self.cooling_rate * n as f32;
        self.lon.slope = slope;
        self.lat.slope = slope;
        self.angle.slope = slope;
    }
}

impl MicrotubuleBindingPotential for MicrotubulePotential {
    fn calculate_bind(&self, dr: &Vector3D<f32>) -> f32 {
        self.lon.energy(dr)
    }

    fn calculate_lat_bind(&self, dr: &Vector3D<f32>) -> f32 {
        self.lat.energy(dr)
    }

    fn calculate_deform(&self, dr1: &Vector3D<f32>, dr2: &Vector3D<f32>, dz: &Vector3D<f32>) -> f32 {
        self.angle.energy(dr1, dr2, dz)
    }
}

#[derive(Clone)]
/// Microtubule binding potential with a Lennard-Jones-like longitudinal/lateral
/// distance boundary: unlike `MicrotubulePotential`'s hard trapezoidal boundary, the
/// energy softly saturates at `energy_inf` as the distance grows beyond `dist_max`,
/// allowing molecules to separate further apart than the cutoff distance if that is
/// favorable elsewhere.
pub struct MicrotubulePotentialLJ {
    lon: LennardJonesLikeBoundary,
    lat: LennardJonesLikeBoundary,
    angle: AsymmetricCosineBoundary,
    cooling_rate: f32,
}

impl MicrotubulePotentialLJ {
    pub fn new(
        lon_dist_min: f32,
        lon_dist_max: f32,
        lat_dist_min: f32,
        lat_dist_max: f32,
        lon_ang_min: f32,
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
                angle: AsymmetricCosineBoundary::new(lon_ang_min, lon_ang_max, 0.0)?,
                cooling_rate,
            }
        )
    }

    pub fn unbounded() -> Self {
        Self {
            lon: LennardJonesLikeBoundary::unbounded(),
            lat: LennardJonesLikeBoundary::unbounded(),
            angle: AsymmetricCosineBoundary::unbounded(),
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

    pub fn with_lon_ang(&self, min: f32, max: f32) -> PyResult<Self> {
        let mut new = self.clone();
        new.angle = AsymmetricCosineBoundary::new(min, max, self.angle.slope)?;
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

impl BindingPotential for MicrotubulePotentialLJ {
    /// Cool the potential by increasing the slope of the trapezoid.
    fn cool(&mut self, n: usize) {
        let slope = self.cooling_rate * n as f32;
        self.lon.slope = slope;
        self.lat.slope = slope;
        self.angle.slope = slope;
    }
}

impl MicrotubuleBindingPotential for MicrotubulePotentialLJ {
    fn calculate_bind(&self, dr: &Vector3D<f32>) -> f32 {
        self.lon.energy(dr)
    }

    fn calculate_lat_bind(&self, dr: &Vector3D<f32>) -> f32 {
        self.lat.energy(dr)
    }

    fn calculate_deform(&self, dr1: &Vector3D<f32>, dr2: &Vector3D<f32>, dz: &Vector3D<f32>) -> f32 {
        self.angle.energy(dr1, dr2, dz)
    }
}
