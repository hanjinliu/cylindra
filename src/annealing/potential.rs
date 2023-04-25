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

pub struct BoxPotential2D {
    lon_dist_min2: f32,
    lon_dist_max2: f32,
    lat_dist_min2: f32,
    lat_dist_max2: f32,
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

struct HarmonicPotential2D {
    halfk0: f32,
    halfk1: f32,
    r0: f32,
    r1: f32,
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
