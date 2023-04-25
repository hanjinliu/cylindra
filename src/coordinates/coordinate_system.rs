use num::traits::real::Real;
use super::vector::Vector3D;

pub struct CoordinateSystem<T> {
    pub origin: Vector3D<T>,
    pub ez: Vector3D<T>,
    pub ey: Vector3D<T>,
    pub ex: Vector3D<T>,
}

impl<T> CoordinateSystem<T> {
    pub fn new(origin: Vector3D<T>, ez: Vector3D<T>, ey: Vector3D<T>, ex: Vector3D<T>) -> Self {
        Self {
            origin,
            ez,
            ey,
            ex,
        }
    }
    pub fn update(&mut self, origin: Vector3D<T>, ez: Vector3D<T>, ey: Vector3D<T>, ex: Vector3D<T>) {
        self.origin = origin;
        self.ez = ez;
        self.ey = ey;
        self.ex = ex;
    }
}

impl<T: Real> CoordinateSystem<T> {
    pub fn at(&self, z: T, y: T, x: T) -> Vector3D<T> {
        self.origin + self.ez * z + self.ey * y + self.ex * x
    }
    pub fn at_vec(&self, vec: Vector3D<T>) -> Vector3D<T> {
        self.origin + self.ez * vec.z + self.ey * vec.y + self.ex * vec.x
    }
}
