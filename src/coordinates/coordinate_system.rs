use num::traits::real::Real;
use super::vector::Vector3D;

#[derive(Clone)]
/// A local coordinate system around a molecule.
/// origin: the origin of the local coordinate in the world coordinate.
/// ez: the unit vector of z axis in the world coordinate.
/// ey: the unit vector of y axis in the world coordinate.
/// ex: the unit vector of x axis in the world coordinate.
pub struct CoordinateSystem<T> {
    pub origin: Vector3D<T>,
    pub ez: Vector3D<T>,
    pub ey: Vector3D<T>,
    pub ex: Vector3D<T>,
}

impl<T: Default> Default for CoordinateSystem<T> {
    fn default() -> Self {
        Self {
            origin: Vector3D::default(),
            ez: Vector3D::default(),
            ey: Vector3D::default(),
            ex: Vector3D::default(),
        }
    }
}

impl<T> CoordinateSystem<T> {
    pub fn new(origin: Vector3D<T>, ez: Vector3D<T>, ey: Vector3D<T>, ex: Vector3D<T>) -> Self {
        Self { origin, ez, ey, ex }
    }

    /// Update (re-initialize) the coordinate system with given parameters.
    pub fn update(&mut self, origin: Vector3D<T>, ez: Vector3D<T>, ey: Vector3D<T>, ex: Vector3D<T>) {
        self.origin = origin;
        self.ez = ez;
        self.ey = ey;
        self.ex = ex;
    }
}

impl<T: Real> CoordinateSystem<T> {
    /// Get the world coordinate of the position (z, y, x) in the local coordinate.
    pub fn at(&self, z: T, y: T, x: T) -> Vector3D<T> {
        self.origin + self.ez * z + self.ey * y + self.ex * x
    }

    /// Vector version of `at`.
    pub fn at_vec(&self, vec: Vector3D<T>) -> Vector3D<T> {
        self.origin + self.ez * vec.z + self.ey * vec.y + self.ex * vec.x
    }

    /// Initialize the coordinate system.
    /// This method creates an invalid coordinate system. The `update` method must be called
    /// before using it.
    pub fn zeros() -> CoordinateSystem<T> {
        CoordinateSystem::new(
            Vector3D::new(T::from(0).unwrap(), T::from(0).unwrap(), T::from(0).unwrap()),
            Vector3D::new(T::from(0).unwrap(), T::from(0).unwrap(), T::from(0).unwrap()),
            Vector3D::new(T::from(0).unwrap(), T::from(0).unwrap(), T::from(0).unwrap()),
            Vector3D::new(T::from(0).unwrap(), T::from(0).unwrap(), T::from(0).unwrap()),
        )
    }
}
