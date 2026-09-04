use num::traits::real::Real;
use super::vector::Vector3D;
use crate::hash::HashMap2D;

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
    // Because `origin + ez * z + ey * y + ex * x` is frequently used,
    // we cache the results for `origin + ez * z + ey * y` and `ex * x`
    pub cache_ozy: HashMap2D<Vector3D<T>>,
    pub cache_x: Vec<Vector3D<T>>,
}

impl<T: Default> Default for CoordinateSystem<T> {
    fn default() -> Self {
        Self {
            origin: Vector3D::default(),
            ez: Vector3D::default(),
            ey: Vector3D::default(),
            ex: Vector3D::default(),
            cache_ozy: HashMap2D::new(),
            cache_x: Vec::new(),
        }
    }
}

impl<T> CoordinateSystem<T> {
    pub fn new(origin: Vector3D<T>, ez: Vector3D<T>, ey: Vector3D<T>, ex: Vector3D<T>) -> Self {
        Self {
            origin,
            ez,
            ey,
            ex,
            cache_ozy: HashMap2D::new(),
            cache_x: Vec::new(),
        }
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

    pub fn at_fast(&self, z: usize, y: usize, x: usize) -> Vector3D<T> {
        let ozy = self.cache_ozy[(z, y)];
        let ox = self.cache_x[x];
        ozy + ox
    }

    /// Faster version of `at_vec` using cached vectors.
    pub fn at_vec_fast(&self, vec: Vector3D<isize>) -> Vector3D<T> {
        let ozy = self.cache_ozy[(vec.z, vec.y)];
        let ox = self.cache_x[vec.x as usize];
        ozy + ox
    }

    pub fn with_cache(&self, max_z: usize, max_y: usize, max_x: usize) -> Self {
        let mut cache_ozy = HashMap2D::from_shape(max_y, max_x);
        for z in 0..max_z {
            for y in 0..max_y {
                cache_ozy.insert(
                    (z, y),
                    self.ez * T::from(z).unwrap() + self.origin + self.ey * T::from(y).unwrap());
            }
        }
        let mut cache_x = Vec::with_capacity(max_x);
        for x in 0..max_x {
            cache_x.push(self.ex * T::from(x).unwrap());
        }
        Self {
            origin: self.origin,
            ez: self.ez,
            ey: self.ey,
            ex: self.ex,
            cache_ozy,
            cache_x,
        }
    }
}
