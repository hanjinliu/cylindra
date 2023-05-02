use num::traits::real::Real;
use numpy::ndarray::{Array1, ArrayView1};

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
/// A simple 3D vector.
pub struct Vector3D<T> {
    pub z: T,
    pub y: T,
    pub x: T,
}

impl<T> Vector3D<T> {
    pub fn new(z: T, y: T, x: T) -> Vector3D<T> {
        Vector3D { z, y, x }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
///////////   Casting   /////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

impl From<Vector3D<isize>> for Vector3D<f32>{
    fn from(other: Vector3D<isize>) -> Vector3D<f32> {
        Vector3D {
            z: other.z as f32,
            y: other.y as f32,
            x: other.x as f32,
        }
    }
}

impl From<Vector3D<usize>> for Vector3D<isize>{
    fn from(other: Vector3D<usize>) -> Vector3D<isize> {
        Vector3D {
            z: other.z as isize,
            y: other.y as isize,
            x: other.x as isize,
        }
    }
}

impl From<Vector3D<usize>> for Vector3D<f32>{
    fn from(other: Vector3D<usize>) -> Vector3D<f32> {
        Vector3D {
            z: other.z as f32,
            y: other.y as f32,
            x: other.x as f32,
        }
    }
}

impl<T: Clone> From<Array1<T>> for Vector3D<T> {
    fn from(other: Array1<T>) -> Vector3D<T> {
        // let other = other.into_raw_vec();
        Vector3D {
            z: other[0].clone(),
            y: other[1].clone(),
            x: other[2].clone(),
        }
    }
}

impl<T: Clone> From<ArrayView1<'_, T>> for Vector3D<T> {
    fn from(other: ArrayView1<T>) -> Vector3D<T> {
        Vector3D {
            z: other[0].clone(),
            y: other[1].clone(),
            x: other[2].clone(),
        }
    }
}

impl<T> From<(T, T, T)> for Vector3D<T> {
    fn from(other: (T, T, T)) -> Vector3D<T> {
        Vector3D {
            z: other.0,
            y: other.1,
            x: other.2,
        }
    }
}

impl<T> Into<(T, T, T)> for Vector3D<T> {
    fn into(self) -> (T, T, T) {
        (self.z, self.y, self.x)
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
///////////   Vector metrics   //////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

impl<T: Real> Vector3D<T> {
    pub fn dot(&self, other: &Vector3D<T>) -> T {
        self.z * other.z + self.y * other.y + self.x * other.x
    }

    pub fn length(&self) -> T {
        self.length2().sqrt()
    }

    pub fn length2(&self) -> T {
        self.z * self.z + self.y * self.y + self.x * self.x
    }

    /// Angle between two vectors.
    pub fn angle(&self, other: Vector3D<T>) -> T {
        let dot_prd = self.dot(&other);
        let a2 = self.length2();
        let b2 = other.length2();
        let ab = (a2 * b2).sqrt();
        return dot_prd / (a2 + b2 - ab - ab);
    }

    /// Return the unit vector of `self`.
    pub fn normed(&self) -> Vector3D<T> {
        let len = self.length();
        Vector3D::new(self.z / len, self.y / len, self.x / len)
    }

    /// Get the squared distance between point `self` and the plane defined by a normal
    /// vector `norm` and the in-plane point `other`.
    pub fn point_to_plane_distance2(
        &self,
        norm: &Vector3D<T>,
        other: &Vector3D<T>,
    ) -> T where T: std::ops::Sub<T>{
        let dr = Vector3D::new(self.z - other.z, self.y - other.y, self.x - other.x);
        dr.dot(&norm).abs()
    }
}

/////////////////////////////////////////////////////////////////////////////////////////
///////////   Operators   ///////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

impl<T: Real> std::ops::Add<T> for Vector3D<T> {
    type Output = Vector3D<T>;

    fn add(self, other: T) -> Vector3D<T> {
        Vector3D::new(self.z + other, self.y + other, self.x + other)
    }
}

impl<T: Real> std::ops::Add<Vector3D<T>> for Vector3D<T> {
    type Output = Vector3D<T>;

    fn add(self, other: Vector3D<T>) -> Vector3D<T> {
        Vector3D::new(self.z + other.z, self.y + other.y, self.x + other.x)
    }
}

impl<T: Real> std::ops::Add<&Vector3D<T>> for Vector3D<T> {
    type Output = Vector3D<T>;

    fn add(self, other: &Vector3D<T>) -> Vector3D<T> {
        Vector3D::new(self.z + other.z, self.y + other.y, self.x + other.x)
    }
}

impl<T: std::ops::Sub<Output = T> + Copy> std::ops::Sub<T> for Vector3D<T> {
    type Output = Vector3D<T>;

    fn sub(self, other: T) -> Vector3D<T> {
        Vector3D::new(self.z - other, self.y - other, self.x - other)
    }
}

impl<T: Real> std::ops::Sub<Vector3D<T>> for Vector3D<T> {
    type Output = Vector3D<T>;

    fn sub(self, other: Vector3D<T>) -> Vector3D<T> {
        Vector3D::new(self.z - other.z, self.y - other.y, self.x - other.x)
    }
}

impl<T: Real> std::ops::Sub<&Vector3D<T>> for &Vector3D<T> {
    type Output = Vector3D<T>;

    fn sub(self, other: &Vector3D<T>) -> Vector3D<T> {
        Vector3D::new(self.z - other.z, self.y - other.y, self.x - other.x)
    }
}

impl<T: Real> std::ops::Mul<T> for Vector3D<T> {
    type Output = Vector3D<T>;

    fn mul(self, other: T) -> Vector3D<T> {
        Vector3D::new(self.z * other, self.y * other, self.x * other)
    }
}

impl<T: Real> std::ops::Div<T> for Vector3D<T> {
    type Output = Vector3D<T>;

    fn div(self, other: T) -> Vector3D<T> {
        Vector3D::new(self.z / other, self.y / other, self.x / other)
    }
}
