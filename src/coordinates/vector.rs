use num::traits::real::Real;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
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

impl From<Vector3D<usize>> for Vector3D<f32>{
    fn from(other: Vector3D<usize>) -> Vector3D<f32> {
        Vector3D {
            z: other.z as f32,
            y: other.y as f32,
            x: other.x as f32,
        }
    }
}

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

    pub fn angle(&self, other: Vector3D<T>) -> T {
        let dot_prd = self.dot(&other);
        let a2 = self.length2();
        let b2 = other.length2();
        let ab = (a2 * b2).sqrt();
        return dot_prd / (a2 + b2 - ab - ab);
    }

    pub fn normed(&self) -> Vector3D<T> {
        let len = self.length();
        Vector3D::new(self.z / len, self.y / len, self.x / len)
    }

    pub fn point_to_plane_distance2(&self, norm: &Vector3D<T>, other: &Vector3D<T>) -> T where T: std::ops::Sub<T>{
        let dr = Vector3D::new(self.z - other.z, self.y - other.y, self.x - other.x);
        dr.dot(&norm).abs()
    }
}

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
