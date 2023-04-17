#ifndef COORDS_H
#define COORDS_H

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// NOTE: The definition of Vector3D and CoordinateSystem cannot be moved into another
// cpp file because they are templated.

// A 3-D vector (or coordinate) object.
// Implemented with simple operations.
template <typename T>
class Vector3D {	
	public:
		T z; T y; T x;
		Vector3D<T> operator+(Vector3D<T> &other);
		Vector3D<T> operator+(T other);
		Vector3D<T> operator-(Vector3D<T> &other);
		Vector3D<T> operator-(T other);
		Vector3D<T> operator*(T other);
		Vector3D<T> operator/(T other);
        T dot(Vector3D<T> other);
        T angle(Vector3D<T> other);
		T length2();
		T length();
		Vector3D<T> normed();
		Vector3D(T z0, T y0, T x0) {z=z0; y=y0; x=x0;}
		Vector3D() : Vector3D<T>(0, 0, 0) {}
		Vector3D(Vector3D<T> &vec) : Vector3D<T>(vec.z, vec.y, vec.x) {}
		Vector3D(py::array_t<T> &vec): Vector3D<T>(*vec.data(0), *vec.data(1), *vec.data(2)) {}
		void update(T z0, T y0, T x0) {z=z0; y=y0; x=x0;};
		py::array_t<T> asarray();
		T pointToPlaneDistance2(Vector3D<T> &norm, Vector3D<T> &other);
};


// vector + vector
template<typename T>
inline Vector3D<T> Vector3D<T>::operator+(Vector3D<T> &other) {
   return Vector3D<T>(z + other.z, y + other.y, x + other.x);
}

// vector + scalar
template<typename T>
inline Vector3D<T> Vector3D<T>::operator+(T other) {
   return Vector3D<T>(z + other, y + other, x + other);
}

// vector - vector
template<typename T>
inline Vector3D<T> Vector3D<T>::operator-(Vector3D<T> &other) {
   return Vector3D<T>(z - other.z, y - other.y, x - other.x);
}

// vector - scalar
template<typename T>
inline Vector3D<T> Vector3D<T>::operator-(T other) {
   return Vector3D<T>(z - other, y - other, x - other);
}

// vector * scalar
template<typename T>
inline Vector3D<T> Vector3D<T>::operator*(T other) {
   return Vector3D<T>(z * other, y * other, x * other);
}

// vector / scalar
template<typename T>
inline Vector3D<T> Vector3D<T>::operator/(T other) {
   return Vector3D<T>(z / other, y / other, x / other);
}

// dot product of two vectors
template<typename T>
inline T Vector3D<T>::dot(Vector3D<T> other) {
    return z * other.z + y * other.y + x * other.x;
}

// the square of length.
template<typename T>
T Vector3D<T>::length2() {
	return z*z + y*y + x*x;
}

// return the cosine angle between two vectors.
template<typename T>
T Vector3D<T>::angle(Vector3D<T> other) {
    T dot_prod = dot(other);
    T a2 = length2();
    T b2 = other.length2();
    T ab = std::sqrt(a2 * b2);
    return dot_prod / (a2 + b2 - 2 * ab);
}

// length of the vector.
template<typename T>
T Vector3D<T>::length() {
	return std::sqrt(length2());
}

template<typename T>
Vector3D<T> Vector3D<T>::normed() {
	T len = length();
	return *this / len;
}

// convert a vector object into np.ndarray
template<typename T>
py::array_t<T> Vector3D<T>::asarray() {
	auto arr_ = py::array_t<T>{{3}};
	auto arr = arr_.mutable_unchecked<1>();
	arr(0) = z; arr(1) = y; arr(2) = x;
	return arr_;
}

/// Calculate the square of distance between the point `other` and the plane defined by the
/// normal vector `norm` and the point `this`. 
template<typename T>
T Vector3D<T>::pointToPlaneDistance2(Vector3D<T> &norm, Vector3D<T> &other) {
	return std::abs((other - *this).dot(norm));
}

// A CoordinateSystem defines a local Cartesian coordinate system.
template <typename T>
class CoordinateSystem {
	public:
		Vector3D<T> origin; // world coordinate of the origin
		Vector3D<T> ez;		// world coordinate of z-axis unit vector
		Vector3D<T> ey;		// world coordinate of y-axis unit vector
		Vector3D<T> ex;		// world coordinate of x-axis unit vector
		Vector3D<T> at(T, T, T);
		Vector3D<T> at(Vector3D<T>);
		CoordinateSystem(Vector3D<T> &origin_, Vector3D<T> &ez_, Vector3D<T> &ey_, Vector3D<T> &ex_) {
			origin = origin_; ez = ez_; ey = ey_; ex = ex_;
		}
		CoordinateSystem(py::array_t<T> &origin, py::array_t<T> &ez, py::array_t<T> &ey, py::array_t<T> &ex)
			: CoordinateSystem(Vector3D<T>(origin), Vector3D<T>(ez), Vector3D<T>(ey), Vector3D<T>(ex))
		{}
		CoordinateSystem()
			: CoordinateSystem(Vector3D<T>(), Vector3D<T>(), Vector3D<T>(), Vector3D<T>())
		{}
		void update(Vector3D<T> &origin_, Vector3D<T> &ez_, Vector3D<T> &ey_, Vector3D<T> &ex_) {
			origin = origin_; ez = ez_; ey = ey_; ex = ex_;
		}
		void update(py::array_t<T> &origin, py::array_t<T> &ez, py::array_t<T> &ey, py::array_t<T> &ex) {
			return update(Vector3D<T>(&origin), Vector3D<T>(&ez), Vector3D<T>(&ey), Vector3D<T>(&ex));
		}
};

// Convert local coordinates to world coordinates.
template <typename T>
Vector3D<T> CoordinateSystem<T>::at(T z, T y, T x) {
	return origin + ez * z + ey * y + ex * x;
}

// Convert local coordinates to world coordinates, using 3D vector.
template <typename T>
Vector3D<T> CoordinateSystem<T>::at(Vector3D<T> vec) {
	return origin + ez * vec.z + ey * vec.y + ex * vec.x;
}

#endif
