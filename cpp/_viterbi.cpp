#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using ssize_t = Py_ssize_t;

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
		T length2();
		T length();
		Vector3D(T z0, T y0, T x0){z=z0; y=y0; x=x0;}
		Vector3D() : Vector3D<T>(0, 0, 0){}
		Vector3D(Vector3D<T> &vec) : Vector3D<T>(vec.z, vec.y, vec.x){}
		Vector3D(py::array_t<T> &vec): Vector3D<T>(vec.data(0), vec.data(1), vec.data(2)){}
		void update(T z0, T y0, T x0){z=z0; y=y0; x=x0;};
		py::array_t<T> asarray();
};

// vector + vector
template<typename T>
Vector3D<T> Vector3D<T>::operator+(Vector3D<T> &other) {
   return Vector3D<T>(z + other.z, y + other.y, x + other.x);
}

// vector + scalar
template<typename T>
Vector3D<T> Vector3D<T>::operator+(T other) {
   return Vector3D<T>(z + other, y + other, x + other);
}

// vector - vector
template<typename T>
Vector3D<T> Vector3D<T>::operator-(Vector3D<T> &other) {
   return Vector3D<T>(z - other.z, y - other.y, x - other.x);
}

// vector - scalar
template<typename T>
Vector3D<T> Vector3D<T>::operator-(T other) {
   return Vector3D<T>(z - other, y - other, x - other);
}

// vector * scalar
template<typename T>
Vector3D<T> Vector3D<T>::operator*(T other) {
   return Vector3D<T>(z * other, y * other, x * other);
}

// vector / scalar
template<typename T>
Vector3D<T> Vector3D<T>::operator/(T other) {
   return Vector3D<T>(z / other, y / other, x / other);
}

template<typename T>
T Vector3D<T>::length2() {
	return z*z + y*y + x*x;
}

template<typename T>
T Vector3D<T>::length() {
	return std::sqrt(length2());
}

// convert a vector object into np.ndarray
template<typename T>
py::array_t<T> Vector3D<T>::asarray() {
	auto arr_ = py::array_t<T>{{3}};
	auto arr = arr_.mutable_unchecked<1>();
	arr(0) = z; arr(1) = y; arr(2) = x;
	return arr_;
}

template <typename T>
class CoordinateSystem {
	public:
		Vector3D<T> origin; // world coordinate of the origin
		Vector3D<T> ez;		// world coordinate of z-axis unit vector
		Vector3D<T> ey;		// world coordinate of y-axis unit vector
		Vector3D<T> ex;		// world coordinate of x-axis unit vector
		Vector3D<T> at(T, T, T);
		Vector3D<T> at(Vector3D<T>);
		CoordinateSystem(Vector3D<T> &origin, Vector3D<T> &ez, Vector3D<T> &ey, Vector3D<T> &ex) {
			origin = origin; ez = ez; ey = ey; ex = ex;
		}
		CoordinateSystem(py::array_t<T> &origin, py::array_t<T> &ez, py::array_t<T> &ey, py::array_t<T> &ex)
			: CoordinateSystem(Vector3D<T>(origin), Vector3D<T>(ez), Vector3D<T>(ey), Vector3D<T>(ex))
		{}
		CoordinateSystem()
			: CoordinateSystem(Vector3D<T>(), Vector3D<T>(), Vector3D<T>(), Vector3D<T>())
		{}
		void update(Vector3D<T> &origin_, Vector3D<T> &ez_, Vector3D<T> &ey_, Vector3D<T> &ex_){
			origin = origin_; ez = ez_; ey = ey_; ex = ex_;
		}
		void update(py::array_t<T> &origin, py::array_t<T> &ez, py::array_t<T> &ey, py::array_t<T> &ex){
			return update(Vector3D<T>(&origin), Vector3D<T>(&ez), Vector3D<T>(&ey), Vector3D<T>(&ex));
		}
};

// Convert local coordinates to world coordinates.
template <typename T>
Vector3D<T> CoordinateSystem<T>::at(T z, T y, T x){
	return origin + ez * z + ey * y + ex * x;
}

// Convert local coordinates to world coordinates, using 3D vector.
template <typename T>
Vector3D<T> CoordinateSystem<T>::at(Vector3D<T> vec){
	return origin + ez * vec.z + ey * vec.y + ex * vec.x;
}

std::tuple<py::array_t<ssize_t>, double> viterbi(
	py::array_t<double> score,
	py::array_t<double> origin,
	py::array_t<double> zvec,
	py::array_t<double> yvec,
	py::array_t<double> xvec,
	double dist_min,  // NOTE: upsample factor must be considered
	double dist_max
)
{
	auto dist_min2 = dist_min * dist_min;
	auto dist_max2 = dist_max * dist_max;

	// get buffers
	py::buffer_info _score_info = score.request();

	// score has shape (N, Z, Y, X)
	ssize_t nmole = _score_info.shape[0];
	ssize_t nz = _score_info.shape[1];
	ssize_t ny = _score_info.shape[2];
	ssize_t nx = _score_info.shape[3];

	// prepare arrays
	auto state_sequence_ = py::array_t<ssize_t>{{nmole, ssize_t(3)}};
	auto viterbi_lattice_ = py::array_t<double>{{nmole, nz, ny, nx}};
	auto state_sequence = state_sequence_.mutable_unchecked<2>();
	auto viterbi_lattice = viterbi_lattice_.mutable_unchecked<4>();
	auto nogil = py::gil_scoped_release{};  // without GIL

	// initialization at t = 0
	for (auto z = 0; z < nz; ++z) {
	for (auto y = 0; y < ny; ++y) {
	for (auto x = 0; x < nx; ++x) {
		viterbi_lattice(0, z, y, x) = *score.data(0, z, y, x);
	}}}
	
	// Allocation of arrays of coordinate system.
	// Offsets and orientations of local coordinates of score landscape are well-defined by this.
	auto coords = new CoordinateSystem<double>[nmole];
	
	for (auto t = 0; t < nmole; ++t) {
		auto _ori = Vector3D<double>(*origin.data(t, 0), *origin.data(t, 1), *origin.data(t, 2));
		auto _ez = Vector3D<double>(*zvec.data(t, 0), *zvec.data(t, 1), *zvec.data(t, 2));
		auto _ey = Vector3D<double>(*yvec.data(t, 0), *yvec.data(t, 1), *yvec.data(t, 2));
		auto _ex = Vector3D<double>(*xvec.data(t, 0), *xvec.data(t, 1), *xvec.data(t, 2));
		coords[t].update(_ori, _ez, _ey, _ex);
	}

	// forward
	for (auto t = 1; t < nmole; ++t) {
		for (auto z1 = 0; z1 < nz; ++z1) {
		for (auto y1 = 0; y1 < ny; ++y1) {
		for (auto x1 = 0; x1 < nx; ++x1) {
			auto max = -std::numeric_limits<double>::infinity();
			bool neighbor_found = false;
			for (auto z0 = 0; z0 < nz; ++z0) {
			for (auto y0 = 0; y0 < ny; ++y0) {
			for (auto x0 = 0; x0 < nx; ++x0) {
				auto distance2 = (coords[t-1].at(z0, y0, x0) - coords[t].at(z1, y1, x1)).length2();
				if (distance2 < dist_min2 || dist_max2 < distance2) {
					continue;
				}
				neighbor_found = true;
				max = std::max(max, viterbi_lattice(t - 1, z0, y0, x0));
			}}}
			
			if (!neighbor_found) {
				char buf[128];
				std::sprintf(buf, "No neighbor found between %d and %d.", t-1, t);
				throw py::value_error(buf);
			}
			auto next_score = score.data(t, z1, y1, x1);
			viterbi_lattice(t, z1, y1, x1) = max + *next_score;
		}}}
	}

	// find maximum score
	double max_score = -std::numeric_limits<double>::infinity();
	auto prev = Vector3D<int>(0, 0, 0);
	
	for (auto z = 0; z < nz; ++z) {
	for (auto y = 0; y < ny; ++y) {
	for (auto x = 0; x < nx; ++x) {
		auto s = viterbi_lattice(nmole - 1, z, y, x);
		if (s > max_score) {
			max_score = s;
			prev.z = z;
			prev.y = y;
			prev.x = x;
		}
	}}}

	state_sequence(nmole-1, 0) = prev.z;
	state_sequence(nmole-1, 1) = prev.y;
	state_sequence(nmole-1, 2) = prev.x;

	// backward tracking
	for (auto t = nmole - 2; t >= 0; --t) {
		double max = -std::numeric_limits<double>::infinity();
		auto argmax = Vector3D<int>(0, 0, 0);
		auto point_prev = coords[t+1].at(prev.z, prev.y, prev.x);
		for (auto z0 = 0; z0 < nz; ++z0) {
		for (auto y0 = 0; y0 < ny; ++y0) {
		for (auto x0 = 0; x0 < nx; ++x0) {
			auto distance2 = (point_prev - coords[t].at(z0, y0, x0)).length2();
			if (distance2 < dist_min2 || dist_max2 < distance2) {
				continue;
			}
			auto value = viterbi_lattice(t, z0, y0, x0);
			if (max < value) {
				max = value;
				argmax = Vector3D<int>(z0, y0, x0);
			}
		}}}
		
		prev = argmax;
		state_sequence(t, 0) = prev.z;
		state_sequence(t, 1) = prev.y;
		state_sequence(t, 2) = prev.x;
	}

	return {state_sequence_, max_score};
}


PYBIND11_MODULE(_cpp_ext, m) {
	m.doc() = "C++ extensions";
  	m.def("viterbi", &viterbi, "Viterbi algorithm.");
}
