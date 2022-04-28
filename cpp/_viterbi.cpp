#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "coords.h"

namespace py = pybind11;
using ssize_t = Py_ssize_t;

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
			auto end_point = coords[t].at(z1, y1, x1);
			for (auto z0 = 0; z0 < nz; ++z0) {
			for (auto y0 = 0; y0 < nx; ++y0) {
				// If distances are not in the range of [dist_min, dist_max] at the edges, i.e., 
				// x=0 and x=nx-1, then other points are not in the range either.
				// Since valid range of distance is relatively small, this check largely improves
				// performance.
				auto distance2_0 = (coords[t-1].at(z0, y0, 0) - end_point).length2();
				auto distance2_1 = (coords[t-1].at(z0, y0, nx-1) - end_point).length2();
				bool is_0_smaller = distance2_0 < dist_min2;
				bool is_0_larger = dist_max2 < distance2_0;
				bool is_1_smaller = distance2_1 < dist_min2;
				bool is_1_larger = dist_max2 < distance2_1;
				auto both_smaller = is_0_smaller && is_1_smaller;
				auto both_larger = is_0_larger && is_1_larger;
				
				if (both_smaller || both_larger) {
					continue;
				}

				if (!is_0_smaller && !is_0_larger) {
					if (!is_1_smaller && !is_1_larger) {
						max = std::max(
							{
								max, 
								viterbi_lattice(t - 1, z0, y0, 0), 
								viterbi_lattice(t - 1, z0, y0, nx-1)
							}
						);
					}
					else {
						max = std::max(max, viterbi_lattice(t - 1, z0, y0, 0));
					}
					neighbor_found = true;
				}
				else if (!is_1_smaller && !is_1_larger) {
					max = std::max(max, viterbi_lattice(t - 1, z0, y0, nx-1));
					neighbor_found = true;
				}

				for (auto x0 = 1; x0 < nx-1; ++x0) {
					auto distance2 = (coords[t-1].at(z0, y0, x0) - end_point).length2();
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

std::tuple<py::array_t<ssize_t>, double> viterbiAngularConstraint(
	py::array_t<double> score,
	py::array_t<double> origin,
	py::array_t<double> zvec,
	py::array_t<double> yvec,
	py::array_t<double> xvec,
	double dist_min,  // NOTE: upsample factor must be considered
	double dist_max,
	double skew_max
)
{
	auto dist_min2 = dist_min * dist_min;
	auto dist_max2 = dist_max * dist_max;
	auto cos_skew = std::cos(skew_max);

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
		auto origin_vector = coords[t-1].origin - coords[t].origin;
		auto b2 = origin_vector.length2();
		for (auto z1 = 0; z1 < nz; ++z1) {
		for (auto y1 = 0; y1 < ny; ++y1) {
		for (auto x1 = 0; x1 < nx; ++x1) {
			auto max = -std::numeric_limits<double>::infinity();
			bool neighbor_found = false;
			auto end_point = coords[t].at(z1, y1, x1);
			for (auto z0 = 0; z0 < nz; ++z0) {
			for (auto y0 = 0; y0 < nx; ++y0) {
				// If distances are not in the range of [dist_min, dist_max] at the edges, i.e., 
				// x=0 and x=nx-1, then other points are not in the range either.
				// Since valid range of distance is relatively small, this check largely improves
				// performance.
				auto distance2_0 = (coords[t-1].at(z0, y0, 0) - end_point).length2();
				auto distance2_1 = (coords[t-1].at(z0, y0, nx-1) - end_point).length2();
				bool is_0_smaller = distance2_0 < dist_min2;
				bool is_0_larger = dist_max2 < distance2_0;
				bool is_1_smaller = distance2_1 < dist_min2;
				bool is_1_larger = dist_max2 < distance2_1;
				auto both_smaller = is_0_smaller && is_1_smaller;
				auto both_larger = is_0_larger && is_1_larger;
				
				if (both_smaller || both_larger) {
					continue;
				}

				for (auto x0 = 0; x0 < nx; ++x0) {
					auto vec = coords[t-1].at(z0, y0, x0) - end_point;
					auto distance2 = vec.length2();

					if (distance2 < dist_min2 || dist_max2 < distance2) {
						continue;
					}

					auto dot_prod = vec.dot(origin_vector);
					auto a2 = vec.length2();
					auto ab = std::sqrt(a2 * b2);
					auto cos = std::abs(dot_prod / (a2 + b2 - 2 * ab));

					if (cos < cos_skew) {
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
			// TODO: angle check
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
