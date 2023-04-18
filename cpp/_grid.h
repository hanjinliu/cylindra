#ifndef _GRID
#define _GRID

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_coords.h"
#include "_constraint.h"

namespace py = pybind11;
using ssize_t = Py_ssize_t;

/// A 1D grid for Viterbi alignment.
/// This class contains the score landscape, coordinate systems and the shape.
class ViterbiGrid {
    public:
        py::array_t<float> score; // A 4D array, where score_array[n, z, y, x] is the score of the n-th molecule at the grid point (z, y, x).
        CoordinateSystem<double>* coords;  // coordinate system of each molecule
        ssize_t nmole, nz, ny, nx;  // number of molecules, number of grid points in z, y, x directions

        ViterbiGrid (
            py::array_t<float> &score_array,
            py::array_t<float> &origin,
            py::array_t<float> &zvec,
            py::array_t<float> &yvec,
            py::array_t<float> &xvec
        ) {
            score = score_array;

            // get buffers
            py::buffer_info _score_info = score.request();

            // score has shape (N, Z, Y, X)
            nmole = _score_info.shape[0];
            nz = _score_info.shape[1];
            ny = _score_info.shape[2];
            nx = _score_info.shape[3];

            // check the input shapes
            if (origin.shape(0) != nmole || origin.shape(1) != 3) {
                throw py::value_error(
                    "Shape of 'origin' must be (" + std::to_string(nmole) + ", 3) "
                    "but got (" + std::to_string(origin.shape(0)) + ", " + std::to_string(origin.shape(1)) + ")."
                );
            } else if (zvec.shape(0) != nmole || zvec.shape(1) != 3) {
                throw py::value_error(
                    "Shape of 'zvec' must be (" + std::to_string(nmole) + ", 3) "
                    "but got (" + std::to_string(zvec.shape(0)) + ", " + std::to_string(zvec.shape(1)) + ")."
                );
            } else if (yvec.shape(0) != nmole || yvec.shape(1) != 3) {
                throw py::value_error(
                    "Shape of 'yvec' must be (" + std::to_string(nmole) + ", 3) "
                    "but got (" + std::to_string(yvec.shape(0)) + ", " + std::to_string(yvec.shape(1)) + ")."
                );
            } else if (xvec.shape(0) != nmole || xvec.shape(1) != 3) {
                throw py::value_error(
                    "Shape of 'xvec' must be (" + std::to_string(nmole) + ", 3) "
                    "but got (" + std::to_string(xvec.shape(0)) + ", " + std::to_string(xvec.shape(1)) + ")."
                );
            } else if (nmole < 2 || nz < 2 || ny < 2 || nx < 2) {
                throw py::value_error(
                    "Invalid shape of 'score': (" 
                    + std::to_string(nmole) + ", " + std::to_string(nz) + ", " 
                    + std::to_string(ny) + ", " + std::to_string(nx) + ")."
                );
            }

            // Allocation of arrays of coordinate system.
            // Offsets and orientations of local coordinates of score landscape are well-defined by this.
            auto _coords = new CoordinateSystem<double>[nmole];
            
            for (auto t = 0; t < nmole; ++t) {
                auto _ori = Vector3D<double>(*origin.data(t, 0), *origin.data(t, 1), *origin.data(t, 2));
                auto _ez = Vector3D<double>(*zvec.data(t, 0), *zvec.data(t, 1), *zvec.data(t, 2));
                auto _ey = Vector3D<double>(*yvec.data(t, 0), *yvec.data(t, 1), *yvec.data(t, 2));
                auto _ex = Vector3D<double>(*xvec.data(t, 0), *xvec.data(t, 1), *xvec.data(t, 2));
                _coords[t].update(_ori, _ez, _ey, _ex);
            }

			coords = _coords;
        };

        std::tuple<py::array_t<ssize_t>, double> viterbiSimple(double dist_min, double dist_max);
        std::tuple<py::array_t<ssize_t>, double> viterbi(double dist_min, double dist_max);
        std::tuple<py::array_t<ssize_t>, double> viterbi(double dist_min, double dist_max, double skew_max);
		std::string pyRepr() {
			return "ViterbiGrid(nmole=" + std::to_string(nmole) + ", nz=" + std::to_string(nz)
				+ ", ny=" + std::to_string(ny) + ", nx=" + std::to_string(nx) + ")";
		}
		
		#pragma warning(push)
		#pragma warning(disable:4244)
		/// Get the world coordinates of the point (z, y, x) in the local coordinate system of the n-th molecule.
		py::array_t<double> worldPos(ssize_t n, ssize_t z, ssize_t y, ssize_t x) {
			auto _pos = coords[n].at(Vector3D<double>(z, y, x));
			py::array_t<double> pos = py::array_t<double>(3);
            auto pos_mut = pos.mutable_unchecked<1>();
            pos_mut(0) = _pos.z;
            pos_mut(1) = _pos.y;
            pos_mut(2) = _pos.x;
			return pos;
		}
		#pragma warning(pop)
};

std::tuple<py::array_t<ssize_t>, double> ViterbiGrid::viterbiSimple(
	double dist_min,  // NOTE: upsample factor must be considered
	double dist_max
)
{
	if (dist_min >= dist_max) {
		throw py::value_error("`dist_min` must be smaller than `dist_max`.");
	}
	auto dist_min2 = dist_min * dist_min;
	auto dist_max2 = dist_max * dist_max;

	// prepare arrays
	auto state_sequence_ = py::array_t<ssize_t>{{nmole, ssize_t(3)}};
	auto state_sequence = state_sequence_.mutable_unchecked<2>();
	
	// Prepare the Viterbi lattice and initialize the initial states.
	auto viterbi_lattice_ = py::array_t<float>{{nmole, nz, ny, nx}};
	auto viterbi_lattice = viterbi_lattice_.mutable_unchecked<4>();

	// initialization at t = 0
	for (auto z = 0; z < nz; ++z) {
        for (auto y = 0; y < ny; ++y) {
            for (auto x = 0; x < nx; ++x) {
                viterbi_lattice(0, z, y, x) = *score.data(0, z, y, x);
            }
        }
    }

	auto constraint = Constraint(nz, ny, nx, dist_min2, dist_max2);
	py::gil_scoped_release nogil;  // without GIL

	// forward
	for (auto t = 1; t < nmole; ++t) {
		// iterate over all the end points
		auto coord_prev = coords[t - 1];
		auto coord = coords[t];
		for (auto z1 = 0; z1 < nz; ++z1) {
		for (auto y1 = 0; y1 < ny; ++y1) {
		for (auto x1 = 0; x1 < nx; ++x1) {
			auto max = -std::numeric_limits<float>::infinity();
			auto end_point = coord.at(z1, y1, x1);
			// iterate over all the start points
			for (auto y0 = 0; y0 < ny; ++y0) {
				if (constraint.fastCheckLongitudinal(coord_prev, end_point, y0) > 0) {
					continue;
				}
				// Calculate distances from all the possible start points.
				for (auto z0 = 0; z0 < nz; ++z0) {
				for (auto x0 = 0; x0 < nx; ++x0) {
					if (constraint.checkConstraint(coord_prev.at(z0, y0, x0), end_point)) {
						continue;
					}
					max = std::max(max, viterbi_lattice(t - 1, z0, y0, x0));
				}}
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

	state_sequence(nmole - 1, 0) = prev.z;
	state_sequence(nmole - 1, 1) = prev.y;
	state_sequence(nmole - 1, 2) = prev.x;

	// backward tracking
	for (auto t = nmole - 2; t >= 0; --t) {
		double max = -std::numeric_limits<double>::infinity();
		auto argmax = Vector3D<int>(-1, -1, -1);
		auto point_prev = coords[t + 1].at(prev.z, prev.y, prev.x);
		auto coord = coords[t];
		for (auto z0 = 0; z0 < nz; ++z0) {
		for (auto y0 = 0; y0 < ny; ++y0) {
		for (auto x0 = 0; x0 < nx; ++x0) {
			if (constraint.checkConstraint(coord.at(z0, y0, x0), point_prev)) {
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


std::tuple<py::array_t<ssize_t>, double> ViterbiGrid::viterbi(
	double dist_min, double dist_max
)
{
	return viterbiSimple(dist_min, dist_max);
}

std::tuple<py::array_t<ssize_t>, double> ViterbiGrid::viterbi(
	double dist_min,  // NOTE: upsample factor must be considered
	double dist_max,
	double skew_max  // NOTE: this parameter must be in radian
)
{
	if (dist_min >= dist_max) {
		throw py::value_error("`dist_min` must be smaller than `dist_max`.");
	} else if (skew_max <= 0.0 || skew_max > 3.14159) {
		throw py::value_error("`skew_max` must be in (0, pi/2)");
	}
	auto dist_min2 = dist_min * dist_min;
	auto dist_max2 = dist_max * dist_max;
	auto cos_skew_max = std::cos(skew_max);

	// prepare arrays
	auto state_sequence_ = py::array_t<ssize_t>{{nmole, ssize_t(3)}};
	auto state_sequence = state_sequence_.mutable_unchecked<2>();
	
	// Prepare the Viterbi lattice and initialize the initial states.
	auto viterbi_lattice_ = py::array_t<float>{{nmole, nz, ny, nx}};
	auto viterbi_lattice = viterbi_lattice_.mutable_unchecked<4>();

	// initialization at t = 0
	for (auto z = 0; z < nz; ++z) {
        for (auto y = 0; y < ny; ++y) {
            for (auto x = 0; x < nx; ++x) {
                viterbi_lattice(0, z, y, x) = *score.data(0, z, y, x);
            }
        }
    }

	auto constraint = AngleConstraint(nz, ny, nx, dist_min2, dist_max2, cos_skew_max);
	py::gil_scoped_release nogil;  // without GIL

	// forward
	for (auto t = 1; t < nmole; ++t) {
		auto coord_prev = coords[t - 1];
		auto coord = coords[t];
		auto origin_vector = coord_prev.origin - coord.origin;
		auto origin_dist2 = origin_vector.length2();
		for (auto z1 = 0; z1 < nz; ++z1) {
		for (auto y1 = 0; y1 < ny; ++y1) {
		for (auto x1 = 0; x1 < nx; ++x1) {
			auto max = -std::numeric_limits<float>::infinity();
			auto end_point = coord.at(z1, y1, x1);
			// iterate over all the start points
			for (auto y0 = 0; y0 < nx; ++y0) {
				if (constraint.fastCheckLongitudinal(coord_prev, end_point, y0) > 0) {
					continue;
				}
				// Calculate distances from all the possible start points.
				for (auto z0 = 0; z0 < nz; ++z0) {
				for (auto x0 = 0; x0 < nx; ++x0) {
					if (
						constraint.checkConstraint(
							coord.at(z0, y0, x0),
							end_point,
							origin_vector,
							origin_dist2
						)
					) {
						continue;
					}
					max = std::max(max, viterbi_lattice(t - 1, z0, y0, x0));
				}}
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

	state_sequence(nmole - 1, 0) = prev.z;
	state_sequence(nmole - 1, 1) = prev.y;
	state_sequence(nmole - 1, 2) = prev.x;

	// backward tracking
	for (auto t = nmole - 2; t >= 0; --t) {
		double max = -std::numeric_limits<double>::infinity();
		auto argmax = Vector3D<int>(-1, -1, -1);
		auto coord_prev = coords[t + 1];
		auto point_prev = coord_prev.at(prev.z, prev.y, prev.x);
		auto coord = coords[t];
		auto origin_vector = coord.origin - coord_prev.origin;
		auto origin_dist2 = origin_vector.length2();
		for (auto z0 = 0; z0 < nz; ++z0) {
		for (auto y0 = 0; y0 < ny; ++y0) {
		for (auto x0 = 0; x0 < nx; ++x0) {
			if (
				constraint.checkConstraint(
					coord.at(z0, y0, x0),
					point_prev,
					origin_vector,
					origin_dist2
				)
			) {
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

#endif
