#ifndef _GRID2D
#define _GRID2D

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_coords.h"
#include "_cylindric.h"

namespace py = pybind11;
using ssize_t = Py_ssize_t;

class Coords2DGrid {
    public:
        CoordinateSystem<double>* coords;  // flattened coordinate system array
        ssize_t naxial, nang;
        CoordinateSystem<double> at(ssize_t y, ssize_t a) {
            if (y < 0 || y >= naxial || a < 0 || a >= nang) {
                throw py::index_error("Index out of range.");  // TODO: remove this later
            }
            return coords[y * nang + a];
        }
        Coords2DGrid() {};
        Coords2DGrid(ssize_t _naxial, ssize_t _nang) {
            naxial = _naxial;
            nang = _nang;
            coords = new CoordinateSystem<double>[naxial * nang];
        };
};

/// A 2D grid for Viterbi alignment.
/// This class contains the score landscape, coordinate systems and the shape.
class ViterbiGrid2D {
    public:
        py::array_t<double> score; // A 5D array, where score_array[ny, na, z, y, x] is the score of the (ny, na) molecule at the grid point (z, y, x).
        Coords2DGrid coords;  // coordinate system of each molecule
        ssize_t naxial, nang, nz, ny, nx;  // number of molecules, number of grid points in z, y, x directions
        ssize_t nrise;
        ViterbiGrid2D (
            py::array_t<double> score_array,
            py::array_t<double> origin,
            py::array_t<double> zvec,
            py::array_t<double> yvec,
            py::array_t<double> xvec,
            ssize_t n_rise
        ) {
            score = score_array;
            nrise = n_rise;

            // get buffers
            py::buffer_info _score_info = score.request();

            // score has shape (N, Z, Y, X)
            naxial = _score_info.shape[0];
            nang = _score_info.shape[1];
            nz = _score_info.shape[2];
            ny = _score_info.shape[3];
            nx = _score_info.shape[4];

            // check the input shapes
            if (origin.shape(0) != naxial || origin.shape(1) != nang || origin.shape(2) != 3) {
                throw py::value_error("Shape of 'origin' is wrong.");
            } else if (zvec.shape(0) != naxial || zvec.shape(1) != nang || zvec.shape(2) != 3) {
                throw py::value_error("Shape of 'zvec' is wrong.");
            } else if (yvec.shape(0) != naxial || yvec.shape(1) != nang || yvec.shape(2) != 3) {
                throw py::value_error("Shape of 'yvec' is wrong.");
            } else if (xvec.shape(0) != naxial || xvec.shape(1) != nang || xvec.shape(2) != 3) {
                throw py::value_error("Shape of 'xvec' is wrong.");
            } else if (naxial < 2 || nang < 2 || nz < 2 || ny < 2 || nx < 2) {
                throw py::value_error(
                    "Invalid shape of 'score': (" 
                    + std::to_string(naxial) + std::to_string(nang) + ", " + std::to_string(nz) + ", " 
                    + std::to_string(ny) + ", " + std::to_string(nx) + ")."
                );
            }

            // Allocation of arrays of coordinate system.
            // Offsets and orientations of local coordinates of score landscape are well-defined by this.
            Coords2DGrid coords(naxial, nang);
            
            for (auto t = 0; t < naxial; ++t) {
                for (auto s = 0; s < nang; ++s) {
                    auto _ori = Vector3D<double>(*origin.data(t, s, 0), *origin.data(t, s, 1), *origin.data(t, s, 2));
                    auto _ez = Vector3D<double>(*zvec.data(t, s, 0), *zvec.data(t, s, 1), *zvec.data(t, s, 2));
                    auto _ey = Vector3D<double>(*yvec.data(t, s, 0), *yvec.data(t, s, 1), *yvec.data(t, s, 2));
                    auto _ex = Vector3D<double>(*xvec.data(t, s, 0), *xvec.data(t, s, 1), *xvec.data(t, s, 2));
                    coords.at(t, s).update(_ori, _ez, _ey, _ex);
                }
            }
        };

        std::tuple<py::array_t<ssize_t>, double> viterbi(
            double dist_min, 
            double dist_max,
            double skew_max
        );
        auto prepViterbiLattice();
        auto getGeometry() {
            CylinderGeometry geometry(naxial, nang, nrise);
            return geometry;
        }
};

/// Prepare the Viterbi lattice and initialize the initial states.
/// Return the mutable reference of the Viterbi lattice.
auto ViterbiGrid2D::prepViterbiLattice() {
    auto viterbi_lattice_ = py::array_t<double>{{naxial, nang, nz, ny, nx}};
	auto viterbi_lattice = viterbi_lattice_.mutable_unchecked<5>();

	// initialization at t = 0
    for (auto a = 0; a < nang; ++a) {
        for (auto z = 0; z < nz; ++z) {
            for (auto y = 0; y < ny; ++y) {
                for (auto x = 0; x < nx; ++x) {
                    viterbi_lattice(0, a, z, y, x) = *score.data(0, a, z, y, x);
                }
            }
        }
    }
    return viterbi_lattice;
}


std::tuple<py::array_t<ssize_t>, double> ViterbiGrid2D::viterbi(
	double dist_min,  // NOTE: upsample factor must be considered
	double dist_max,
	double skew_max,  // NOTE: this parameter must be in radian
    double lat_dist_min,
    double lat_dist_max,
    double lat_,
)
{
	auto dist_min2 = dist_min * dist_min;
	auto dist_max2 = dist_max * dist_max;
	auto cos_skew_max = std::cos(skew_max);

	// prepare arrays
	auto state_sequence_ = py::array_t<ssize_t>{{naxial, nang, ssize_t(3)}};
	auto state_sequence = state_sequence_.mutable_unchecked<3>();
	auto viterbi_lattice = prepViterbiLattice();
    auto geometry = getGeometry();
	py::gil_scoped_release nogil;  // without GIL

	// forward
	for (auto t = 1; t < naxial; ++t) {
        for (auto s = 0; s < nang; ++s) {
            s = geometry.convertAngular(s);
            auto sources = geometry.sourceOf(t, s);
            auto t0 = sources.lon.first;
            auto s0 = sources.lon.second;
            auto origin_vector_lon = coords.at(t0, s0).origin - coords.at(t, s).origin;
            auto b2 = origin_vector_lon.length2();
            
            auto src_lon = sources.lon;
            if (sources.hasLateral()) {
                // TODO: calculate next score for lon & lat
                auto t1 = sources.lat.first;
                auto s1 = sources.lat.second;
                auto origin_vector_lat = coords.at(t1, s1).origin - coords.at(t, s).origin;


            } else {
                for (auto z1 = 0; z1 < nz; ++z1) {
                for (auto y1 = 0; y1 < ny; ++y1) {
                for (auto x1 = 0; x1 < nx; ++x1) {
                    auto max = -std::numeric_limits<double>::infinity();
                    bool neighbor_found = false;
                    auto end_point = coords.at(t, s).at(z1, y1, x1);
                    for (auto z0 = 0; z0 < nz; ++z0) {
                    for (auto y0 = 0; y0 < nx; ++y0) {
                        // If distances are not in the range of [dist_min, dist_max] at the edges, i.e., 
                        // x=0 and x=nx-1, then other points are not in the range either.
                        // Since valid range of distance is relatively small, this check largely improves
                        // performance.
                        auto distance2_0 = (coords.at(t0, s0).at(static_cast<double>(z0), static_cast<double>(y0), 0.0) - end_point).length2();
                        auto distance2_1 = (coords.at(t0, s0).at(static_cast<double>(z0), static_cast<double>(y0), static_cast<double>(nx-1)) - end_point).length2();
                        bool is_0_smaller = distance2_0 < dist_min2;
                        bool is_1_smaller = distance2_1 < dist_min2;
                        if (is_0_smaller && is_1_smaller) {
                            continue;
                        }
                        bool is_0_larger = dist_max2 < distance2_0;
                        bool is_1_larger = dist_max2 < distance2_1;
                        if (is_0_larger && is_1_larger) {
                            continue;
                        }

                        for (auto x0 = 0; x0 < nx; ++x0) {
                            auto vec = coords.at(t0, s0).at(z0, y0, x0) - end_point;
                            auto a2 = vec.length2();

                            if (a2 < dist_min2 || dist_max2 < a2) {
                                // check distance between two points
                                continue;
                            }

                            // Use formula: a.dot(b) = |a|*|b|*cos(C)
                            auto ab = std::sqrt(a2 * b2);
                            auto cos = vec.dot(origin_vector_lon) / ab;

                            if (cos < cos_skew_max) {
                                // check angle of displacement vector of origins and that of
                                // points of interests. Smaller cosine means larger skew.
                                continue;
                            }

                            neighbor_found = true;
                            max = std::max(max, viterbi_lattice(t - 1, z0, y0, x0));
                        }
                    }}
                
                    if (!neighbor_found) {
                        viterbi_lattice(t, z1, y1, x1) = -std::numeric_limits<double>::infinity();
                    }
                    auto next_score = score.data(t, z1, y1, x1);
                    viterbi_lattice(t, z1, y1, x1) = max + *next_score;
                }
            
            }}}
        }
    }

	// find maximum score
	double max_score = -std::numeric_limits<double>::infinity();
	auto prev = Vector3D<int>(0, 0, 0);
    auto index_end = geometry.indexEnd();
	
    for (auto z = 0; z < nz; ++z) {
    for (auto y = 0; y < ny; ++y) {
    for (auto x = 0; x < nx; ++x) {
        auto s = viterbi_lattice(index_end.y, index_end.a, z, y, x);
        if (s > max_score) {
            max_score = s;
            prev.z = z;
            prev.y = y;
            prev.x = x;
        }
    }}}

	state_sequence(index_end.y, index_end.a, 0) = prev.z;
	state_sequence(index_end.y, index_end.a, 1) = prev.y;
	state_sequence(index_end.y, index_end.a, 2) = prev.x;

	// backward tracking
    // TODO: consider 2D
	for (auto t = naxial - 2; t >= 0; --t) {
		double max = -std::numeric_limits<double>::infinity();
		auto argmax = Vector3D<int>(0, 0, 0);
		auto point_prev = coords[t+1].at(prev.z, prev.y, prev.x);
		auto origin_vector = coords[t].origin - coords[t+1].origin;
		auto b2 = origin_vector.length2();
		for (auto z0 = 0; z0 < nz; ++z0) {
		for (auto y0 = 0; y0 < ny; ++y0) {
		for (auto x0 = 0; x0 < nx; ++x0) {
			auto vec = coords[t].at(z0, y0, x0) - point_prev;
			auto a2 = vec.length2();

			if (a2 < dist_min2 || dist_max2 < a2) {
				// check distance.
				continue;
			}

			auto ab = std::sqrt(a2 * b2);
			auto cos = vec.dot(origin_vector) / ab;
			if (cos < cos_skew_max) {
				// check angle.
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
