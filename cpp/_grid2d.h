#ifndef _GRID2D
#define _GRID2D

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_coords.h"
#include "_cylindric.h"
#include "_constraint.h"

namespace py = pybind11;
using ssize_t = Py_ssize_t;

/// A 2D array of CoordinateSystem objects.
/// Pointer of pointers cannot be safely handled by C++. This class mimics the
/// 2D get-item operations on a flattened array.
class Coords2DGrid {
    public:
        CoordinateSystem<double>* coords;  // flattened coordinate system array
        ssize_t naxial, nang;
        CoordinateSystem<double> at(ssize_t y, ssize_t a) {
            return coords[y * nang + a];
        }
        CoordinateSystem<double> at(std::pair<ssize_t, ssize_t> y_a) {
            return at(y_a.first, y_a.second);
        }
        CoordinateSystem<double>* at_mut(ssize_t y, ssize_t a) {
            return &coords[y * nang + a];
        }
        Coords2DGrid() : naxial(0), nang(0) {};
        Coords2DGrid(ssize_t _naxial, ssize_t _nang) {
            naxial = _naxial;
            nang = _nang;
            coords = new CoordinateSystem<double>[naxial * nang];
            for (auto i = 0; i < naxial * nang; ++i) {
                coords[i] = CoordinateSystem<double>();
            }
        };
};

#pragma warning(push)
#pragma warning(disable:4244)

Vector3D<double> stateAt(
    py::detail::unchecked_mutable_reference<long long, 3i64> &state_array,
    ssize_t y,
    ssize_t a
)
{
    auto z_ = state_array(y, a, 0);
    auto y_ = state_array(y, a, 1);
    auto x_ = state_array(y, a, 2);
    return Vector3D<double>(z_, y_, x_);
}

Vector3D<double> stateAt(
    py::detail::unchecked_mutable_reference<long long, 3i64> &state_array,
    std::pair<ssize_t, ssize_t> y_a
)
{
    return stateAt(state_array, y_a.first, y_a.second);
}

#pragma warning(pop)

/// A 2D grid for Viterbi alignment.
/// This class contains the score landscape, coordinate systems and the shape.
class ViterbiGrid2D {
    public:
        py::array_t<float> score; // A 5D array, where score_array[ny, na, z, y, x] is the score of the (ny, na) molecule at the grid point (z, y, x).
        Coords2DGrid coords;  // coordinate system of each molecule
        ssize_t naxial, nang, nz, ny, nx;  // number of molecules, number of grid points in z, y, x directions
        ssize_t nrise;
        ViterbiGrid2D (
            py::array_t<float> &score_array,
            py::array_t<float> &origin,
            py::array_t<float> &zvec,
            py::array_t<float> &yvec,
            py::array_t<float> &xvec,
            ssize_t n_rise
        );

        std::tuple<py::array_t<ssize_t>, double> viterbi(
            double dist_min,
            double dist_max,
            double lat_dist_min,
            double lat_dist_max
        );

        /// Get the geometry object of the grid.
        auto getGeometry() {
            CylinderGeometry geometry(naxial, nang, nrise);
            return geometry;
        };

        /// Return all the pairs of indices that are connected longitudinally.
        auto allLongitudinalPairs() {
            return getGeometry().allLongitudinalPairs();
        };

        /// Return all the pairs of indices that are connected laterally.
        auto allLateralPairs() {
            return getGeometry().allLateralPairs();
        };

        py::array_t<double> allLongitudinalDistances(py::array_t<int> &states);
        py::array_t<double> allLateralDistances(py::array_t<int> &states);

        std::string pyRepr() {
			return "ViterbiGrid(naxial=" + std::to_string(naxial) + ", nang=" + std::to_string(nang)
                + ", nz=" + std::to_string(nz) + ", ny=" + std::to_string(ny)
                + ", nx=" + std::to_string(nx) + ")";
		};

		#pragma warning(push)
		#pragma warning(disable:4244)
		/// Get the world coordinates of the point (z, y, x) in the local coordinate system of the molecule
        /// at (lon, lat).
		py::array_t<double> worldPos(ssize_t lon, ssize_t lat, ssize_t z, ssize_t y, ssize_t x) {
			auto _pos = coords.at(lon, lat).at(Vector3D<double>(z, y, x));
            py::array_t<double> pos = py::array_t<double>(3);
            auto pos_mut = pos.mutable_unchecked<1>();
            pos_mut(0) = _pos.z;
            pos_mut(1) = _pos.y;
            pos_mut(2) = _pos.x;
			return pos;
		}
        #pragma warning(pop)
};

ViterbiGrid2D::ViterbiGrid2D (
    py::array_t<float> &score_array,
    py::array_t<float> &origin,
    py::array_t<float> &zvec,
    py::array_t<float> &yvec,
    py::array_t<float> &xvec,
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
    Coords2DGrid _coords(naxial, nang);

    for (auto t = 0; t < naxial; ++t) {
        for (auto s = 0; s < nang; ++s) {
            auto _ori = Vector3D<double>(*origin.data(t, s, 0), *origin.data(t, s, 1), *origin.data(t, s, 2));
            auto _ez = Vector3D<double>(*zvec.data(t, s, 0), *zvec.data(t, s, 1), *zvec.data(t, s, 2));
            auto _ey = Vector3D<double>(*yvec.data(t, s, 0), *yvec.data(t, s, 1), *yvec.data(t, s, 2));
            auto _ex = Vector3D<double>(*xvec.data(t, s, 0), *xvec.data(t, s, 1), *xvec.data(t, s, 2));
            (*_coords.at_mut(t, s)).update(_ori, _ez, _ey, _ex);
        }
    }
    coords = _coords;
};

/// @brief 2D, distance-constrained Viterbi alignment on a cylindric grid.
/// @param dist_min Minimum distance between two longitudinally consecutive molecules.
/// @param dist_max Maximum distance between two longitudinally consecutive molecules.
/// @param lat_dist_min Minimum distance between two latitudinally consecutive molecules.
/// @param lat_dist_max Maximum distance between two latitudinally consecutive molecules.
/// @return {state_sequence, score} State sequence and the score of the optimal path.
std::tuple<py::array_t<ssize_t>, double> ViterbiGrid2D::viterbi(
	double dist_min,  // NOTE: upsample factor must be considered
	double dist_max,
    double lat_dist_min,
    double lat_dist_max
)
{
    #pragma warning(push)
    #pragma warning(disable:4244)

    if (dist_min >= dist_max) {
		throw py::value_error("`dist_min` must be smaller than `dist_max`.");
	} else if (lat_dist_min >= lat_dist_max) {
		throw py::value_error("`lat_dist_min` must be smaller than `lat_dist_max`.");
	}
	auto dist_min2 = dist_min * dist_min;
	auto dist_max2 = dist_max * dist_max;
    auto lat_dist_min2 = lat_dist_min * lat_dist_min;
    auto lat_dist_max2 = lat_dist_max * lat_dist_max;

	// prepare arrays
	auto state_sequence_ = py::array_t<ssize_t>{{naxial, nang, ssize_t(3)}};
	auto state_sequence = state_sequence_.mutable_unchecked<3>();
    for (auto t = 0; t < naxial; ++t) {
        for (auto s = 0; s < nang; ++s) {
            for (auto i = 0; i < 3; ++i) {
                state_sequence(t, s, i) = -1;
            }
        }
    }

    // For 2D grid, scores of the initial position will be initialized during the
    // for-loop in the `viterbi` method.
    auto viterbi_lattice_ = py::array_t<float>{{naxial, nang, nz, ny, nx}};
	auto viterbi_lattice = viterbi_lattice_.mutable_unchecked<5>();

    auto geometry = getGeometry();
    auto constraint_lon = Constraint(nz, ny, nx, dist_min2, dist_max2);
    auto constraint_lat = Constraint(nz, ny, nx, lat_dist_min2, lat_dist_max2);
	py::gil_scoped_release nogil;  // without GIL

	// forward
	for (auto t1 = 0; t1 < naxial; ++t1) {
        for (auto _s1 = 0; _s1 < nang; ++_s1) {
            auto s1 = geometry.convertAngular(_s1);
            auto sources = geometry.sourceForward(t1, s1);
            auto coord_end = coords.at(t1, s1);
            if (sources.hasLateral() && sources.hasLongitudinal()) {
                // If (t1, s1) has both longitudinal and lateral sources, then we have to check
                // the distances between (t1, s1, z1, y1, x1) and (t0, s0, z0, y0, x0) for each
                // (z1, y1, x1, z0, y0, x0), and for each source as (t0, s0). If either distance
                // is out of range, then the state (t1, s1, z1, y1, x1) is invalid (score is -inf).

                // "o" means "longitudinal", "a" means "lateral".

                auto t0o = sources.lon.first;
                auto s0o = sources.lon.second;
                auto t0a = sources.lat.first;
                auto s0a = sources.lat.second;
                auto coord_o = coords.at(t0o, s0o);
                auto coord_a = coords.at(t0a, s0a);
                for (auto z1 = 0; z1 < nz; ++z1) {
                for (auto y1 = 0; y1 < ny; ++y1) {
                for (auto x1 = 0; x1 < nx; ++x1) {
                    auto max = -std::numeric_limits<float>::infinity();
                    auto end_point = coord_end.at(z1, y1, x1);
                    for (auto y0o = 0; y0o < ny; ++y0o) {
                        if (constraint_lon.fastCheckLongitudinal(coord_o, end_point, y0o) > 0) {
                            continue;
                        }
                        for (auto x0a = 0; x0a < nx; ++x0a) {
                            if (constraint_lat.fastCheckLateral(coord_a, end_point, x0a) > 0) {
                                continue;
                            }
                            for (auto z0o = 0; z0o < nz; ++z0o) {
                            for (auto x0o = 0; x0o < nx; ++x0o) {
                                if (constraint_lon.checkConstraint(coord_o.at(z0o, y0o, x0o), end_point)) {
                                    continue;
                                }
                                for (auto z0a = 0; z0a < nz; ++z0a) {
                                for (auto y0a = 0; y0a < ny; ++y0a) {
                                    if (constraint_lat.checkConstraint(coord_a.at(z0a, y0a, x0a), end_point)) {
                                        continue;
                                    }
                                    // | | | | |
                                    // + + + + + -
                                    // + + + + + -
                                    // + + + + + -
                                    max = std::max(
                                        max,
                                        viterbi_lattice(t0o, s0o, z0o, y0o, x0o)
                                            + viterbi_lattice(t0a, s0a, z0a, y0a, x0a)
                                            - viterbi_lattice(t0o, s0a, z0a, y0a, x0a)
                                    );
                                }}
                            }}
                        }
                    }

                    auto next_score = score.data(t1, s1, z1, y1, x1);
                    viterbi_lattice(t1, s1, z1, y1, x1) = max + *next_score;
                }}}  // end of x1, y1, z1

            } else if (sources.hasLongitudinal()) {
                // if (t1, s1) has only longitudinal source (first or last |nRise| molecules),
                // then we have to check the distances between (t1, s1, z1, y1, x1) and
                // (t0, s0, z0, y0, x0) for each (z1, y1, x1, z0, y0, x0), only for the longitudinal
                // source (t0, s0).

                auto t0 = sources.lon.first;
                auto s0 = sources.lon.second;
                auto coord = coords.at(t0, s0);
                auto coord_ey = coord.ey.normed();
                for (auto z1 = 0; z1 < nz; ++z1) {
                for (auto y1 = 0; y1 < ny; ++y1) {
                for (auto x1 = 0; x1 < nx; ++x1) {
                    auto max = -std::numeric_limits<float>::infinity();
                    auto end_point = coord_end.at(z1, y1, x1);
                    for (auto y0 = 0; y0 < nx; ++y0) {
                        if (constraint_lon.fastCheckLongitudinal(coord, end_point, y0) > 0) {
                            continue;
                        }
                        for (auto z0 = 0; z0 < nz; ++z0) {
                        for (auto x0 = 0; x0 < nx; ++x0) {
                            if (constraint_lon.checkConstraint(coord.at(z0, y0, x0), end_point)) {
                                continue;
                            }
                            max = std::max(max, viterbi_lattice(t0, s0, z0, y0, x0));
                        }
                    }}

                    auto next_score = score.data(t1, s1, z1, y1, x1);
                    viterbi_lattice(t1, s1, z1, y1, x1) = max + *next_score;
                }}}  // end of x1, y1, z1

            } else if (sources.hasLateral()) {
                // if (t1, s1) has only lateral source (first ring of molecules),
                // then we have to check the distances between (t1, s1, z1, y1, x1) and
                // (t0, s0, z0, y0, x0) for each (z1, y1, x1, z0, y0, x0), only for the lateral
                // source (t0, s0).

                auto t0 = sources.lat.first;
                auto s0 = sources.lat.second;
                auto coord = coords.at(t0, s0);
                auto coord_ex = coord.ex.normed();
                for (auto z1 = 0; z1 < nz; ++z1) {
                for (auto y1 = 0; y1 < ny; ++y1) {
                for (auto x1 = 0; x1 < nx; ++x1) {
                    auto max = -std::numeric_limits<float>::infinity();
                    auto end_point = coord_end.at(z1, y1, x1);
                    for (auto x0 = 0; x0 < nx; ++x0) {
                        if (constraint_lat.fastCheckLateral(coord, end_point, x0) > 0) {
                            continue;
                        }
                        for (auto y0 = 0; y0 < nx; ++y0) {
                        for (auto z0 = 0; z0 < nz; ++z0) {
                            if (constraint_lat.checkConstraint(coord.at(z0, y0, x0), end_point)) {
                                continue;
                            }
                            max = std::max(max, viterbi_lattice(t0, s0, z0, y0, x0));
                        }}
                    }

                    auto next_score = score.data(t1, s1, z1, y1, x1);
                    viterbi_lattice(t1, s1, z1, y1, x1) = max + *next_score;
                }}}  // end of x1, y1, z1
            } else {
                // No source. Just copy the score.
                for (auto z = 0; z < nz; ++z) {
                    for (auto y = 0; y < ny; ++y) {
                        for (auto x = 0; x < nx; ++x) {
                            viterbi_lattice(t1, s1, z, y, x) = *score.data(t1, s1, z, y, x);
                        }
                    }
                }
            }
        }  // end of s1
    }  // end of t1

	double max_score = -std::numeric_limits<double>::infinity();

    // backward tracking
	for (auto t0 = naxial - 1; t0 >= 0; --t0) {
        for (auto _s0 = nang - 1; _s0 >= 0; --_s0) {
            auto s0 = geometry.convertAngular(_s0);
            auto coord = coords.at(t0, s0);
            auto bsrc = geometry.sourceBackward(t0, s0);
            auto max = -std::numeric_limits<float>::infinity();
            auto argmax = Vector3D<int>(-1, -1, -1);

            if (bsrc.hasLongitudinal() && bsrc.hasLateral()) {
                // Find the maximum position with the constraint of the distance from
                // the backward sources.
                auto argmaxo = stateAt(state_sequence, bsrc.lon);
                if (argmaxo.z < 0) continue;
                auto argmaxa = stateAt(state_sequence, bsrc.lat);
                if (argmaxa.z < 0) continue;

                auto point_prev_lon = coords.at(bsrc.lon).at(argmaxo);
                auto point_prev_lat = coords.at(bsrc.lat).at(argmaxa);
                for (auto z0 = 0; z0 < nz; ++z0) {
                for (auto y0 = 0; y0 < ny; ++y0) {
                for (auto x0 = 0; x0 < nx; ++x0) {
                    auto point0 = coord.at(z0, y0, x0);
                    if (constraint_lon.checkConstraint(point0, point_prev_lon)) {
                        continue;
                    }
                    if (constraint_lat.checkConstraint(point0, point_prev_lat)) {
                        continue;
                    }
                    auto current = viterbi_lattice(t0, s0, z0, y0, x0);
                    if (max < current) {
                        max = current;
                        argmax = Vector3D<int>(z0, y0, x0);
                    }
                }}}
            } else if (bsrc.hasLongitudinal()) {
                auto argmaxo = stateAt(state_sequence, bsrc.lon);
                if (argmaxo.z < 0) continue;
                auto point_prev_lon = coords.at(bsrc.lon).at(argmaxo);

                for (auto z0 = 0; z0 < nz; ++z0) {
                for (auto y0 = 0; y0 < ny; ++y0) {
                for (auto x0 = 0; x0 < nx; ++x0) {
                    if (constraint_lon.checkConstraint(coord.at(z0, y0, x0), point_prev_lon)) {
                        continue;
                    }
                    auto current = viterbi_lattice(t0, s0, z0, y0, x0);
                    if (max < current) {
                        max = current;
                        argmax = Vector3D<int>(z0, y0, x0);
                    }
                }}}

            } else if (bsrc.hasLateral()) {
                auto argmaxa = stateAt(state_sequence, bsrc.lat);
                if (argmaxa.z < 0) {
                    // no source
                    continue;
                }
                auto point_prev_lat = coords.at(bsrc.lat).at(argmaxa);

                for (auto z0 = 0; z0 < nz; ++z0) {
                for (auto y0 = 0; y0 < ny; ++y0) {
                for (auto x0 = 0; x0 < nx; ++x0) {
                    if (constraint_lat.checkConstraint(coord.at(z0, y0, x0), point_prev_lat)) {
                        continue;
                    }
                    auto current = viterbi_lattice(t0, s0, z0, y0, x0);
                    if (max < current) {
                        max = current;
                        argmax = Vector3D<int>(z0, y0, x0);
                    }
                }}}

            } else {
                for (auto z0 = 0; z0 < nz; ++z0) {
                for (auto y0 = 0; y0 < ny; ++y0) {
                for (auto x0 = 0; x0 < nx; ++x0) {
                    auto current = viterbi_lattice(t0, s0, z0, y0, x0);
                    if (max < current) {
                        max = current;
                        argmax = Vector3D<int>(z0, y0, x0);
                    }
                }}}
                max_score = max;
            }

            state_sequence(t0, s0, 0) = argmax.z;
            state_sequence(t0, s0, 1) = argmax.y;
            state_sequence(t0, s0, 2) = argmax.x;
	    }
    }
    #pragma warning(pop)
	return {state_sequence_, max_score};
}


py::array_t<double> ViterbiGrid2D::allLongitudinalDistances(py::array_t<int> &states) {
    auto pairs = allLongitudinalPairs();
    auto n = pairs.size();
    auto result = py::array_t<double>(n);
    auto ptr = result.mutable_data();
    for (auto i = 0; i < n; ++i) {
        auto idx0 = pairs[i].first;
        auto z0 = states.at(idx0.y, idx0.a, 0);
        auto y0 = states.at(idx0.y, idx0.a, 1);
        auto x0 = states.at(idx0.y, idx0.a, 2);
        auto pos0 = worldPos(idx0.y, idx0.a, z0, y0, x0);
        Vector3D<double> vec0(pos0);

        auto idx1 = pairs[i].second;
        auto z1 = states.at(idx1.y, idx1.a, 0);
        auto y1 = states.at(idx1.y, idx1.a, 1);
        auto x1 = states.at(idx1.y, idx1.a, 2);
        auto pos1 = worldPos(idx1.y, idx1.a, z1, y1, x1);
        Vector3D<double> vec1(pos1);

        ptr[i] = (vec1 - vec0).length();
    }
    return result;
}

py::array_t<double> ViterbiGrid2D::allLateralDistances(py::array_t<int> &states) {
    auto pairs = allLateralPairs();
    auto n = pairs.size();
    auto result = py::array_t<double>(n);
    auto ptr = result.mutable_data();
    for (auto i = 0; i < n; ++i) {
        auto idx0 = pairs[i].first;
        auto z0 = states.at(idx0.y, idx0.a, 0);
        auto y0 = states.at(idx0.y, idx0.a, 1);
        auto x0 = states.at(idx0.y, idx0.a, 2);
        auto pos0 = worldPos(idx0.y, idx0.a, z0, y0, x0);
        Vector3D<double> vec0(pos0);

        auto idx1 = pairs[i].second;
        auto z1 = states.at(idx1.y, idx1.a, 0);
        auto y1 = states.at(idx1.y, idx1.a, 1);
        auto x1 = states.at(idx1.y, idx1.a, 2);
        auto pos1 = worldPos(idx1.y, idx1.a, z1, y1, x1);
        Vector3D<double> vec1(pos1);

        ptr[i] = (vec1 - vec0).length();
    }
    return result;
}

#endif
