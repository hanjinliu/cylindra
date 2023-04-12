#ifndef _GRID2D
#define _GRID2D

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_coords.h"
#include "_cylindric.h"

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
            if (y < 0 || y >= naxial || a < 0 || a >= nang) {
                // TODO: remove this in the future. Should be unchecked.
                auto msg = "Index out of range. Grid shape is (y=" + std::to_string(naxial)
                    + ", a=" + std::to_string(nang) + "but accessed at y = "
                    + std::to_string(y) + ", a = " + std::to_string(a);
                throw py::index_error(msg);
            }
            return coords[y * nang + a];
        }
        Coords2DGrid() : naxial(0), nang(0) {};
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
                    _coords.at(t, s).update(_ori, _ez, _ey, _ex);
                }
            }
            coords = _coords;
        };

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
	
    // For 2D grid, scores of the initial position will be initialized during the
    // for-loop in the `viterbi` method.
    auto viterbi_lattice_ = py::array_t<float>{{naxial, nang, nz, ny, nx}};
	auto viterbi_lattice = viterbi_lattice_.mutable_unchecked<5>();

    auto geometry = getGeometry();
	py::gil_scoped_release nogil;  // without GIL

	// forward
	for (auto t1 = 0; t1 < naxial; ++t1) {
        for (auto _s1 = 0; _s1 < nang; ++_s1) {
            auto s1 = geometry.convertAngular(_s1);
            auto sources = geometry.sourceForward(t1, s1);

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
                for (auto z1 = 0; z1 < nz; ++z1) {
                for (auto y1 = 0; y1 < ny; ++y1) {
                for (auto x1 = 0; x1 < nx; ++x1) {
                    auto max = -std::numeric_limits<float>::infinity();
                    auto end_point = coords.at(t1, s1).at(z1, y1, x1);
                    for (auto y0o = 0; y0o < nx; ++y0o) {
                        // If the length from point (x1, y1, z1) to the four corners at y=y0 is all
                        // shorter than dist_min, then any point in the plane is invalid, considering
                        // the convexity of the shell-range created by [dist_min, dist_max].
                        auto coord = coords.at(t0o, s0o);
                        auto point_0y0 = coord.at(0.0, y0o, 0.0);
                        auto dist2_00 = (point_0y0 - end_point).length2();
                        auto dist2_01 = (coord.at(0.0, y0o, nx-1) - end_point).length2();
                        auto dist2_10 = (coord.at(nz-1, y0o, 0.0) - end_point).length2();
                        auto dist2_11 = (coord.at(nz-1, y0o, nx-1) - end_point).length2();
                        if (
                            dist2_00 < dist_min2 
                            && dist2_01 < dist_min2 
                            && dist2_10 < dist_min2 
                            && dist2_11 < dist_min2
                        ) {
                            continue;
                        }

                        // If the length of perpendicular line drawn from point (x1, y1, z1) to the
                        // plane of (_, y0, _) is longer than dist_max, then any point in the plane
                        // is invalid.
                        if (point_0y0.pointToPlaneDistance2(coord.ey, end_point) > dist_max2) {
                            continue;  // break?
                        }

                        for (auto x0a = 0; x0a < nx; ++x0a) {
                            // check the distance between two points to speed up
                            auto point_00x = coords.at(t0a, s0a).at(0.0, 0.0, x0a);
                            dist2_00 = (point_00x - end_point).length2();
                            dist2_01 = (coords.at(t0a, s0a).at(0.0, 0.0, nx-1) - end_point).length2();
                            dist2_10 = (coords.at(t0a, s0a).at(nz-1, 0.0, 0.0) - end_point).length2();
                            dist2_11 = (coords.at(t0a, s0a).at(nz-1, 0.0, nx-1) - end_point).length2();
                            if (
                                dist2_00 < lat_dist_min2 
                                && dist2_01 < lat_dist_min2 
                                && dist2_10 < lat_dist_min2 
                                && dist2_11 < lat_dist_min2
                            ) {
                                continue;
                            }
                            if (point_00x.pointToPlaneDistance2(coords.at(t0a, s0a).ex, end_point) > lat_dist_max2) {
                                continue;  // break?
                            }

                            for (auto z0o = 0; z0o < nz; ++z0o) {
                            for (auto z0a = 0; z0a < nz; ++z0a) {
                            for (auto y0a = 0; y0a < ny; ++y0a) {
                            for (auto x0o = 0; x0o < nx; ++x0o) {
                                auto vec_o = coord.at(z0o, y0o, x0o) - end_point;
                                auto a2o = vec_o.length2();

                                if (a2o < dist_min2 || dist_max2 < a2o) {
                                    // check distance between two points
                                    continue;
                                }
                                
                                auto vec_a = coords.at(t0a, s0a).at(z0a, y0a, x0a) - end_point;
                                auto a2a = vec_a.length2();

                                if (a2a < lat_dist_min2 || lat_dist_max2 < a2a) {
                                    // check distance between two points
                                    continue;
                                }

                                max = std::max(
                                    max,
                                    viterbi_lattice(t0o, s0o, z0o, y0o, x0o)
                                    + viterbi_lattice(t0a, s0a, z0a, y0a, x0a)  // TODO: add??
                                );
                            }}}}  // end of x0o, y0a, z0a, z0o
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
                for (auto z1 = 0; z1 < nz; ++z1) {
                for (auto y1 = 0; y1 < ny; ++y1) {
                for (auto x1 = 0; x1 < nx; ++x1) {
                    auto max = -std::numeric_limits<float>::infinity();
                    auto end_point = coords.at(t1, s1).at(z1, y1, x1);
                    for (auto y0 = 0; y0 < nx; ++y0) {
                    for (auto z0 = 0; z0 < nz; ++z0) {
                        // If the length from point (x1, y1, z1) to the four corners at y=y0 is all
                        // shorter than dist_min, then any point in the plane is invalid, considering
                        // the convexity of the shell-range created by [dist_min, dist_max].
                        auto point_0y0 = coords.at(t0, s0).at(0.0, y0, 0.0);
                        auto dist2_00 = (point_0y0 - end_point).length2();
                        auto dist2_01 = (coords.at(t0, s0).at(0.0, y0, nx-1) - end_point).length2();
                        auto dist2_10 = (coords.at(t0, s0).at(nz-1, y0, 0.0) - end_point).length2();
                        auto dist2_11 = (coords.at(t0, s0).at(nz-1, y0, nx-1) - end_point).length2();
                        if (
                            dist2_00 < dist_min2 
                            && dist2_01 < dist_min2 
                            && dist2_10 < dist_min2 
                            && dist2_11 < dist_min2
                        ) {
                            continue;
                        }

                        // If the length of perpendicular line drawn from point (x1, y1, z1) to the
                        // plane of (_, y0, _) is longer than dist_max, then any point in the plane
                        // is invalid.
                        if (point_0y0.pointToPlaneDistance2(coords.at(t0, s0).ey, end_point) > dist_max2) {
                            continue;  // break?
                        }

                        for (auto x0 = 0; x0 < nx; ++x0) {
                            auto vec = coords.at(t0, s0).at(z0, y0, x0) - end_point;
                            auto a2 = vec.length2();

                            if (a2 < dist_min2 || dist_max2 < a2) {
                                // check distance between two points
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
                for (auto z1 = 0; z1 < nz; ++z1) {
                for (auto y1 = 0; y1 < ny; ++y1) {
                for (auto x1 = 0; x1 < nx; ++x1) {
                    auto max = -std::numeric_limits<float>::infinity();
                    auto end_point = coords.at(t1, s1).at(z1, y1, x1);
                    for (auto x0 = 0; x0 < nx; ++x0) {
                        // check the distance between two points to speed up
                        auto point_00x = coords.at(t0, s0).at(0.0, 0.0, x0);
                        auto dist2_00 = (point_00x - end_point).length2();
                        auto dist2_01 = (coords.at(t0, s0).at(0.0, ny-1, x0) - end_point).length2();
                        auto dist2_10 = (coords.at(t0, s0).at(nz-1, 0.0, x0) - end_point).length2();
                        auto dist2_11 = (coords.at(t0, s0).at(nz-1, ny-1, x0) - end_point).length2();
                        if (
                            dist2_00 < lat_dist_min2 
                            && dist2_01 < lat_dist_min2 
                            && dist2_10 < lat_dist_min2 
                            && dist2_11 < lat_dist_min2
                        ) {
                            continue;
                        }
                        if (point_00x.pointToPlaneDistance2(coords.at(t0, s0).ex, end_point) > lat_dist_max2) {
                            continue;  // break?
                        }

                        for (auto y0 = 0; y0 < nx; ++y0) {
                        for (auto z0 = 0; z0 < nz; ++z0) {
                            auto vec = coords.at(t0, s0).at(z0, y0, x0) - end_point;
                            auto a2 = vec.length2();

                            if (a2 < lat_dist_min2 || lat_dist_max2 < a2) {
                                // check distance between two points
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
    auto index_end = geometry.indexEnd();

    // backward tracking    
	for (auto t0 = naxial - 1; t0 >= 0; --t0) {
        for (auto _s0 = 0; _s0 < nang; ++_s0) {
            auto s0 = geometry.convertAngular(_s0);
            auto bsrc = geometry.sourceBackward(t0, s0);
            double max = -std::numeric_limits<double>::infinity();
            auto argmax = Vector3D<int>(0, 0, 0);
            if (bsrc.hasLongitudinal() && bsrc.hasLateral()) {
                // Find the maximum position with the constraint of the distance from
                // the backward sources.
                auto t1o = bsrc.lon.first;
                auto s1o = bsrc.lon.second;
                auto t1a = bsrc.lat.first;
                auto s1a = bsrc.lat.second;
                auto zargmaxo = state_sequence(t1o, s1o, 0);
                auto yargmaxo = state_sequence(t1o, s1o, 1);
                auto xargmaxo = state_sequence(t1o, s1o, 2);
                auto zargmaxa = state_sequence(t1a, s1a, 0);
                auto yargmaxa = state_sequence(t1a, s1a, 1);
                auto xargmaxa = state_sequence(t1a, s1a, 2);
                auto point_prev_lon = coords.at(t1o, s1o).at(zargmaxo, yargmaxo, xargmaxo);
                auto point_prev_lat = coords.at(t1a, s1a).at(zargmaxa, yargmaxa, xargmaxa);

                for (auto z0 = 0; z0 < nz; ++z0) {
                for (auto y0 = 0; y0 < ny; ++y0) {
                for (auto x0 = 0; x0 < nx; ++x0) {
                    auto point0 = coords.at(t0, s0).at(z0, y0, x0);
                    auto a2_lon = (point0 - point_prev_lon).length2();

                    if (a2_lon < dist_min2 || dist_max2 < a2_lon) {
                        // check distance.
                        continue;
                    }

                    auto a2_lat = (point0 - point_prev_lat).length2();

                    if (a2_lat < lat_dist_min2 || lat_dist_max2 < a2_lat) {
                        // check distance.
                        continue;
                    }
                    if (max < viterbi_lattice(t0, s0, z0, y0, x0)) {
                        max = viterbi_lattice(t0, s0, z0, y0, x0);
                        argmax = Vector3D<int>(z0, y0, x0);
                    }
                }}}

            } else if (bsrc.hasLongitudinal()) {
                auto t1 = bsrc.lon.first;
                auto s1 = bsrc.lon.second;
                auto zargmax = state_sequence(t1, s1, 0);
                auto yargmax = state_sequence(t1, s1, 1);
                auto xargmax = state_sequence(t1, s1, 2);
                auto point_prev_lon = coords.at(t1, s1).at(zargmax, yargmax, xargmax);

                for (auto z0 = 0; z0 < nz; ++z0) {
                for (auto y0 = 0; y0 < ny; ++y0) {
                for (auto x0 = 0; x0 < nx; ++x0) {
                    auto vec = coords.at(t0, s0).at(z0, y0, x0) - point_prev_lon;
                    auto a2 = vec.length2();

                    if (a2 < dist_min2 || dist_max2 < a2) {
                        // check distance.
                        continue;
                    }
                    if (max < viterbi_lattice(t0, s0, z0, y0, x0)) {
                        max = viterbi_lattice(t0, s0, z0, y0, x0);
                        argmax = Vector3D<int>(z0, y0, x0);
                    }
                }}}

            } else if (bsrc.hasLateral()) {
                auto t1 = bsrc.lat.first;
                auto s1 = bsrc.lat.second;
                auto zargmax = state_sequence(t1, s1, 0);
                auto yargmax = state_sequence(t1, s1, 1);
                auto xargmax = state_sequence(t1, s1, 2);
                auto point_prev_lat = coords.at(t1, s1).at(zargmax, yargmax, xargmax);

                for (auto z0 = 0; z0 < nz; ++z0) {
                for (auto y0 = 0; y0 < ny; ++y0) {
                for (auto x0 = 0; x0 < nx; ++x0) {
                    auto vec = coords.at(t0, s0).at(z0, y0, x0) - point_prev_lat;
                    auto a2 = vec.length2();

                    if (a2 < lat_dist_min2 || lat_dist_max2 < a2) {
                        // check distance.
                        continue;
                    }
                    if (max < viterbi_lattice(t0, s0, z0, y0, x0)) {
                        max = viterbi_lattice(t0, s0, z0, y0, x0);
                        argmax = Vector3D<int>(z0, y0, x0);
                    }
                }}}

            } else {
                for (auto z0 = 0; z0 < nz; ++z0) {
                for (auto y0 = 0; y0 < ny; ++y0) {
                for (auto x0 = 0; x0 < nx; ++x0) {
                    if (max < viterbi_lattice(t0, s0, z0, y0, x0)) {
                        max = viterbi_lattice(t0, s0, z0, y0, x0);
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

#endif
