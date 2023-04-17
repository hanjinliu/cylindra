#ifndef _CONSTRAINT_H
#define _CONSTRAINT_H

#pragma once

#include <pybind11/pybind11.h>
#include "_coords.h"

namespace py = pybind11;
using ssize_t = Py_ssize_t;

#pragma warning(push)
#pragma warning(disable:4244)

class AbstractConstraint {
    public:
        ssize_t nz, ny, nx;
};

class Constraint : public AbstractConstraint {
    public:
        ssize_t nz, ny, nx;
        double dist_min2, dist_max2;
        Constraint() : nz(0), ny(0), nx(0), dist_min2(-1), dist_max2(-1) {};
        Constraint(ssize_t _nz, ssize_t _ny, ssize_t _nx, double _dist_min2, double _dist_max2) {
            nz = _nz;
            ny = _ny;
            nx = _nx;
            dist_min2 = _dist_min2;
            dist_max2 = _dist_max2;
        }
        int fastCheckLongitudinal(CoordinateSystem<double>&, Vector3D<double>&, ssize_t);
        int fastCheckLateral(CoordinateSystem<double>&, Vector3D<double>&, ssize_t);
        bool checkConstraint(Vector3D<double>&, Vector3D<double>&);
};

inline int Constraint::fastCheckLongitudinal(
    CoordinateSystem<double> &coord, 
    Vector3D<double> &end_point, 
    ssize_t y0
) {
    auto point_0y0 = coord.at(0.0, y0, 0.0);
    auto dist2_00 = (point_0y0 - end_point).length2();
    auto dist2_01 = (coord.at(0.0, y0, nx-1) - end_point).length2();
    auto dist2_10 = (coord.at(nz-1, y0, 0.0) - end_point).length2();
    auto dist2_11 = (coord.at(nz-1, y0, nx-1) - end_point).length2();
    if (
        dist2_00 < dist_min2 
        && dist2_01 < dist_min2 
        && dist2_10 < dist_min2 
        && dist2_11 < dist_min2
    ) {
        return 1;
    }

    // If the length of perpendicular line drawn from point (x1, y1, z1) to the
    // plane of (_, y0, _) is longer than dist_max, then any point in the plane
    // is invalid.
    if (point_0y0.pointToPlaneDistance2(coord.ey, end_point) > dist_max2) {
        return 2;
    }
    return 0;
}

inline int Constraint::fastCheckLateral(
    CoordinateSystem<double> &coord, 
    Vector3D<double> &end_point, 
    ssize_t x0
) {
    // If the length from point (x1, y1, z1) to the four corners at y=y0 is all
    // shorter than dist_min, then any point in the plane is invalid, considering
    // the convexity of the shell-range created by [dist_min, dist_max].
    auto point_00x = coord.at(0.0, 0.0, x0);
    auto dist2_00 = (point_00x - end_point).length2();
    auto dist2_01 = (coord.at(0.0, ny-1, x0) - end_point).length2();
    auto dist2_10 = (coord.at(nz-1, 0.0, x0) - end_point).length2();
    auto dist2_11 = (coord.at(nz-1, ny-1, x0) - end_point).length2();
    if (
        dist2_00 < dist_min2 
        && dist2_01 < dist_min2 
        && dist2_10 < dist_min2 
        && dist2_11 < dist_min2
    ) {
        return 1;
    }
    if (point_00x.pointToPlaneDistance2(coord.ex, end_point) > dist_max2) {
        return 2;
    }
    return 0;
}

inline bool Constraint::checkConstraint(
    Vector3D<double> &start_point, 
    Vector3D<double> &end_point
) {
    auto dist2 = (start_point - end_point).length2();
    if (dist2 < dist_min2 || dist_max2 < dist2) {
        return true;
    }
    return false;
}


class AngleConstraint : public Constraint {
    public:
        double cos_max;
        AngleConstraint(ssize_t _nz, ssize_t _ny, ssize_t _nx, double _dist_min2, double _dist_max2, double _cos_max) {
            nz = _nz;
            ny = _ny;
            nx = _nx;
            dist_min2 = _dist_min2;
            dist_max2 = _dist_max2;
            cos_max = _cos_max;
        }
        bool checkConstraint(Vector3D<double>&, Vector3D<double>&, Vector3D<double>&, double);
};

inline bool AngleConstraint::checkConstraint(
    Vector3D<double> &start_point, 
    Vector3D<double> &end_point,
    Vector3D<double> &origin_vector,
    double origin_dist2
) {
    auto vec = end_point - start_point;
    auto dist2 = vec.length2();
    if (dist2 < dist_min2 || dist_max2 < dist2) {
        return true;
    }

    // Use formula: a.dot(b) = |a|*|b|*cos(C)
    auto ab = std::sqrt(dist2 * origin_dist2);
    auto cos = vec.dot(origin_vector) / ab;

    if (cos < cos_max) {
        // check angle of displacement vector of origins and that of
        // points of interests. Smaller cosine means larger skew.
        return true;
    }
    return false;
}

#pragma warning(pop)
#endif
