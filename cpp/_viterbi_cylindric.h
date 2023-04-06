#ifndef _VITERBI_CYLINDRIC_H
#define _VITERBI_CYLINDRIC_H

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_coords.h"
#include "_cylindric.h"

namespace py = pybind11;
using ssize_t = Py_ssize_t;

/// A iterator-like class for the proper ordering of the cylindric indices.
class CylindricIterator {
    public:
        CylinderGeometry geometry;
        ssize_t count() { return geometry.nY * geometry.nA; };
        std::tuple<ssize_t, ssize_t> nth(ssize_t i);
        std::vector<std::tuple<ssize_t, ssize_t>> sourceOf(ssize_t y, ssize_t a);
        CylindricIterator(CylinderGeometry _geometry) : geometry(_geometry) {};
        CylindricIterator(ssize_t nY, ssize_t nA, ssize_t nRise) : geometry(nY, nA, nRise) {
            if (nRise == 0) {
                throw py::value_error("zero-rise is not implemented yet.");
            }
        };
};

/// @brief Return the indices corresponding to the i-th element of the iterator.
/// @param i The index of the element.
/// @return A tuple of indices (y, a).
std::tuple<ssize_t, ssize_t> CylindricIterator::nth(ssize_t i) {
    ssize_t y = i / geometry.nA;
    ssize_t a = i % geometry.nA;
    return std::make_tuple(y, a);
}

# endif
