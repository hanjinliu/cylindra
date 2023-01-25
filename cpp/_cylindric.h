#ifndef _CYLINDRIC_H
#define _CYLINDRIC_H

#pragma once

#include <vector>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using ssize_t = Py_ssize_t;
namespace py = pybind11;

// tuple of unsigned integers
struct Index {
    ssize_t y, a;
    Index(ssize_t y, ssize_t a) : y(y), a(a) {}
    bool isValid(ssize_t nY, ssize_t nA) {
        return y >= 0 && y < nY && a >= 0 && a < nA;
    }
};

// Tuple of signed integers.
// This struct is used for not-resolved-yet indices.
struct SignedIndex {
    ssize_t y, a;
    SignedIndex(ssize_t y, ssize_t a) : y(y), a(a) {}
};

// Class that defines a geometry of a cylinder.
// A "geometry" defines the connectivity of an assembly of molecules.
// Currently only connectivity=1 is supported.
class CylinderGeometry {
    public:
        ssize_t nY, nA, nRise;
        CylinderGeometry(ssize_t _nY, ssize_t _nA, ssize_t _nRise) {
            nY = _nY;
            nA = _nA;
            nRise = _nRise;
        };
        CylinderGeometry(ssize_t nY, ssize_t nA) : CylinderGeometry(nY, nA, 0) {};
        std::vector<Index> getNeighbor(ssize_t, ssize_t);
        std::vector<Index> getNeighbors(std::vector<Index>);
        Index getIndex(ssize_t, ssize_t);

    private:
        ssize_t compress(Index);
        Index decompress(ssize_t);
        SignedIndex getSignedIndex(ssize_t, ssize_t);
};

inline SignedIndex CylinderGeometry::getSignedIndex(ssize_t y, ssize_t a) {
    while (a >= nA) {
        a -= nA;
        y += nRise;
    }
    while (a < 0) {
        a += nA;
        y -= nRise;
    }
    return SignedIndex(y, a);
}

inline Index CylinderGeometry::getIndex(ssize_t y, ssize_t a) {
    auto idx = getSignedIndex(y, a);
    if (idx.y < 0 || nY <= idx.y) {
        char msg[64];
        sprintf(msg, "Index (%lld, %lld) out of bounds.", idx.y, idx.a);
        throw py::index_error(msg);
    }
    return Index(idx.y, idx.a);
}

// Get neighbors of a given index.
// This function considers the connectivity of the cylinder and returns the
// corrected indices.
inline std::vector<Index> CylinderGeometry::getNeighbor(ssize_t y, ssize_t a) {
    std::vector<Index> neighbors;
    auto idx = getSignedIndex(y, a);

    if (y > 0) {
        auto index = Index(y - 1, a);
        if (index.isValid(nY, nA)) {
            neighbors.push_back(Index(y - 1, a));
        }
    }
    if (y < nY - 1) {
        auto index = Index(y + 1, a);
        if (index.isValid(nY, nA)) {
            neighbors.push_back(index);
        }
    }

    // left neighbor
    auto index_l = (a > 0) ? Index(y, a - 1) : Index(y - nRise, nA - 1);
    if (index_l.isValid(nY, nA)) {
        neighbors.push_back(index_l);
    }
    
    // right neighbor
    auto index_r = (a < nA - 1) ? Index(y, a + 1) : Index(y + nRise, 0);
    if (index_r.isValid(nY, nA)) {
        neighbors.push_back(index_r);
    }

    return neighbors;
}

// Get all the unique neighbors of a list of indices.
inline std::vector<Index> CylinderGeometry::getNeighbors(std::vector<Index> indices) {
    std::set<ssize_t> uniqueNeighbors;
    // add all the neighbor candidates
    for (auto index : indices) {
        auto new_neighbors = getNeighbor(index.y, index.a);
        for (auto neighbor : new_neighbors) {
            ssize_t compressed = compress(neighbor);
            uniqueNeighbors.insert(compressed);
        }
    }
    
    // remove inputs
    for (auto index : indices) {
        uniqueNeighbors.erase(compress(index));
    }

    // convert to a vector
    std::vector<Index> neighbors;
    for (ssize_t neighbor : uniqueNeighbors) {
        neighbors.push_back(decompress(neighbor));
    }
    return neighbors;
}

// Compresses the index tuple into a single ssize_t.
// This method returns the absolute index of the index tuple.
inline ssize_t CylinderGeometry::compress(Index idx) {
    return idx.y + idx.a * nY;
}

// Decompresses the index tuple from a single ssize_t
inline Index CylinderGeometry::decompress(ssize_t val) {
    ssize_t y = val % nY;
    ssize_t a = val / nY;
    return Index(y, a);
}

#endif
