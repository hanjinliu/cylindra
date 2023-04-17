#ifndef _CYLINDRIC_H
#define _CYLINDRIC_H

#pragma once

#include <vector>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using ssize_t = Py_ssize_t;
namespace py = pybind11;

// --- Index struct -----------------------------------------------------------

/// Struct of unsigned integers for an index on a cylinder.
struct Index {
    ssize_t y, a;
    Index(ssize_t y, ssize_t a) : y(y), a(a) {}

    /// Check if the index is valid for a given geometry.
    bool isValid(ssize_t nY, ssize_t nA) {
        return y >= 0 && y < nY && a >= 0 && a < nA;
    }
    /// __repr__ for python 
    std::string pyRepr() {
        return "Index(y=" + std::to_string(y) + ", a=" + std::to_string(a) + ")";
    }

    /// __eq__ for python 
    bool pyEq(std::pair<ssize_t, ssize_t> other) {
        return y == other.first && a == other.second;
    }
};

/// Struct of signed integers for an index on a cylinder.
/// This struct is used for not-resolved-yet indices.
struct SignedIndex {
    ssize_t y, a;
    SignedIndex(ssize_t y, ssize_t a) : y(y), a(a) {}
};

/// List of source indices.
/// This class is used to store the indices of the longitudinal and lateral sources.
/// The lateral source is optional.
struct Sources {
    std::pair<ssize_t, ssize_t> lon;  // longitudinal source
    std::pair<ssize_t, ssize_t> lat;  // lateral source

    Sources() : lon({-1, -1}), lat({-1, -1}) {};  // default constructor
    Sources(std::pair<ssize_t, ssize_t> _lon) : lon(_lon), lat({-1, -1}) {};
    Sources(std::pair<ssize_t, ssize_t> _lon, std::pair<ssize_t, ssize_t> _lat) : lon(_lon), lat(_lat) {};
    
    /// Check if the source has a longitudinal contact.
    bool hasLongitudinal() { return lon.first >= 0; };

    /// Check if the source has a lateral contact.
    bool hasLateral() { return lat.first >= 0; };

    /// Update the longitudinal source.
    void setValue(std::pair<ssize_t, ssize_t> _lon) {
        setValue(_lon, {-1, -1});
    };

    /// Update the longitudinal and lateral source.
    void setValue(std::pair<ssize_t, ssize_t> _lon, std::pair<ssize_t, ssize_t> _lat) {
        // check if the values exists on the grid. Replace out-of-bound values with {-1, -1}.
        if (_lon.first < 0 || _lon.second < 0) {
            _lon = {-1, -1};
        }
        if (_lat.first < 0 || _lat.second < 0) {
            _lat = {-1, -1};
        }
        lon = _lon;
        lat = _lat;
    };

    /// __eq__ for python
    /// This function provides the way to check values on the Python side.
    bool pyEq(const std::vector<std::pair<ssize_t, ssize_t>> &vec) {
        if (vec.size() == 0) {
            return lon.first == -1 && lon.second == -1
                && lat.first == -1 && lat.second == -1;
        }
        if (vec.size() == 1) {
            return lon.first == vec[0].first && lon.second == vec[0].second
                && lat.first == -1 && lat.second == -1;
        }
        else if (vec.size() == 2) {
            return lon.first == vec[0].first && lon.second == vec[0].second
                && lat.first == vec[1].first && lat.second == vec[1].second;
        }
        else {
            return false;
        }
    };

    /// __repr__ for python
    /// Sources(lon=(0, 1), lat=(2, 3)) ... if both sources are defined
    /// Sources(lon=(0, 1), lat=None) ... if only the longitudinal source is defined
    /// Sources(lon=None, lat=None) ... if no source is defined
    std::string pyRepr() {
        std::string repr = "Sources(";
        if (hasLongitudinal()) {
            repr += "lon=(y=" + std::to_string(lon.first) + ", a=" + std::to_string(lon.second) + ")";
        } else {
            repr += "lon=None";
        }
        if (hasLateral()) {
            repr += ", lat=(y=" + std::to_string(lat.first) + ", a=" + std::to_string(lat.second) + ")";
        } else {
            repr += ", lat=None";
        }
        repr += ")";
        return repr;
    };
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
        CylinderGeometry() : CylinderGeometry(0, 0, 0) {};
        CylinderGeometry(ssize_t nY, ssize_t nA) : CylinderGeometry(nY, nA, 0) {};
        std::vector<Index> getNeighbor(ssize_t, ssize_t);
        std::vector<Index> getNeighbors(std::vector<Index>);
        std::vector<Index> getNeighbors(std::vector<std::pair<ssize_t, ssize_t>>);
        Index getIndex(ssize_t, ssize_t);
        ssize_t count() { return nY * nA; };
        Sources sourceForward(ssize_t y, ssize_t a);
        Sources sourceBackward(ssize_t y, ssize_t a);
        Index indexStart();
        Index indexEnd();
        ssize_t convertAngular(ssize_t ang);
        std::vector<std::pair<Index, Index>> allLongitudinalPairs();
        std::vector<std::pair<Index, Index>> allLateralPairs();
        std::string pyRepr() {
            return "CylinderGeometry(nY=" + std::to_string(nY) +
                ", nA=" + std::to_string(nA) + ", nRise=" + std::to_string(nRise) + ")";
        };

    private:
        ssize_t compress(Index);
        Index decompress(ssize_t);
        SignedIndex getSignedIndex(ssize_t, ssize_t);
};

/// @brief Unchecked normalization of the input index considering the rise number.
/// @return SignedIndex object with the normalized index.
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

/// @brief Normalize the input index considering the rise number.
/// @return UnsignedIndex object with the normalized index.
inline Index CylinderGeometry::getIndex(ssize_t y, ssize_t a) {
    auto idx = getSignedIndex(y, a);
    if (idx.y < 0 || nY <= idx.y) {
        auto msg = "Index (" + std::to_string(idx.y) + ", " + std::to_string(idx.a) + ") out of bounds.";
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

inline std::vector<Index> CylinderGeometry::getNeighbors(std::vector<std::pair<ssize_t, ssize_t>> indices) {
    std::vector<Index> idx;
    for (auto index : indices) {
        idx.push_back(getIndex(index.first, index.second));
    }
    return getNeighbors(idx);
}

/// @brief Return the indices corresponding to the i-th element of the iterator.
/// @param y Longitudinal coordinate of which source will be returned.
/// @param a Angular coordinate of which source will be returned.
/// @return A tuple of indices (y, a).
inline Sources CylinderGeometry::sourceForward(ssize_t y, ssize_t a) {
    Sources sources;
    if (nRise >= 0) {
        if (a > 0) {
            sources.setValue({y - 1, a}, {y, a - 1});
        } else {
            auto y0 = y - nRise;
            if (y0 >= 0) {
                sources.setValue({y - 1, a}, {y0, nA - 1});
            } else {
                sources.setValue({y - 1, a});
            }
        }
    } else {
        if (a < nA - 1) {
            sources.setValue({y - 1, a}, {y, a + 1});
        } else {
            auto y0 = y + nRise;
            if (y0 >= 0) {
                sources.setValue({y - 1, a}, {y0, 0});
            } else {
                sources.setValue({y - 1, a});
            }
        }
    }
    return sources;
}

inline Sources CylinderGeometry::sourceBackward(ssize_t y, ssize_t a) {
    Sources sources;
    auto ynext = y + 1;
    if (ynext >= nY) {
        ynext = -1;
    }
    if (nRise >= 0) {
        if (a < nA - 1) {
            sources.setValue({ynext, a}, {y, a + 1});
        } else {
            auto y0 = y + nRise;
            if (y0 < nY) {
                sources.setValue({ynext, a}, {y0, 0});
            } else {
                sources.setValue({ynext, a});
            }
        }
    } else {
        if (a > 0) {
            sources.setValue({ynext, a}, {y, a - 1});
        } else {
            auto y0 = y - nRise;
            if (y0 < nY) {
                sources.setValue({ynext, a}, {y0, nA - 1});
            } else {
                sources.setValue({ynext, a});
            }
        }
    }
    return sources;
}

/// Return the starting index of Viterbi lattice.
inline Index CylinderGeometry::indexStart() {
    if (nRise >= 0) {
        return Index(0, 0);
    } else {
        return Index(0, nA - 1); 
    }
}

/// Return the ending index of Viterbi lattice.
inline Index CylinderGeometry::indexEnd() {
    if (nRise >= 0) {
        return Index(nY - 1, nA - 1);
    } else {
        return Index(nY - 1, 0);
    }
}

/// Reverse angular iteration if nRise is negative.
inline ssize_t CylinderGeometry::convertAngular(ssize_t ang) {
    if (nRise >= 0) {
        return ang;
    } else {
        return nA - ang - 1;
    }
}

/// Return all the pairs of indices that are connected longitudinally.
std::vector<std::pair<Index, Index>> CylinderGeometry::allLongitudinalPairs() {
    std::vector<std::pair<Index, Index>> pairs;
    for (ssize_t y = 0; y < nY; ++y) {
        for (ssize_t a = 0; a < nA; ++a) {
            auto idx1 = Index(y, a);
            auto sources = sourceForward(y, a);
            if (sources.hasLongitudinal()) {
                auto idx0 = Index(sources.lon.first, sources.lon.second);
                pairs.push_back({idx0, idx1});
            }
        }
    }
    return pairs;
}


/// Return all the pairs of indices that are connected laterally.
std::vector<std::pair<Index, Index>> CylinderGeometry::allLateralPairs() {
    std::vector<std::pair<Index, Index>> pairs;
    for (ssize_t y = 0; y < nY; ++y) {
        for (ssize_t a = 0; a < nA; ++a) {
            auto idx1 = Index(y, a);
            auto sources = sourceForward(y, a);
            if (sources.hasLateral()) {
                auto idx0 = Index(sources.lat.first, sources.lat.second);
                pairs.push_back({idx0, idx1});
            }
        }
    }
    return pairs;
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
