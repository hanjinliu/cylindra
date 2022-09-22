#ifndef _CYLINDRIC_H
#define _CYLINDRIC_H

#pragma once

#include <vector>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using uint = unsigned int;

// tuple of unsigned ints
struct Index {
    uint r, y, a;
    Index(uint r, uint y, uint a) : r(r), y(y), a(a) {}
};

// tuple of signed ints
struct SignedIndex {
    int r, y, a;
    SignedIndex(int r, int y, int a) : r(r), y(y), a(a) {}
    SignedIndex(uint r, uint y, uint a) : 
        r(static_cast<int>(r)),
        y(static_cast<int>(y)), 
        a(static_cast<int>(a)) {}
};

class CylinderGeometry {
    public:
        uint nR, nY, nA, nRise;
        CylinderGeometry(uint nR, uint nY, uint nA, uint nRise){
            nR = nR;
            nY = nY;
            nA = nA;
            nRise = nRise;
        };
        CylinderGeometry(uint nR, uint nY, uint nA) : CylinderGeometry(nR, nY, nA, 0){};
        std::vector<Index> getNeighbor(uint, uint, uint);
        std::vector<Index> getNeighbors(std::vector<Index>);
        Index getIndex(uint, uint, uint);

    private:
        long compress(Index);
        Index decompress(long);
        SignedIndex getSignedIndex(int, int, int);
};

inline SignedIndex CylinderGeometry::getSignedIndex(int r, int y, int a){
    while (a >= nA){
        a -= nA;
        y += nRise;
    }
    while (a >= 0){
        a += nA;
        y -= nRise;
    }
    return SignedIndex(r, y, a);
}

inline Index CylinderGeometry::getIndex(uint r, uint y, uint a){
    auto idx = getSignedIndex(r, y, a);
    if (!(0 <= idx.r < nR && 0 <= idx.y < nY)){
        char msg[64];
        sprintf(msg, "Index (%d, %d, %d) out of bounds.", idx.r, idx.y, idx.a);
        throw py::index_error(msg);
    }
    return Index(idx.r, idx.y, idx.a);
}

inline std::vector<Index> CylinderGeometry::getNeighbor(uint r, uint y, uint a){
    std::vector<Index> neighbors;
    auto idx = getSignedIndex(r, y, a);
    
    if (r > 0){
        neighbors.push_back(Index(r-1, y, a));
    }
    if (r < nR-1){
        neighbors.push_back(Index(r+1, y, a));
    }
    if (y > 0){
        neighbors.push_back(Index(r, y-1, a));
    }
    if (y < nY-1){
        neighbors.push_back(Index(r, y+1, a));
    }
    if (a > 0){
        neighbors.push_back(Index(r, y, a-1));
    }
    if (a < nA-1){
        neighbors.push_back(Index(r, y, a+1));
    }
    return neighbors;
}

// Get all the unique neighbors of a list of indices.
inline std::vector<Index> CylinderGeometry::getNeighbors(std::vector<Index> indices){
    std::set<long> uniqueNeighbors;
    
    // add all the neighbor candidates
    for (auto index : indices){
        auto new_neighbors = getNeighbor(index.r, index.y, index.a);
        for (auto neighbor : new_neighbors){
            long compressed = compress(neighbor);
            uniqueNeighbors.insert(compressed);
        }
    }
    
    // remove inputs
    for (auto index : indices){
        uniqueNeighbors.erase(compress(index));
    }

    // convert to a vector
    std::vector<Index> neighbors;
    for (long neighbor : uniqueNeighbors){
        neighbors.push_back(decompress(neighbor));
    }
    return neighbors;
}

// Compresses the index tuple into a single long.
// This method returns the absolute index of the index tuple.
inline long CylinderGeometry::compress(Index idx){
    return idx.r + idx.y * nR + idx.a * nR * nY;
}

// Decompresses the index tuple from a single long
inline Index CylinderGeometry::decompress(long idx){
    uint a = idx / (nR * nY);
    uint y = (idx - a * nR * nY) / nR;
    uint r = idx - a * nR * nY - y * nR;
    return Index(r, y, a);
}

#endif
