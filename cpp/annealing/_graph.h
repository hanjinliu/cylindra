#ifndef _GRAPH_H
#define _GRAPH_H

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "_random.h"
#include "_binding_potential.h"
#include "../_coords.h"
#include "../_cylindric.h"

template <typename S>
struct ShiftResult {
    size_t index;
    S state;
    float dE;
    ShiftResult(size_t i, S s, float e) : index(i), state(s), dE(e) {}
};

/// @brief Abstract class for an undirected graph with scores.
template <typename Sn, typename Se, typename T>
class AbstractGraph {
    public:
        std::vector<std::vector<size_t>> edges;  // id of connected edges of the i-th node
        std::vector<std::pair<size_t, size_t>> edgeEnds;  // end nodes of the i-th edge
        std::vector<Sn> nodeState;
        std::vector<Se> edgeState;

        AbstractGraph() {};

        virtual size_t nodeCount() { return nodeState.size(); };
        virtual size_t edgeCount() { return edgeEnds.size(); };

        void addEdge(size_t i, size_t j, Se edgestate) {
            edges[i].push_back(edgeEnds.size());
            edges[j].push_back(edgeEnds.size());
            edgeEnds.push_back(std::make_pair(i, j));
        }

        virtual Sn randomLocalNeighBorState(Sn &nodestate, RandomNumberGenerator &rng) { return nodestate; };

        /// Returns the "potential score" of molecule at `pos` of given `state`.
        virtual T potential(Sn &nodestate) { return 0; };

        /// Returns the "binding score" between adjacent molecules.
        virtual T binding(Sn &nodestate0, Sn &nodestate1, Se &edgestate) { return 0.0; };

        /// Returns the total score of the current graph state.
        T totalEnergy();

        ShiftResult<Sn> tryRandomShift(RandomNumberGenerator&);
        void applyShift(ShiftResult<Sn> &result) {
            nodeState[result.index] = result.state;
        }
};

template <typename Sn, typename Se, typename T>
T AbstractGraph<Sn, Se, T>::totalEnergy() {
    T score = 0;
    for (auto s : nodeState) {
        score += potential(s);
    }
    for (size_t i = 0; i < edgeEnds.size(); ++i) {
        auto ends = edgeEnds[i];
        score += binding(nodeState[ends.first], nodeState[ends.second], edgeState[i]);
    }
    return score;
};

template <typename Sn, typename Se, typename T>
ShiftResult<Sn> AbstractGraph<Sn, Se, T>::tryRandomShift(RandomNumberGenerator &rng) {
    auto idx = rng.uniformInt(nodeCount());
    Sn state_old = nodeState[idx];
    T E_old = potential(state_old);
    Sn state_new = randomLocalNeighBorState(state_old, rng);
    T E_new = potential(state_new);
    auto connected_edges = edges[idx];

    for (auto edgeid : connected_edges) {
        auto ends = edgeEnds[edgeid];
        auto other_idx = (ends.first == idx) ? ends.second : ends.first;
        Sn other_state = nodeState[other_idx];
        E_old += binding(state_old, other_state, edgeState[edgeid]);
        E_new += binding(state_new, other_state, edgeState[edgeid]);
    }

    auto dE = E_new - E_old;
    return ShiftResult<Sn>(idx, state_new, dE);
}

template <typename T>
class Grid2D {
    public:
        T* coords;  // flattened coordinate system array
        ssize_t naxial, nang;
        T at(ssize_t y, ssize_t a) {
            return coords[y * nang + a];
        }
        T at(std::pair<ssize_t, ssize_t> y_a) {
            return at(y_a.first, y_a.second);
        }
        T at1d(ssize_t i) {
            return coords[i];
        }
        void setAt(ssize_t y, ssize_t a, T &val) {
            coords[y * nang + a] = val;
        }
        ssize_t size() {
            return naxial * nang;
        }
        Grid2D() : naxial(0), nang(0) {};
        Grid2D(ssize_t _naxial, ssize_t _nang) {
            naxial = _naxial;
            nang = _nang;
            coords = new T[naxial * nang];
            for (auto i = 0; i < naxial * nang; ++i) {
                coords[i] = T();
            }
        };
};

struct _NodeState {
    Index index;
    Vector3D<int> shift;
    // _NodeState() {
    //     index = Index(0, 0);
    //     shift = Vector3D<int>(0, 0, 0);
    // };
    _NodeState(Index i, Vector3D<int> s) : index(i), shift(s) {};
};


class CylindricGraph : public AbstractGraph<_NodeState, EdgeType, float> {
    private:
        CylinderGeometry geometry;
        Grid2D<CoordinateSystem<double>> coords;
        py::array_t<float> score;  // 5D array
        BindingPotential2D bindingPotential;
        Vector3D<int> localShape;

    public:
        CylindricGraph() {};
        CylindricGraph(
            py::array_t<float> &score,
            py::array_t<float> &origin,
            py::array_t<float> &zvec,
            py::array_t<float> &yvec,
            py::array_t<float> &xvec,
            int nrise
        ) {
            update(score, origin, zvec, yvec, xvec, nrise);
        }

        void update(py::array_t<float>&, py::array_t<float>&, py::array_t<float>&, py::array_t<float>&, py::array_t<float>&, int nrise);

        _NodeState randomLocalNeighBorState(_NodeState &state, RandomNumberGenerator &rng) override {
            auto idx = state.index;
            auto shift = state.shift;
            auto shift_new = rng.randShift(shift, localShape);
            return _NodeState(idx, shift_new);
        }

        float potential(_NodeState &nodestate) override {
            auto idx = nodestate.index;
            auto vec = nodestate.shift;
            return *score.data(idx.y, idx.a, vec.z, vec.y, vec.x);
        }

        float binding(
            _NodeState &pos1,
            _NodeState &pos2,
            EdgeType &type
        ) override {
            auto vec1 = pos1.shift;
            auto vec2 = pos2.shift;
            auto coord1 = coords.at(pos1.index.y, pos1.index.a);
            auto coord2 = coords.at(pos2.index.y, pos2.index.a);
            auto dr = coord1.at(vec1.z, vec1.y, vec1.x) - coord2.at(vec2.z, vec2.y, vec2.x);
            return bindingPotential(dr.length2(), type);
        }

        py::array_t<int> getShifts() {
            auto out = py::array_t<int>{{geometry.nY, geometry.nA, ssize_t(3)}};
            for (auto state : nodeState) {
                auto y = state.index.y;
                auto a = state.index.a;
                auto shift = state.shift;
                out.mutable_at(y, a, 0) = shift.z;
                out.mutable_at(y, a, 1) = shift.y;
                out.mutable_at(y, a, 2) = shift.x;
            }
            return out;
        }
};

void CylindricGraph::update(
    py::array_t<float> &score,
    py::array_t<float> &origin,
    py::array_t<float> &zvec,
    py::array_t<float> &yvec,
    py::array_t<float> &xvec,
    int nrise
) {
    auto score_shape = score.request().shape;
    if (score_shape.size() != 5) {
        throw py::value_error("Score array must be 5D");
    } else if (origin.shape(0) != score.shape(0) || zvec.shape(0) != score.shape(0) || yvec.shape(0) != score.shape(0) || xvec.shape(0) != score.shape(0)) {
        throw py::value_error("Score array and vectors must have the same first dimension");
    } else if (origin.shape(1) != score.shape(1) || zvec.shape(1) != score.shape(1) || yvec.shape(1) != score.shape(1) || xvec.shape(1) != score.shape(1)) {
        throw py::value_error("Score array and vectors must have the same second dimension");
    } else if (origin.shape(2) != 3 || zvec.shape(2) != 3 || yvec.shape(2) != 3 || xvec.shape(2) != 3) {
        throw py::value_error("Vectors must be 3D");
    }
    auto naxial = score_shape[0];
    auto nang = score_shape[1];

    geometry = CylinderGeometry(naxial, nang, nrise);
    localShape = Vector3D<int>(score_shape[2], score_shape[3], score_shape[4]);

    nodeState.clear();
    edgeState.clear();
    edges.clear();
    edgeEnds.clear();

    auto center = localShape / 2;
    for (auto y = 0; y < geometry.nY; ++y) {
        for (auto a = 0; a < geometry.nA; ++a) {
            auto idx = Index(y, a);
            nodeState.push_back(_NodeState(idx, center));
        }
    }

    for (auto pair : geometry.allLongitudinalPairs()) {
        auto idx0 = geometry.nA * pair.first.y + pair.first.a;
        auto idx1 = geometry.nA * pair.second.y + pair.second.a;
        addEdge(idx0, idx1, EdgeType::Longitudinal);
    }

    for (auto pair : geometry.allLateralPairs()) {
        auto idx0 = geometry.nA * pair.first.y + pair.first.a;
        auto idx1 = geometry.nA * pair.second.y + pair.second.a;
        addEdge(idx0, idx1, EdgeType::Lateral);
    }

    Grid2D<CoordinateSystem<double>> _coords(naxial, nang);

    for (auto t = 0; t < naxial; ++t) {
        for (auto s = 0; s < nang; ++s) {
            auto _ori = Vector3D<double>(*origin.data(t, s, 0), *origin.data(t, s, 1), *origin.data(t, s, 2));
            auto _ez = Vector3D<double>(*zvec.data(t, s, 0), *zvec.data(t, s, 1), *zvec.data(t, s, 2));
            auto _ey = Vector3D<double>(*yvec.data(t, s, 0), *yvec.data(t, s, 1), *yvec.data(t, s, 2));
            auto _ex = Vector3D<double>(*xvec.data(t, s, 0), *xvec.data(t, s, 1), *xvec.data(t, s, 2));
            _coords.setAt(t, s, CoordinateSystem<double>(_ori, _ez, _ey, _ex));
        }
    }

    this->score = score;
    this->coords = _coords;
}


#endif
