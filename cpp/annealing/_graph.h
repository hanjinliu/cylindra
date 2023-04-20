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
    protected:
        std::vector<std::vector<size_t>> edges;  // id of connected edges of the i-th node
        std::vector<std::pair<size_t, size_t>> edgeEnds;  // end nodes of the i-th edge
        std::vector<Sn> nodeState;
        std::vector<Se> edgeState;

    public:
        AbstractGraph() {};

        size_t nodeCount() { return nodeState.size(); };
        size_t edgeCount() { return edgeState.size(); };

        /// Add a node of given state to the graph.
        void addNode(const Sn &nodestate) {
            nodeState.push_back(nodestate);
            edges.push_back(std::vector<size_t>());
        }

        /// add an edge between i-th and j-th nodes.
        void addEdge(size_t i, size_t j, const Se &edgestate) {
            if (i >= nodeCount() || j >= nodeCount()) {
                throw py::index_error(
                    "There are " + std::to_string(nodeCount()) +
                    " nodes, but trying to add an edge between " + std::to_string(i) +
                    " and " + std::to_string(j) + "."
                );
            }
            edges[i].push_back(edgeEnds.size());
            edges[j].push_back(edgeEnds.size());
            edgeEnds.push_back(std::make_pair(i, j));
            edgeState.push_back(edgestate);
        }

        Sn &nodeStateAt(size_t i) & { return nodeState[i]; };
        Se &edgeStateAt(size_t i) & { return edgeState[i]; };
        std::pair<size_t, size_t> edgeEndsAt(size_t i) & { return edgeEnds[i]; };

        virtual Sn randomLocalNeighBorState(const Sn &nodestate, RandomNumberGenerator &rng) { throw py::attribute_error("randomLocalNeighBorState() is not implemented.");};

        /// Returns the internal potential energy of molecule at `pos` of given `state`.
        virtual T internal(const Sn &nodestate) { return 0; };

        /// Returns the binding potential energy between adjacent molecules.
        virtual T binding(const Sn &nodestate0, const Sn &nodestate1, const Se &edgestate) { return 0; };

        // Check if the current state of this graph is ready.
        virtual void checkGraph() { throw py::attribute_error("checkGraph() is not implemented.");};

        void applyShift(ShiftResult<Sn> &result) {
            nodeState[result.index] = result.state;
        }

        std::vector<std::pair<size_t, size_t>> getEdgeEnds() { return edgeEnds; };

        /// Clear all the nodes and edges of the graph.
        void clearGraph() {
            nodeState.clear();
            edgeState.clear();
            edges.clear();
            edgeEnds.clear();
        }
};

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

struct NodeState {
    Index index;
    Vector3D<int> shift;
    NodeState(Index i, Vector3D<int> s) : index(i), shift(s) {};
};


class CylindricGraph : public AbstractGraph<NodeState, EdgeType, float> {
    private:
        CylinderGeometry geometry;
        Grid2D<CoordinateSystem<double>> coords;
        py::array_t<float> score;  // 5D array
        BoxPotential2D bindingPotential;
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

        NodeState randomLocalNeighBorState(const NodeState &state, RandomNumberGenerator &rng) override {
            auto idx = state.index;
            auto shift = state.shift;
            auto shift_new = rng.randShift(shift, localShape);
            return NodeState(idx, shift_new);
        }

        float internal(const NodeState &nodestate) override {
            auto idx = nodestate.index;
            auto vec = nodestate.shift;
            return *score.data(idx.y, idx.a, vec.z, vec.y, vec.x);
        }

        float binding(
            const NodeState &pos1,
            const NodeState &pos2,
            const EdgeType &type
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
            for (auto i = 0; i < nodeCount(); ++i) {
                auto state = nodeState[i];
                auto y = state.index.y;
                auto a = state.index.a;
                auto shift = state.shift;
                out.mutable_at(y, a, 0) = shift.z;
                out.mutable_at(y, a, 1) = shift.y;
                out.mutable_at(y, a, 2) = shift.x;
            }
            return out;
        }

        py::array_t<float> getLongitudinalDistances() { return getDistances(EdgeType::Longitudinal); }
        py::array_t<float> getLateralDistances() { return getDistances(EdgeType::Lateral); }

        py::array_t<float> getDistances(EdgeType type) {
            std::vector<float> out;
            for (auto i = 0; i < edgeCount(); ++i) {
                auto state = edgeState[i];
                if (state != type) continue;
                auto nodes = edgeEnds[i];
                auto pos1 = nodeState[nodes.first];
                auto pos2 = nodeState[nodes.second];
                auto coord1 = coords.at(pos1.index.y, pos1.index.a);
                auto coord2 = coords.at(pos2.index.y, pos2.index.a);
                auto dr = coord1.at(pos1.shift.z, pos1.shift.y, pos1.shift.x) - coord2.at(pos2.shift.z, pos2.shift.y, pos2.shift.x);
                out.push_back(dr.length());
            }
            return py::array_t<float>(out.size(), out.data());
        }

        BoxPotential2D potentialModel() { return bindingPotential; }

        void setPotentialModel(BoxPotential2D &model) { bindingPotential = model; }

        void checkGraph() override {
            if (nodeCount() == 0) {
                throw py::value_error("graph is empty.");
            } else if (edgeCount() == 0) {
                throw py::value_error("graph has no edges.");
            } else if (!bindingPotential.isConcrete()) {
                throw py::value_error("binding potential is not concrete.");
            } else if (edgeEnds.size() != edgeCount()) {
                throw py::value_error("edgeEnds.size() != edgeCount()");
            } else if (edges.size() != nodeCount()) {
                throw py::value_error("edges.size() != nodeCount");
            }
        }

        float totalEnergy();
        ShiftResult<NodeState> tryRandomShift(RandomNumberGenerator &rng);
        void initialize() {
            auto center = localShape / 2;
            for (auto y = 0; y < geometry.nY; ++y) {
                for (auto a = 0; a < geometry.nA; ++a) {
                    auto idx = Index(y, a);
                    auto i = geometry.nA * y + a;
                    nodeState[i] = NodeState(idx, center);
                }
            }
        }

};

void CylindricGraph::update(
    py::array_t<float> &score_array,
    py::array_t<float> &origin,
    py::array_t<float> &zvec,
    py::array_t<float> &yvec,
    py::array_t<float> &xvec,
    int nrise
) {
    auto score_shape = score_array.request().shape;
    if (score_shape.size() != 5) {
        throw py::value_error("Score array must be 5D");
    } else if (origin.shape(0) != score_array.shape(0) || zvec.shape(0) != score_array.shape(0) || yvec.shape(0) != score_array.shape(0) || xvec.shape(0) != score_array.shape(0)) {
        throw py::value_error("Score array and vectors must have the same first dimension");
    } else if (origin.shape(1) != score_array.shape(1) || zvec.shape(1) != score_array.shape(1) || yvec.shape(1) != score_array.shape(1) || xvec.shape(1) != score_array.shape(1)) {
        throw py::value_error("Score array and vectors must have the same second dimension");
    } else if (origin.shape(2) != 3 || zvec.shape(2) != 3 || yvec.shape(2) != 3 || xvec.shape(2) != 3) {
        throw py::value_error("Vectors must be 3D");
    }
    auto naxial = score_shape[0];
    auto nang = score_shape[1];

    geometry = CylinderGeometry(naxial, nang, nrise);
    localShape = Vector3D<int>(score_shape[2], score_shape[3], score_shape[4]);

    clearGraph();

    auto center = localShape / 2;
    for (auto y = 0; y < geometry.nY; ++y) {
        for (auto a = 0; a < geometry.nA; ++a) {
            auto idx = Index(y, a);
            addNode(NodeState(idx, center));
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

    this->score = score_array;
    this->coords = _coords;
}


float CylindricGraph::totalEnergy() {
    float score = 0;
    for (auto i = 0; i < nodeCount(); ++i) {
        score += internal(nodeState[i]);
    }
    for (size_t i = 0; i < edgeCount(); ++i) {
        auto ends = edgeEnds[i];
        score += binding(nodeState[ends.first], nodeState[ends.second], edgeState[i]);
    }
    return score;
};

ShiftResult<NodeState> CylindricGraph::tryRandomShift(RandomNumberGenerator &rng) {
    auto idx = rng.uniformInt(nodeCount());
    auto state_old = nodeState[idx];
    auto E_old = internal(state_old);
    auto state_new = randomLocalNeighBorState(state_old, rng);
    auto E_new = internal(state_new);
    auto connected_edges = edges[idx];

    for (auto edgeid : connected_edges) {
        auto ends = edgeEnds[edgeid];
        auto other_idx = (ends.first == idx) ? ends.second : ends.first;
        NodeState other_state = nodeState[other_idx];
        E_old += binding(state_old, other_state, edgeState[edgeid]);
        E_new += binding(state_new, other_state, edgeState[edgeid]);
    }

    auto dE = E_new - E_old;
    return ShiftResult<NodeState>(idx, state_new, dE);
}

#endif
