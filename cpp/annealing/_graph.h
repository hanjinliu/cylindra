#ifndef _GRAPH_H
#define _GRAPH_H

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <set>
#include <unordered_map>

#include "_random.h"
#include "_binding_potential.h"
#include "../_coords.h"
#include "../_cylindric.h"

template <typename S>
struct ShiftResult {
    S state;
    float dGain;
    ShiftResult(S _state, float _dGain) : state(_state), dGain(_dGain) {}
};

/// @brief Abstract class for an undirected graph with scores.
/// @tparam N type of node (e.g. 3D vector for a 3D grid)
/// @tparam S type of internal state (e.g. bool for up/down)
/// @tparam T type of score (e.g. float for negative energy)
template <typename N, typename S, typename T>
class AbstractGraph {
    protected:
        std::unordered_map<N, std::set<N>> neighborsMap;  // to speed up neighbor lookup
        std::unordered_map<std::pair<N, N>, int> edgeType;
    public:
        std::vector<N> nodes;

        AbstractGraph() {
            this->nodes = std::vector<N>();
            this->neighborsMap = std::unordered_map<N, std::set<N>>();
        };

        void addEdge(N &node0, N &node1) {
            this->neighborsMap[node0].insert(node1);
            this->neighborsMap[node1].insert(node0);
        };

        /// Returns the number of nodes.
        virtual int count() { return 0; };

        /// Returns the local state of the molecule at `pos`.
        virtual S localState(N &pos) { return S(); };

        /// Sets the local state of the molecule at `pos`.
        virtual void setLocalState(N &pos, S &state) { return; };

        virtual S randomLocalNeighBorState(N &pos, S &state, RandomNumberGenerator &rng) { return S(); };

        /// Returns the "potential score" of molecule at `pos` of given `state`.
        virtual T potential(N &pos, S &state) { return 0.0; };

        /// Returns the "binding score" between adjacent molecules.
        virtual T binding(N &pos0, N &pos1, S &state0, S &state1, int type) { return 0.0; };

        /// Returns the total score of the current graph state.
        T totalEnergy() {
            for (int i = 0; i < count(); i++) {
                auto _node = nodes[i];
                auto _state = localState(_node);
                auto score = potential(_node, _state);
                for (auto neighbor : neighborsMap[_node]) {
                    score += binding(_node, neighbor, _state, localState(neighbor));
                }
            }
        };

        ShiftResult<S> randomShift(RandomNumberGenerator &rng) {
            auto idx = rng.uniformInt(graph.count());
            auto pos = nodes[idx];
            auto state_old = localState(pos);
            auto score_old = potential(pos, state_old);
            auto state_new = randomLocalNeighBorState(pos, state_old, rng);
            auto score_new = potential(pos, state_new);
            for (auto n: neighbors(pos)) {
                auto nstate = localState(n);
                score_old += binding(pos, n, state_old, nstate);
                score_new += binding(pos, n, state_new, nstate);
            }
            auto dGain = score_new - score_old;
            return ShiftResult(state_new, dGain);
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

class CylindricGraph : public AbstractGraph<Index, Vector3D<int>, float> {
    private:
        CylinderGeometry geometry;
        Grid2D<Vector3D<int>> states;
        Grid2D<CoordinateSystem<double>> coords;
        py::array_t<float> score;
        BindingPotential2D bindingPotential;
        std::vector<ssize_t> localShape;

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
            auto nodes = std::vector<Index>();
            auto edges = std::unordered_map<Index, std::set<Index>>();
            for (auto y = 0; y < geometry.nY; ++y) {
                for (auto a = 0; a < geometry.nA; ++a) {
                    auto idx = Index(y, a);
                    nodes.push_back(idx);
                    auto neighbors = geometry.getNeighbor(idx.y, idx.a);
                    for (auto n : neighbors) {
                        edges[idx].insert(n);
                    }
                }
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
            this->localShape = std::vector<ssize_t>({score_shape[2], score_shape[3], score_shape[4]});
            this->nodes = nodes;
            this->neighborsMap = edges;
        }

        int count() override {
            return geometry.nY * geometry.nA;
        }

        Vector3D<int> localState(Index &pos) override {
            return states.at(pos.y, pos.a);
        }

        void setLocalState(Index &pos, Vector3D<int> &state) override {
            states.setAt(pos.y, pos.a, state);
        }

        Vector3D<int> randomLocalNeighBorState(Index &pos, Vector3D<int> &state, RandomNumberGenerator &rng) override {
            auto neighbors = geometry.getNeighbor(pos.y, pos.a);
            auto n = neighbors[rng.uniformInt(neighbors.size())];
            return states.at(n.y, n.a);
        }

        float potential(Index &pos, Vector3D<int> &state) override {
            auto vec = states.at(pos.y, pos.a);
            return *score.data(vec.z, vec.y, vec.x);
        }

        float binding(
            Index &pos1,
            Index &pos2,
            Vector3D<int> &state1,
            Vector3D<int> &state2,
            int type
        ) override {
            auto vec1 = states.at(pos1.y, pos1.a);
            auto vec2 = states.at(pos2.y, pos2.a);
            auto coord1 = coords.at(pos1.y, pos1.a);
            auto coord2 = coords.at(pos2.y, pos2.a);
            auto dr = coord1.at(vec1.z, vec1.y, vec1.x) - coord2.at(vec2.z, vec2.y, vec2.x);

            auto edge = std::make_pair(pos1, pos2);
            auto ptr = edgeType.find(edge);

            int type;
            if (ptr != edgeType.end()) {
                type = ptr->second;
            } else {
                type = 0;
            }
            return bindingPotential(dr.length2(), type);
        }

        py::array_t<int> getShifts() {
            auto out = py::array_t<int>{{geometry.nY, geometry.nA, ssize_t(3)}};
            for (auto i = 0; i < geometry.nY; ++i) {
                for (auto j = 0; j < geometry.nA; ++j) {
                    auto shift = states.at(i, j);
                    out.mutable_at(i, j, 0) = shift.x;
                    out.mutable_at(i, j, 1) = shift.y;
                    out.mutable_at(i, j, 2) = shift.z;
                }
            }
            return out;
        }
};



#endif
