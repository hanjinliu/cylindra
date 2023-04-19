#ifndef _ANNEAL_H
#define _ANNEAL_H

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include "_random.h"
#include "_graph.h"
#include "_reservoir.h"
#include "../_coords.h"

using ssize_t = Py_ssize_t;

template <typename Sn, typename Se, typename T>
class AbstractAnnealingModel {
    private:
        std::vector<T> energies;
    protected:
        AbstractGraph<Sn, Se, T> graph;
        AbstractReservoir reservoir;
        RandomNumberGenerator rng;

        void proceed();
        void setRandomState(int seed) { this.rng = RandomNumberGenerator(seed); }

    public:
        void setGraph(AbstractGraph<Sn, Se, T> graph) { this->graph = graph; }
        void setReservoir(AbstractReservoir reservoir) { this->reservoir = reservoir; }

        void setTemperature(float temperature) { reservoir.setTemperature(temperature); }

        py::array_t<float> getEnergyArray() {
            return py::array_t<float>(energies.size(), energies.data());
        }

        void simulate(ssize_t nsteps);
        T totalEnergy() { return graph.totalEnergy(); }

};

template <typename Sn, typename Se, typename T>
void AbstractAnnealingModel<Sn, Se, T>::proceed() {
    auto idx = rng.uniformInt(graph.nodeCount());
    auto result = graph.tryRandomShift(rng);
    auto prob = reservoir.prob(result.dE);
    auto last_energy = energies[energies.size() - 1];

    if (rng.bernoulli(prob)) {
        // accept shift
        graph.applyShift(result);
        energies.push_back(last_energy + result.dE);
    } else {
        // reject shift
        energies.push_back(last_energy);
    }
}

template <typename Sn, typename Se, typename T>
void AbstractAnnealingModel<Sn, Se, T>::simulate(ssize_t nsteps) {
    for (auto i = 0; i < nsteps; ++i) {
        proceed();
        reservoir.cool();
    }
}

class CylindricAnnealingModel : public AbstractAnnealingModel<_NodeState, EdgeType, float> {
    protected:
        CylindricGraph graph;
    public:
        CylindricAnnealingModel(int seed) {
            this->reservoir = Reservoir();
            this->rng = RandomNumberGenerator(seed);
        }

        void setReservoir(
            float temperature,
            float cooling_rate,
            float min_temperature
        ) {
            this->reservoir = Reservoir(temperature, cooling_rate, min_temperature);
        }

        void setGraph(
            py::array_t<float> &score,
            py::array_t<float> &origin,
            py::array_t<float> &zvec,
            py::array_t<float> &yvec,
            py::array_t<float> &xvec,
            int nrise
        ) {
            this->graph.update(score, origin, zvec, yvec, xvec, nrise);
        }

        py::array_t<int> getShifts() { return graph.getShifts(); }
};

#endif
