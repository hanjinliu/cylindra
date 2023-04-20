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
    protected:
        AbstractGraph<Sn, Se, T> graph;
        AbstractReservoir reservoir;
        RandomNumberGenerator rng;

        void proceed();
        void setRandomState(int seed) { this.rng = RandomNumberGenerator(seed); }

    public:
        AbstractAnnealingModel& setGraph(AbstractGraph<Sn, Se, T> graph) & {
            this->graph = graph;
            return *this;
        }
        AbstractAnnealingModel& setReservoir(AbstractReservoir reservoir) & {
            this->reservoir = reservoir;
            return *this;
        }
        AbstractAnnealingModel& setTemperature(float temperature) & {
            reservoir.setTemperature(temperature);
            return *this;
        }

        void simulate(ssize_t nsteps);
        T totalEnergy() { return graph.totalEnergy(); }
};

template <typename Sn, typename Se, typename T>
void AbstractAnnealingModel<Sn, Se, T>::proceed() {
    auto idx = rng.uniformInt(graph.nodeCount());
    auto result = graph.tryRandomShift(rng);
    auto prob = reservoir.prob(result.dE);

    if (rng.bernoulli(prob)) {
        // accept shift
        graph.applyShift(result);
    }
}

template <typename Sn, typename Se, typename T>
void AbstractAnnealingModel<Sn, Se, T>::simulate(ssize_t nsteps) {
    if (nsteps < 0) {
        throw py::value_error("nsteps must be non-negative.");
    }
    graph.checkGraph();
    for (auto i = 0; i < nsteps; ++i) {
        proceed();
        reservoir.cool();
    }
}

class CylindricAnnealingModel : public AbstractAnnealingModel<NodeState, EdgeType, float> {
    protected:
        CylindricGraph graph;
    public:
        CylindricAnnealingModel(int seed) {
            this->reservoir = Reservoir();
            this->rng = RandomNumberGenerator(seed);
        }

        CylindricAnnealingModel& setReservoir(
            float temperature,
            float cooling_rate,
            float min_temperature = 0.0
        ) & {
            this->reservoir = Reservoir(temperature, cooling_rate, min_temperature);
            return *this;
        }

        CylindricAnnealingModel& setGraph(
            py::array_t<float> &score,
            py::array_t<float> &origin,
            py::array_t<float> &zvec,
            py::array_t<float> &yvec,
            py::array_t<float> &xvec,
            int nrise
        ) & {
            this->graph.update(score, origin, zvec, yvec, xvec, nrise);
            return *this;
        }

        CylindricAnnealingModel& setBoxPotential(
            double lon_dist_min,
            double lon_dist_max,
            double lat_dist_min,
            double lat_dist_max
        ) & {
            BoxPotential2D model(lon_dist_min, lon_dist_max, lat_dist_min, lat_dist_max);
            this->graph.setPotentialModel(model);
            return *this;
        }

        py::array_t<int> getShifts() { return graph.getShifts(); }
};

#endif
