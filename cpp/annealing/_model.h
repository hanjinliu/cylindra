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

class AbstractAnnealingModel {
    protected:
        RandomNumberGenerator rng;

        virtual void proceed() {};
        void setRandomState(int seed) { rng = RandomNumberGenerator(seed); }

    public:
        virtual void simulate(ssize_t nsteps) {};
};

class CylindricAnnealingModel : public AbstractAnnealingModel {
    protected:
        CylindricGraph graph;
        Reservoir reservoir;
    public:
        CylindricAnnealingModel(int seed) {
            setReservoir(1.0, 0.995);
            this->rng = RandomNumberGenerator(seed);
        }

        CylindricAnnealingModel& setReservoir(
            float temperature,
            float cooling_rate,
            float min_temperature = 0.0
        ) & {
            Reservoir rv(temperature, cooling_rate, min_temperature);
            reservoir = rv;
            return *this;
        }

        CylindricGraph getGraph() { return graph; }
        CylindricAnnealingModel& setGraph(
            py::array_t<float> &score,
            py::array_t<float> &origin,
            py::array_t<float> &zvec,
            py::array_t<float> &yvec,
            py::array_t<float> &xvec,
            int nrise
        ) & {
            CylindricGraph gr(score, origin, zvec, yvec, xvec, nrise);
            graph = gr;
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
        float totalEnergy() { return graph.totalEnergy(); }
        void simulate(ssize_t nsteps) override;
        void proceed() override;
        void initialize() {
            graph.initialize();
            reservoir.initialize();
        };

};

void CylindricAnnealingModel::proceed() {
    auto idx = rng.uniformInt(graph.nodeCount());
    auto result = graph.tryRandomShift(rng);
    auto prob = reservoir.prob(result.dE);
    if (rng.bernoulli(prob)) {
        // accept shift
        graph.applyShift(result);
    }
}

void CylindricAnnealingModel::simulate(ssize_t nsteps) {
    if (nsteps < 0) {
        throw py::value_error("nsteps must be non-negative.");
    }
    graph.checkGraph();
    for (auto i = 0; i < nsteps; ++i) {
        proceed();
        reservoir.cool();
    }
}

#endif
