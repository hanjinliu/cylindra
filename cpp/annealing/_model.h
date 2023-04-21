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

enum class OptimizationState {
    NOT_CONVERGED,
    CONVERGED,
    FAILED,
};

class AbstractAnnealingModel {
    protected:
        RandomNumberGenerator rng;
        OptimizationState optimization_state;

        virtual bool proceed() { return false; };
        void setRandomState(int seed) { rng = RandomNumberGenerator(seed); }

    public:
        virtual void simulate(ssize_t nsteps) {};
};

class CylindricAnnealingModel : public AbstractAnnealingModel {
    protected:
        CylindricGraph graph;
        Reservoir reservoir;
        size_t REJECT_LIMIT = 200;
    public:
        CylindricAnnealingModel(int seed) {
            setReservoir(1.0, 0.995);
            this->rng = RandomNumberGenerator(seed);
            optimization_state = OptimizationState::NOT_CONVERGED;
        }

        Reservoir getReservoir() { return reservoir; }
        CylindricAnnealingModel& setReservoir(
            float temperature,
            float time_constant,
            float min_temperature = 0.0
        ) & {
            Reservoir rv(temperature, time_constant, min_temperature);
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
        bool proceed() override;
        void initialize() {
            graph.initialize();
            reservoir.initialize();
            optimization_state = OptimizationState::NOT_CONVERGED;
        };

        std::string getOptimizationState() {
            switch (optimization_state) {
                case OptimizationState::NOT_CONVERGED:
                    return "not_converged";
                case OptimizationState::CONVERGED:
                    return "converged";
                case OptimizationState::FAILED:
                    return "failed";
                default:
                    throw py::value_error("Unknown optimization state.");
            }
        }
};

/// Proceed the simulation step by one. Return true if the shift is accepted.
bool CylindricAnnealingModel::proceed() {
    auto idx = rng.uniformInt(graph.nodeCount());
    auto result = graph.tryRandomShift(rng);
    auto prob = reservoir.prob(result.dE);
    if (rng.bernoulli(prob)) {
        // accept shift
        graph.applyShift(result);
        return true;
    } else {
        return false;
    }
}

/// Run simulation for the given number of steps.
void CylindricAnnealingModel::simulate(ssize_t nsteps) {
    if (nsteps < 0) {
        throw py::value_error("nsteps must be non-negative.");
    }
    graph.checkGraph();
    size_t reject_count = 0;
	py::gil_scoped_release nogil;  // without GIL
    for (auto i = 0; i < nsteps; ++i) {
        if (proceed()) {
            reject_count = 0;
        } else {
            reject_count++;
        }
        if (reject_count > REJECT_LIMIT) {
            if (totalEnergy() == std::numeric_limits<float>::infinity()) {
                optimization_state = OptimizationState::FAILED;
            } else {
                optimization_state = OptimizationState::CONVERGED;
            }
            break;
        }
        reservoir.cool();
    }
}

#endif
