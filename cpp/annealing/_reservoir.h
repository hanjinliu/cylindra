#ifndef _RESERVOIR_H
#define _RESERVOIR_H

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

using ssize_t = Py_ssize_t;
namespace py = pybind11;

/// @brief Abstract class for reservoirs
/// A reservoir implements a temperature schedule and a probability function for the
/// given energy.
class AbstractReservoir {
    public:
        virtual void cool() {};
        virtual float prob(float dE) { return 0.0; };
};

/// @brief Basic reservoir class.
/// This reservoir implements a exponential cooling schedule and the Boltzmann-type
/// energy function.
class Reservoir : public AbstractReservoir {
    private:
        double initial_temperature;
        double temperature0;
        double temperature;
        double time_constant;
        double min_temperature;
    public:
        /// Default constructor.
        Reservoir() {
            initial_temperature = 1.0;
            temperature0 = 1.0;
            temperature = 1.0;
            time_constant = 10000;
            min_temperature = 0.0;
        }

        Reservoir(double temperature, double time_constant, double min_temperature = 0.0) {
            // Check values
            if (min_temperature < 0) {
                throw py::value_error("Minimum temperature must be positive");
            } else if (temperature < min_temperature) {
                throw py::value_error("Initial temperature must be greater than minimum temperature");
            } else if (time_constant <= 0) {
                throw py::value_error("Time constant must be positive.");
            }
            // initial temperature
            initial_temperature = temperature;
            this->time_constant = time_constant;
            this->min_temperature = min_temperature;
            temperature0 = initial_temperature - min_temperature;
            initialize();
        }

        /// Cool the reservoir by one step.
        void cool(size_t n) {
            temperature = temperature0 * exp(-static_cast<double>(n) / time_constant) + min_temperature;
        }

        /// Calculate the transition probability for the energy change dE.
        float prob(float dE) override {
            return (dE < 0) ? 1 : static_cast<float>(exp(-dE / temperature));
        }

        float getTemperature() { return temperature; }

        /// Initialize the reservoir.
        void initialize() {
            temperature = initial_temperature;
        }
};

#endif
