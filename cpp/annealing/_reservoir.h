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
    protected:
        double temperature;
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
        double temperature;
        double cooling_rate;
        double min_temperature;
    public:
        Reservoir() {
            this->temperature = 1.0;
            this->cooling_rate = 0.99;
            this->min_temperature = 0.0;
        }

        Reservoir(double temperature, double cooling_rate, double min_temperature = 0.0) {
            if (min_temperature < 0) {
                throw py::value_error("Minimum temperature must be positive");
            } else if (temperature < min_temperature) {
                throw py::value_error("Initial temperature must be greater than minimum temperature");
            } else if (cooling_rate <= 0 || cooling_rate >= 1) {
                throw py::value_error("Cooling rate must be in (0, 1)");
            }
            this->initial_temperature = temperature;
            this->temperature = temperature;
            this->cooling_rate = cooling_rate;
            this->min_temperature = min_temperature;
        }

        void cool() {
            temperature = std::max(temperature * cooling_rate, min_temperature);
        }

        float prob(float dE) override {
            return (dE < 0) ? 1 : static_cast<float>(exp(-dE / temperature));
        }

        float getTemperature() { return temperature; }
        void initialize() { temperature = initial_temperature; }
};

#endif
