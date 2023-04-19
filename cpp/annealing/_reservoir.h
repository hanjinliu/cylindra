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
        void setTemperature(double temperature) {
            if (temperature <= 0) {
                throw py::value_error("Temperature must be positive");
            }
            this->temperature = temperature;
        }
};

/// @brief Basic reservoir class.
/// This reservoir implements a exponential cooling schedule and the Boltzmann-type
/// energy function.
class Reservoir : public AbstractReservoir {
    private:
        double temperature;
        double cooling_rate;
        double min_temperature;
        unsigned int chunk;
        unsigned int chunk_count;
    public:
        Reservoir() {
            this->temperature = 1.0;
            this->cooling_rate = 0.99;
            this->min_temperature = 0.0;
            this->chunk = 1;
            this->chunk_count = 0;
        }
        Reservoir(double temperature, double cooling_rate, double min_temperature = 0.0) {
            if (min_temperature <= 0) {
                throw py::value_error("Minimum temperature must be positive");
            } else if (temperature < min_temperature) {
                throw py::value_error("Initial temperature must be greater than minimum temperature");
            } else if (cooling_rate <= 0 || cooling_rate >= 1) {
                throw py::value_error("Cooling rate must be in (0, 1)");
            }
            this->temperature = temperature;
            this->cooling_rate = cooling_rate;
            this->min_temperature = min_temperature;
            this->chunk = 1;
            this->chunk_count = 0;
        }
        void cool() {
            chunk_count++;
            if (chunk_count == chunk) {
                temperature = std::max(temperature * cooling_rate, min_temperature);
                chunk_count = 0;
            }
        }
        float prob(float dE) override {
            return (dE < 0) ? 1.0 : static_cast<float>(exp(-dE / temperature));
        }
};

#endif
