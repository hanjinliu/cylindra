#ifndef _ANNEAL_H
#define _ANNEAL_H

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include "_random.h"
#include "_cylindric.h"
#include "_coords.h"

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


struct ShiftResult {
    Vector3D<int> shift;
    float dGain;
    ShiftResult(Vector3D<int> _shift, float _dE) {
        shift = _shift;
        dGain = _dE;
    }
};

class AbstractReservoir {
    double temperature;
    virtual void cool() {};
};

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
        float prob(float dGain) {
            return (dGain > 0) ? 1.0 : exp(dGain/ temperature);
        }
        void setTemperature(double temperature) {
            if (temperature <= 0) {
                throw py::value_error("Temperature must be positive");
            }
            this->temperature = temperature;
        }
};

class AnnealingModel {
    private:
        py::array_t<float> score;
        Grid2D<CoordinateSystem<double>> coords;
        Grid2D<Vector3D<int>> shifts;
        CylinderGeometry geometry;
        Reservoir reservoir;
        RandomNumberGenerator rng;
        Vector3D<int> localShape;

        Index randomPosition() {
            auto random_index = rng.uniformInt(coords.size());
            return Index(random_index / coords.nang, random_index % coords.nang);
        }

        ShiftResult randomShift(ssize_t y, ssize_t a);

        void tryAccept() {
            auto idx = randomPosition();
            auto result = randomShift(idx.y, idx.a);
            float dGain = result.dGain;
            auto prob = reservoir.prob(dGain);
            auto last_gain = gains[gains.size() - 1];

            if (rng.bernoulli(prob)) {
                // accept shift
                shifts.setAt(idx.y, idx.a, result.shift);
                gains.push_back(last_gain + dGain);
            } else {
                // reject shift
                gains.push_back(last_gain);
            }
        }
        bool terminate() { return false; }; // TODO: just for now!

    public:
        std::vector<float> gains;
        AnnealingModel(py::array_t<float>&, ssize_t, ssize_t);
        AnnealingModel withSeed(ssize_t);

        void setReservoir(Reservoir reservoir) {
            this->reservoir = reservoir;
        }

        void optimize(ssize_t nsteps);
        float getGain();
        py::array_t<float> getGainArray() {
            return py::array_t<float>(gains.size(), gains.data());
        }
        py::array_t<int> getShifts();
        float upperLimit();

};

AnnealingModel::AnnealingModel(py::array_t<float> &score, ssize_t nrise, ssize_t seed = 0) {
    this->score = score;
    auto score_shape = score.request().shape;
    // check dimensionality
    if (score_shape.size() != 5) {
        throw py::value_error("Score array must be 5D");
    }
    auto naxial = score_shape[0];
    auto nang = score_shape[1];
    coords = Grid2D<CoordinateSystem<double>>(naxial, nang);
    shifts = Grid2D<Vector3D<int>>(naxial, nang);
    geometry = CylinderGeometry(naxial, nang, nrise);
    reservoir = Reservoir();
    rng = RandomNumberGenerator(seed);
    localShape = Vector3D<int>(score_shape[2], score_shape[3], score_shape[4]);

    // initialize shifts to the center.

    for (auto y = 0; y < naxial; ++y) {
        for (auto a = 0; a < nang; ++a) {
            auto shift = localShape / 2;
            shifts.setAt(y, a, shift);
        }
    }

    gains = std::vector<float>({getGain()});

}

/// Apply random shift by a single voxel to the molecule at position (y, a).
/// For example, (2, 2, 3) will be shifted to such as (2, 2, 2) or (1, 2, 3) with equal probability.
inline ShiftResult AnnealingModel::randomShift(ssize_t y, ssize_t a) {
    auto shift = shifts.at(y, a);
    auto score_old = *score.data(y, a, shift.z, shift.y, shift.x);
    auto new_shift = rng.randShift(shift, localShape);
    auto score_new = *score.data(y, a, new_shift.z, new_shift.y, new_shift.x);
    auto dGain = score_new - score_old;
    return ShiftResult(new_shift, dGain);
}

void AnnealingModel::optimize(ssize_t nsteps = 10000) {
    for (auto i = 0; i < nsteps; ++i) {
        tryAccept();
        reservoir.cool();
        if (terminate()) {
            break;
        }
    }
}

float AnnealingModel::getGain() {
    float gain = 0.0;
    for (auto y = 0; y < shifts.naxial; ++y) {
        for (auto a = 0; a < shifts.nang; ++a) {
            auto shift = shifts.at(y, a);
            gain += *score.data(y, a, shift.z, shift.y, shift.x);
        }
    }
    return gain;
}

py::array_t<int> AnnealingModel::getShifts() {
    auto out = py::array_t<int>{{shifts.naxial, shifts.nang, ssize_t(3)}};
    for (auto i = 0; i < shifts.naxial; ++i) {
        for (auto j = 0; j < shifts.nang; ++j) {
            auto shift = shifts.at(i, j);
            out.mutable_at(i, j, 0) = shift.x;
            out.mutable_at(i, j, 1) = shift.y;
            out.mutable_at(i, j, 2) = shift.z;
        }
    }
    return out;
}

float AnnealingModel::upperLimit() {
    float gain = 0.0;
    for (auto ax = 0; ax < shifts.naxial; ++ax) {
        for (auto an = 0; an < shifts.nang; ++an) {
            float max0 = -std::numeric_limits<float>::infinity();
            for (auto z = 0; z < localShape.z; ++z) {
                for (auto y = 0; y < localShape.y; ++y) {
                    for (auto x = 0; x < localShape.x; ++x) {
                        auto _score = *score.data(ax, an, z, y, x);
                        if (_score > max0) {
                            max0 = _score;
                        }
                    }
                }
            }
            gain += max0;
        }
    }
    return gain;
}

#endif
