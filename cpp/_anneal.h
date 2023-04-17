#ifndef _ANNEAL
#define _ANNEAL

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <random>
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
        T* at_mut(ssize_t y, ssize_t a) {
            return &coords[y * nang + a];
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
    Vector3D<double> shift;
    double dE;
    ShiftResult(Vector3D<double> _shift, double _dE) {
        shift = _shift;
        dE = _dE;
    }
};

class AbstractReservoir {
    double temperature;
    virtual void cool();
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
        double prob(double dE) {
            return dE < 0 ? 1.0 : exp(-dE/ temperature);
        } 
};

class AnnealingModel {
    private:
        Grid2D<CoordinateSystem<double>> coords;
        Grid2D<Vector3D<double>> shifts;
        CylinderGeometry geometry;
        Reservoir reservoir;
    public:
        ssize_t seed;
    
        AnnealingModel(ssize_t naxial, ssize_t nang, ssize_t nrise, ssize_t seed) {
            coords = Grid2D<CoordinateSystem<double>>(naxial, nang);
            shifts = Grid2D<Vector3D<double>>(naxial, nang);
            geometry = CylinderGeometry(naxial, nang, nrise);
            reservoir = Reservoir();
            this->seed = 0;
        };

        void setReservoir(Reservoir reservoir) {
            this->reservoir = reservoir;
        }

        Index randomPosition() {
            std::random_device rd;
            std::default_random_engine gen(rd());
            std::uniform_int_distribution<> dist(0, coords.size() - 1);
            int random_index = dist(gen);
            return Index(random_index / coords.nang, random_index % coords.nang);
        }

        ShiftResult randomShift(ssize_t y, ssize_t a);

        void tryAccept() {
            auto idx = randomPosition();
            auto result = randomShift(idx.y, idx.a);
            auto dE = result.dE;
            double prob = reservoir.prob(dE);
            double rand_val = static_cast<double>(rand()) / RAND_MAX;
            if (rand_val < prob) {
                auto ptr = shifts.at_mut(idx.y, idx.a);
                ptr = &result.shift;
            }
        }
        bool terminate();
        void anneal(ssize_t nsteps = 10000) {
            for (auto i = 0; i < nsteps; ++i) {
                tryAccept();
                reservoir.cool();
                if (terminate()) {
                    break;
                }
            }
        }

};

#endif
