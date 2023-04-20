#ifndef _BINDING_POTENTIAL_H
#define _BINDING_POTENTIAL_H

#pragma once

#include <numeric>
#include <cmath>

template <typename Se, typename T>
class AbstractBindingPotential {
    public:
        virtual T operator()(double dist2, const Se &type) { return 0.0; }
        virtual bool isConcrete() { return false; }
};

/// Enum of edge connection types used for cylindric grids.
enum class EdgeType {
    Longitudinal,
    Lateral,
};

class BindingPotential2D : AbstractBindingPotential<EdgeType, float> {
    public:
        virtual bool isConcrete() { return false; }
        virtual float longitudinal(double dist2) { return 0.0; };
        virtual float lateral(double dist2) { return 0.0; };
        float operator()(double dist2, const EdgeType &type) override {
            if (type == EdgeType::Longitudinal) {
                return longitudinal(dist2);
            } else {
                return lateral(dist2);
            }
        }
};

class BoxPotential2D : public BindingPotential2D {
    private:
        double lon_dist_min2;
        double lon_dist_max2;
        double lat_dist_min2;
        double lat_dist_max2;

    public:
        bool isConcrete() override { return true; }
        BoxPotential2D() {
            this->lon_dist_min2 = 0;
            this->lon_dist_max2 = std::numeric_limits<double>::infinity();
            this->lat_dist_min2 = 0;
            this->lat_dist_max2 = std::numeric_limits<double>::infinity();
        }
        BoxPotential2D(double lon_dist_min, double lon_dist_max, double lat_dist_min, double lat_dist_max) {
            this->lon_dist_min2 = lon_dist_min * lon_dist_min;
            this->lon_dist_max2 = lon_dist_max * lon_dist_max;
            this->lat_dist_min2 = lat_dist_min * lat_dist_min;
            this->lat_dist_max2 = lat_dist_max * lat_dist_max;
        }

        float longitudinal(double dist2) override {
            if (dist2 < lon_dist_min2 || lon_dist_max2 < dist2) {
                return std::numeric_limits<float>::infinity();
            } else {
                return 0;
            }
        }

        float lateral(double dist2) override {
            if (dist2 < lat_dist_min2 || lat_dist_max2 < dist2) {
                return std::numeric_limits<float>::infinity();
            } else {
                return 0;
            }
        }
};

class HarmonicPotential2D : BindingPotential2D {
    private:
        double halfk0;
        double halfk1;
        double r0;
        double r1;

    public:
        bool isConcrete() override { return true; }
        HarmonicPotential2D(double k0, double k1, double r0, double r1) {
            this->halfk0 = k0 / 2;
            this->halfk1 = k1 / 2;
            this->r0 = r0;
            this->r1 = r1;
        }

        float longitudinal(double dist2) override {
            auto x = std::sqrt(dist2) - r0;
            return static_cast<float>(halfk0 * x * x);
        }

        float lateral(double dist2) override {
            auto x = std::sqrt(dist2) - r1;
            return static_cast<float>(halfk1 * x * x);
        }
};

#endif
