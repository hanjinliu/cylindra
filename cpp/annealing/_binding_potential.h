#ifndef _BINDING_POTENTIAL_H
#define _BINDING_POTENTIAL_H

#pragma once

#include <numeric>
#include <cmath>

class AbstractBindingPotential {
    virtual double operator()(double dist2, int type) { return 0.0; }
};

class BindingPotential2D : AbstractBindingPotential {
    public:
        virtual double longitudinal(double dist2) { return 0.0; };
        virtual double lateral(double dist2) { return 0.0; };
        double operator()(double dist2, int type) override {
            if (type == 0) {
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
        double longitudinal(double dist2) override {
            if (dist2 < lon_dist_min2 || lon_dist_max2 < dist2) {
                return std::numeric_limits<double>::infinity();
            } else {
                return 0.0;
            }
        }

        double lateral(double dist2) override {
            if (dist2 < lat_dist_min2 || lat_dist_max2 < dist2) {
                return std::numeric_limits<double>::infinity();
            } else {
                return 0.0;
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
        HarmonicPotential2D(double k0, double k1, double r0, double r1) {
            this->halfk0 = k0 / 2;
            this->halfk1 = k1 / 2;
            this->r0 = r0;
            this->r1 = r1;
        }

        double longitudinal(double dist2) override {
            auto x = std::sqrt(dist2) - r0;
            return halfk0 * x * x;
        }

        double lateral(double dist2) override {
            auto x = std::sqrt(dist2) - r1;
            return halfk1 * x * x;
        }
};

#endif
