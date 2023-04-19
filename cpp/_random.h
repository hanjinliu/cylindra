#ifndef _RANDOM_H
#define _RANDOM_H

#include <random>
#include "_coords.h"

class RandomNumberGenerator {
    private:
        std::default_random_engine eng;
    public:
        RandomNumberGenerator() {
            std::random_device rd;
            eng = std::default_random_engine(rd());
        }
        RandomNumberGenerator(int seed) {
            eng = std::default_random_engine(seed);
        }

        bool bernoulli(double ptrue) {
            std::bernoulli_distribution dist(ptrue);
            return dist(eng);
        }

        int uniformInt(int max) {
            return uniformInt(0, max);
        }

        int uniformInt(int min, int max) {
            std::uniform_int_distribution<int> dist(min, max - 1);
            return dist(eng);
        }

        std::tuple<int, int, int> randShift(std::tuple<int, int, int> src, std::tuple<int, int, int> shape) {
            auto _src = Vector3D<int>(std::get<0>(src), std::get<1>(src), std::get<2>(src));
            auto _shape = Vector3D<int>(std::get<0>(shape), std::get<1>(shape), std::get<2>(shape));
            auto shift = randShift(_src, _shape);
            return std::make_tuple(shift.z, shift.y, shift.x);
        }

        Vector3D<int> randShift(Vector3D<int> src, Vector3D<int> shape) {
            std::vector<Vector3D<int>> neighbors;

            if (0 < src.z && src.z < shape.z - 1) {
                neighbors.push_back(Vector3D<int>(src.z - 1, src.y, src.x));
                neighbors.push_back(Vector3D<int>(src.z + 1, src.y, src.x));
            } else if (src.z == 0) {
                neighbors.push_back(Vector3D<int>(src.z + 1, src.y, src.x));
            } else {
                neighbors.push_back(Vector3D<int>(src.z - 1, src.y, src.x));
            }

            if (0 < src.y && src.y < shape.y - 1) {
                neighbors.push_back(Vector3D<int>(src.z, src.y - 1, src.x));
                neighbors.push_back(Vector3D<int>(src.z, src.y + 1, src.x));
            } else if (src.y == 0) {
                neighbors.push_back(Vector3D<int>(src.z, src.y + 1, src.x));
            } else {
                neighbors.push_back(Vector3D<int>(src.z, src.y - 1, src.x));
            }

            if (0 < src.x && src.x < shape.x - 1) {
                neighbors.push_back(Vector3D<int>(src.z, src.y, src.x - 1));
                neighbors.push_back(Vector3D<int>(src.z, src.y, src.x + 1));
            } else if (src.x == 0) {
                neighbors.push_back(Vector3D<int>(src.z, src.y, src.x + 1));
            } else {
                neighbors.push_back(Vector3D<int>(src.z, src.y, src.x - 1));
            }

            return neighbors[uniformInt(neighbors.size())];
        }

        double uniform() {
            return uniform(0.0, 1.0);
        }

        double uniform(double min, double max) {
            std::uniform_real_distribution<double> dist(min, max);
            return dist(eng);
        }
};

#endif
