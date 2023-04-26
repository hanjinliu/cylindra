
// class RandomNumberGenerator {
//     private:
//         std::default_random_engine eng;
//     public:
//         RandomNumberGenerator() {
//             std::random_device rd;
//             eng = std::default_random_engine(rd());
//         }
//         RandomNumberGenerator(int seed) {
//             eng = std::default_random_engine(seed);
//         }

//         bool bernoulli(double ptrue) {
//             std::bernoulli_distribution dist(ptrue);
//             return dist(eng);
//         }

//         size_t uniformInt(size_t max) {
//             return uniformInt(0, max);
//         }

//         size_t uniformInt(size_t min, size_t max) {
//             std::uniform_int_distribution<size_t> dist(min, max - 1);
//             return dist(eng);
//         }


//         Vector3D<int> randShift(const Vector3D<int> src, const Vector3D<int> shape) {
//             std::vector<Vector3D<int>> neighbors;

//             if (0 < src.z && src.z < shape.z - 1) {
//                 neighbors.push_back(Vector3D<int>(src.z - 1, src.y, src.x));
//                 neighbors.push_back(Vector3D<int>(src.z + 1, src.y, src.x));
//             } else if (src.z == 0) {
//                 neighbors.push_back(Vector3D<int>(src.z + 1, src.y, src.x));
//             } else {
//                 neighbors.push_back(Vector3D<int>(src.z - 1, src.y, src.x));
//             }

//             if (0 < src.y && src.y < shape.y - 1) {
//                 neighbors.push_back(Vector3D<int>(src.z, src.y - 1, src.x));
//                 neighbors.push_back(Vector3D<int>(src.z, src.y + 1, src.x));
//             } else if (src.y == 0) {
//                 neighbors.push_back(Vector3D<int>(src.z, src.y + 1, src.x));
//             } else {
//                 neighbors.push_back(Vector3D<int>(src.z, src.y - 1, src.x));
//             }

//             if (0 < src.x && src.x < shape.x - 1) {
//                 neighbors.push_back(Vector3D<int>(src.z, src.y, src.x - 1));
//                 neighbors.push_back(Vector3D<int>(src.z, src.y, src.x + 1));
//             } else if (src.x == 0) {
//                 neighbors.push_back(Vector3D<int>(src.z, src.y, src.x + 1));
//             } else {
//                 neighbors.push_back(Vector3D<int>(src.z, src.y, src.x - 1));
//             }

//             return neighbors[uniformInt(neighbors.size())];
//         }

//         double uniform() {
//             return uniform(0.0, 1.0);
//         }

//         double uniform(double min, double max) {
//             std::uniform_real_distribution<double> dist(min, max);
//             return dist(eng);
//         }
// };

use rand::SeedableRng;
use rand::prelude::Distribution;
use rand::rngs::StdRng;

use crate::coordinates::Vector3D;

#[derive(Clone)]
pub struct RandomNumberGenerator {
    rng: StdRng
}

impl RandomNumberGenerator {
    pub fn new(seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { rng }
    }

    pub fn bernoulli(&mut self, ptrue: f32) -> bool {
        let dist = rand::distributions::Bernoulli::new(ptrue as f64).unwrap();
        dist.sample(&mut self.rng)
    }

    pub fn uniform_int(&mut self, max: usize) -> usize {
        let dist = rand::distributions::Uniform::new(0, max);
        dist.sample(&mut self.rng)
    }

    pub fn rand_shift(&mut self, src: &Vector3D<isize>, shape: &Vector3D<isize>) -> Vector3D<isize> {
        let mut neighbors = Vec::new();

        if 0 < src.z && src.z < shape.z - 1 {
            neighbors.push(Vector3D::new(src.z - 1, src.y, src.x));
            neighbors.push(Vector3D::new(src.z + 1, src.y, src.x));
        } else if src.z == 0 {
            neighbors.push(Vector3D::new(src.z + 1, src.y, src.x));
        } else {
            neighbors.push(Vector3D::new(src.z - 1, src.y, src.x));
        }

        if 0 < src.y && src.y < shape.y - 1 {
            neighbors.push(Vector3D::new(src.z, src.y - 1, src.x));
            neighbors.push(Vector3D::new(src.z, src.y + 1, src.x));
        } else if src.y == 0 {
            neighbors.push(Vector3D::new(src.z, src.y + 1, src.x));
        } else {
            neighbors.push(Vector3D::new(src.z, src.y - 1, src.x));
        }

        if 0 < src.x && src.x < shape.x - 1 {
            neighbors.push(Vector3D::new(src.z, src.y, src.x - 1));
            neighbors.push(Vector3D::new(src.z, src.y, src.x + 1));
        } else if src.x == 0 {
            neighbors.push(Vector3D::new(src.z, src.y, src.x + 1));
        } else {
            neighbors.push(Vector3D::new(src.z, src.y, src.x - 1));
        }

        neighbors[self.uniform_int(neighbors.len())]
    }
}
