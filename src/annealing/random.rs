use rand::SeedableRng;
use rand::prelude::Distribution;
use rand::rngs::StdRng;

use crate::coordinates::Vector3D;

/// A custom random number generator similar to np.random.
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
        let dist = match rand::distributions::Bernoulli::new(ptrue as f64){
            Ok(dist) => dist,
            Err(_) => panic!("Bernoulli distribution failed {}", ptrue),
        };
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
