use rand::SeedableRng;
use rand::prelude::Distribution;
use rand::rngs::StdRng;

use crate::coordinates::{Vector3D, list_neighbors};

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

    pub fn uniform_vec(&mut self, shape: &Vector3D<isize>) -> Vector3D<isize> {
        let z = self.uniform_int(shape.z as usize);
        let y = self.uniform_int(shape.y as usize);
        let x = self.uniform_int(shape.x as usize);
        Vector3D::new(z as isize, y as isize, x as isize)
    }

    pub fn rand_shift(&mut self, src: &Vector3D<isize>, shape: &Vector3D<isize>) -> Vector3D<isize> {
        let neighbors = list_neighbors(src, shape);
        neighbors[self.uniform_int(neighbors.len())]
    }
}
