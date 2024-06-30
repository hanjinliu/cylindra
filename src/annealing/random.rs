use rand::SeedableRng;
use rand::prelude::Distribution;
use rand::Rng;
use numpy::ndarray::Array3;
use mt19937::MT19937;

use crate::coordinates::{Vector3D, list_neighbors};

// get a uniform distribution over the range [0, n)
fn get_uniform(n: usize) -> rand::distributions::Uniform<usize> {
    rand::distributions::Uniform::new(0, n)
}

#[derive(Clone)]
struct NeighborList {
    neighbors: Array3<Vec<Vector3D<isize>>>,
}

impl NeighborList {
    fn new(shape: Vector3D<isize>) -> Self {
        let sh = (shape.z as usize, shape.y as usize, shape.x as usize);
        let mut arr = Array3::from_elem(sh, Vec::new());
        for z in 0..shape.z {
            for y in 0..shape.y {
                for x in 0..shape.x {
                    let src = Vector3D::new(z, y, x);
                    let neighbors = list_neighbors(&src, &shape);
                    arr[[z as usize, y as usize, x as usize]] = neighbors;
                }
            }
        }
        Self { neighbors: arr }
    }

    fn empty() -> Self {
        Self { neighbors: Array3::from_elem((0, 0, 0), Vec::new()) }
    }

    fn at(&self, src: &Vector3D<isize>) -> &Vec<Vector3D<isize>> {
        &self.neighbors[[src.z as usize, src.y as usize, src.x as usize]]
    }
}

/// A custom random number generator with a neighbor cache.
pub struct RandomNumberGenerator {
    rng: MT19937,
    seed: u64,
    neighbor_list: NeighborList,
}

impl Clone for RandomNumberGenerator {
    /// Clone the random number generator.
    fn clone(&self) -> Self {
        let rng = MT19937::seed_from_u64(self.seed);
        Self { rng, seed: self.seed, neighbor_list: self.neighbor_list.clone() }
    }
}

const BUF: usize = 2;
const SCALE: f32 = (1 << (24 - BUF)) as f32;

impl RandomNumberGenerator {
    pub fn new(seed: u64) -> Self {
        let rng = MT19937::seed_from_u64(seed);
        Self { rng, seed, neighbor_list: NeighborList::empty() }
    }

    /// Create a new random number generator with a different seed.
    /// As the shape does not change, the neighbor list can be cloned.
    pub fn with_seed(&self, seed: u64) -> Self {
        let rng = MT19937::seed_from_u64(seed);
        Self { rng, seed, neighbor_list: self.neighbor_list.clone() }
    }

    /// Shape of the neighbor list.
    pub fn shape(&self) -> &[usize] {
        self.neighbor_list.neighbors.shape()
    }

    /// Set the shape of the neighbor list.
    pub fn set_shape(&mut self, shape: (usize, usize, usize)) {
        let (z, y, x) = shape;
        self.neighbor_list = NeighborList::new(Vector3D::new(z as isize, y as isize, x as isize));
    }

    /// Sample a random number from a Bernoulli distribution.
    /// This is a naive implementation instead of using a Bernoulli distribution from
    /// the rand crate. This implementation directly use f32, and is safer when values
    /// are sampled many times (Bernoulli sampling using rand crate returns different
    /// results even with the same seed!).
    pub fn bernoulli(&mut self, ptrue: f32) -> bool {
        let p_int = (ptrue * SCALE).floor() as u32;
        let v: u32 = self.rng.gen();
        (v >> (BUF + 8)) < p_int
    }

    /// Sample a random positive integer from a uniform distribution.
    pub fn uniform_int(&mut self, max: usize) -> usize {
        let dist = rand::distributions::Uniform::new(0, max);
        dist.sample(&mut self.rng)
    }

    /// Sample a random integer vector from a uniform distribution.
    pub fn uniform_vec(&mut self, shape: &Vector3D<isize>) -> Vector3D<isize> {
        let dist = rand::distributions::Uniform::new(0, shape.z);
        let z = dist.sample(&mut self.rng);
        let dist = rand::distributions::Uniform::new(0, shape.y);
        let y = dist.sample(&mut self.rng);
        let dist = rand::distributions::Uniform::new(0, shape.x);
        let x = dist.sample(&mut self.rng);
        Vector3D::new(z, y, x)
    }

    /// Sample a random shift from the given source position.
    pub fn rand_shift(&mut self, src: &Vector3D<isize>) -> Vector3D<isize> {
        let neighbors = self.neighbor_list.at(src);
        let idx = get_uniform(neighbors.len()).sample(&mut self.rng);
        neighbors[idx]
    }
}
