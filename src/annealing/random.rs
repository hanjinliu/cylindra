use rand::SeedableRng;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use numpy::ndarray::Array3;

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
    rng: StdRng,
    seed: u64,
    neighbor_list: NeighborList,
}

impl Clone for RandomNumberGenerator {
    /// Clone the random number generator.
    fn clone(&self) -> Self {
        let rng = StdRng::seed_from_u64(self.seed);
        Self { rng, seed: self.seed, neighbor_list: self.neighbor_list.clone() }
    }
}

impl RandomNumberGenerator {
    pub fn new(seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { rng, seed, neighbor_list: NeighborList::empty() }
    }

    pub fn with_seed(&self, seed: u64) -> Self {
        let rng = StdRng::seed_from_u64(seed);
        Self { rng, seed, neighbor_list: self.neighbor_list.clone() }
    }

    pub fn shape(&self) -> &[usize] {
        self.neighbor_list.neighbors.shape()
    }

    pub fn set_shape(&mut self, shape: (usize, usize, usize)) {
        let (z, y, x) = shape;
        self.neighbor_list = NeighborList::new(Vector3D::new(z as isize, y as isize, x as isize));
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
        let dist = rand::distributions::Uniform::new(0, shape.z);
        let z = dist.sample(&mut self.rng);
        let dist = rand::distributions::Uniform::new(0, shape.y);
        let y = dist.sample(&mut self.rng);
        let dist = rand::distributions::Uniform::new(0, shape.x);
        let x = dist.sample(&mut self.rng);
        Vector3D::new(z, y, x)
    }

    pub fn rand_shift(&mut self, src: &Vector3D<isize>) -> Vector3D<isize> {
        let neighbors = self.neighbor_list.at(src);
        let idx = get_uniform(neighbors.len()).sample(&mut self.rng);
        neighbors[idx]
    }
}
