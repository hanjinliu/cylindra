use core::slice::Iter;
use numpy::{ndarray::{ArcArray, Array1}, Ix4};
use pyo3::PyResult;

use crate::{
    annealing::random::RandomNumberGenerator,
    coordinates::Vector3D,
    cylindric::Index,
    value_error,
};


pub struct ShiftResult<S> {
    pub index: usize,
    pub state: S,
    pub energy_diff: f32,
}


#[derive(Clone)]
/// GraphComponents represents a concrete graph structure and the assigned
/// states of nodes and edges.
pub struct GraphComponents<Sn, Se> {
    edges: Vec<Vec<usize>>,  // edges[i]: list of edge indices connected to node i
    edge_ends: Vec<(usize, usize)>,  // edge_ends[j]: two nodes connected by edge j
    node_states: Vec<Sn>,
    edge_states: Vec<Se>,
}

type Shift = Vector3D<isize>;

impl<Sn, Se> GraphComponents<Sn, Se> {
    pub fn empty() -> Self {
        Self {
            edges: Vec::new(),
            edge_ends: Vec::new(),
            node_states: Vec::new(),
            edge_states: Vec::new(),
        }
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.node_states.len()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edge_states.len()
    }

    pub fn iter_nodes(&self) -> Iter<'_, Sn> {
        self.node_states.iter()
    }

    /// Add a new node of state node_state to the graph.
    pub fn add_node(&mut self, node_state: Sn) {
        self.node_states.push(node_state);
        self.edges.push(Vec::new());
    }

    /// Add a new edge of state edge_state to the graph, connecting nodes i and j.
    pub fn add_edge(&mut self, i: usize, j: usize, edge_state: Se) {
        let count = self.edge_count();
        self.edges[i].push(count);
        self.edges[j].push(count);
        self.edge_ends.push((i, j));
        self.edge_states.push(edge_state);
    }

    /// Clear the graph.
    pub fn clear(&mut self) {
        self.edges.clear();
        self.edge_ends.clear();
        self.node_states.clear();
        self.edge_states.clear();
    }

    /// Return the node state at index i.
    pub fn node_state(&self, i: usize) -> &Sn {
        &self.node_states[i]
    }

    /// Set a new node state to the node at index i.
    pub fn set_node_state(&mut self, i: usize, node_state: Sn) {
        self.node_states[i] = node_state;
    }

    /// Return the edge state at index i.
    pub fn edge_state(&self, i: usize) -> &Se {
        &self.edge_states[i]
    }

    /// Return all the edge IDs connected to node i.
    pub fn connected_edge_indices(&self, i: usize) -> Iter<'_, usize> {
        self.edges[i].iter()
    }

    pub fn edge_end(&self, i: usize) -> (usize, usize) {
        self.edge_ends[i]
    }
}

#[derive(Clone)]
pub struct Node1D<S: Clone> {
    pub index: usize,
    pub state: S,
}

#[derive(Clone)]
pub struct Node2D<S: Clone> {
    pub index: Index,
    pub state: S,
}

pub trait GraphTrait<N: Clone, E: Clone> {
    fn components(&self) -> &GraphComponents<N, E>;
    fn components_mut(&mut self) -> &mut GraphComponents<N, E>;
    /// Internal energy of a node.
    fn internal(&self, node: &N) -> f32;
    /// Binding energy between two nodes.
    fn binding(&self, node_0: &N, node_1: &N, typ: &E) -> f32;
    /// Return a random neighbor state of a given node.
    fn random_local_neighbor_state(&self, node: &N, rng: &mut RandomNumberGenerator) -> N;
    /// Initialize the graph state
    fn initialize(&mut self) -> &Self;
    /// Return the shape of each graph node.
    fn local_shape(&self) -> Shift;
    /// Set the energy landscape array to the graph.
    fn set_energy_landscape(&mut self, energy: ArcArray<f32, Ix4>) -> PyResult<&Self>;

    /// Energy difference by shifting a state of node at idx.
    fn energy_diff_by_shift(
        &self,
        idx: usize,
        state_old: &N,
        state_new: &N,
    ) -> f32;

    fn energy(&self) -> f32 {
        let mut energy = 0.0;
        let graph = self.components();
        for i in 0..graph.node_count() {
            energy += self.internal(&graph.node_state(i));
        }
        for i in 0..graph.edge_count() {
            let edge = graph.edge_end(i);
            let node_0 = graph.node_state(edge.0);
            let node_1 = graph.node_state(edge.1);
            energy += self.binding(&node_0, &node_1, &graph.edge_state(i));
        }
        energy
    }

    /// Apply the shift result to the graph.
    fn apply_shift(&mut self, result: &ShiftResult<N>) {
        self.components_mut().set_node_state(result.index, result.state.clone());
    }


    /// Randomly choose a node and a possible neighbor shift. This method does not actually
    /// update the graph but just calculate the resulting energy difference.
    fn try_random_shift(
        &self,
        rng: &mut RandomNumberGenerator,
    ) -> ShiftResult<N> {
        let graph = self.components();
        let idx = rng.uniform_int(graph.node_count());
        let state_old = graph.node_state(idx);
        let state_new = self.random_local_neighbor_state(&state_old, rng);
        let de = self.energy_diff_by_shift(idx, &state_old, &state_new);
        ShiftResult { index: idx, state: state_new, energy_diff: de }
    }
}

pub trait CylindricGraphTrait<S: Clone, E: Clone>: GraphTrait<Node2D<S>, E> {
    /// Calculate the longitidinal and lateral binding energies of the graph.
    fn binding_energies(&self) -> (Array1<f32>, Array1<f32>);
    /// List all the possible neighbor states of a given node.
    fn list_neighbors(&self, node: &Node2D<S>) -> Vec<S>;

    /// Calculate the total energy of the graph.
    fn check_graph(&self) -> PyResult<()> {
        if self.components().node_count() < 2 {
            return value_error!("Graph has less than 2 nodes");
        }
        Ok(())
    }

    /// Try all the available shifts and return the best shift.
    fn try_all_shifts(&self) -> ShiftResult<Node2D<S>> {
        let graph = self.components();
        let mut best_shift = ShiftResult { index: 0, state: graph.node_state(0).clone(), energy_diff: f32::INFINITY };
        for idx in 0..graph.node_count() {
            let state_old = graph.node_state(idx);
            let neighbors = self.list_neighbors(&state_old);
            for nbr in neighbors.iter() {
                let index = state_old.index.clone();
                let state_new = Node2D { index, state: nbr.clone() };
                let de = self.energy_diff_by_shift(idx, &state_old, &state_new);
                if best_shift.energy_diff > de {
                    best_shift = ShiftResult { index: idx, state: state_new, energy_diff: de };
                }
            }
        }
        best_shift
    }
}

/// The minimum but enough shape of HashMap2D for the given indices.
pub fn shape_for_indices(indices: &Vec<Index>) -> (usize, usize) {
    let mut max_y = 0;
    let mut max_a = 0;
    for idx in indices.iter() {
        if idx.y > max_y {
            max_y = idx.y;
        }
        if idx.a > max_a {
            max_a = idx.a;
        }
    }
    ((max_y + 1) as usize, (max_a + 1) as usize)
}
