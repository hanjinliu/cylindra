use core::slice::Iter;
use std::{sync::Arc, collections::HashMap};
use numpy::{
    ndarray::{Array1, Array2, Array, s, ArcArray, ArcArray2}, Ix3, Ix4
};
use pyo3::PyResult;

use crate::{
    value_error,
    coordinates::{Vector3D, CoordinateSystem, list_neighbors},
    cylindric::Index,
    annealing::{
        potential::{TrapezoidalPotential2D, BindingPotential2D, EdgeType},
        random::RandomNumberGenerator,
    }
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
pub struct NodeState {
    index: Index,
    shift: Vector3D<isize>,
}


#[derive(Clone)]
pub struct CylindricGraph {
    components: GraphComponents<NodeState, EdgeType>,
    coords: Arc<HashMap<Index, CoordinateSystem<f32>>>,
    energy: Arc<HashMap<Index, Array<f32, Ix3>>>,
    pub binding_potential: TrapezoidalPotential2D,
    pub local_shape: Vector3D<isize>,
}

impl CylindricGraph {
    /// Create a graph with no nodes or edges.
    pub fn empty() -> Self {
        Self {
            components: GraphComponents::empty(),
            coords: Arc::new(HashMap::new()),
            energy: Arc::new(HashMap::new()),
            binding_potential: TrapezoidalPotential2D::unbounded(),
            local_shape: Vector3D::new(0, 0, 0),
        }
    }

    /// Construct a graph from a cylindric parameters.
    pub fn construct(
        &mut self,
        indices: Vec<Index>,
        na: isize,
        nrise: isize,
    ) -> PyResult<&Self> {
        self.components.clear();

        let mut index_to_id: HashMap<Index, usize> = HashMap::new();
        for i in 0..indices.len() {
            let idx = indices[i].clone();
            index_to_id.insert(idx.clone(), i);
            self.components.add_node(NodeState { index: idx, shift: Vector3D::new(0, 0, 0) });
        }

        for (idx, i) in index_to_id.iter() {
            let neighbors = idx.get_neighbors(na, nrise);
            for neighbor in neighbors.y_iter() {
                match index_to_id.get(&neighbor) {
                    Some(j) => {
                        if i < j {
                            self.components.add_edge(*i, *j, EdgeType::Longitudinal);
                        }
                    }
                    None => {}
                }
            }
            for neighbor in neighbors.a_iter() {
                match index_to_id.get(&neighbor) {
                    Some(j) => {
                        if i < j {
                            self.components.add_edge(*i, *j, EdgeType::Lateral);
                        }
                    }
                    None => {}
                }
            }
        }
        Ok(self)
    }

    pub fn set_coordinates(
        &mut self,
        origin: ArcArray2<f32>,
        zvec: ArcArray2<f32>,
        yvec: ArcArray2<f32>,
        xvec: ArcArray2<f32>,
    ) -> PyResult<&Self> {
        let n_nodes = self.components.node_count();
        if origin.shape() != [n_nodes, 3] {
            return value_error!("origin has wrong shape");
        } else if zvec.shape() != [n_nodes, 3] {
            return value_error!("zvec has wrong shape");
        } else if yvec.shape() != [n_nodes, 3] {
            return value_error!("yvec has wrong shape");
        } else if xvec.shape() != [n_nodes, 3] {
            return value_error!("xvec has wrong shape");
        }
        let mut _coords: HashMap<Index, CoordinateSystem<f32>> = HashMap::new();
        for i in 0..n_nodes {
            let node = self.components.node_state(i);
            let idx = node.index.clone();

            _coords.insert(
                idx,
                CoordinateSystem::new(
                    origin.slice(s![i, ..]).into(),
                    zvec.slice(s![i, ..]).into(),
                    yvec.slice(s![i, ..]).into(),
                    xvec.slice(s![i, ..]).into(),
                )
            );
        }
        self.coords = Arc::new(_coords);
        Ok(self)
    }

    /// Set the energy landscape array to the graph.
    pub fn set_energy_landscape(&mut self, energy: ArcArray<f32, Ix4>) -> PyResult<&Self> {
        let n_nodes = self.components.node_count();
        let shape = energy.shape();
        if shape[0] != n_nodes {
            return value_error!(
                format!("`energy` has wrong shape, Expected ({n_nodes}, ...) but got {shape:?}.")
            );
        }

        let (_nz, _ny, _nx) = (shape[1], shape[2], shape[3]);
        self.local_shape = Vector3D::new(_nz, _ny, _nx).into();
        let center: Vector3D<isize> = Vector3D::new(_nz / 2, _ny / 2, _nx / 2).into();
        let mut _energy: HashMap<Index, Array<f32, Ix3>> = HashMap::new();
        for i in 0..n_nodes {
            let node_state = self.components.node_state(i);
            let idx = &node_state.index;
            _energy.insert(idx.clone(), energy.slice(s![i, .., .., ..]).to_owned());
            self.components.set_node_state(i, NodeState { index: idx.clone(), shift: center.clone() })
        }
        self.energy = Arc::new(_energy);
        Ok(self)
    }

    /// Cool down the binding potential.
    pub fn cool(&mut self, n: usize) {
        self.binding_potential.cool(n);
    }

    /// Get the graph components.
    pub fn components(&self) -> &GraphComponents<NodeState, EdgeType> {
        &self.components
    }

    /// Calculate the internal energy of a node state.
    /// # Arguments
    /// * `node_state` - The node state of interest.
    pub fn internal(&self, node_state: &NodeState) -> f32 {
        let idx = &node_state.index;
        let vec = node_state.shift;
        self.energy[&idx][[vec.z as usize, vec.y as usize, vec.x as usize]]
    }

    /// Calculate the binding energy between two nodes.
    /// # Arguments
    /// * `node_state0` - The node state of the first node.
    /// * `node_state1` - The node state of the second node.
    /// * `typ` - The type of the edge between the two nodes.
    pub fn binding(&self, node_state0: &NodeState, node_state1: &NodeState, typ: &EdgeType) -> f32 {
        let vec1 = node_state0.shift;
        let vec2 = node_state1.shift;
        let coord1 = &self.coords[&node_state0.index];
        let coord2 = &self.coords[&node_state1.index];
        let dr = coord1.at_vec(vec1.into()) - coord2.at_vec(vec2.into());
        // ey is required for the angle constraint.
        let ey = coord2.origin - coord1.origin;
        self.binding_potential.calculate(&dr, &ey, typ)
    }

    /// Return a random neighbor state of a given node state.
    pub fn random_local_neighbor_state(&self, node_state: &NodeState, rng: &mut RandomNumberGenerator) -> NodeState {
        let idx = node_state.index.clone();
        let shift = node_state.shift;
        let shift_new = rng.rand_shift(&shift);
        NodeState { index: idx, shift: shift_new }
    }

    /// Return the current shifts of the graph.
    pub fn get_shifts(&self) -> Array2<isize> {
        let graph = self.components();
        let n_nodes = graph.node_count();
        let mut shifts = Array2::<isize>::zeros((n_nodes as usize, 3));
        for i in 0..n_nodes {
            let state = graph.node_state(i);
            let shift = state.shift;
            shifts[[i, 0]] = shift.z;
            shifts[[i, 1]] = shift.y;
            shifts[[i, 2]] = shift.x;
        }
        shifts
    }

    /// Set shifts to each node.
    pub fn set_shifts(&mut self, shifts: &Array2<isize>) -> PyResult<&Self> {
        let n_nodes = self.components.node_count();
        if shifts.shape() != [n_nodes as usize, 3] {
            return value_error!("shifts has wrong shape");
        }
        for i in 0..n_nodes {
            let mut state = self.components.node_state(i).clone();
            state.shift.z = shifts[[i, 0]];
            state.shift.y = shifts[[i, 1]];
            state.shift.x = shifts[[i, 2]];
            self.components.set_node_state(i, state);
        }
        Ok(self)
    }

    pub fn set_shifts_arc(&mut self, shifts: &ArcArray2<isize>) -> PyResult<&Self> {
        let n_nodes = self.components.node_count();
        if shifts.shape() != [n_nodes as usize, 3] {
            return value_error!("shifts has wrong shape");
        }
        for i in 0..n_nodes {
            let mut state = self.components.node_state(i).clone();
            state.shift.z = shifts[[i, 0]];
            state.shift.y = shifts[[i, 1]];
            state.shift.x = shifts[[i, 2]];
            self.components.set_node_state(i, state);
        }
        Ok(self)
    }

    fn get_distances(&self, typ: &EdgeType) -> Array1<f32> {
        if self.coords.len() == 0 {
            panic!("Coordinates not set.")
        }
        let graph = self.components();
        let mut distances = Vec::new();
        for i in 0..graph.edge_count() {
            if graph.edge_state(i) != typ {
                continue;
            }
            let edge = graph.edge_end(i);
            let pos0 = graph.node_state(edge.0);
            let pos1 = graph.node_state(edge.1);

            let coord0 = &self.coords[&pos0.index];
            let coord1 = &self.coords[&pos1.index];
            let dr = coord0.at_vec(pos0.shift.into()) - coord1.at_vec(pos1.shift.into());
            distances.push(dr.length())
        }
        Array1::from(distances)
    }

    fn get_angles(&self, typ: &EdgeType) -> Array1<f32> {
        if self.coords.len() == 0 {
            panic!("Coordinates not set.")
        }
        let graph = self.components();
        let mut angles = Array1::<f32>::zeros(graph.node_count());
        for i in 0..graph.node_count() {
            let mut neighbors = Vec::new();
            for k in graph.connected_edge_indices(i) {
                if graph.edge_state(*k) != typ {
                    continue;
                }
                let edge_end = graph.edge_end(*k);
                if edge_end.0 == i {
                    neighbors.push(edge_end.1);
                } else {
                    neighbors.push(edge_end.0);
                }
            }
            if neighbors.len() != 2 {
                angles[i] = -1.0;
            } else {
                //      (c)
                //     /   \
                //  (l)     (r)
                let pos_c = graph.node_state(i);
                let pos_l = graph.node_state(neighbors[0]);
                let pos_r = graph.node_state(neighbors[1]);

                let coord_c = &self.coords[&pos_c.index];
                let coord_l = &self.coords[&pos_l.index];
                let coord_r = &self.coords[&pos_r.index];

                let dr_l = coord_c.at_vec(pos_c.shift.into()) - coord_l.at_vec(pos_l.shift.into());
                let dr_r = coord_c.at_vec(pos_c.shift.into()) - coord_r.at_vec(pos_r.shift.into());
                angles[i] = dr_l.angle(&dr_r);
            }

        }
        angles
    }

    /// Set a box potential model to the graph.
    pub fn set_potential_model(&mut self, model: TrapezoidalPotential2D) -> &Self {
        self.binding_potential = model;
        self
    }

    /// Calculate the local energy at the given index.
    pub fn energy_at(&self, i: usize) -> f32 {
        let mut energy = 0.0;
        let graph = self.components();
        energy += self.internal(&graph.node_state(i));
        for j in graph.connected_edge_indices(i) {
            let edge = graph.edge_end(*j);
            let node_state0 = graph.node_state(edge.0);
            let node_state1 = graph.node_state(edge.1);
            energy += self.binding(&node_state0, &node_state1, &graph.edge_state(i));
        }
        energy
    }

    /// Calculate the total energy of the graph.
    pub fn energy(&self) -> f32 {
        let mut energy = 0.0;
        let graph = self.components();
        for i in 0..graph.node_count() {
            energy += self.internal(&graph.node_state(i));
        }
        for i in 0..graph.edge_count() {
            let edge = graph.edge_end(i);
            let node_state0 = graph.node_state(edge.0);
            let node_state1 = graph.node_state(edge.1);
            energy += self.binding(&node_state0, &node_state1, &graph.edge_state(i));
        }
        energy
    }

    pub fn binding_energies(&self) -> (Array1<f32>, Array1<f32>) {
        let graph = self.components();
        let mut eng_lon = Array1::zeros(graph.node_count());
        let mut eng_lat = Array1::zeros(graph.node_count());
        for idx in 0..graph.edge_count() {
            // node0 ---- edge ---- node1
            let edge = graph.edge_end(idx);
            let estate = graph.edge_state(idx);
            let node_state0 = graph.node_state(edge.0);
            let node_state1 = graph.node_state(edge.1);
            let eng = self.binding(&node_state0, &node_state1, &estate);
            match estate {
                EdgeType::Longitudinal => {
                    eng_lon[edge.0] += eng;
                    eng_lon[edge.1] += eng;
                }
                EdgeType::Lateral => {
                    eng_lat[edge.0] += eng;
                    eng_lat[edge.1] += eng;
                }
            }
        }
        (eng_lon, eng_lat)
    }

    /// Randomly choose a node and a possible neighbor shift. This method does not actually
    /// update the graph but just calculate the resulting energy difference.
    pub fn try_random_shift(&self, rng: &mut RandomNumberGenerator) -> ShiftResult<NodeState> {
        let graph = self.components();
        let idx = rng.uniform_int(graph.node_count());
        let state_old = graph.node_state(idx);
        let state_new = self.random_local_neighbor_state(&state_old, rng);
        let de = self.energy_diff_by_shift(idx, &state_old, &state_new);
        ShiftResult { index: idx, state: state_new, energy_diff: de }
    }

    /// Energy difference by shifting a state of node at idx.
    fn energy_diff_by_shift(&self, idx: usize, state_old: &NodeState, state_new: &NodeState) -> f32 {
        let graph = self.components();
        let mut e_old = self.internal(&state_old);
        let mut e_new = self.internal(&state_new);
        for edge_id in graph.connected_edge_indices(idx) {
            let edge_id = *edge_id;
            let ends = graph.edge_end(edge_id);
            let other_idx = if ends.0 == idx { ends.1 } else { ends.0 };
            let other_state = graph.node_state(other_idx);
            e_old += self.binding(&state_old, &other_state, &graph.edge_state(edge_id));
            e_new += self.binding(&state_new, &other_state, &graph.edge_state(edge_id));
        }
        e_new - e_old
    }

    /// Try all the available shifts and return the best shift.
    pub fn try_all_shifts(&self) -> ShiftResult<NodeState> {
        let graph = self.components();
        let mut best_shift = ShiftResult { index: 0, state: graph.node_state(0).clone(), energy_diff: f32::INFINITY };
        for idx in 0..graph.node_count() {
            let state_old = graph.node_state(idx);
            let neighbors = list_neighbors(&state_old.shift, &self.local_shape);
            for nbr in neighbors.iter() {
                let index = state_old.index.clone();
                let state_new = NodeState { index: index, shift: nbr.clone() };
                let de = self.energy_diff_by_shift(idx, &state_old, &state_new);
                if best_shift.energy_diff > de {
                    best_shift = ShiftResult { index: idx, state: state_new, energy_diff: de };
                }
            }
        }
        best_shift
    }

    /// Apply the shift result to the graph.
    pub fn apply_shift(&mut self, result: &ShiftResult<NodeState>) {
        self.components.set_node_state(result.index, result.state.clone());
    }

    /// Initialize the node states to the center of each local coordinates.
    pub fn initialize(&mut self) -> &Self {
        let center = Vector3D::new(self.local_shape.z / 2, self.local_shape.y / 2, self.local_shape.x / 2);
        for i in 0..self.components.node_count() {
            let node = self.components.node_state(i);
            let idx = node.index.clone();
            self.components.set_node_state(i, NodeState { index: idx, shift: center.clone() });
        }
        self
    }

    pub fn get_longitudinal_distances(&self) -> Array1<f32> {
        self.get_distances(&EdgeType::Longitudinal)
    }

    pub fn get_lateral_distances(&self) -> Array1<f32> {
        self.get_distances(&EdgeType::Lateral)
    }

    pub fn get_longitudinal_angles(&self) -> Array1<f32> {
        self.get_angles(&EdgeType::Longitudinal)
    }

    pub fn get_lateral_angles(&self) -> Array1<f32> {
        self.get_angles(&EdgeType::Lateral)
    }

    pub fn get_edge_states(&self) -> (Array2<f32>, Array2<f32>, Array1<i32>) {
        let mut out0 = Array2::<f32>::zeros((self.components.edge_count(), 3));
        let mut out1 = Array2::<f32>::zeros((self.components.edge_count(), 3));
        let mut out2 = Array1::<i32>::zeros(self.components.edge_count());
        for i in 0..self.components.edge_count() {
            let edge_type = self.components.edge_state(i);
            let ends = self.components.edge_end(i);
            let node0 = self.components.node_state(ends.0);
            let node1 = self.components.node_state(ends.1);
            let coord0 = self.coords[&node0.index].at_vec(node0.shift.into());
            let coord1 = self.coords[&node1.index].at_vec(node1.shift.into());
            out0[[i, 0]] = coord0.z;
            out0[[i, 1]] = coord0.y;
            out0[[i, 2]] = coord0.x;
            out1[[i, 0]] = coord1.z;
            out1[[i, 1]] = coord1.y;
            out1[[i, 2]] = coord1.x;
            out2[i] = match edge_type {
                EdgeType::Longitudinal => 0,
                EdgeType::Lateral => 1,
            }
        }

        (out0, out1, out2)
    }

    pub fn check_graph(&self) -> PyResult<()> {
        if self.components().node_count() < 2 {
            return value_error!("Graph has less than 2 nodes");
        }
        Ok(())
    }
}
