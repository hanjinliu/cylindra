use std::sync::Arc;
use numpy::{
    ndarray::{Array1, Array2, Array, s, ArcArray, ArcArray2}, Ix3, Ix4
};
use pyo3::PyResult;
use super::traits::{
    GraphComponents, GraphTrait, Node1D, ShiftResult
};

use crate::{
    value_error,
    coordinates::{Vector3D, CoordinateSystem, list_neighbors},
    hash::HashMap1D,
    annealing::{
        potential::{StiffFilamentPotential, BindingPotential, EdgeType},
        random::RandomNumberGenerator,
    }
};

type Shift = Vector3D<isize>;
#[derive(Clone)]
pub struct FilamentousGraph {
    components: GraphComponents<Node1D<Shift>, EdgeType>,
    coords: Arc<HashMap1D<CoordinateSystem<f32>>>,
    energy: Arc<HashMap1D<Array<f32, Ix3>>>,
    pub binding_potential: StiffFilamentPotential,
    pub local_shape: Vector3D<isize>,
}

impl FilamentousGraph {
    /// Create a graph with no nodes or edges.
    pub fn empty() -> Self {
        Self {
            components: GraphComponents::empty(),
            coords: Arc::new(HashMap1D::new()),
            energy: Arc::new(HashMap1D::new()),
            binding_potential: StiffFilamentPotential::unbounded(),
            local_shape: Vector3D::new(0, 0, 0),
        }
    }


    /// Construct a graph from a cylindric parameters.
    pub fn construct(&mut self, num: usize) -> PyResult<&Self> {
        self.components.clear();
        for i in 0..num {
            self.components.add_node(Node1D { index: i, state: Vector3D::new(0, 0, 0) });
        }
        for i in 0..num - 1 {
            self.components.add_edge(i, i + 1, EdgeType::Longitudinal);
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

        let mut _coords: HashMap1D<CoordinateSystem<f32>> = HashMap1D::from_shape(n_nodes);
        for i in 0..n_nodes {
            let node = self.components.node_state(i);
            _coords.insert(
                node.index,
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
        let mut _energy: HashMap1D<Array<f32, Ix3>> = HashMap1D::from_shape(n_nodes);
        for i in 0..n_nodes {
            _energy.insert(i, energy.slice(s![i, .., .., ..]).to_owned());
            self.components.set_node_state(i, Node1D { index: i, state: center.clone() })
        }
        self.energy = Arc::new(_energy);
        Ok(self)
    }

    /// Cool down the binding potential.
    pub fn cool(&mut self, n: usize) {
        self.binding_potential.cool(n);
    }

    /// Return the current shifts of the graph.
    pub fn get_shifts(&self) -> Array2<isize> {
        let graph = self.components();
        let n_nodes = graph.node_count();
        let mut shifts = Array2::<isize>::zeros((n_nodes as usize, 3));
        for i in 0..n_nodes {
            let state = graph.node_state(i);
            let shift = state.state;
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
            state.state.z = shifts[[i, 0]];
            state.state.y = shifts[[i, 1]];
            state.state.x = shifts[[i, 2]];
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
            state.state.z = shifts[[i, 0]];
            state.state.y = shifts[[i, 1]];
            state.state.x = shifts[[i, 2]];
            self.components.set_node_state(i, state);
        }
        Ok(self)
    }

    pub fn get_distances(&self) -> Array1<f32> {
        if self.coords.len() == 0 {
            panic!("Coordinates not set.")
        }
        let graph = self.components();
        let mut distances = Vec::new();
        for i in 0..graph.edge_count() {
            let edge = graph.edge_end(i);
            let pos0 = graph.node_state(edge.0);
            let pos1 = graph.node_state(edge.1);

            let coord0 = &self.coords[pos0.index as isize];
            let coord1 = &self.coords[pos1.index as isize];
            let dr = coord0.at_vec(pos0.state.into()) - coord1.at_vec(pos1.state.into());
            distances.push(dr.length())
        }
        Array1::from(distances)
    }

    pub fn get_angles(&self) -> Array1<f32> {
        if self.coords.len() == 0 {
            panic!("Coordinates not set.")
        }
        let graph = self.components();
        let mut angles = Array1::<f32>::zeros(graph.node_count());
        for i in 0..graph.node_count() {
            let mut neighbors = Vec::new();
            for k in graph.connected_edge_indices(i) {
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

                let coord_c = &self.coords[pos_c.index as isize];
                let coord_l = &self.coords[pos_l.index as isize];
                let coord_r = &self.coords[pos_r.index as isize];

                let dr_l = coord_c.at_vec(pos_c.state.into()) - coord_l.at_vec(pos_l.state.into());
                let dr_r = coord_c.at_vec(pos_c.state.into()) - coord_r.at_vec(pos_r.state.into());
                angles[i] = dr_l.angle(&dr_r);
            }

        }
        angles
    }

    /// Set a box potential model to the graph.
    pub fn set_potential_model(&mut self, model: StiffFilamentPotential) -> &Self {
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

    pub fn binding_energies(&self) -> Array1<f32> {
        let graph = self.components();
        let mut engs = Array1::zeros(graph.node_count());
        for idx in 0..graph.edge_count() {
            // node0 ---- edge ---- node1
            let edge = graph.edge_end(idx);
            let estate = graph.edge_state(idx);
            let node_state0 = graph.node_state(edge.0);
            let node_state1 = graph.node_state(edge.1);
            let eng = self.binding(&node_state0, &node_state1, &estate);
            engs[edge.0] += eng;
            engs[edge.1] += eng;
        }
        engs
    }

    /// Try all the available shifts and return the best shift.
    pub fn try_all_shifts(&self) -> ShiftResult<Node1D<Shift>> {
        let graph = self.components();
        let mut best_shift = ShiftResult { index: 0, state: graph.node_state(0).clone(), energy_diff: f32::INFINITY };
        for idx in 0..graph.node_count() {
            let state_old = graph.node_state(idx);
            let neighbors = list_neighbors(&state_old.state, &self.local_shape);
            for nbr in neighbors.iter() {
                let index = state_old.index.clone();
                let state_new = Node1D { index, state: nbr.clone() };
                let de = self.energy_diff_by_shift(idx, &state_old, &state_new);
                if best_shift.energy_diff > de {
                    best_shift = ShiftResult { index: idx, state: state_new, energy_diff: de };
                }
            }
        }
        best_shift
    }

    /// Calculate the deforming energy.
    fn deforming(
        &self,
        node_state_prev: &Node1D<Shift>,
        node_state: &Node1D<Shift>,
        node_state_next: &Node1D<Shift>,
    ) -> f32 {
        let vec = node_state.state;
        let vec1 = node_state_prev.state;
        let vec2 = node_state_next.state;
        let coord = &self.coords[node_state.index as isize];
        let coord1 = &self.coords[node_state_prev.index as isize];
        let coord2 = &self.coords[node_state_next.index as isize];
        let dr1 = coord.at_vec(vec.into()) - coord1.at_vec(vec1.into());
        let dr2 = coord.at_vec(vec.into()) - coord2.at_vec(vec2.into());
        self.binding_potential.calculate_deform(&dr1, &dr2)
    }

    pub fn energy(&self) -> f32 {
        let graph = self.components();
        let mut energy = 0.0;
        for i in 0..graph.node_count() {
            energy += self.internal(&graph.node_state(i));
        }
        for i in 0..graph.edge_count() {
            let edge = graph.edge_end(i);
            let node_0 = graph.node_state(edge.0);
            let node_1 = graph.node_state(edge.1);
            energy += self.binding(&node_0, &node_1, &graph.edge_state(i));
        }
        for i in 1..graph.node_count() - 1 {
            let prev_state = graph.node_state(i - 1);
            let next_state = graph.node_state(i + 1);
            energy += self.deforming(&prev_state, &graph.node_state(i), &next_state);
        }
        energy
    }

    pub fn check_graph(&self) -> PyResult<()> {
        if self.components().node_count() < 2 {
            return value_error!("Graph has less than 2 nodes");
        }
        Ok(())
    }
}

impl GraphTrait<Node1D<Shift>, EdgeType> for FilamentousGraph {
    /// Get the graph components.
    fn components(&self) -> &GraphComponents<Node1D<Shift>, EdgeType> {
        &self.components
    }

    fn components_mut(&mut self) -> &mut GraphComponents<Node1D<Shift>, EdgeType> {
        &mut self.components
    }

    /// Calculate the internal energy of a node state.
    /// # Arguments
    /// * `node_state` - The node state of interest.
    fn internal(&self, node_state: &Node1D<Shift>) -> f32 {
        let idx = node_state.index;
        let vec = node_state.state;
        self.energy[idx as isize][[vec.z as usize, vec.y as usize, vec.x as usize]]
    }

    /// Calculate the binding energy between two nodes.
    /// # Arguments
    /// * `node_state0` - The node state of the first node.
    /// * `node_state1` - The node state of the second node.
    fn binding(
        &self,
        node_state0: &Node1D<Shift>,
        node_state1: &Node1D<Shift>,
        _: &EdgeType,
    ) -> f32 {
        let vec1 = node_state0.state;
        let vec2 = node_state1.state;
        let coord1 = &self.coords[node_state0.index as isize];
        let coord2 = &self.coords[node_state1.index as isize];
        let dr = coord1.at_vec(vec1.into()) - coord2.at_vec(vec2.into());
        self.binding_potential.calculate_bind(&dr)
    }

    fn energy_diff_by_shift(
        &self,
        idx: usize,
        state_old: &Node1D<Shift>,
        state_new: &Node1D<Shift>,
    ) -> f32 {
        let graph = self.components();
        let mut e_old = self.internal(&state_old);
        let mut e_new = self.internal(&state_new);
        for edge_id in graph.connected_edge_indices(idx) {
            let edge_id = *edge_id;
            let ends = graph.edge_end(edge_id);
            let other_idx = if ends.0 == idx { ends.1 } else { ends.0 };
            let other_state = graph.node_state(other_idx);
            e_old += self.binding(&state_old, &other_state, graph.edge_state(edge_id));
            e_new += self.binding(&state_new, &other_state, graph.edge_state(edge_id));
        }
        if 0 < idx && idx < graph.node_count() - 1 {
            let state_prev = graph.node_state(idx - 1);
            let state_next = graph.node_state(idx + 1);
            e_old += self.deforming(&state_prev, &state_old, &state_next);
            e_new += self.deforming(&state_prev, &state_new, &state_next);
            if 1 < idx {
                let state_prevprev = graph.node_state(idx - 2);
                e_old += self.deforming(&state_prevprev, &state_prev, &state_old);
                e_new += self.deforming(&state_prevprev, &state_prev, &state_new);
            }
            if idx < graph.node_count() - 2 {
                let state_nextnext = graph.node_state(idx + 2);
                e_old += self.deforming(&state_next, &state_old, &state_nextnext);
                e_new += self.deforming(&state_next, &state_new, &state_nextnext);
            }
        }
        e_new - e_old
    }

    /// Return a random neighbor state of a given node state.
    fn random_local_neighbor_state(
        &self,
        node_state: &Node1D<Shift>,
        rng: &mut RandomNumberGenerator,
    ) -> Node1D<Shift> {
        let shift = node_state.state;
        let shift_new = rng.rand_shift(&shift);
        Node1D { index: node_state.index, state: shift_new }
    }

    /// Initialize the node states to the center of each local coordinates.
    fn initialize(&mut self) -> &Self {
        let center = Vector3D::new(self.local_shape.z / 2, self.local_shape.y / 2, self.local_shape.x / 2);
        for i in 0..self.components.node_count() {
            let node = self.components.node_state(i);
            let idx = node.index.clone();
            self.components.set_node_state(i, Node1D { index: idx, state: center.clone() });
        }
        self
    }


    fn local_shape(&self) -> Vector3D<isize> {
        self.local_shape
    }

    /// Set the energy landscape array to the graph.
    fn set_energy_landscape(&mut self, energy: ArcArray<f32, Ix4>) -> PyResult<&Self> {
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
        let mut _energy: HashMap1D<Array<f32, Ix3>> = HashMap1D::from_shape(n_nodes);
        for i in 0..n_nodes {
            _energy.insert(i, energy.slice(s![i, .., .., ..]).to_owned());
            self.components.set_node_state(i, Node1D { index: i, state: center.clone() })
        }
        self.energy = Arc::new(_energy);
        Ok(self)
    }

}
