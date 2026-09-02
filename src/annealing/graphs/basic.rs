use std::sync::Arc;
use numpy::{
    ndarray::{Array1, Array2, Array, s, ArcArray, ArcArray2}, Ix3, Ix4
};
use pyo3::PyResult;
use super::traits::{
    shape_for_indices, CylindricGraphTrait, GraphComponents, GraphTrait, Node2D, ShiftResult
};

use crate::{
    value_error,
    coordinates::{Vector3D, CoordinateSystem, list_neighbors},
    cylindric::Index,
    hash::HashMap2D,
    annealing::{
        potential::{
            TrapezoidalPotential2D,
            LennardJonesLikePotential2D,
            BindingPotential2D,
            EdgeScalars,
            EdgeType,
            edge_scalars,
    },
        random::RandomNumberGenerator,
    }
};

type Shift = Vector3D<isize>;

#[derive(Clone)]
/// Geometry of an edge that never changes once the coordinates are set.
/// `ey` is the vector connecting the origins of the two local coordinate systems, and
/// is required for the angle constraint. Because the angle energy only depends on
/// `cos(dr, ey).abs()`, the sign of `ey` does not matter.
pub struct EdgeGeometry {
    pub ey: Vector3D<f32>,
    pub ey_len: f32,
}

#[derive(Clone)]
pub struct CylindricalGraph<T: BindingPotential2D> {
    components: GraphComponents<Node2D<Shift>, EdgeType>,
    coords: Arc<HashMap2D<CoordinateSystem<f32>>>,
    edge_geometry: Arc<Vec<EdgeGeometry>>,
    // Cached quantities of the *current* state of the graph. Because the binding
    // potential is cooled at every iteration, the energies themselves cannot be cached,
    // but these state-dependent scalars can, which makes re-evaluating the current
    // energy of a node free of any coordinate arithmetic.
    edge_scalars: Vec<EdgeScalars>,
    internal_energies: Vec<f32>,
    energy: Arc<HashMap2D<Array<f32, Ix3>>>,
    pub binding_potential: T,
    pub local_shape: Shift,
}

impl<T> CylindricalGraph<T> where T: BindingPotential2D {
    /// Construct a graph from a cylindric parameters.
    pub fn construct(
        &mut self,
        indices: Vec<Index>,
        npf: isize,
        nrise: isize,
    ) -> PyResult<&Self> {
        self.components.clear();
        let (ny, na) = shape_for_indices(&indices);
        let mut index_to_id: HashMap2D<usize> = HashMap2D::from_shape(ny, na);
        for i in 0..indices.len() {
            let idx = indices[i].clone();
            index_to_id.insert(idx.as_tuple_usize(), i);
            self.components.add_node(Node2D { index: idx, state: Vector3D::new(0, 0, 0) });
        }
        for (idx, i) in index_to_id.iter() {
            let neighbors = Index::new(idx.0 as isize, idx.1 as isize).get_neighbors(npf, nrise);
            for neighbor in neighbors.y_iter() {
                match index_to_id.get((neighbor.y, neighbor.a)) {
                    Some(j) => {
                        if i < j {
                            self.components.add_edge(*i, *j, EdgeType::Longitudinal);
                        }
                    }
                    None => {}
                }
            }
            for neighbor in neighbors.a_iter() {
                match index_to_id.get((neighbor.y, neighbor.a)) {
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

        let (ny, na) = self.outer_shape();
        let mut _coords: HashMap2D<CoordinateSystem<f32>> = HashMap2D::from_shape(ny, na);
        for i in 0..n_nodes {
            let node = self.components.node_state(i);
            _coords.insert(
                node.index.as_tuple_usize(),
                CoordinateSystem::new(
                    origin.slice(s![i, ..]).into(),
                    zvec.slice(s![i, ..]).into(),
                    yvec.slice(s![i, ..]).into(),
                    xvec.slice(s![i, ..]).into(),
                )
            );
        }
        self.coords = Arc::new(_coords);
        self.update_edge_geometry();
        Ok(self)
    }

    /// Cache the coordinate-dependent, state-independent part of each edge.
    fn update_edge_geometry(&mut self) {
        let n_edges = self.components.edge_count();
        let mut geometry = Vec::with_capacity(n_edges);
        for i in 0..n_edges {
            let (i0, i1) = self.components.edge_end(i);
            let idx0 = &self.components.node_state(i0).index;
            let idx1 = &self.components.node_state(i1).index;
            let ey = self.coords[(idx1.y, idx1.a)].origin - self.coords[(idx0.y, idx0.a)].origin;
            geometry.push(EdgeGeometry { ey, ey_len: ey.length() });
        }
        self.edge_geometry = Arc::new(geometry);
        self.refresh_state_cache();
    }

    /// Calculate the scalars that describe the current geometry of the edge `edge_id`.
    fn calc_edge_scalars(&self, edge_id: usize) -> EdgeScalars {
        let (i0, i1) = self.components.edge_end(edge_id);
        let node0 = self.components.node_state(i0);
        let node1 = self.components.node_state(i1);
        let coord0 = &self.coords[(node0.index.y, node0.index.a)];
        let coord1 = &self.coords[(node1.index.y, node1.index.a)];
        let dr = coord0.at_vec_fast(node0.state.into()) - coord1.at_vec_fast(node1.state.into());
        let geom = &self.edge_geometry[edge_id];
        edge_scalars(
            &dr, dr.length(), &geom.ey, geom.ey_len, self.components.edge_state(edge_id),
        )
    }

    /// Binding energy of the edge `edge_id` when one of its ends takes `node_state` and
    /// the other one takes `other_state`.
    fn binding_at(
        &self,
        node_state: &Node2D<Shift>,
        other_state: &Node2D<Shift>,
        edge_id: usize,
    ) -> f32 {
        let coord_self = &self.coords[(node_state.index.y, node_state.index.a)];
        let coord_other = &self.coords[(other_state.index.y, other_state.index.a)];
        let dr = coord_self.at_vec_fast(node_state.state.into())
            - coord_other.at_vec_fast(other_state.state.into());
        // `ey` only depends on the origins of the local coordinate systems, so that it
        // is cached for each edge.
        let geom = &self.edge_geometry[edge_id];
        self.binding_potential.calculate_with_lengths(
            &dr, dr.length(), &geom.ey, geom.ey_len, self.components.edge_state(edge_id),
        )
    }

    /// Recalculate all the caches that depend on the current node states.
    /// This method must be called whenever the node states, the coordinates or the
    /// energy landscape are updated by anything other than `apply_shift`.
    fn refresh_state_cache(&mut self) {
        let n_nodes = self.components.node_count();
        let n_edges = self.components.edge_count();
        // The caches require the energy landscape, because the local coordinate caches
        // used by `at_vec_fast` are only built in `set_energy_landscape`.
        if self.energy.len() != n_nodes || n_nodes == 0 {
            self.internal_energies = vec![0.0; n_nodes];
            self.edge_scalars = vec![EdgeScalars::default(); n_edges];
            return;
        }
        self.internal_energies = (0..n_nodes)
            .map(|i| self.internal(self.components.node_state(i)))
            .collect();
        self.edge_scalars = (0..n_edges).map(|i| self.calc_edge_scalars(i)).collect();
    }

    /// Cool down the binding potential.
    pub fn cool(&mut self, n: usize) {
        self.binding_potential.cool(n);
    }

    /// If the graph is a cylinder with (ny, na) nodes, return (ny, na).
    fn outer_shape(&self) -> (usize, usize) {
        let mut ny = 0;
        let mut na = 0;
        for node in self.components().iter_nodes() {
            let idx = &node.index;
            if idx.y > ny {
                ny = idx.y;
            }
            if idx.a > na {
                na = idx.a;
            }
        }
        (ny as usize + 1, na as usize + 1)
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

            let coord0 = &self.coords[(pos0.index.y, pos0.index.a)];
            let coord1 = &self.coords[(pos1.index.y, pos1.index.a)];
            let dr = coord0.at_vec(pos0.state.into()) - coord1.at_vec(pos1.state.into());
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

                let coord_c = &self.coords[(pos_c.index.y, pos_c.index.a)];
                let coord_l = &self.coords[(pos_l.index.y, pos_l.index.a)];
                let coord_r = &self.coords[(pos_r.index.y, pos_r.index.a)];

                let dr_l = coord_c.at_vec(pos_c.state.into()) - coord_l.at_vec_fast(pos_l.state.into());
                let dr_r = coord_c.at_vec(pos_c.state.into()) - coord_r.at_vec_fast(pos_r.state.into());
                angles[i] = dr_l.angle(&dr_r);
            }

        }
        angles
    }

    /// Set a box potential model to the graph.
    pub fn set_potential_model(&mut self, model: T) -> &Self {
        self.binding_potential = model;
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
            let coord0 = self.coords[(node0.index.y, node0.index.a)].at_vec(node0.state.into());
            let coord1 = self.coords[(node1.index.y, node1.index.a)].at_vec(node1.state.into());
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

    /// Return the current shifts of the graph.
    pub fn get_shifts(&self) -> Array2<isize> {
        let graph = self.components();
        let n_nodes = graph.node_count();
        let mut shifts = Array2::<isize>::zeros((n_nodes as usize, 3));
        for i in 0..n_nodes {
            let node = graph.node_state(i);
            let shift = node.state;
            shifts[[i, 0]] = shift.z;
            shifts[[i, 1]] = shift.y;
            shifts[[i, 2]] = shift.x;
        }
        shifts
    }

    /// Set shifts to each node.
    pub fn set_shifts(&mut self, shifts: &Array2<isize>) -> PyResult<&Self> {
        let n_nodes = self.components().node_count();
        if shifts.shape() != [n_nodes as usize, 3] {
            return value_error!("shifts has wrong shape");
        }
        for i in 0..n_nodes {
            let node = Node2D {
                index: self.components().node_state(i).index.clone(),
                state: Vector3D::new(shifts[[i, 0]], shifts[[i, 1]], shifts[[i, 2]]),
            };
            self.components_mut().set_node_state(i, node);
        }
        self.refresh_state_cache();
        Ok(self)
    }

    pub fn set_shifts_arc(&mut self, shifts: &ArcArray2<isize>) -> PyResult<&Self> {
        let n_nodes = self.components().node_count();
        if shifts.shape() != [n_nodes as usize, 3] {
            return value_error!("shifts has wrong shape");
        }
        for i in 0..n_nodes {
            let node = Node2D {
                index: self.components().node_state(i).index.clone(),
                state: Vector3D::new(shifts[[i, 0]], shifts[[i, 1]], shifts[[i, 2]]),
            };
            self.components_mut().set_node_state(i, node);
        }
        self.refresh_state_cache();
        Ok(self)
    }

}

impl CylindricalGraph<TrapezoidalPotential2D> {
    /// Create a graph with no nodes or edges.
    pub fn empty() -> Self {
        Self {
            components: GraphComponents::empty(),
            coords: Arc::new(HashMap2D::new()),
            edge_geometry: Arc::new(Vec::new()),
            edge_scalars: Vec::new(),
            internal_energies: Vec::new(),
            energy: Arc::new(HashMap2D::new()),
            binding_potential: TrapezoidalPotential2D::unbounded(),
            local_shape: Vector3D::new(0, 0, 0),
        }
    }
}


impl CylindricalGraph<LennardJonesLikePotential2D> {
    /// Create a graph with no nodes or edges.
    pub fn empty() -> Self {
        Self {
            components: GraphComponents::empty(),
            coords: Arc::new(HashMap2D::new()),
            edge_geometry: Arc::new(Vec::new()),
            edge_scalars: Vec::new(),
            internal_energies: Vec::new(),
            energy: Arc::new(HashMap2D::new()),
            binding_potential: LennardJonesLikePotential2D::unbounded(),
            local_shape: Vector3D::new(0, 0, 0),
        }
    }
}

impl<T> GraphTrait<Node2D<Shift>, EdgeType> for CylindricalGraph<T> where T: BindingPotential2D {
    /// Get the graph components.
    fn components(&self) -> &GraphComponents<Node2D<Shift>, EdgeType> {
        &self.components
    }

    fn components_mut(&mut self) -> &mut GraphComponents<Node2D<Shift>, EdgeType> {
        &mut self.components
    }

    /// Calculate the internal energy of a node state.
    /// # Arguments
    /// * `node_state` - The node state of interest.
    fn internal(&self, node_state: &Node2D<Shift>) -> f32 {
        let idx = &node_state.index;
        let vec = node_state.state;
        self.energy[(idx.y, idx.a)][[vec.z as usize, vec.y as usize, vec.x as usize]]
    }

    /// Calculate the binding energy between two nodes.
    /// # Arguments
    /// * `node_state0` - The node state of the first node.
    /// * `node_state1` - The node state of the second node.
    /// * `typ` - The type of the edge between the two nodes.
    fn binding(
        &self,
        node_state0: &Node2D<Shift>,
        node_state1: &Node2D<Shift>,
        typ: &EdgeType,
    ) -> f32 {
        let vec1 = node_state0.state;
        let vec2 = node_state1.state;
        let coord1 = &self.coords[(node_state0.index.y, node_state0.index.a)];
        let coord2 = &self.coords[(node_state1.index.y, node_state1.index.a)];
        let dr = coord1.at_vec_fast(vec1.into()) - coord2.at_vec_fast(vec2.into());
        // ey is required for the angle constraint.
        let ey = coord2.origin - coord1.origin;
        self.binding_potential.calculate(&dr, &ey, typ)
    }

    /// Return a random neighbor state of a given node state.
    fn random_local_neighbor_state(
        &self,
        node_state: &Node2D<Shift>,
        rng: &mut RandomNumberGenerator,
    ) -> Node2D<Shift> {
        let idx = node_state.index.clone();
        let shift = node_state.state;
        let shift_new = rng.rand_shift(&shift);
        Node2D { index: idx, state: shift_new }
    }

    /// Energy difference by shifting a state of node at idx.
    /// `state_old` must be the current state of the node at `idx`, which is guaranteed
    /// by every caller. Its energy is therefore taken from the caches instead of being
    /// calculated again from the coordinates.
    fn energy_diff_by_shift(
        &self,
        idx: usize,
        state_old: &Node2D<Shift>,
        state_new: &Node2D<Shift>,
    ) -> f32 {
        debug_assert_eq!(self.internal_energies[idx], self.internal(&state_old));
        let graph = self.components();
        let mut e_old = self.internal_energies[idx];
        let mut e_new = self.internal(&state_new);
        for edge_id in graph.connected_edge_indices(idx) {
            let edge_id = *edge_id;
            let ends = graph.edge_end(edge_id);
            let other_idx = if ends.0 == idx { ends.1 } else { ends.0 };
            let other_state = graph.node_state(other_idx);
            let typ = graph.edge_state(edge_id);
            e_old += self.binding_potential.energy_of(&self.edge_scalars[edge_id], typ);
            e_new += self.binding_at(&state_new, &other_state, edge_id);
        }
        e_new - e_old
    }

    /// Update the node state and all the caches that depend on it.
    fn apply_shift(&mut self, result: ShiftResult<Node2D<Shift>>) {
        let idx = result.index;
        self.components.set_node_state(idx, result.state);
        let eng = self.internal(self.components.node_state(idx));
        self.internal_energies[idx] = eng;
        for k in 0..self.components.connected_edge_count(idx) {
            let edge_id = self.components.connected_edge_id(idx, k);
            let scalars = self.calc_edge_scalars(edge_id);
            self.edge_scalars[edge_id] = scalars;
        }
    }

    /// Initialize the node states to the center of each local coordinates.
    fn initialize(&mut self) -> &Self {
        let center = Vector3D::new(self.local_shape.z / 2, self.local_shape.y / 2, self.local_shape.x / 2);
        for i in 0..self.components.node_count() {
            let node = self.components.node_state(i);
            let idx = node.index.clone();
            self.components.set_node_state(i, Node2D { index: idx, state: center.clone() });
        }
        self.refresh_state_cache();
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

        // Initialize all the cache
        let ny = self.outer_shape().0;
        let na = self.outer_shape().1;
        let mut new_coords = HashMap2D::from_shape(ny, na);
        for (index, coord) in self.coords.iter() {
            new_coords.insert(index, coord.with_cache(_nz, _ny, _nx));
        }
        self.coords = Arc::new(new_coords);

        self.local_shape = Vector3D::new(_nz, _ny, _nx).into();
        let center: Shift = Vector3D::new(_nz / 2, _ny / 2, _nx / 2).into();
        let (ny_out, na_out) = self.outer_shape();
        let mut _energy: HashMap2D<Array<f32, Ix3>> = HashMap2D::from_shape(ny_out, na_out);
        for i in 0..n_nodes {
            let node = self.components.node_state(i);
            let idx = &node.index;
            _energy.insert(idx.as_tuple_usize(), energy.slice(s![i, .., .., ..]).to_owned());
            self.components.set_node_state(i, Node2D { index: idx.clone(), state: center.clone() })
        }
        self.energy = Arc::new(_energy);
        self.refresh_state_cache();
        Ok(self)
    }
}

impl<T> CylindricGraphTrait<Shift, EdgeType> for CylindricalGraph<T> where T: BindingPotential2D {
    fn binding_energies(&self) -> (Array1<f32>, Array1<f32>) {
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
    fn list_neighbors(&self, node_state: &Node2D<Shift>) -> Vec<Shift> {
        list_neighbors(&node_state.state, &self.local_shape)
    }
}
