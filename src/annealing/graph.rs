use std::sync::Arc;
use numpy::{
    ndarray::{Array1, Array3, s, ArcArray},
    Ix3, Ix5
};
use pyo3::PyResult;

use crate::{
    value_error,
    coordinates::{Vector3D, CoordinateSystem},
    cylindric::{Index, CylinderGeometry},
    annealing::{
        potential::{BoxPotential2D, BindingPotential2D, EdgeType},
        random::RandomNumberGenerator,
    }
};


pub struct ShiftResult<S> {
    pub index: usize,
    pub state: S,
    pub energy_diff: f32,
}


#[derive(Clone)]
pub struct GraphComponents<Sn, Se> {
    edges: Vec<Vec<usize>>,
    edge_ends: Vec<(usize, usize)>,
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

    pub fn node_count(&self) -> usize {
        self.node_states.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edge_states.len()
    }

    pub fn add_node(&mut self, node_state: Sn) {
        self.node_states.push(node_state);
        self.edges.push(Vec::new());
    }

    pub fn add_edge(&mut self, i: usize, j: usize, edge_state: Se) {
        let count = self.edge_count();
        self.edges[i].push(count);
        self.edges[j].push(count);
        self.edge_ends.push((i, j));
        self.edge_states.push(edge_state);
    }

    pub fn clear(&mut self) {
        self.edges.clear();
        self.edge_ends.clear();
        self.node_states.clear();
        self.edge_states.clear();
    }

    pub fn node_state(&self, i: usize) -> &Sn {
        &self.node_states[i]
    }

    pub fn set_node_state(&mut self, i: usize, node_state: Sn) {
        self.node_states[i] = node_state;
    }

    pub fn edge_state(&self, i: usize) -> &Se {
        &self.edge_states[i]
    }

    pub fn edge(&self, i: usize) -> &Vec<usize> {
        &self.edges[i]
    }

    pub fn edge_end(&self, i: usize) -> (usize, usize) {
        self.edge_ends[i]
    }
}


struct Grid2D<T> {
    coords: Vec<T>,
    ny: usize,
    na: usize,
}

impl<T> Grid2D<T> {
    pub fn at(&self, y: usize, a: usize) -> &T {
        &self.coords[y * self.na + a]
    }

    pub fn at_mut(&mut self, y: usize, a: usize) -> &mut T {
        &mut self.coords[y * self.na + a]
    }
}

impl Grid2D<CoordinateSystem<f32>> {
    pub fn init(naxial: usize, nang: usize) -> Self {
        let coords = vec![CoordinateSystem::zeros(); naxial * nang];
        Self { coords, ny: naxial, na: nang }
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
    geometry: CylinderGeometry,
    coords: Arc<Grid2D<CoordinateSystem<f32>>>,
    score: ArcArray<f32, Ix5>,
    binding_potential: BoxPotential2D,
    local_shape: Vector3D<isize>,
}

impl CylindricGraph {
    pub fn empty() -> Self {
        Self {
            components: GraphComponents::empty(),
            geometry: CylinderGeometry::new(0, 0, 0),
            coords: Arc::new(Grid2D::init(0, 0)),
            score: ArcArray::zeros((0, 0, 0, 0, 0)),
            binding_potential: BoxPotential2D::unbounded(),
            local_shape: Vector3D::new(0, 0, 0),
        }
    }

    pub fn update(
        &mut self,
        score: ArcArray<f32, Ix5>,
        origin: ArcArray<f32, Ix3>,
        zvec: ArcArray<f32, Ix3>,
        yvec: ArcArray<f32, Ix3>,
        xvec: ArcArray<f32, Ix3>,
        nrise: isize,
    ) -> PyResult<&Self> {
        let score_dim = score.raw_dim();
        let (ny, na) = (score_dim[0], score_dim[1]);
        let (_nz, _ny, _nx) = (score_dim[2], score_dim[3], score_dim[4]);

        if origin.shape() != &[ny, na, 3] {
            return value_error!("origin shape mismatch");
        } else if zvec.shape() != &[ny, na, 3] {
            return value_error!("zvec shape mismatch");
        } else if yvec.shape() != &[ny, na, 3] {
            return value_error!("yvec shape mismatch");
        } else if xvec.shape() != &[ny, na, 3] {
            return value_error!("xvec shape mismatch");
        }

        let mut coords: Grid2D<CoordinateSystem<f32>> = Grid2D::init(ny, na);
        for y in 0..coords.ny {
            for a in 0..coords.na {
                coords.at_mut(y, a).origin = origin.slice(s![y, a, ..]).into();
                coords.at_mut(y, a).ez = zvec.slice(s![y, a, ..]).into();
                coords.at_mut(y, a).ey = yvec.slice(s![y, a, ..]).into();
                coords.at_mut(y, a).ex = xvec.slice(s![y, a, ..]).into();
            }
        }

        self.geometry = CylinderGeometry::new(ny as isize, na as isize, nrise);
        self.coords = Arc::new(coords);
        self.score = score;
        self.local_shape = Vector3D::new(_nz, _ny, _nx).into();

        self.components.clear();

        let center: Vector3D<isize> = Vector3D::new(_nz / 2, _ny / 2, _nx / 2).into();
        for y in 0..self.geometry.ny {
            for a in 0..self.geometry.na {
                let idx = Index::new(y, a);
                self.components.add_node(NodeState { index: idx, shift: center.clone() });
            }
        }
        for pair in self.geometry.all_longitudinal_pairs().iter() {
            let idx0 = self.geometry.na * pair.0.y + pair.0.a;
            let idx1 = self.geometry.na * pair.1.y + pair.1.a;
            self.components.add_edge(idx0 as usize, idx1 as usize, EdgeType::Longitudinal);
        }
        for pair in self.geometry.all_lateral_pairs().iter() {
            let idx0 = self.geometry.na * pair.0.y + pair.0.a;
            let idx1 = self.geometry.na * pair.1.y + pair.1.a;
            self.components.add_edge(idx0 as usize, idx1 as usize, EdgeType::Lateral);
        }
        Ok(self)
    }

    pub fn graph(&self) -> &GraphComponents<NodeState, EdgeType> {
        &self.components
    }

    /// Calculate the internal energy of a node state.
    pub fn internal(&self, node_state: &NodeState) -> f32 {
        let idx = &node_state.index;
        let vec = node_state.shift;
        self.score[[idx.y as usize, idx.a as usize, vec.z as usize, vec.y as usize, vec.x as usize]]
    }

    /// Calculate the binding energy between two nodes.
    pub fn binding(&self, node_state0: &NodeState, node_state1: &NodeState, typ: &EdgeType) -> f32 {
        let vec1 = node_state0.shift;
        let vec2 = node_state1.shift;
        let coord1 = self.coords.at(node_state0.index.y as usize, node_state0.index.a as usize);
        let coord2 = self.coords.at(node_state1.index.y as usize, node_state1.index.a as usize);
        let dr = coord1.at_vec(vec1.into()) - coord2.at_vec(vec2.into());
        self.binding_potential.calculate(dr.length2(), typ)
    }

    pub fn random_local_neighbor_state(&self, node_state: &NodeState, rng: &mut RandomNumberGenerator) -> NodeState {
        let idx = node_state.index.clone();
        let shift = node_state.shift;
        let shift_new = rng.rand_shift(&shift, &self.local_shape);
        NodeState { index: idx, shift: shift_new }
    }

    pub fn get_shifts(&self) -> Array3<isize> {
        let mut shifts = Array3::<isize>::zeros(
            (self.geometry.ny as usize, self.geometry.na as usize, 3)
        );
        let graph = self.graph();
        for i in 0..graph.node_count() {
            let state = graph.node_state(i);
            let y = state.index.y;
            let a = state.index.a;
            let shift = state.shift;
            shifts[[y as usize, a as usize, 0]] = shift.z;
            shifts[[y as usize, a as usize, 1]] = shift.y;
            shifts[[y as usize, a as usize, 2]] = shift.x;
        }
        shifts
    }

    fn get_distances(&self, typ: &EdgeType) -> Array1<f32> {
        let graph = self.graph();
        let mut distances = Vec::new();
        for i in 0..graph.edge_count() {
            if graph.edge_state(i) != typ {
                continue;
            }
            let edge = graph.edge_end(i);
            let pos0 = graph.node_state(edge.0);
            let pos1 = graph.node_state(edge.1);
            let coord0 = self.coords.at(pos0.index.y as usize, pos0.index.a as usize);
            let coord1 = self.coords.at(pos1.index.y as usize, pos1.index.a as usize);
            let dr = coord0.at_vec(pos0.shift.into()) - coord1.at_vec(pos1.shift.into());
            distances.push(dr.length())
        }
        Array1::from(distances)
    }

    pub fn set_potential_model(&mut self, model: BoxPotential2D) -> &Self {
        self.binding_potential = model;
        self
    }

    pub fn energy(&self) -> f32 {
        let mut energy = 0.0;
        let graph = self.graph();
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

    pub fn try_random_shift(&self, rng: &mut RandomNumberGenerator) -> ShiftResult<NodeState> {
        let graph = self.graph();
        let idx = rng.uniform_int(graph.node_count());
        let state_old = graph.node_state(idx);
        let mut e_old = self.internal(&state_old);
        let state_new = self.random_local_neighbor_state(&state_old, rng);
        let mut e_new = self.internal(&state_new);
        let connected_edges = graph.edge(idx);
        for edge_id in connected_edges {
            let edge_id = *edge_id;
            let ends = graph.edge_end(edge_id);
            let other_idx = if ends.0 == idx { ends.1 } else { ends.0 };
            let other_state = graph.node_state(other_idx);
            e_old += self.binding(&state_old, &other_state, &graph.edge_state(edge_id));
            e_new += self.binding(&state_new, &other_state, &graph.edge_state(edge_id));
        }
        let de = e_new - e_old;
        ShiftResult { index: idx, state: state_new, energy_diff: de }
    }

    pub fn apply_shift(&mut self, result: &ShiftResult<NodeState>) {
        self.components.set_node_state(result.index, result.state.clone());
    }


    pub fn initialize(&mut self) -> &Self {
        let center = Vector3D::new(self.local_shape.z / 2, self.local_shape.y / 2, self.local_shape.x / 2);
        let (ny, na) = (self.geometry.ny, self.geometry.na);
        for y in 0..ny {
            for a in 0..na {
                let idx = Index::new(y, a);
                let i = na * y + a;
                self.components.set_node_state(
                    i as usize,
                    NodeState { index: idx, shift: center }
                );
            }
        }
        self
    }

    pub fn get_longitudinal_distances(&self) -> Array1<f32> {
        self.get_distances(&EdgeType::Longitudinal)
    }

    pub fn get_lateral_distances(&self) -> Array1<f32> {
        self.get_distances(&EdgeType::Lateral)
    }

    pub fn check_graph(&self) -> PyResult<()> {
        if self.graph().node_count() < 2 {
            return value_error!("Graph has less than 2 nodes");
        }
        Ok(())
    }
}
