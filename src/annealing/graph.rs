use numpy::ndarray::{Array1, Array3, Array5, Axis, s};

use crate::coordinates::{Vector3D, CoordinateSystem};
use crate::cylindric::{Index, CylinderGeometry};
use crate::annealing::{
    potential::{EmptyPotential2D, BindingPotential2D, EdgeType},
    random::RandomNumberGenerator,
};

pub struct ShiftResult<S> {
    index: usize,
    state: S,
    dE: f32,
}


struct GraphComponents<Sn, Se> {
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


// /// @brief Abstract class for an undirected graph with scores.
// template <typename Sn, typename Se, typename T>
// class AbstractGraph {
//     protected:
//         std::vector<std::vector<size_t>> edges;  // id of connected edges of the i-th node
//         std::vector<std::pair<size_t, size_t>> edgeEnds;  // end nodes of the i-th edge
//         std::vector<Sn> nodeState;
//         std::vector<Se> edgeState;

//     public:
//         AbstractGraph() {};

//         size_t nodeCount() { return nodeState.size(); };
//         size_t edgeCount() { return edgeState.size(); };

//         /// Add a node of given state to the graph.
//         void addNode(const Sn &nodestate) {
//             nodeState.push_back(nodestate);
//             edges.push_back(std::vector<size_t>());
//         }

//         /// add an edge between i-th and j-th nodes.
//         void addEdge(size_t i, size_t j, const Se &edgestate) {
//             if (i >= nodeCount() || j >= nodeCount()) {
//                 throw py::index_error(
//                     "There are " + std::to_string(nodeCount()) +
//                     " nodes, but trying to add an edge between " + std::to_string(i) +
//                     " and " + std::to_string(j) + "."
//                 );
//             }
//             edges[i].push_back(edgeEnds.size());
//             edges[j].push_back(edgeEnds.size());
//             edgeEnds.push_back(std::make_pair(i, j));
//             edgeState.push_back(edgestate);
//         }

//         Sn &nodeStateAt(size_t i) & { return nodeState[i]; };
//         Se &edgeStateAt(size_t i) & { return edgeState[i]; };
//         std::pair<size_t, size_t> edgeEndsAt(size_t i) & { return edgeEnds[i]; };

//         virtual Sn randomLocalNeighBorState(const Sn &nodestate, RandomNumberGenerator &rng) { throw py::attribute_error("randomLocalNeighBorState() is not implemented.");};

//         /// Returns the internal potential energy of molecule at `pos` of given `state`.
//         virtual T internal(const Sn &nodestate) { return 0; };

//         /// Returns the binding potential energy between adjacent molecules.
//         virtual T binding(const Sn &nodestate0, const Sn &nodestate1, const Se &edgestate) { return 0; };

//         // Check if the current state of this graph is ready.
//         virtual void checkGraph() { throw py::attribute_error("checkGraph() is not implemented.");};

//         void applyShift(ShiftResult<Sn> &result) {
//             nodeState[result.index] = result.state;
//         }

//         std::vector<std::pair<size_t, size_t>> getEdgeEnds() { return edgeEnds; };

//         /// Clear all the nodes and edges of the graph.
//         void clearGraph() {
//             nodeState.clear();
//             edgeState.clear();
//             edges.clear();
//             edgeEnds.clear();
//         }
// };

struct Grid2D<T> {
    coords: Vec<T>,
    naxial: usize,
    nang: usize,
}

impl<T> Grid2D<T> {
    pub fn at(&self, y: usize, a: usize) -> &T {
        &self.coords[y * self.nang + a]
    }

    pub fn at_mut(&mut self, y: usize, a: usize) -> &mut T {
        &mut self.coords[y * self.nang + a]
    }

    pub fn size(&self) -> usize {
        self.naxial * self.nang
    }
}

impl Grid2D<CoordinateSystem<f32>> {
    pub fn init(naxial: usize, nang: usize) -> Self {
        let coords = vec![CoordinateSystem::zeros(); naxial * nang];
        Self { coords, naxial, nang }
    }
}

#[derive(Clone)]
pub struct NodeState {
    index: Index,
    shift: Vector3D<isize>,
}

pub struct CylindricGraph {
    components: GraphComponents<NodeState, EdgeType>,
    geometry: CylinderGeometry,
    coords: Grid2D<CoordinateSystem<f32>>,
    score: Array5<f32>,
    binding_potential: Box<dyn BindingPotential2D>,
    local_shape: Vector3D<isize>,
}

impl CylindricGraph {
    pub fn new(
        score: Array5<f32>,
        origin: Array3<f32>,
        zvec: Array3<f32>,
        yvec: Array3<f32>,
        xvec: Array3<f32>,
        nrise: isize,
    ) -> Self {
        let (ny, na) = (score.len_of(Axis(0)), score.len_of(Axis(1)));
        let (_nz, _ny, _nx) = (score.len_of(Axis(2)), score.len_of(Axis(3)), score.len_of(Axis(4)));
        let mut coords: Grid2D<CoordinateSystem<f32>> = Grid2D::init(ny, na);
        for y in 0..ny {
            for a in 0..na {
                coords.at_mut(y, a).origin = origin.slice(s![y, a, ..]).into();
                coords.at_mut(y, a).ez = zvec.slice(s![y, a, ..]).into();
                coords.at_mut(y, a).ey = yvec.slice(s![y, a, ..]).into();
                coords.at_mut(y, a).ex = xvec.slice(s![y, a, ..]).into();
            }
        }
        Self {
            components: GraphComponents::empty(),
            geometry: CylinderGeometry::new(ny as isize, na as isize, nrise),
            coords,
            score,
            binding_potential: Box::new(EmptyPotential2D{}),
            local_shape: Vector3D::new(_nz, _ny, _nx).into(),
        }
    }

    fn graph(&self) -> &GraphComponents<NodeState, EdgeType> {
        &self.components
    }

    fn graph_mut(&mut self) -> &mut GraphComponents<NodeState, EdgeType> {
        &mut self.components
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
        let dr = coord1.at(vec1.z as f32, vec1.y as f32, vec1.x as f32) - coord2.at(vec2.z as f32, vec2.y as f32, vec2.x as f32);
        self.binding_potential.calculate(dr.length2(), typ)
    }

    pub fn random_local_neighbor_state(&self, node_state: &NodeState, rng: &mut RandomNumberGenerator) -> NodeState {
        let idx = node_state.index.clone();
        let shift = node_state.shift;
        let shift_new = rng.rand_shift(&shift, &self.local_shape);
        NodeState { index: idx, shift: shift_new }
    }

    pub fn get_shifts(&self) -> Array3<isize> {
        let mut shifts = Array3::<isize>::zeros((self.geometry.ny as usize, self.geometry.na as usize, 3));
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
        let mut distances = Array1::<f32>::zeros(graph.edge_count());
        for i in 0..graph.edge_count() {
            let edge = graph.edge_end(i);
            let node_state0 = graph.node_state(edge.0);
            let node_state1 = graph.node_state(edge.1);
            distances[i] = self.binding(&node_state0, &node_state1, typ);
        }
        distances
    }

    pub fn get_longitudinal_distances(&self) -> Array1<f32> {
        self.get_distances(&EdgeType::Longitudinal)
    }

    pub fn get_lateral_distances(&self) -> Array1<f32> {
        self.get_distances(&EdgeType::Lateral)
    }

    pub fn potential_model(&self) -> &dyn BindingPotential2D {
        &*self.binding_potential
    }

    pub fn set_potential_model(&mut self, model: Box<dyn BindingPotential2D>) {
        self.binding_potential = model;
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
        let e_old = self.internal(&state_old);
        let state_new = self.random_local_neighbor_state(&state_old, rng);
        let e_new = self.internal(&state_new);
        let connected_edges = graph.edge(idx);
        let mut e_old = e_old;
        let mut e_new = e_new;
        for edge_id in connected_edges {
            let edge_id = *edge_id;
            let ends = graph.edge_end(edge_id);
            let other_idx = if ends.0 == idx { ends.1 } else { ends.0 };
            let other_state = graph.node_state(other_idx);
            e_old += self.binding(&state_old, &other_state, &graph.edge_state(edge_id));
            e_new += self.binding(&state_new, &other_state, &graph.edge_state(edge_id));
        }
        let de = e_new - e_old;
        ShiftResult { index: idx, state: state_new, dE: de }
    }

    pub fn initialize(&mut self) {
        let center = Vector3D::new(self.local_shape.z / 2, self.local_shape.y / 2, self.local_shape.x / 2);
        let (ny, na) = (self.geometry.ny, self.geometry.na);
        let graph = self.graph_mut();
        for y in 0..ny {
            for a in 0..na {
                let idx = Index::new(y, a);
                let i = na * y + a;
                graph.set_node_state(
                    i.try_into().unwrap(),
                    NodeState { index: idx, shift: center }
                );
            }
        }
    }


}
