use pyo3::{prelude::*};

/// An edge on the cylindric lattice.
///
/// `iface` is the protofilament the edge is anchored to and `idx` is the
/// longitudinal index (`nth`) of the lower molecule of the pair. A longitudinal
/// edge at `(p, n)` joins `(n, p)` and `(n + 1, p)`; a lateral edge at `(p, n)`
/// joins `(n, p)` and `(n + shift, p + 1 mod npf)`, where `shift` is the start
/// number when the edge crosses the seam and zero otherwise.
#[derive(Clone, Copy, Debug)]
pub struct LatticeEdge {
    pub iface: u32,
    pub idx: i32,
    pub src: u32,
    pub dst: u32,
}

/// Edges grouped by interface, each group sorted by longitudinal index.
struct EdgeIndex {
    groups: Vec<Vec<(i32, usize)>>,
}

impl EdgeIndex {
    fn new(edges: Vec<(u32, i32, u32, u32)>, npf: u32) -> Self {
        let mut groups = vec![Vec::new(); npf as usize];
        for (id, e) in edges.iter().enumerate() {
            let e = LatticeEdge{iface: e.0, idx: e.1, src: e.2, dst: e.3};
            assert!(e.iface < npf, "interface {} out of range (npf = {})", e.iface, npf);
            groups[e.iface as usize].push((e.idx, id));
        }
        for g in groups.iter_mut() {
            g.sort_unstable_by_key(|&(idx, _)| idx);
        }
        Self { groups }
    }

    fn contains(&self, iface: u32, idx: i32) -> bool {
        self.groups[iface as usize]
            .binary_search_by_key(&idx, |&(i, _)| i)
            .is_ok()
    }
}

/// Longitudinal offset applied to edges that cross the seam.
#[inline]
fn seam_shift(iface: u32, npf: u32, start: i32) -> i32 {
    if iface + 1 == npf { start } else { 0 }
}

/// Whether the 2x2 plaquette anchored at the lateral edge `(iface, idx)` closes.
///
/// The plaquette consists of two lateral edges at `idx` and `idx + 1` on
/// `iface`, plus the longitudinal edges that join their endpoints on each of
/// the two protofilaments.
fn has_plaquette(
    lat: &EdgeIndex,
    long: &EdgeIndex,
    iface: u32,
    idx: i32,
    npf: u32,
    start: i32,
) -> bool {
    let next = (iface + 1) % npf;
    let shift = seam_shift(iface, npf, start);
    lat.contains(iface, idx)
        && lat.contains(iface, idx + 1)
        && long.contains(iface, idx)
        && long.contains(next, idx + shift)
}

/// Emit the ids of edges that belong to a maximal run of at least `min_run`
/// consecutive indices. `sorted` must be sorted by index and free of duplicates.
fn runs_of(sorted: &[(i32, usize)], min_run: usize, out: &mut Vec<usize>) {
    let mut head = 0usize;
    for k in 1..=sorted.len() {
        let broken = k == sorted.len() || sorted[k].0 - sorted[k - 1].0 != 1;
        if broken {
            if k - head >= min_run {
                out.extend(sorted[head..k].iter().map(|&(_, id)| id));
            }
            head = k;
        }
    }
}

/// Select longitudinal edges belonging to a run of at least `min_run` contacts.
///
/// Returns positions into `edges`, in ascending order.
#[pyfunction]
pub fn activate_longitudinal(
    edges: Vec<(u32, i32, u32, u32)>,
    npf: u32,
    min_run: usize,
) -> Vec<usize> {
    let index = EdgeIndex::new(edges, npf);
    let mut out = Vec::new();
    for g in index.groups.iter() {
        runs_of(g, min_run, &mut out);
    }
    out.sort_unstable();
    out
}

/// Select lateral edges that are supported by at least one closed plaquette and
/// belong to a run of at least `min_run` such edges.
///
/// Returns positions into `lat_edges`, in ascending order.
#[pyfunction]
pub fn activate_lateral(
    lat_edges: Vec<(u32, i32, u32, u32)>,
    long_edges: Vec<(u32, i32, u32, u32)>,
    npf: u32,
    start: i32,
    min_run: usize,
) -> Vec<usize> {
    let lat = EdgeIndex::new(lat_edges, npf);
    let long = EdgeIndex::new(long_edges, npf);

    let mut out = Vec::new();
    let mut supported: Vec<(i32, usize)> = Vec::new();
    for (iface, g) in lat.groups.iter().enumerate() {
        let iface = iface as u32;
        supported.clear();
        for &(idx, id) in g.iter() {
            if has_plaquette(&lat, &long, iface, idx, npf, start)
                || has_plaquette(&lat, &long, iface, idx - 1, npf, start)
            {
                supported.push((idx, id));
            }
        }
        runs_of(&supported, min_run, &mut out);
    }
    out.sort_unstable();
    out
}
