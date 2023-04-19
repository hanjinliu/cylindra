#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "_alleviate.h"
#include "_cylindric.h"
#include "_grid.h"
#include "_grid2d.h"

namespace py = pybind11;

// NOTE: Linker errors when defining functions in cpp file could not be resolved...
// Therefore, all functions are defined in the header file for now.

// define pybind11 module
PYBIND11_MODULE(_cpp_ext, m) {
	m.doc() = "C++ extensions for cylindra";
  	m.def("alleviate", &alleviate, "Alleviate coordinates on a cylindric grid.");

    py::class_<ViterbiGrid>(m, "ViterbiGrid")
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>>(),
             py::arg("score_array"), py::arg("origin"), py::arg("zvec"), py::arg("yvec"), py::arg("xvec"))
        .def("viterbi", py::overload_cast<double, double>(&ViterbiGrid::viterbi), py::arg("dist_min"), py::arg("dist_max"))
        .def("viterbi", py::overload_cast<double, double, py::none>(&ViterbiGrid::viterbi), py::arg("dist_min"), py::arg("dist_max"), py::arg("skew_max") = py::none())
        .def("viterbi", py::overload_cast<double, double, double>(&ViterbiGrid::viterbi), py::arg("dist_min"), py::arg("dist_max"), py::arg("skew_max"))
        .def("world_pos", &ViterbiGrid::worldPos, py::arg("n"), py::arg("z"), py::arg("y"), py::arg("x"))
        .def("__repr__", &ViterbiGrid::pyRepr);

    py::class_<ViterbiGrid2D>(m, "ViterbiGrid2D")
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, ssize_t>(),
             py::arg("score_array"), py::arg("origin"), py::arg("zvec"), py::arg("yvec"), py::arg("xvec"), py::arg("nrise"))
        .def("viterbi", &ViterbiGrid2D::viterbi, py::arg("dist_min"), py::arg("dist_max"), py::arg("lat_dist_min"), py::arg("lat_dist_max"))
        .def("world_pos", &ViterbiGrid2D::worldPos, py::arg("lon"), py::arg("lat"), py::arg("z"), py::arg("y"), py::arg("x"))
        .def("all_longitudinal_pairs", &ViterbiGrid2D::allLongitudinalPairs)
        .def("all_lateral_pairs", &ViterbiGrid2D::allLateralPairs)
        .def("all_longitudinal_distances", &ViterbiGrid2D::allLongitudinalDistances, py::arg("states"))
        .def("all_lateral_distances", &ViterbiGrid2D::allLateralDistances, py::arg("states"))
        .def("__repr__", &ViterbiGrid2D::pyRepr);

    // `CylinderGeometry` is exported mainly for testing
    py::class_<CylinderGeometry>(m, "CylinderGeometry")
        .def(py::init<ssize_t, ssize_t, ssize_t>(), py::arg("ny"), py::arg("na"), py::arg("nrise"))
        .def("source_forward", &CylinderGeometry::sourceForward, py::arg("y"), py::arg("a"))
        .def("source_backward", &CylinderGeometry::sourceBackward, py::arg("y"), py::arg("a"))
        .def("get_neighbors", py::overload_cast<std::vector<std::pair<ssize_t, ssize_t>>>(&CylinderGeometry::getNeighbors), py::arg("indices"))
        .def("get_index", &CylinderGeometry::getIndex, py::arg("y"), py::arg("a"))
        .def("convert_angular", &CylinderGeometry::convertAngular, py::arg("ang"))
        // .def("all_longitudinal_pairs", &ViterbiGrid2D::allLongitudinalPairs)
        // .def("all_lateral_pairs", &ViterbiGrid2D::allLateralPairs)
        .def("__repr__", &CylinderGeometry::pyRepr);

    // `Sources` is exported mainly for testing
    py::class_<Sources>(m, "Sources")
        .def("has_longitudinal", &Sources::hasLongitudinal)
        .def("has_lateral", &Sources::hasLateral)
        .def("__repr__", &Sources::pyRepr)
        .def("__eq__", &Sources::pyEq);

    // `Index` is exported mainly for testing
    py::class_<Index>(m, "Index")
        .def(py::init<ssize_t, ssize_t>(), py::arg("y"), py::arg("a"))
        .def("is_valid", &Index::isValid, py::arg("y"), py::arg("a"))
        .def_readwrite("y", &Index::y)
        .def_readwrite("a", &Index::a)
        .def("__repr__", &Index::pyRepr)
        .def("__eq__", &Index::pyEq);
}
