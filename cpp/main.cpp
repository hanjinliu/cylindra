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
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>>())
        .def("viterbi", py::overload_cast<double, double>(&ViterbiGrid::viterbi))
        .def("viterbi", py::overload_cast<double, double, double>(&ViterbiGrid::viterbi))
        .def("world_pos", &ViterbiGrid::worldPos)
        .def("__repr__", &ViterbiGrid::pyRepr);
    
    py::class_<ViterbiGrid2D>(m, "ViterbiGrid2D")
        .def(py::init<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, ssize_t>())
        .def("viterbi", &ViterbiGrid2D::viterbi)
        .def("world_pos", &ViterbiGrid2D::worldPos)
        .def("__repr__", &ViterbiGrid2D::pyRepr);
    
    // `CylinderGeometry` is exported mainly for testing
    py::class_<CylinderGeometry>(m, "CylinderGeometry")
        .def(py::init<ssize_t, ssize_t, ssize_t>())
        .def("source_forward", &CylinderGeometry::sourceForward)
        .def("source_backward", &CylinderGeometry::sourceBackward)
        .def("get_neighbors", &CylinderGeometry::getNeighbors)
        .def("get_index", &CylinderGeometry::getIndex)
        .def("convert_angular", &CylinderGeometry::convertAngular)
        .def("__repr__", &CylinderGeometry::pyRepr);

    // `Sources` is exported mainly for testing
    py::class_<Sources>(m, "Sources")
        .def("has_longitudinal", &Sources::hasLongitudinal)
        .def("has_lateral", &Sources::hasLateral)
        .def("__repr__", &Sources::pyRepr)
        .def("__eq__", &Sources::pyEq);

    // `Index` is exported mainly for testing
    py::class_<Index>(m, "Index")
        .def(py::init<ssize_t, ssize_t>())
        .def("is_valid", &Index::isValid)
        .def("__repr__", &Index::pyRepr)
        .def("__eq__", &Index::pyEq);
}
