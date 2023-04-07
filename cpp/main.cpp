#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_alleviate.h"
#include "_cylindric.h"
#include "_grid.h"

namespace py = pybind11;

// NOTE: Linker errors when defining functions in cpp file could not be resolved...
// Therefore, all functions are defined in the header file for now.

// define pybind11 module
PYBIND11_MODULE(_cpp_ext, m) {
	m.doc() = "C++ extensions";
  	m.def("alleviate", &alleviate, "Alleviate coordinates on a cylindric grid.");
    py::class_<ViterbiGrid>(m, "ViterbiGrid")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>())
        .def("viterbi_simple", &ViterbiGrid::viterbiSimple)
        .def("viterbi", &ViterbiGrid::viterbi);
    py::class_<CylinderGeometry>(m, "CylinderGeometry")
        .def(py::init<ssize_t, ssize_t, ssize_t>())
        .def("source_of", &CylinderGeometry::sourceOf)
        .def("__repr__", &CylinderGeometry::pyRepr);
    py::class_<Sources>(m, "Sources")
        .def("has_longitudinal", &Sources::hasLongitudinal)
        .def("has_lateral", &Sources::hasLateral)
        .def("__repr__", &Sources::pyRepr);
    py::class_<Index>(m, "Index")
        .def(py::init<ssize_t, ssize_t>())
        .def("is_valid", &Index::isValid)
        .def("__repr__", &Index::pyRepr);
}
