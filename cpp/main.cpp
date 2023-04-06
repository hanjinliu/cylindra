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
  	m.def("alleviate", &alleviate, "Alleviate coordinates.");
    py::class_<ViterbiGrid>(m, "ViterbiGrid")
        .def(py::init<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>>())
        .def("viterbi_simple", &ViterbiGrid::viterbiSimple)
        .def("viterbi", &ViterbiGrid::viterbi);
  	// m.def("viterbi", &viterbi, "Viterbi algorithm for alignment.");
  	// m.def("viterbiAngularConstraint", &viterbiAngularConstraint, "Viterbi algorithm for alignment.");
}
