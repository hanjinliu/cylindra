#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_viterbi.h"
#include "_cylindric.h"

namespace py = pybind11;
using uint = unsigned int;
using ssize_t = Py_ssize_t;


// NOTE: Linker errors when defining functions in cpp file could not be resolved...
// Therefore, all functions are defined in the header file for now.

std::vector<Index> arrayToIndices(py::array_t<ssize_t> array){
    py::buffer_info _arr_info = array.request();
    ssize_t nidx = _arr_info.shape[0];
    ssize_t nd = _arr_info.shape[1];

    std::vector<Index> indices;

    for (ssize_t i = 0; i < nidx; ++i){
        indices.push_back(
            Index(
                static_cast<uint>(*array.data(i, 0)),
                static_cast<uint>(*array.data(i, 1)), 
                static_cast<uint>(*array.data(i, 2))
            )
        );
    }
    return indices;
}

py::array_t<ssize_t> alleviate(
    py::array_t<ssize_t> arr,
    py::array_t<ssize_t> label,
    ssize_t iterations
)
{
    // get buffers
	py::buffer_info _arr_info = arr.request();
    
	ssize_t nr = _arr_info.shape[0];
	ssize_t ny = _arr_info.shape[1];
	ssize_t na = _arr_info.shape[2];
    ssize_t ndim = _arr_info.shape[3];

    auto indices = arrayToIndices(label);
    CylinderGeometry geometry(nr, ny, na);

    for (int i = 0; i < iterations; ++i){
        auto neighbors = geometry.getNeighbors(indices);
        for (auto neighbor : neighbors){
            auto curnbr = geometry.getNeighbor(neighbor.r, neighbor.y, neighbor.a);
            auto nCurnbr = curnbr.size();
            ssize_t sumR = 0, sumY = 0, sumA = 0;
            for (auto nbr : curnbr){
                sumR += *arr.data(nbr.r, nbr.y, nbr.a, 0);
                sumY += *arr.data(nbr.r, nbr.y, nbr.a, 1);
                sumA += *arr.data(nbr.r, nbr.y, nbr.a, 2);
            }

            *arr.mutable_data(neighbor.r, neighbor.y, neighbor.a, 0) = sumR / nCurnbr;
            *arr.mutable_data(neighbor.r, neighbor.y, neighbor.a, 1) = sumY / nCurnbr;
            *arr.mutable_data(neighbor.r, neighbor.y, neighbor.a, 2) = sumA / nCurnbr;  // border!
        }
        // concatenate vectors
        indices.insert(indices.end(), neighbors.begin(), neighbors.end());
    }
}


// define pybind11 module
PYBIND11_MODULE(_cpp_ext, m) {
	m.doc() = "C++ extensions";
  	m.def("alleviate", &alleviate, "Alleviate coordinates.");
  	m.def("viterbi", &viterbi, "Viterbi algorithm for alignment.");
  	m.def("viterbiAngularConstraint", &viterbiAngularConstraint, "Viterbi algorithm for alignment.");
}
