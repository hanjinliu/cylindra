#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "_cylindric.h"

namespace py = pybind11;
using ssize_t = Py_ssize_t;

const double PI = 3.14159265358979323846;

// Convert (N, 2) pyarray in to a vector of Index objects
// As a pseudo-code example, arrayToIndices(np.array([[2, 3], [5, 4]])) will return 
// {Index(2, 3), Index(5, 4)}.
std::vector<Index> arrayToIndices(py::array_t<ssize_t> array) {
    py::buffer_info _arr_info = array.request();
    ssize_t nidx = _arr_info.shape[0];

    std::vector<Index> indices;

    for (ssize_t i = 0; i < nidx; ++i){
        indices.push_back(
            Index(*array.data(i, 0), *array.data(i, 1))
        );
    }
    return indices;
}

// Alleviate molecule displacements by iterative local-averaging algorithm.
// Molecule positions labeled by the argument `label` will not be moved. The other
// molecules will be averaged by the surroudning molecules. This procedure will be
// repeated `iterations` times.
py::array_t<double> alleviate(
    py::array_t<double> arr,
    py::array_t<ssize_t> label,
	ssize_t nrise,
    ssize_t iterations
) {
    // get buffers
	py::buffer_info _arr_info = arr.request();
    
	ssize_t ny = _arr_info.shape[0];
	ssize_t na = _arr_info.shape[1];
	ssize_t ndim = _arr_info.shape[2];

    auto indices = arrayToIndices(label);

    // Create geometry. A geometry will consider the connectivity of molecules, 
    // especially at the boundary of the cylinder.
    CylinderGeometry geometry(ny, na, nrise);

	auto arr_data = arr.mutable_unchecked<3>();

    for (int i = 0; i < iterations; ++i) {
        auto neighbors = geometry.getNeighbors(indices);
        for (auto neighbor : neighbors) {
			// Get current neighbors
            auto curNeighbor = geometry.getNeighbor(neighbor.y, neighbor.a);
            auto nCurNeighbor = static_cast<double>(curNeighbor.size());

			// Alleviate inter-molecule distances by taking the average of the neighbors
            // Note that along the angular axis, averaging should be calculated in the
            // complex plane (in which mean([30 deg, 330 deg]) should return 0 deg, not 
            // 180 deg).
            double sumR = 0.0, sumY = 0.0, sumACos = 0.0, sumASin = 0.0;
            for (auto nbr : curNeighbor) {
                sumR += *arr.data(nbr.y, nbr.a, 0);
                sumY += *arr.data(nbr.y, nbr.a, 1);
                double a = *arr.data(nbr.y, nbr.a, 2);
                sumACos += std::cos(a);
                sumASin += std::sin(a);
            }

            arr_data(neighbor.y, neighbor.a, 0) = sumR / nCurNeighbor;
            arr_data(neighbor.y, neighbor.a, 1) = sumY / nCurNeighbor;
            double theta = std::atan2(sumASin, sumACos);
            if (theta < 0) {
                theta += 2 * PI;
            }
            arr_data(neighbor.y, neighbor.a, 2) = theta;
        }
        // concatenate vectors
        indices.insert(indices.end(), neighbors.begin(), neighbors.end());
    }

	return arr;
}
