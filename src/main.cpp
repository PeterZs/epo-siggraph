#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <array>
#include <iostream>
#include <list>
#include <vector>

using points_t = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using tetra_t = Eigen::Matrix<int64_t, Eigen::Dynamic, 4, Eigen::RowMajor>;

using neighborhood_lists_t = std::vector<std::list<int64_t>>;
PYBIND11_MAKE_OPAQUE(std::vector<std::list<int64_t>>);

struct EikonalTetraSolver {
	neighborhood_lists_t neighborhood_lists;

	EikonalTetraSolver(Eigen::Ref<points_t> points, Eigen::Ref<tetra_t> tetra)
	{
		neighborhood_lists.resize(points.rows());
		for (Eigen::Index t = 0; t < tetra.rows(); ++t) {
			for (Eigen::Index j = 0; j < tetra.cols(); ++j) {
				int64_t v = tetra(t, j);
				auto & lst = neighborhood_lists[v];
				if (std::find(lst.begin(), lst.end(), t) == lst.end()) {
					lst.push_back(t);
				}
			}
		}
	}
};

PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

	py::bind_vector<std::vector<std::list<int64_t>>>(m, "NeighborhoodLists");

	py::class_<EikonalTetraSolver>(m, "EikonalTetraSolver")
		.def(py::init<Eigen::Ref<points_t>, Eigen::Ref<tetra_t>>())
		.def_readonly("neighborhood_lists", &EikonalTetraSolver::neighborhood_lists);

    // m.def("add", &add, R"pbdoc(
    //     Add two numbers

    //     Some other explanation about the add function.
    // )pbdoc");

    // m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
    //     Subtract two numbers

    //     Some other explanation about the subtract function.
    // )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
