#include "pybind11_common.hpp"

#include "eikonal.hpp"
#include "eikonal_adaptive_gauss_seidel.hpp"
#include "eikonal_dijkstra_like.hpp"

void init_eikonal(py::module & m);
void init_eikonal_adaptive_gauss_seidel(py::module & m);
void init_eikonal_dijkstra_like(py::module & m);
void init_tetra_mesh(py::module & m);

PYBIND11_MODULE(olim, m) {
  m.doc() = R"pbdoc(
olim
----

TODO!
)pbdoc";

  init_eikonal(m);
  init_eikonal_adaptive_gauss_seidel(m);
  init_eikonal_dijkstra_like(m);
  init_tetra_mesh(m);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
