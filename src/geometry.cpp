#include "geometry.hpp"

#if USING_PYBIND11
#include "pybind11_common.hpp"
#endif

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_3.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;

std::vector<std::array<int64_t, 3>>
olim::geometry::spherical_delaunay(Eigen::Ref<points_t> points)
{
  std::vector<K::Point_3> points_;
  points_.reserve(points.rows());
  for (int64_t i = 0; i < points.rows(); ++i) {
    points_.emplace_back(points(i, 0), points(i, 1), points(i, 2));
  }

  CGAL::Surface_mesh<K::Point_3> mesh;
  CGAL::convex_hull_3(points_.begin(), points_.end(), mesh);

  std::vector<std::array<int64_t, 3>> face_indices;
  face_indices.reserve(mesh.number_of_faces());

  for (auto face_index: mesh.faces()) {
    auto face_halfedge = mesh.halfedge(face_index);
    std::array<int64_t, 3> ind;
    int64_t i = 0;
    for (auto halfedge: CGAL::halfedges_around_face(face_halfedge, mesh)) {
      auto vertex_index = CGAL::target(halfedge, mesh);
      ind[i++] = static_cast<int64_t>(vertex_index);
    }
    face_indices.push_back(ind);
  }

  return face_indices;
}

#if USING_PYBIND11
void init_geometry(py::module & m) {
  m.def("spherical_delaunay", &olim::geometry::spherical_delaunay);
}
#endif
