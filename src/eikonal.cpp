#include "eikonal.hpp"

#if USING_PYBIND11
#include "pybind11_common.hpp"
#endif

olim::Eikonal::Eikonal(Eigen::Ref<points_t> points,
                       Eigen::Ref<tetras_t> tetras,
                       Eigen::Ref<slowness_t> slowness):
  points {points},
  slowness {slowness},
  mesh {points, tetras}
{
  values.setConstant(
    mesh.num_points,
    std::numeric_limits<double>::infinity()
    );
}

double olim::Eikonal::line(int64_t i, int64_t i0) const {
  double L = (points.row(i0) - points.row(i)).norm();
  double s = slowness[i];
  double s0 = slowness[i0];
  double u0 = values[i0];
  return u0 + (s + s0)*L/2;
}

std::pair<double, double> olim::Eikonal::tri(int64_t i, Edge edge) const {
  auto x0 = points.row(edge.i0) - points.row(i);
  auto dx = points.row(edge.i1) - points.row(edge.i0);
  auto x0_sq = x0.squaredNorm();
  auto dx_sq = dx.squaredNorm();
  auto x0_dx = x0.dot(dx);

  double u0 = values[edge.i0];
  double du = values[edge.i1] - u0;

  double s0 = slowness[edge.i0];
  double s1 = slowness[edge.i1];
  double s_hat = slowness[i];
  double s = (s_hat + (s0 + s1)/2)/2;

  double L = std::sqrt((x0_sq*dx_sq - x0_dx*x0_dx)/(dx_sq - du*du/(s*s)));
  double lam = L*du/(s*dx_sq) - x0_dx/dx_sq;

  s = (s_hat + s0 + (s1 - s0)*lam)/2; // mp1 correction
  double u = u0 + s*(x0_sq + lam*x0_dx)/L;

  return {u, lam};
}

std::pair<double, Eigen::Vector2d>
olim::Eikonal::tetra(int64_t i, Face face) const {
  auto x0 = points.row(face.i0) - points.row(i);
  auto dx1 = points.row(face.i1) - points.row(face.i0);
  auto dx2 = points.row(face.i2) - points.row(face.i0);

  double u0 = values[face.i0];
  Eigen::Vector2d dU = {values[face.i1] - u0, values[face.i2] - u0};

  double s0 = slowness[face.i0];
  double s1 = slowness[face.i1];
  double s2 = slowness[face.i2];
  double s_hat = slowness[i];
  double s = (s_hat + (s0 + s1 + s2)/3)/2;

  // Compute reduced QR decomposition

  Eigen::Matrix<double, 3, 2> Q;
  Eigen::Matrix<double, 2, 2> R;
  R(1, 0) = 0;
  R(0, 0) = dx1.norm();
  Q.col(0) = dx1/R(0, 0);
  R(0, 1) = Q.col(0).dot(dx2);
  Q.col(1) = dx2.transpose() - R(0, 1)*Q.col(0);
  R(1, 1) = Q.col(1).norm();
  Q.col(1).normalize();

  auto x0_sq = x0.squaredNorm();
  auto Qt_x0 = Q.transpose()*x0.transpose();
  auto Qt_x0_sq = Qt_x0.squaredNorm();

  auto Rt_inv_dU_s = R.transpose().triangularView<Eigen::Lower>().solve(dU)/s;

  auto L = std::sqrt((x0_sq - Qt_x0_sq)/(1 - Rt_inv_dU_s.squaredNorm()));
  auto lam = -R.triangularView<Eigen::Upper>().solve(Qt_x0 + L*Rt_inv_dU_s);

  s = (s_hat + s0 + (s1 - s0)*lam(0) + (s2 - s0)*lam(1))/2; // mp1 correction
  auto u = u0 + s*x0.dot(x0 + lam(0)*dx1  + lam(1)*dx2)/L;

  return {u, lam};
}

bool olim::Eikonal::in_simplex(double lam) const {
  return 0 <= lam && lam <= 1;
}

bool olim::Eikonal::in_simplex(Eigen::Array2d lam) const {
  return lam(0) >= 0 && lam(1) >= 0 && lam(0) + lam(1) <= 1;
}

double olim::Eikonal::update(int64_t i) const {
  std::set<int64_t> neib;
  for (int64_t j: mesh.vv_neib[i]) {
    neib.insert(j);
  }

  std::unordered_set<Edge> edges;
  for (Face face: stencils[i]) {
    for (Edge edge: face.get_edges()) {
      edges.insert(edge);
    }
  }

  double u = inf<double>(), u_new;

  for (Face face: stencils[i]) {
    if (do_tetra_update(face)) {
      Eigen::Vector2d lam;
      std::tie(u_new, lam) = tetra(i, face);
      if (in_simplex(lam) && u_new < u) {
        u = u_new;
        for (Edge edge: face.get_edges()) {
          edges.erase(edge);
          neib.erase(edge.i0);
        }
      }
    }
  }

  for (Edge edge: edges) {
    if (do_tri_update(edge)) {
      double lam;
      std::tie(u_new, lam) = tri(i, edge);
      if (in_simplex(lam) && u_new < u) {
        u = u_new;
        neib.erase(edge.i0);
        neib.erase(edge.i1);
      }
    }
  }

  for (int64_t i0: neib) {
    if (do_line_update(i0)) {
      u_new = line(i, i0);
      if (u_new < u) {
        u = u_new;
      }
    }
  }

  return u;
}

void olim::Eikonal::solve() {
  while (step()) {}
}

#if USING_PYBIND11
void init_eikonal(py::module & m) {
  py::class_<olim::Eikonal>(m, "Eikonal")
    .def("line", &olim::Eikonal::line)
    .def("tri", &olim::Eikonal::tri)
    .def("tetra", &olim::Eikonal::tetra)
    .def("update", &olim::Eikonal::update)
    .def("add_boundary_point", &olim::Eikonal::add_boundary_point)
    .def("commit", &olim::Eikonal::commit)
    .def("step", &olim::Eikonal::step)
    .def("solve", &olim::Eikonal::solve)
    .def_readonly("mesh", &olim::Eikonal::mesh)
    .def_readonly("slowness", &olim::Eikonal::slowness)
    .def_readonly("stencils", &olim::Eikonal::stencils)
    .def_readonly("values", &olim::Eikonal::values);
}
#endif
