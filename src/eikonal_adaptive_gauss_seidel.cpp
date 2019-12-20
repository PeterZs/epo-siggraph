#include "eikonal_adaptive_gauss_seidel.hpp"

#if USING_PYBIND11
#include "pybind11_common.hpp"
#endif

olim::EikonalAdaptiveGaussSeidel::EikonalAdaptiveGaussSeidel(
  Eigen::Ref<points_t> points,
  Eigen::Ref<tetras_t> tetras,
  Eigen::Ref<slowness_t> slowness,
  double tol):
  Eikonal {points, tetras, slowness},
  tol {tol}
{
  build_stencils();
}

void olim::EikonalAdaptiveGaussSeidel::build_stencils() {
  stencils.resize(mesh.num_points);
  for (int64_t i = 0; i < mesh.num_points; ++i) {
    for (int64_t t: mesh.vt_neib[i]) {
      auto face = mesh.get_opposite_face(t, i);
      stencils[i].insert(face);
    }
  }
}

void olim::EikonalAdaptiveGaussSeidel::push_back(int64_t i) {
  if (enqueued.find(i) == enqueued.end()) {
    queue.push(i);
    enqueued.insert(i);
  }
}

void olim::EikonalAdaptiveGaussSeidel::pop_front() {
  enqueued.erase(queue.front());
  queue.pop();
}

opt_t<int64_t> olim::EikonalAdaptiveGaussSeidel::front() const {
  return queue.empty() ?
    boost::none :
    opt_t<int64_t> {queue.front()};
}

void olim::EikonalAdaptiveGaussSeidel::add_boundary_point(int64_t i, double value) {
  values[i] = value;
  boundary.insert(i);
}

void olim::EikonalAdaptiveGaussSeidel::commit() {
  for (int64_t i: boundary) {
    for (int64_t j: mesh.vv_neib[i]) {
      if (boundary.find(j) == boundary.end()) {
        push_back(j);
      }
    }
  }
}

bool olim::EikonalAdaptiveGaussSeidel::do_line_update(int64_t i0) const {
  return std::isfinite(values[i0]);
}

bool olim::EikonalAdaptiveGaussSeidel::do_tri_update(Edge edge) const {
  return std::isfinite(values[edge.i0]) && std::isfinite(values[edge.i1]);
}

bool olim::EikonalAdaptiveGaussSeidel::do_tetra_update(Face face) const {
  return std::isfinite(values[face.i0]) && std::isfinite(values[face.i1]) &&
    std::isfinite(values[face.i2]);
}

bool olim::EikonalAdaptiveGaussSeidel::step() {
  int64_t i;
  if (opt_t<int64_t> opt_i = front()) {
    i = *opt_i;
    pop_front();
  } else {
    return false;
  }

  double u = update(i);

  if (fabs(u - values[i]) > tol) {
    for (int64_t j: mesh.vv_neib[i]) {
      push_back(j);
    }
  }

  values[i] = u;

  return true;
}

#if USING_PYBIND11
void init_eikonal_adaptive_gauss_seidel(py::module & m) {
  py::class_<olim::EikonalAdaptiveGaussSeidel,
             olim::Eikonal>(m, "EikonalAdaptiveGaussSeidel")
    .def(py::init<
           Eigen::Ref<points_t>,
           Eigen::Ref<tetras_t>,
           Eigen::Ref<slowness_t>,
           double
         >())
    .def_readonly("queue", &olim::EikonalAdaptiveGaussSeidel::queue)
    .def_readonly("enqueued", &olim::EikonalAdaptiveGaussSeidel::enqueued)
    .def_readonly("boundary", &olim::EikonalAdaptiveGaussSeidel::boundary)
    .def_readonly("tol", &olim::EikonalAdaptiveGaussSeidel::tol);
}
#endif
