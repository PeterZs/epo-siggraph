#include "eikonal_dijkstra_like.hpp"

#if USING_PYBIND11
#include "pybind11_common.hpp"
#endif

olim::EikonalDijkstraLike::EikonalDijkstraLike(Eigen::Ref<points_t> points,
                                               Eigen::Ref<tetras_t> tetras,
                                               Eigen::Ref<slowness_t> slowness):
  Eikonal {points, tetras, slowness}
{
  build_stencils();

  states.resize(mesh.num_points, State::Far);
}

bool olim::EikonalDijkstraLike::is_causal(int64_t i, Face face) {
  (void) i;
  (void) face;
  throw std::runtime_error("blah");
}

void olim::EikonalDijkstraLike::build_stencils() {
  stencils.resize(mesh.num_points);
  for (int64_t i = 0; i < mesh.num_points; ++i) {
    for (int64_t t: mesh.vt_neib[i]) {
      auto face = mesh.get_opposite_face(t, i);
      if (is_causal(i, face)) {
        stencils[i].insert(face);
      } else {
        modified_stencils.insert(i);
        // TODO: need to check for degenerate case: i.e., opposite
        // vertex isn't in the cone spanned by the original update
        for (auto new_face: mesh.split_tetra(t, i)) {
          if (!is_causal(i, new_face)) {
            puts("WARNING: noncausal update after split");
          }
          stencils[i].insert(face);
        }
      }
    }
  }
}

void olim::EikonalDijkstraLike::push_back(int64_t i) {
  heap.push_back(i);
  std::push_heap(
    heap.begin(),
    heap.end(),
    [&] (int64_t const & i, int64_t const & j) {
      return values[i] > values[j];
    }
  );
}

void olim::EikonalDijkstraLike::pop_front() {
  std::pop_heap(
    heap.begin(),
    heap.end(),
    [&] (int64_t const & i, int64_t const & j) {
      return values[i] > values[j];
    }
  );
  heap.pop_back();
}

opt_t<int64_t> olim::EikonalDijkstraLike::front() const {
  return heap.empty() ?
    boost::none :
    opt_t<int64_t> {heap[0]};
}

void olim::EikonalDijkstraLike::add_boundary_point(int64_t i, double value) {
  values[i] = value;
  states[i] = State::Trial;
}

void olim::EikonalDijkstraLike::commit() {
  for (int64_t i = 0; i < mesh.num_points; ++i) {
    if (states[i] == State::Trial) {
      push_back(i);
    }
  }
}

bool olim::EikonalDijkstraLike::is_valid(int64_t i) const {
  return states[i] == State::Valid;
}

bool olim::EikonalDijkstraLike::do_line_update(int64_t i0) const {
  return is_valid(i0);
}

bool olim::EikonalDijkstraLike::do_tri_update(Edge edge) const {
  return is_valid(edge.i0) && is_valid(edge.i1);
}

bool olim::EikonalDijkstraLike::do_tetra_update(Face face) const {
  return is_valid(face.i0) && is_valid(face.i1) && is_valid(face.i2);
}

bool olim::EikonalDijkstraLike::step() {
  int64_t i;
  if (opt_t<int64_t> opt_i = front()) {
    i = *opt_i;
    pop_front();
  } else {
    return false;
  }

  states[i] = State::Valid;

  for (int64_t j: mesh.vv_neib[i]) {
    if (states[j] == State::Far) {
      states[j] = State::Trial;
      push_back(j);
    }
  }

  for (int64_t j: mesh.vv_neib[i]) {
    if (states[j] == State::Trial) {
      values[j] = update(j);
    }
  }

  return true;
}

#if USING_PYBIND11
void init_state(py::module & m) {
  py::enum_<olim::State>(m, "State")
    .value("Far", olim::State::Far)
    .value("Trial", olim::State::Trial)
    .value("Valid", olim::State::Valid)
    .value("Boundary", olim::State::Boundary)
    .value("Shadow", olim::State::Shadow)
    .export_values();
}

void init_eikonal_dijkstra_like(py::module & m) {
  py::class_<olim::EikonalDijkstraLike, olim::Eikonal>(m, "EikonalDijkstraLike")
    .def(py::init<
           Eigen::Ref<points_t>,
           Eigen::Ref<tetras_t>,
           Eigen::Ref<slowness_t>
         >())
    .def_readonly("states", &olim::EikonalDijkstraLike::states)
    .def_readonly("heap", &olim::EikonalDijkstraLike::heap)
    .def_readonly("modified_stencils",
                  &olim::EikonalDijkstraLike::modified_stencils);
}
#endif
