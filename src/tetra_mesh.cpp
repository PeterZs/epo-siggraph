#include "tetra_mesh.hpp"

#if USING_PYBIND11
#include "pybind11_common.hpp"
#endif

olim::TetraMesh::TetraMesh(Eigen::Ref<points_t> points,
                           Eigen::Ref<tetras_t> tetras):
  num_points {points.rows()},
  num_tetras {tetras.rows()},
  points {points},
  tetras {tetras}
{
  // build vert->tetra neighborhoods

  vt_neib.resize(num_points);
  for (int64_t s = 0; s < num_tetras; ++s) {
    for (int64_t t = 0; t < 4; ++t) {
      vt_neib[tetras(s, t)].insert(s);
    }
  }

  // build vert->vert neighborhoods

  vv_neib.resize(num_points);
  for (int64_t i = 0; i < num_points; ++i) {
    for (int64_t s: vt_neib[i]) {
      for (int64_t t = 0; t < 4; ++t) {
        vv_neib[i].insert(tetras(s, t));
      }
    }
    vv_neib[i].erase(i);
  }

  // Build tetra->tetra neighborhoods
  //
  // We do this by iterating over each tetrahedron, taking each face
  // of a tetrahedron, and intersecting the tetrahedron
  // neighborhoods of each face vertex---the result contains the
  // index of the original tetrahedron, and possibly one other
  // tetrahedron, which is consequently face-adjacent. If there is
  // no other element of this set, then the tetrahedron is at the
  // boundary of the domain.

  tt_neib.resize(num_tetras);
  for (int64_t t = 0; t < num_tetras; ++t) {
    for (int64_t f = 0; f < 4; ++f) {
      auto face = get_face(t, f);

      std::set<int64_t> tmp1;
      std::set_intersection(
        vt_neib[face[0]].begin(), vt_neib[face[0]].end(),
        vt_neib[face[1]].begin(), vt_neib[face[1]].end(),
        std::inserter(tmp1, tmp1.end())
        );

      std::set<int64_t> tmp2;
      std::set_intersection(
        tmp1.begin(), tmp1.end(),
        vt_neib[face[2]].begin(), vt_neib[face[2]].end(),
        std::inserter(tmp2, tmp2.end())
        );

      tmp2.erase(t);

      if (tmp2.size() != 0 && tmp2.size() != 1) {
        printf("BAD: size = %lu\n", tmp2.size());
      }

      tt_neib[t][f] = tmp2.size() == 0 ? NO_INDEX : *tmp2.begin();
    }
  }
}

olim::Face olim::TetraMesh::get_face(int64_t t, int64_t f) const {
  assert(f >= 0);
  assert(f <= 3);
  if (f == 0) {
    return {tetras(t, 0), tetras(t, 1), tetras(t, 2)};
  } else if (f == 1) {
    return {tetras(t, 0), tetras(t, 1), tetras(t, 3)};
  } else if (f == 2) {
    return {tetras(t, 0), tetras(t, 2), tetras(t, 3)};
  } else {
    return {tetras(t, 1), tetras(t, 2), tetras(t, 3)};
  }
}

olim::Face olim::TetraMesh::get_opposite_face(int64_t t, int64_t i) const {
  Face face;
  for (int64_t a = 0, b = 0, j; a < 4; ++a) {
    if ((j = tetras(t, a)) != i) {
      face[b++] = j;
    }
  }
  return face;
}

bool olim::TetraMesh::tetra_contains_point(int64_t t, int64_t i) const {
  return tetras(t, 0) == i || tetras(t, 1) == i ||
    tetras(t, 2) == i || tetras(t, 3) == i;
}

std::array<olim::Face, 3>
olim::TetraMesh::split_tetra(int64_t t, int64_t i) const {
  // find the tetrahedron neighboring `t` that doesn't contain `i`:
  // this is the tetrahedron "opposite" `i`
  int64_t t_new;
  for (int64_t s: tt_neib[t]) {
    if (s != NO_INDEX && !tetra_contains_point(s, i)) {
      t_new = s;
      break;
    }
  }

  // get the face that we're splitting
  Face current_face = get_opposite_face(t, i);

  // find the other faces of the new tetrahedron that aren't equal
  // to the original face that's being split
  std::array<Face, 3> faces;
  for (int64_t f = 0, a = 0; f < 4; ++f) {
    Face face = get_face(t_new, f);
    if (current_face == face) {
      continue;
    }
    faces[a++] = face;
  }

  return faces;
}

#if USING_PYBIND11
void init_face(py::module & m) {
  py::class_<olim::Face>(m, "Face")
    .def(py::init<int64_t, int64_t, int64_t>())
    .def("__repr__", &olim::Face::to_string);
}

void init_tetra_mesh(py::module & m) {
  py::class_<olim::TetraMesh>(m, "TetraMesh")
    .def(py::init<Eigen::Ref<points_t>, Eigen::Ref<tetras_t>>())
    .def_readonly("vt_neib", &olim::TetraMesh::vt_neib)
    .def_readonly("tt_neib", &olim::TetraMesh::tt_neib);
}
#endif
