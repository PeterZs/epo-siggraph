#include <pybind11/pybind11.h>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <list>
#include <queue>
#include <unordered_set>
#include <vector>

#include <boost/functional/hash.hpp>
#include <boost/optional.hpp>

template <class T>
constexpr T inf() {
  return std::numeric_limits<T>::infinity();
}

constexpr int64_t NO_INDEX = -1;

constexpr double DEFAULT_TOL = std::numeric_limits<double>::epsilon();

enum class State {
  Far,
  Trial,
  Valid,
  Boundary,
  Shadow
};

using points_t = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using tetras_t = Eigen::Matrix<int64_t, Eigen::Dynamic, 4, Eigen::RowMajor>;
using states_t = Eigen::Array<State, Eigen::Dynamic, 1>;
using values_t = Eigen::ArrayXd;
using slowness_t = Eigen::ArrayXd;
using tetra_nb_t = std::vector<std::list<int64_t>>;

struct Edge
{
  union {
    struct {
      int64_t i0;
      int64_t i1;
    };
    int64_t data[2];
  };

  Edge(): i0 {NO_INDEX}, i1 {NO_INDEX} {}
  Edge(int64_t i0, int64_t i1): i0 {i0}, i1 {i1} {}

  bool operator==(Edge const & edge) const {
    return i0 == edge.i0 && i1 == edge.i1;
  }
};

namespace std
{
  template<> struct hash<Edge>
  {
    std::size_t operator()(Edge const & edge) const noexcept {
      std::size_t seed = 0;
      boost::hash_combine(seed, edge.i0);
      boost::hash_combine(seed, edge.i1);
      return seed;
    }
  };
}

struct Face
{
  union {
    struct {
      int64_t i0;
      int64_t i1;
      int64_t i2;
    };
    int64_t data[3];
  };

  Face(): i0 {NO_INDEX}, i1 {NO_INDEX}, i2 {NO_INDEX} {}
  Face(int64_t i0, int64_t i1, int64_t i2): i0 {i0}, i1 {i1}, i2 {i2} {}

  int64_t & operator[](int64_t pos) {
    return data[pos];
  }

  bool operator==(Face const & face) const {
    return i0 == face.i0 && i1 == face.i1 && i2 == face.i2;
  }

  std::array<Edge, 3> get_edges() const {
    return {Edge {i0, i1}, Edge {i1, i2}, Edge {i2, i0}};
  }
};

namespace std
{
  template<> struct hash<Face>
  {
    std::size_t operator()(Face const & face) const noexcept {
      std::size_t seed = 0;
      boost::hash_combine(seed, face.i0);
      boost::hash_combine(seed, face.i1);
      boost::hash_combine(seed, face.i2);
      return seed;
    }
  };
}

struct TetraMesh
{
  int64_t num_points;
  int64_t num_tetras;

  Eigen::Ref<points_t> points;
  Eigen::Ref<tetras_t> tetras;

  std::vector<std::set<int64_t>> vv_neib;
  std::vector<std::set<int64_t>> vt_neib;
  std::vector<std::array<int64_t, 4>> tt_neib;

  TetraMesh(Eigen::Ref<points_t> points, Eigen::Ref<tetras_t> tetras):
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

  Face get_face(int64_t t, int64_t f) const {
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

  Face get_opposite_face(int64_t t, int64_t i) {
    Face face;
    for (int64_t a = 0, b = 0, j; a < 4; ++a) {
      if ((j = tetras(t, b)) != i) {
        face[b++] = j;
      }
    }
    return face;
  }

  bool tetra_contains_point(int64_t t, int64_t i) {
    return tetras(t, 0) == i || tetras(t, 1) == i ||
      tetras(t, 2) == i || tetras(t, 3) == i;
  }

  std::array<Face, 3> split_tetra(int64_t t, int64_t i) {
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
};

struct Eikonal
{
  Eigen::Ref<points_t> points;
  Eigen::Ref<slowness_t> slowness;

  values_t values;

  TetraMesh mesh;
  std::vector<std::unordered_set<Face>> stencils;

  Eikonal(Eigen::Ref<points_t> points,
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

  virtual ~Eikonal() = default;

  virtual void build_stencils() = 0;
  virtual void push_back(int64_t i) = 0;
  virtual void pop_front() = 0;
  virtual boost::optional<int64_t> front() const = 0;
  virtual void add_boundary_point(int64_t i, double value) = 0;
  virtual void commit() = 0;
  virtual bool do_line_update(int64_t i0) const = 0;
  virtual bool do_tri_update(Edge edge) const = 0;
  virtual bool do_tetra_update(Face face) const = 0;
  virtual bool step() = 0;

  double line(int64_t i, int64_t i0) const {
    double L = (points.row(i0) - points.row(i)).norm();
    double s = slowness[i];
    double s0 = slowness[i0];
    double u0 = values[i0];
    return u0 + (s + s0)*L/2;
  }

  std::pair<double, double> tri(int64_t i, Edge edge) const {
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

  std::pair<double, Eigen::Vector2d> tetra(int64_t i, Face face) const {
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

  bool in_simplex(double lam) const {
    return 0 <= lam && lam <= 1;
  }

  bool in_simplex(Eigen::Array2d lam) const {
    return lam(0) >= 0 && lam(1) >= 0 && lam(0) + lam(1) <= 1;
  }

  double update(int64_t i) const {
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

  void solve() {
    while (step()) {}
  }
};

struct EikonalAdaptiveGaussSeidel: public Eikonal
{
  std::queue<int64_t> queue;
  std::unordered_set<int64_t> enqueued;
  std::unordered_set<int64_t> boundary;

  double tol;

  EikonalAdaptiveGaussSeidel(Eigen::Ref<points_t> points,
                             Eigen::Ref<tetras_t> tetras,
                             Eigen::Ref<slowness_t> slowness,
                             double tol = DEFAULT_TOL):
    Eikonal {points, tetras, slowness},
    tol {tol}
  {
    build_stencils();
  }

  virtual void build_stencils() {
    stencils.resize(mesh.num_points);
    for (int64_t i = 0; i < mesh.num_points; ++i) {
      for (int64_t t: mesh.vt_neib[i]) {
        auto face = mesh.get_opposite_face(t, i);
        stencils[i].insert(face);
      }
    }
  }

  virtual void push_back(int64_t i) {
    if (enqueued.find(i) == enqueued.end()) {
      queue.push(i);
      enqueued.insert(i);
    }
  }

  virtual void pop_front() {
    enqueued.erase(queue.front());
    queue.pop();
  }

  virtual boost::optional<int64_t> front() const {
    return queue.empty() ?
      boost::none :
      boost::optional<int64_t> {queue.front()};
  }

  virtual void add_boundary_point(int64_t i, double value) {
    values[i] = value;
    boundary.insert(i);
  }

  virtual void commit() {
    for (int64_t i: boundary) {
      for (int64_t j: mesh.vv_neib[i]) {
        if (boundary.find(j) == boundary.end()) {
          push_back(j);
        }
      }
    }
  }

  virtual bool do_line_update(int64_t i0) const {
    return std::isfinite(values[i0]);
  }

  virtual bool do_tri_update(Edge edge) const {
    return std::isfinite(values[edge.i0]) && std::isfinite(values[edge.i1]);
  }

  virtual bool do_tetra_update(Face face) const {
    return std::isfinite(values[face.i0]) && std::isfinite(values[face.i1]) &&
      std::isfinite(values[face.i2]);
  }

  virtual bool step() {
    int64_t i;
    if (boost::optional<int64_t> opt_i = front()) {
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
};

struct EikonalDijkstraLike: public Eikonal
{
  states_t states;
  std::vector<int64_t> heap;
  std::set<int64_t> modified_stencils;

  EikonalDijkstraLike(Eigen::Ref<points_t> points,
                      Eigen::Ref<tetras_t> tetras,
                      Eigen::Ref<slowness_t> slowness):
    Eikonal {points, tetras, slowness}
  {
    build_stencils();

    states.setConstant(mesh.num_points, State::Far);
  }

  bool is_causal(int64_t i, Face face) {
    throw std::runtime_error("blah");
  }

  virtual void build_stencils() {
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

  virtual void push_back(int64_t i) {
    heap.push_back(i);
    std::push_heap(
      heap.begin(),
      heap.end(),
      [&] (int64_t const & i, int64_t const & j) {
        return values[i] > values[j];
      }
    );
  }

  virtual void pop_front() {
    std::pop_heap(
      heap.begin(),
      heap.end(),
      [&] (int64_t const & i, int64_t const & j) {
        return values[i] > values[j];
      }
    );
    heap.pop_back();
  }

  virtual boost::optional<int64_t> front() const {
    return heap.empty() ?
      boost::none :
      boost::optional<int64_t> {heap[0]};
  }

  virtual void add_boundary_point(int64_t i, double value) {
    values[i] = value;
    states[i] = State::Trial;
  }

  virtual void commit() {
    for (int64_t i = 0; i < mesh.num_points; ++i) {
      if (states(i) == State::Trial) {
        push_back(i);
      }
    }
  }

  inline bool is_valid(int64_t i) const {
    return states[i] == State::Valid;
  }

  virtual bool do_line_update(int64_t i0) const {
    return is_valid(i0);
  }

  virtual bool do_tri_update(Edge edge) const {
    return is_valid(edge.i0) && is_valid(edge.i1);
  }

  virtual bool do_tetra_update(Face face) const {
    return is_valid(face.i0) && is_valid(face.i1) && is_valid(face.i2);
  }

  virtual bool step() {
    int64_t i;
    if (boost::optional<int64_t> opt_i = front()) {
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
};

PYBIND11_MAKE_OPAQUE(std::vector<std::list<int64_t>>);
PYBIND11_MAKE_OPAQUE(std::vector<int64_t>);

PYBIND11_MODULE(python_example, m) {
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. CURRENTMODULE:: python_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

  py::bind_vector<std::vector<std::list<int64_t>>>(m, "Neighbors");
  py::bind_vector<std::vector<State>>(m, "StateVector");
  py::bind_vector<std::vector<int64_t>>(m, "Int64Vector");

  py::enum_<State>(m, "State")
    .value("Far", State::Far)
    .value("Trial", State::Trial)
    .value("Valid", State::Valid)
    .value("Boundary", State::Boundary)
    .value("Shadow", State::Shadow)
    .export_values();

  py::class_<TetraMesh>(m, "TetraMesh")
    .def(py::init<Eigen::Ref<points_t>, Eigen::Ref<tetras_t>>())
    .def_readonly("vt_neib", &TetraMesh::vt_neib)
    .def_readonly("tt_neib", &TetraMesh::tt_neib);

  py::class_<Eikonal>(m, "Eikonal")
    .def("line", &Eikonal::line)
    .def("tri", &Eikonal::tri)
    .def("tetra", &Eikonal::tetra)
    .def("update", &Eikonal::update)
    .def("add_boundary_point", &Eikonal::add_boundary_point)
    .def("commit", &Eikonal::commit)
    .def("step", &Eikonal::step)
    .def("solve", &Eikonal::solve)
    .def_readonly("mesh", &Eikonal::mesh)
    .def_readonly("stencils", &Eikonal::stencils)
    .def_readonly("values", &Eikonal::values);

  py::class_<EikonalAdaptiveGaussSeidel, Eikonal>(m, "EikonalAdaptiveGaussSeidel")
    .def(py::init<
           Eigen::Ref<points_t>,
           Eigen::Ref<tetras_t>,
           Eigen::Ref<slowness_t>,
           double
         >())
    .def_readonly("queue", &EikonalAdaptiveGaussSeidel::queue)
    .def_readonly("enqueued", &EikonalAdaptiveGaussSeidel::enqueued)
    .def_readonly("boundary", &EikonalAdaptiveGaussSeidel::boundary)
    .def_readonly("tol", &EikonalAdaptiveGaussSeidel::tol);

  py::class_<EikonalDijkstraLike, Eikonal>(m, "EikonalDijkstraLike")
    .def(py::init<
           Eigen::Ref<points_t>,
           Eigen::Ref<tetras_t>,
           Eigen::Ref<slowness_t>
         >())
    .def_readonly("states", &EikonalDijkstraLike::states)
    .def_readonly("heap", &EikonalDijkstraLike::heap)
    .def_readonly("modified_stencils", &EikonalDijkstraLike::modified_stencils);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
