#pragma once

#include "eikonal.hpp"

namespace olim {

enum class State {
  Far,
  Trial,
  Valid,
  Boundary,
  Shadow
};

struct EikonalDijkstraLike: public Eikonal
{
  std::vector<State> states;
  std::vector<int64_t> heap;
  std::set<int64_t> modified_stencils;

  EikonalDijkstraLike(Eigen::Ref<points_t> points,
                      Eigen::Ref<tetras_t> tetras,
                      Eigen::Ref<slowness_t> slowness);

  bool is_causal(int64_t i, Face face);
  virtual void build_stencils();
  virtual void push_back(int64_t i);
  virtual void pop_front();
  virtual opt_t<int64_t> front() const;
  virtual void add_boundary_point(int64_t i, double value = 0.0);
  virtual void commit();
  inline bool is_valid(int64_t i) const;
  virtual bool do_line_update(int64_t i0) const;
  virtual bool do_tri_update(Edge edge) const;
  virtual bool do_tetra_update(Face face) const;
  virtual bool step();
};

}
