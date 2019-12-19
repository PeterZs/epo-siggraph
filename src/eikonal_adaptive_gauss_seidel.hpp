#pragma once

#include <queue>

#include "eikonal.hpp"

namespace olim
{

struct EikonalAdaptiveGaussSeidel: public Eikonal
{
  std::queue<int64_t> queue;
  std::unordered_set<int64_t> enqueued;
  std::unordered_set<int64_t> boundary;

  double tol;

  EikonalAdaptiveGaussSeidel(Eigen::Ref<points_t> points,
                             Eigen::Ref<tetras_t> tetras,
                             Eigen::Ref<slowness_t> slowness,
                             double tol = DEFAULT_TOL);

  virtual void build_stencils();
  virtual void push_back(int64_t i);
  virtual void pop_front();
  virtual opt_t<int64_t> front() const;
  virtual void add_boundary_point(int64_t i, double value);
  virtual void commit();
  virtual bool do_line_update(int64_t i0) const;
  virtual bool do_tri_update(Edge edge) const;
  virtual bool do_tetra_update(Face face) const;
  virtual bool step();
};

}
