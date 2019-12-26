#pragma once

#include "common.hpp"
#include "tetra_mesh.hpp"

namespace olim
{

struct Eikonal
{
  Eigen::Ref<points_t> points;
  Eigen::Ref<slowness_t> slowness;

  values_t values;

  TetraMesh mesh;
  std::vector<std::unordered_set<Face>> stencils;

  Eikonal(Eigen::Ref<points_t> points, Eigen::Ref<tetras_t> tetras,
          Eigen::Ref<slowness_t> slowness);

  virtual ~Eikonal() = default;

  virtual void build_stencils() = 0;
  virtual void push_back(int64_t i) = 0;
  virtual void pop_front() = 0;
  virtual opt_t<int64_t> front() const = 0;
  virtual void add_boundary_point(int64_t i, double value = 0.0) = 0;
  virtual void commit() = 0;
  virtual bool do_line_update(int64_t i0) const = 0;
  virtual bool do_tri_update(Edge edge) const = 0;
  virtual bool do_tetra_update(Face face) const = 0;
  virtual bool step() = 0;

  double line(int64_t i, int64_t i0) const;
  std::pair<double, double> tri(int64_t i, Edge edge) const;
  std::pair<double, Eigen::Vector2d> tetra(int64_t i, Face face) const;

  bool in_simplex(double lam) const;
  bool in_simplex(Eigen::Array2d lam) const;

  double update(int64_t i) const;
  void solve();
};

}
