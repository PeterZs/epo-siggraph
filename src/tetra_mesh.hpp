#pragma once

#include "common.hpp"

#include <boost/functional/hash.hpp>

namespace olim
{

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

  inline bool operator==(Edge const & edge) const {
    return i0 == edge.i0 && i1 == edge.i1;
  }
};

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

  inline int64_t & operator[](int64_t pos) {
    return data[pos];
  }

  inline bool operator==(Face const & face) const {
    return i0 == face.i0 && i1 == face.i1 && i2 == face.i2;
  }

  inline std::array<Edge, 3> get_edges() const {
    return {Edge {i0, i1}, Edge {i1, i2}, Edge {i2, i0}};
  }
};

struct TetraMesh
{
  int64_t num_points;
  int64_t num_tetras;

  Eigen::Ref<points_t> points;
  Eigen::Ref<tetras_t> tetras;

  std::vector<std::set<int64_t>> vv_neib;
  std::vector<std::set<int64_t>> vt_neib;
  std::vector<std::array<int64_t, 4>> tt_neib;

  TetraMesh(Eigen::Ref<points_t> points, Eigen::Ref<tetras_t> tetras);

  Face get_face(int64_t t, int64_t f) const;
  Face get_opposite_face(int64_t t, int64_t i) const;
  bool tetra_contains_point(int64_t t, int64_t i) const;
  std::array<Face, 3> split_tetra(int64_t t, int64_t i) const;
};

}

namespace std
{

template<> struct hash<olim::Edge>
{
  std::size_t operator()(olim::Edge const & edge) const noexcept {
    std::size_t seed = 0;
    boost::hash_combine(seed, edge.i0);
    boost::hash_combine(seed, edge.i1);
    return seed;
  }
};

template<> struct hash<olim::Face>
{
  std::size_t operator()(olim::Face const & face) const noexcept {
    std::size_t seed = 0;
    boost::hash_combine(seed, face.i0);
    boost::hash_combine(seed, face.i1);
    boost::hash_combine(seed, face.i2);
    return seed;
  }
};

}
