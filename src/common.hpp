#pragma once

#include <unordered_set>
#include <vector>

#include <boost/optional.hpp>

#include <Eigen/Dense>

using points_t = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using slowness_t = Eigen::ArrayXd;
using tetras_t = Eigen::Matrix<int64_t, Eigen::Dynamic, 4, Eigen::RowMajor>;
using values_t = Eigen::ArrayXd;

template <class T>
using opt_t = boost::optional<T>;

template <class T>
constexpr T inf() {
  return std::numeric_limits<T>::infinity();
}

constexpr int64_t NO_INDEX = -1;
constexpr double DEFAULT_TOL = std::numeric_limits<double>::epsilon();
