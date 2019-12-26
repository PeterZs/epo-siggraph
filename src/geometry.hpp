#pragma once

#include "common.hpp"

namespace olim {

namespace geometry {

std::vector<std::array<int64_t, 3>> spherical_delaunay(Eigen::Ref<points_t> points);

}

}
