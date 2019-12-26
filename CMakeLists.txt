project (olim_tetra)

cmake_minimum_required (VERSION 3.9)

find_package (Boost REQUIRED)
find_package (CGAL REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -Wextra -Werror -std=c++14")
set (CMAKE_CXX_VISIBILITY_PRESET hidden)

add_subdirectory (pybind11)
pybind11_add_module (olim MODULE
  src/olim.cpp
  src/common.hpp
  src/eikonal.cpp
  src/eikonal.hpp
  src/eikonal_adaptive_gauss_seidel.cpp
  src/eikonal_adaptive_gauss_seidel.hpp
  src/eikonal_dijkstra_like.cpp
  src/eikonal_dijkstra_like.hpp
  src/geometry.hpp
  src/geometry.cpp
  src/pybind11_common.hpp
  src/tetra_mesh.cpp
  src/tetra_mesh.hpp
  )
target_compile_definitions (olim PRIVATE USING_PYBIND11=1)
target_include_directories (olim PRIVATE ${Boost_INCLUDE_DIRS})
target_link_directories (olim PRIVATE ${Boost_LIBRARY_DIRS})
target_link_libraries (olim PRIVATE ${Boost_LIBRARIES})
target_link_libraries(olim PRIVATE CGAL::CGAL)
target_link_libraries (olim PRIVATE Eigen3::Eigen)
