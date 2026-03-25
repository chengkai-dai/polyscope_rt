cmake_minimum_required(VERSION 3.24)

project(polyscope_rt_package_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(polyscope_rt CONFIG REQUIRED)

add_executable(polyscope_rt_package_smoke main.cpp)
target_link_libraries(polyscope_rt_package_smoke PRIVATE polyscope_rt::plugin)

polyscope_rt_copy_runtime_assets(polyscope_rt_package_smoke)
