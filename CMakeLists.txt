cmake_minimum_required(VERSION 3.24)

project(polyscope_rt VERSION 0.1.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(POLYSCOPE_BACKEND_OPENGL3_EGL OFF CACHE STRING "" FORCE)
set(POLYSCOPE_BACKEND_OPENGL_MOCK ON CACHE BOOL "" FORCE)

add_subdirectory(deps/polyscope)
add_subdirectory(deps/geometry_central)
include(CTest)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(METALFX_FRAMEWORK MetalFX)

set(POLYSCOPE_RT_PUBLIC_HEADERS
  include/polyscope/rt/plugin.h
  include/polyscope/rt/polyscope.h
  include/polyscope/rt/options.h
  include/polyscope/rt/view.h
  include/polyscope/rt/simple_triangle_mesh.h
  include/polyscope/rt/surface_mesh.h
  include/polyscope/rt/point_cloud.h
  include/polyscope/rt/curve_network.h
  include/polyscope/rt/volume_mesh.h
)

set(POLYSCOPE_RT_BUNDLED_UI_HEADERS
  deps/polyscope/deps/imgui/imgui/imconfig.h
  deps/polyscope/deps/imgui/imgui/imgui.h
  deps/polyscope/deps/imgui/imgui/imstb_rectpack.h
  deps/polyscope/deps/imgui/imgui/imstb_textedit.h
  deps/polyscope/deps/imgui/imgui/imstb_truetype.h
  deps/polyscope/deps/imgui/implot/implot.h
  deps/polyscope/deps/imgui/ImGuizmo/ImGuizmo.h
)

add_library(
  polyscope_rt_plugin
  SHARED
  src/scene/direct_rt_curve_registry.cpp
  src/plugin/polyscope_rt_runtime.cpp
  src/plugin/polyscope_rt_runtime_gui.cpp
  src/scene/polyscope_scene_snapshot.cpp
  src/scene/ray_tracing_geometry_primitives.cpp
  src/rendering/metal_path_tracer_backend.mm
  src/rendering/toon_shading.cpp
  src/plugin/polyscope_rt_plugin.cpp
)
set_source_files_properties(src/rendering/metal_path_tracer_backend.mm PROPERTIES
  COMPILE_FLAGS "-Wno-deprecated-declarations"
)
add_library(polyscope_rt::plugin ALIAS polyscope_rt_plugin)
set_target_properties(polyscope_rt_plugin PROPERTIES
  EXPORT_NAME plugin
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)
target_sources(
  polyscope_rt_plugin
  PUBLIC
  FILE_SET public_headers
  TYPE HEADERS
  BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
  FILES ${POLYSCOPE_RT_PUBLIC_HEADERS}
)

target_include_directories(
  polyscope_rt_plugin
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/glm>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/imgui/imgui>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/imgui/implot>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/imgui/ImGuizmo>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/stb
)

target_link_libraries(
  polyscope_rt_plugin
  PRIVATE
  polyscope
  ${METAL_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
  ${QUARTZCORE_FRAMEWORK}
  $<$<BOOL:${METALFX_FRAMEWORK}>:${METALFX_FRAMEWORK}>
)

target_compile_definitions(polyscope_rt_plugin PUBLIC GLM_ENABLE_EXPERIMENTAL)

add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/metal_path_tracer_shaders.metallib
  COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/metal_path_tracer_shaders.metal
          -o ${CMAKE_BINARY_DIR}/metal_path_tracer_shaders.air -std=metal4.0 -O2
  COMMAND xcrun -sdk macosx metallib ${CMAKE_BINARY_DIR}/metal_path_tracer_shaders.air
          -o ${CMAKE_BINARY_DIR}/metal_path_tracer_shaders.metallib
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/rendering/metal_path_tracer_shaders.metal
  COMMENT "Compiling Metal shaders"
)
add_custom_target(metal_shaders DEPENDS ${CMAKE_BINARY_DIR}/metal_path_tracer_shaders.metallib)
execute_process(COMMAND xcrun -sdk macosx --find metal RESULT_VARIABLE METAL_FOUND OUTPUT_QUIET ERROR_QUIET)
if(METAL_FOUND EQUAL 0)
  add_dependencies(polyscope_rt_plugin metal_shaders)
endif()
target_compile_definitions(polyscope_rt_plugin PRIVATE
  POLYSCOPE_RT_BUILD_METALLIB_PATH="${CMAKE_BINARY_DIR}/metal_path_tracer_shaders.metallib"
  POLYSCOPE_RT_METALLIB_FILENAME="metal_path_tracer_shaders.metallib")

target_compile_options(
  polyscope_rt_plugin
  PRIVATE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

add_executable(
  polyscope_rt_surface_mesh_example
  examples/surface_mesh_example.cpp
)

target_link_libraries(
  polyscope_rt_surface_mesh_example
  PRIVATE
  polyscope_rt_plugin
)

add_executable(
  geometry_central_rt_example
  examples/geometry_central_example.cpp
)
target_link_libraries(
  geometry_central_rt_example
  PRIVATE
  polyscope_rt_plugin
  geometry-central
)

target_compile_definitions(
  polyscope_rt_surface_mesh_example
  PRIVATE
  GLM_ENABLE_EXPERIMENTAL
  POLYSCOPE_RT_EXAMPLE_DEFAULT_MESH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/bunny.obj"
)
target_compile_definitions(
  geometry_central_rt_example
  PRIVATE
  GLM_ENABLE_EXPERIMENTAL
  POLYSCOPE_RT_EXAMPLE_DEFAULT_MESH="${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/bunny.obj"
)

if(APPLE)
  target_compile_definitions(polyscope_rt_surface_mesh_example PRIVATE GL_SILENCE_DEPRECATION)
  target_compile_definitions(geometry_central_rt_example PRIVATE GL_SILENCE_DEPRECATION)
  target_compile_definitions(polyscope_rt_plugin PRIVATE GL_SILENCE_DEPRECATION)
endif()

set(POLYSCOPE_RT_SHADER_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/polyscope_rt)
set(POLYSCOPE_RT_PACKAGE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/polyscope_rt)
set(POLYSCOPE_RT_INSTALL_COMPONENT polyscope_rt_sdk)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/polyscope_rtConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/polyscope_rtConfig.cmake
  INSTALL_DESTINATION ${POLYSCOPE_RT_PACKAGE_INSTALL_DIR}
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR POLYSCOPE_RT_SHADER_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/polyscope_rtConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(
  TARGETS polyscope_rt_plugin
  EXPORT polyscope_rtTargets
  COMPONENT ${POLYSCOPE_RT_INSTALL_COMPONENT}
  FILE_SET public_headers DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/include/polyscope
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT ${POLYSCOPE_RT_INSTALL_COMPONENT}
)
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/glm/glm
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT ${POLYSCOPE_RT_INSTALL_COMPONENT}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.hpp"
  PATTERN "*.inl"
  PATTERN "*.cppm"
)
install(
  FILES ${POLYSCOPE_RT_BUNDLED_UI_HEADERS}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT ${POLYSCOPE_RT_INSTALL_COMPONENT}
)
install(
  FILES ${CMAKE_BINARY_DIR}/metal_path_tracer_shaders.metallib
  DESTINATION ${POLYSCOPE_RT_SHADER_INSTALL_DIR}
  COMPONENT ${POLYSCOPE_RT_INSTALL_COMPONENT}
)
install(
  EXPORT polyscope_rtTargets
  NAMESPACE polyscope_rt::
  DESTINATION ${POLYSCOPE_RT_PACKAGE_INSTALL_DIR}
  COMPONENT ${POLYSCOPE_RT_INSTALL_COMPONENT}
)
install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/polyscope_rtConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/polyscope_rtConfigVersion.cmake
  DESTINATION ${POLYSCOPE_RT_PACKAGE_INSTALL_DIR}
  COMPONENT ${POLYSCOPE_RT_INSTALL_COMPONENT}
)

function(polyscope_rt_add_private_test target_name source_file)
  add_executable(${target_name} ${source_file})
  target_link_libraries(${target_name} PRIVATE polyscope_rt_plugin)
  target_include_directories(
    ${target_name}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

if(BUILD_TESTING)
  polyscope_rt_add_private_test(toon_shading_test tests/toon_shading_test.cpp)
  polyscope_rt_add_private_test(polyscope_scene_snapshot_test tests/polyscope_scene_snapshot_test.cpp)
  polyscope_rt_add_private_test(polyscope_rt_namespace_compatibility_test tests/polyscope_rt_namespace_compatibility_test.cpp)
  polyscope_rt_add_private_test(metal_backend_smoke_test tests/metal_backend_smoke_test.cpp)
  polyscope_rt_add_private_test(metal_shader_unit_test tests/metal_shader_unit_test.cpp)
  polyscope_rt_add_private_test(point_cloud_rendering_test tests/point_cloud_rendering_test.cpp)
endif()
