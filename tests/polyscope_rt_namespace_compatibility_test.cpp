#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "glm/glm.hpp"

#include "polyscope/rt/point_cloud.h"
#include "polyscope/rt/polyscope.h"
#include "polyscope/rt/simple_triangle_mesh.h"

#include "scene/polyscope_scene_snapshot.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
  try {
    polyscope::rt::options::programName = "polyscope::rt namespace test";
    polyscope::rt::options::alwaysRedraw = true;
    polyscope::rt::options::maxBounces = 4;

    polyscope::rt::init("openGL_mock");
    require(polyscope::rt::isEnabled(), "ray tracing mode should default to enabled after init");
    polyscope::rt::removeEverything();

    std::vector<glm::vec3> vertices{{-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    std::vector<glm::uvec3> faces{glm::uvec3(0, 1, 2)};
    auto* mesh = polyscope::rt::registerSimpleTriangleMesh("compat_mesh", vertices, faces);
    mesh->setSurfaceColor(glm::vec3(0.3f, 0.6f, 0.9f));

    auto* cloud = polyscope::rt::registerPointCloud("compat_points", std::vector<glm::vec3>{{0.0f, 0.0f, 0.0f}});
    cloud->setPointColor(glm::vec3(0.9f, 0.4f, 0.2f));
    cloud->setPointRadius(0.2, false);

    require(polyscope::rt::isEnabled(), "ray tracing mode should be enabled");
    require(polyscope::rt::hasSimpleTriangleMesh("compat_mesh"), "compat mesh should be visible through rt namespace");
    require(polyscope::rt::hasPointCloud("compat_points"), "compat point cloud should be visible through rt namespace");
    require(!polyscope::rt::state::structures.empty(), "structure registry should be reachable via rt namespace");

    PolyscopeSceneSnapshot snapshot = capturePolyscopeSceneSnapshot();
    require(snapshot.supportedMeshCount == 2, "expected both rt namespace structures in the snapshot");
    require(snapshot.scene.meshes.size() == 2, "expected two RT meshes from rt namespace usage");

    polyscope::rt::shutdown();
    std::cout << "polyscope_rt_namespace_compatibility_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "polyscope_rt_namespace_compatibility_test failed: " << e.what() << std::endl;
    return 1;
  }
}
