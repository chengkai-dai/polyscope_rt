#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "glm/glm.hpp"

#include "polyscope/rt/point_cloud.h"
#include "polyscope/rt/polyscope.h"
#include "polyscope/rt/simple_triangle_mesh.h"

#include "scene/polyscope_scene_snapshot.h"
#include "test_helpers.h"

namespace {

void requireVecNear(glm::vec3 a, glm::vec3 b, float tol, const char* message) {
  requireNear(a.x, b.x, tol, message);
  requireNear(a.y, b.y, tol, message);
  requireNear(a.z, b.z, tol, message);
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
    mesh->setMaterial("wax");

    auto* cloud = polyscope::rt::registerPointCloud("compat_points", std::vector<glm::vec3>{{0.0f, 0.0f, 0.0f}});
    cloud->setPointColor(glm::vec3(0.9f, 0.4f, 0.2f));
    cloud->setPointRadius(0.2, false);

    require(polyscope::rt::isEnabled(), "ray tracing mode should be enabled");
    require(polyscope::rt::hasSimpleTriangleMesh("compat_mesh"), "compat mesh should be visible through rt namespace");
    require(polyscope::rt::hasPointCloud("compat_points"), "compat point cloud should be visible through rt namespace");
    require(!polyscope::rt::state::structures.empty(), "structure registry should be reachable via rt namespace");

    PolyscopeSceneSnapshot snapshot = capturePolyscopeSceneSnapshot();
    require(snapshot.supportedStructureCount == 2, "expected both rt namespace structures in the snapshot");
    require(snapshot.scene.meshes.size() == 1, "expected one RT mesh (triangle mesh only; point cloud → pointClouds)");
    require(snapshot.scene.pointClouds.size() == 1, "expected one RTPointCloud from the point cloud structure");

    polyscope::rt::setMaterial("compat_mesh", 1.0f, 0.08f);
    requireVecNear(mesh->getSurfaceColor(), glm::vec3(0.3f, 0.6f, 0.9f), 1e-5f,
                   "RT-side material override should not mutate raster color while RT is enabled");
    require(mesh->getMaterial() == "wax",
            "RT-side material override should not mutate raster material while RT is enabled");

    polyscope::rt::setBaseColor("compat_mesh", glm::vec4(0.2f, 0.4f, 0.6f, 1.0f));
    requireVecNear(mesh->getSurfaceColor(), glm::vec3(0.2f, 0.4f, 0.6f), 1e-5f,
                   "setBaseColor should immediately synchronize the underlying Polyscope mesh color");

    polyscope::rt::disable();
    requireVecNear(mesh->getSurfaceColor(), glm::vec3(0.2f, 0.4f, 0.6f), 1e-5f,
                   "Disabling RT should preserve colors explicitly synchronized to Polyscope");
    require(mesh->getMaterial() == "wax",
            "Disabling RT should not rewrite the underlying raster material for RT-only overrides");

    polyscope::rt::enable();
    requireVecNear(mesh->getSurfaceColor(), glm::vec3(0.2f, 0.4f, 0.6f), 1e-5f,
                   "Re-enabling RT should preserve colors explicitly set through the RT API");
    requireNear(mesh->getTransparency(), 1.0f, 1e-5f,
                "Re-enabling RT should restore the original raster mesh transparency");
    require(mesh->getMaterial() == "wax",
            "Re-enabling RT should restore the original raster mesh material");

    polyscope::rt::clearMaterialOverride("compat_mesh");
    polyscope::rt::disable();
    requireVecNear(mesh->getSurfaceColor(), glm::vec3(0.2f, 0.4f, 0.6f), 1e-5f,
                   "Clearing the RT override should keep the synchronized raster color in place");
    require(mesh->getMaterial() == "wax",
            "Clearing the RT override should not mutate the raster material state");

    polyscope::rt::shutdown();
    std::cout << "polyscope_rt_namespace_compatibility_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "polyscope_rt_namespace_compatibility_test failed: " << e.what() << std::endl;
    return 1;
  }
}
