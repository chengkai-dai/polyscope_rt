// Tests for native Metal point cloud rendering:
//   1. RTPointCloud is correctly populated from a Polyscope PointCloud.
//   2. RTScene::pointClouds is used instead of generating triangle meshes.
//   3. Empty point clouds are skipped.
//   4. Scene hash changes when point cloud properties change.

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"

#include "rendering/ray_tracing_types.h"
#include "scene/polyscope_scene_snapshot.h"
#include "test_helpers.h"

namespace {

} // namespace

int main() {
  try {
    polyscope::init("openGL_mock");
    polyscope::removeEverything();

    // -----------------------------------------------------------------------
    // Test 1: basic RTPointCloud extraction
    // -----------------------------------------------------------------------
    std::vector<glm::vec3> pts = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
    };
    auto* cloud = polyscope::registerPointCloud("pt_test", pts);
    cloud->setPointColor(glm::vec3(0.8f, 0.2f, 0.1f));
    cloud->setPointRadius(0.05f, false);

    PolyscopeSceneSnapshot snap = capturePolyscopeSceneSnapshot();

    require(snap.scene.meshes.empty(),
            "Point cloud must not generate triangle meshes anymore");
    require(snap.scene.pointClouds.size() == 1,
            "Expected exactly one RTPointCloud");
    require(snap.supportedStructureCount == 1,
            "Point cloud should count as one supported structure");

    const rt::RTPointCloud& rtpc = snap.scene.pointClouds.front();
    require(rtpc.name == "pt_test", "RTPointCloud name mismatch");
    require(rtpc.centers.size() == 3, "RTPointCloud should have 3 centers");
    requireNear(rtpc.radius, 0.05f, 1e-5f, "RTPointCloud radius mismatch");
    requireNear(rtpc.baseColor.r, 0.8f, 1e-5f, "RTPointCloud color.r mismatch");
    requireNear(rtpc.baseColor.g, 0.2f, 1e-5f, "RTPointCloud color.g mismatch");
    requireNear(rtpc.baseColor.b, 0.1f, 1e-5f, "RTPointCloud color.b mismatch");

    // World-space centers: identity transform, so centers == pts
    for (size_t i = 0; i < pts.size(); ++i) {
      requireNear(rtpc.centers[i].x, pts[i].x, 1e-4f, "center.x mismatch");
      requireNear(rtpc.centers[i].y, pts[i].y, 1e-4f, "center.y mismatch");
      requireNear(rtpc.centers[i].z, pts[i].z, 1e-4f, "center.z mismatch");
    }

    // -----------------------------------------------------------------------
    // Test 2: empty point cloud is skipped
    // -----------------------------------------------------------------------
    polyscope::removeEverything();
    std::vector<glm::vec3> emptyPts;
    polyscope::registerPointCloud("empty_pts", emptyPts);

    PolyscopeSceneSnapshot emptySnap = capturePolyscopeSceneSnapshot();
    require(emptySnap.scene.pointClouds.empty(),
            "Empty point cloud must not produce an RTPointCloud");
    require(emptySnap.supportedStructureCount == 0,
            "Empty point cloud should not count as a supported structure");

    // -----------------------------------------------------------------------
    // Test 3: scene hash changes when radius changes
    // -----------------------------------------------------------------------
    polyscope::removeEverything();
    auto* cloud2 = polyscope::registerPointCloud(
        "hash_test", std::vector<glm::vec3>{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}});
    cloud2->setPointRadius(0.10f, false);

    PolyscopeSceneSnapshot snapA = capturePolyscopeSceneSnapshot();
    cloud2->setPointRadius(0.20f, false);
    PolyscopeSceneSnapshot snapB = capturePolyscopeSceneSnapshot();

    require(snapA.scene.hash != snapB.scene.hash,
            "Scene hash must change when point radius changes");

    // -----------------------------------------------------------------------
    // Test 4: disabled point cloud is not exported
    // -----------------------------------------------------------------------
    cloud2->setEnabled(false);
    PolyscopeSceneSnapshot disabledSnap = capturePolyscopeSceneSnapshot();
    require(disabledSnap.scene.pointClouds.empty(),
            "Disabled point cloud must not be exported");

    // -----------------------------------------------------------------------
    // Test 5: colormap enable/disable changes the scene hash
    // Regression for the bug where disabling a scalar quantity colormap did not
    // revert the point cloud to its base color because pc.colors was not hashed.
    // -----------------------------------------------------------------------
    polyscope::removeEverything();
    std::vector<glm::vec3> pts3 = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    auto* cloud3 = polyscope::registerPointCloud("colormap_hash_test", pts3);
    auto* qty = cloud3->addScalarQuantity("scalar", std::vector<float>{0.0f, 1.0f});

    qty->setEnabled(true);
    PolyscopeSceneSnapshot snapColored = capturePolyscopeSceneSnapshot();
    require(!snapColored.scene.pointClouds.front().colors.empty(),
            "Enabled colormap must populate per-point colors");

    qty->setEnabled(false);
    PolyscopeSceneSnapshot snapBase = capturePolyscopeSceneSnapshot();
    require(snapBase.scene.pointClouds.front().colors.empty(),
            "Disabled colormap must produce empty per-point colors (revert to baseColor)");

    require(snapColored.scene.hash != snapBase.scene.hash,
            "Scene hash must differ between colormap-enabled and colormap-disabled states");

    polyscope::shutdown();
    std::cout << "point_cloud_rendering_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "point_cloud_rendering_test FAILED: " << e.what() << std::endl;
    return 1;
  }
}
