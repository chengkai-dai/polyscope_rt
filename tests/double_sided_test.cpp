#include <exception>
#include <iostream>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering_test_utils.h"
#include "test_helpers.h"

namespace {

void testDoubleSidedBackFaceShading(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  auto cfg = standardConfig();
  cfg.lighting.mainLightDirection = {0.0f, 0.0f, -1.0f};
  cfg.lighting.mainLightIntensity = 1.5f;

  auto renderBackFaceQuad = [&](bool doubleSided, uint32_t hash) {
    rt::RTScene scene;
    rt::RTMesh quad = makeCameraFacingQuad(doubleSided ? "double_sided_quad" : "single_sided_quad", false,
                                           {0.9f, 0.9f, 0.9f, 1.0f});
    quad.doubleSided = doubleSided;
    scene.meshes.push_back(quad);
    scene.hash = hash;
    backend.setScene(scene);
    backend.resetAccumulation();
    backend.renderIteration(cfg);
    return backend.downloadRenderBuffer();
  };

  const rt::RenderBuffer singleSided = renderBackFaceQuad(false, 770);
  const rt::RenderBuffer doubleSided = renderBackFaceQuad(true, 771);

  double singleLum = 0.0;
  double doubleLum = 0.0;
  for (const auto& c : singleSided.color) singleLum += c.x + c.y + c.z;
  for (const auto& c : doubleSided.color) doubleLum += c.x + c.y + c.z;

  require(doubleLum > singleLum + 5.0,
          "double-sided back-face quad should become visibly brighter than the single-sided version");
}

void testDoubleSidedEmissiveBackFaceVisible(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  auto cfg = standardConfig();
  cfg.lighting.mainLightIntensity = 0.0f;

  auto renderEmissiveBackFace = [&](bool doubleSided, uint32_t hash) {
    rt::RTScene scene;
    rt::RTMesh quad = makeCameraFacingQuad(doubleSided ? "double_sided_emissive" : "single_sided_emissive", false,
                                           {0.0f, 0.0f, 0.0f, 1.0f});
    quad.doubleSided = doubleSided;
    quad.emissiveFactor = {6.0f, 3.0f, 1.0f};
    scene.meshes.push_back(quad);
    scene.hash = hash;
    backend.setScene(scene);
    backend.resetAccumulation();
    backend.renderIteration(cfg);
    return backend.downloadRenderBuffer();
  };

  const rt::RenderBuffer singleSided = renderEmissiveBackFace(false, 780);
  const rt::RenderBuffer doubleSided = renderEmissiveBackFace(true, 781);
  const glm::vec3 singlePatchColor = averageCenterPatchVec3(cam, singleSided.color);
  const glm::vec3 doublePatchColor = averageCenterPatchVec3(cam, doubleSided.color);
  const double singlePatch = singlePatchColor.x + singlePatchColor.y + singlePatchColor.z;
  const double doublePatch = doublePatchColor.x + doublePatchColor.y + doublePatchColor.z;
  require(doublePatch > singlePatch + 0.05, "double-sided emissive quad should stay visible from the back face");
  require(averageCenterPatchCoverage(cam, doubleSided.objectId) > 0.8,
          "double-sided emissive quad should still register as visible geometry");
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    testDoubleSidedBackFaceShading(*backend);
    testDoubleSidedEmissiveBackFaceVisible(*backend);
    std::cout << "double_sided_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "double_sided_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "double_sided_test FAILED: " << msg << std::endl;
    return 1;
  }
}
