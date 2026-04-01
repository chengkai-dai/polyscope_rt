#include <exception>
#include <iostream>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering_test_utils.h"
#include "test_helpers.h"

namespace {

void testAreaLightVisibleOnMissPath(rt::IRayTracingBackend& backend) {
  auto cam = makeCamera();
  cam.position = {0.0f, 0.0f, 3.0f};
  cam.lookDir = glm::normalize(glm::vec3(0.0f, 0.0f, -1.0f));
  cam.upDir = {0.0f, 1.0f, 0.0f};
  cam.rightDir = {1.0f, 0.0f, 0.0f};
  cam.viewMatrix = glm::lookAt(cam.position, glm::vec3(0.0f, 0.0f, 0.0f), cam.upDir);

  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene empty;
  empty.hash = 900;
  backend.setScene(empty);
  backend.resetAccumulation();

  auto cfg = standardConfig();
  cfg.maxBounces = 1;
  cfg.lighting.backgroundColor = {0.0f, 0.0f, 0.0f};
  cfg.lighting.mainLightIntensity = 0.0f;
  cfg.lighting.environmentIntensity = 0.0f;
  cfg.lighting.enableAreaLight = true;
  cfg.lighting.areaLightCenter = {0.0f, 0.0f, 0.0f};
  cfg.lighting.areaLightU = {0.9f, 0.0f, 0.0f};
  cfg.lighting.areaLightV = {0.0f, 0.9f, 0.0f};
  cfg.lighting.areaLightEmission = {10.0f, 10.0f, 10.0f};

  backend.renderIteration(cfg);
  const rt::RenderBuffer visible = backend.downloadRenderBuffer();

  cfg.lighting.enableAreaLight = false;
  empty.hash = 901;
  backend.setScene(empty);
  backend.resetAccumulation();
  backend.renderIteration(cfg);
  const rt::RenderBuffer hidden = backend.downloadRenderBuffer();

  const glm::vec3 visiblePatch = averageCenterPatchVec3(cam, visible.color, 6);
  const glm::vec3 hiddenPatch = averageCenterPatchVec3(cam, hidden.color, 6);
  require(luminance(visiblePatch) > luminance(hiddenPatch) + 0.1,
          "visible area light should contribute a bright miss-path rectangle when directly viewed");
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    testAreaLightVisibleOnMissPath(*backend);
    std::cout << "visible_light_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "visible_light_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "visible_light_test FAILED: " << msg << std::endl;
    return 1;
  }
}
