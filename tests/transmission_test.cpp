#include <exception>
#include <iostream>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering_test_utils.h"
#include "test_helpers.h"

namespace {

void testTransmissionRevealsBackground(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RenderConfig cfg = standardConfig();
  cfg.maxBounces = 2;
  cfg.lighting.backgroundColor = {0.05f, 0.2f, 0.9f};
  cfg.lighting.environmentIntensity = 0.0f;
  cfg.lighting.ambientFloor = 0.0f;
  cfg.lighting.mainLightIntensity = 0.0f;
  cfg.lighting.enableAreaLight = false;

  rt::RTScene glassScene;
  rt::RTMesh glassSphere = makeSphere(0.75f, 18u, 24u);
  glassSphere.baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
  glassSphere.roughnessFactor = 0.02f;
  glassSphere.metallicFactor = 0.0f;
  glassSphere.transmissionFactor = 1.0f;
  glassSphere.indexOfRefraction = 1.5f;
  glassScene.meshes.push_back(std::move(glassSphere));
  glassScene.hash = 43;
  backend.setScene(glassScene);
  backend.resetAccumulation();
  backend.renderIteration(cfg);
  const rt::RenderBuffer glassBuffer = backend.downloadRenderBuffer();
  const glm::vec3 glassCenter = glassBuffer.color[(cam.height / 2) * cam.width + (cam.width / 2)];

  rt::RTScene opaqueScene;
  rt::RTMesh opaqueSphere = makeSphere(0.75f, 18u, 24u);
  opaqueSphere.baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
  opaqueSphere.roughnessFactor = 0.02f;
  opaqueSphere.metallicFactor = 0.0f;
  opaqueScene.meshes.push_back(std::move(opaqueSphere));
  opaqueScene.hash = 44;
  backend.setScene(opaqueScene);
  backend.resetAccumulation();
  backend.renderIteration(cfg);
  const rt::RenderBuffer opaqueBuffer = backend.downloadRenderBuffer();
  const glm::vec3 opaqueCenter = opaqueBuffer.color[(cam.height / 2) * cam.width + (cam.width / 2)];

  require(glassCenter.z > 0.15f, "glass material should transmit the blue background through the sphere");
  require(glassCenter.z > opaqueCenter.z + 0.05f,
          "glass sphere should reveal more background than an opaque sphere");
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    testTransmissionRevealsBackground(*backend);
    std::cout << "transmission_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "transmission_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "transmission_test FAILED: " << msg << std::endl;
    return 1;
  }
}
