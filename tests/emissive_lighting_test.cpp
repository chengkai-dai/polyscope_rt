#include <exception>
#include <iostream>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering_test_utils.h"
#include "test_helpers.h"

namespace {

void testEmissiveSurfaceVisibleWithoutExplicitLights(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene scene;
  rt::RTMesh emissiveQuad = makeQuad("emissive_quad",
                                     {-0.8f, -0.8f, 0.0f}, {0.8f, -0.8f, 0.0f},
                                     {0.8f, 0.8f, 0.0f}, {-0.8f, 0.8f, 0.0f},
                                     {0.0f, 0.0f, 1.0f});
  emissiveQuad.baseColorFactor = {0.0f, 0.0f, 0.0f, 1.0f};
  emissiveQuad.emissiveFactor = {10.0f, 4.0f, 1.0f};
  scene.meshes.push_back(emissiveQuad);
  scene.hash = 45;
  backend.setScene(scene);
  backend.resetAccumulation();

  auto cfg = standardConfig();
  cfg.maxBounces = 1;
  cfg.lighting.mainLightIntensity = 0.0f;
  backend.renderIteration(cfg);
  const rt::RenderBuffer buffer = backend.downloadRenderBuffer();
  const glm::vec3 center = buffer.color[(cam.height / 2) * cam.width + (cam.width / 2)];
  require(luminance(center) > 0.08, "emissive surface should stay visibly bright without explicit lights");
}

void testEmissiveGeometryLightsReceiver(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene scene;
  rt::RTMesh floor = makeQuad("receiver_floor",
                              {-1.6f, -0.6f, -1.2f}, {1.6f, -0.6f, -1.2f},
                              {1.6f, -0.6f, 1.2f}, {-1.6f, -0.6f, 1.2f},
                              {0.0f, 1.0f, 0.0f});
  floor.baseColorFactor = {0.8f, 0.8f, 0.8f, 1.0f};
  floor.roughnessFactor = 0.8f;
  scene.meshes.push_back(floor);

  rt::RTMesh lightQuad = makeQuad("ceiling_light",
                                  {-0.55f, 1.0f, -0.35f}, {-0.55f, 1.0f, 0.35f},
                                  {0.55f, 1.0f, 0.35f}, {0.55f, 1.0f, -0.35f},
                                  {0.0f, -1.0f, 0.0f});
  lightQuad.baseColorFactor = {0.0f, 0.0f, 0.0f, 1.0f};
  lightQuad.emissiveFactor = {14.0f, 13.0f, 12.0f};
  scene.meshes.push_back(lightQuad);
  scene.hash = 46;
  backend.setScene(scene);
  backend.resetAccumulation();

  auto cfg = standardConfig();
  cfg.samplesPerIteration = 4;
  cfg.maxBounces = 1;
  cfg.lighting.mainLightIntensity = 0.0f;
  backend.renderIteration(cfg);
  const rt::RenderBuffer buffer = backend.downloadRenderBuffer();
  const glm::vec3 center = buffer.color[(cam.height / 2) * cam.width + (cam.width / 2)];
  require(luminance(center) > 0.03, "emissive geometry should directly light nearby diffuse surfaces");
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    testEmissiveSurfaceVisibleWithoutExplicitLights(*backend);
    testEmissiveGeometryLightsReceiver(*backend);
    std::cout << "emissive_lighting_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "emissive_lighting_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "emissive_lighting_test FAILED: " << msg << std::endl;
    return 1;
  }
}
