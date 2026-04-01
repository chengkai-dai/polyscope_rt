#include <cmath>
#include <exception>
#include <iostream>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering_test_utils.h"
#include "test_helpers.h"

namespace {

void testNormalMapAffectsShading(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  auto cfg = standardConfig();
  cfg.lighting.mainLightDirection = {-0.7f, 0.0f, -0.7f};
  cfg.lighting.mainLightIntensity = 1.2f;

  auto renderNormalMappedQuad = [&](bool useNormalMap, uint32_t hash) {
    rt::RTScene scene;
    rt::RTMesh quad = makeCameraFacingQuad(useNormalMap ? "normal_mapped_quad" : "flat_quad", true,
                                           {0.7f, 0.7f, 0.7f, 1.0f});
    quad.roughnessFactor = 0.5f;
    if (useNormalMap) {
      quad.hasNormalTexture = true;
      quad.normalTexture.width = 2;
      quad.normalTexture.height = 2;
      quad.normalTexture.cacheKey = "test_normal_map";
      quad.normalTextureScale = 1.0f;
      const glm::vec3 tangentNormal = glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f));
      const glm::vec4 encoded = glm::vec4(tangentNormal * 0.5f + 0.5f, 1.0f);
      quad.normalTexture.pixels.assign(4, encoded);
    }
    scene.meshes.push_back(quad);
    scene.hash = hash;
    backend.setScene(scene);
    backend.resetAccumulation();
    backend.renderIteration(cfg);
    return backend.downloadRenderBuffer();
  };

  const rt::RenderBuffer flat = renderNormalMappedQuad(false, 790);
  const rt::RenderBuffer mapped = renderNormalMappedQuad(true, 791);

  int diffCount = 0;
  int normalDiffCount = 0;
  for (size_t i = 0; i < flat.color.size(); ++i) {
    if (glm::length(flat.color[i] - mapped.color[i]) > 0.01f) ++diffCount;
    if (glm::length(flat.normal[i] - mapped.normal[i]) > 0.02f) ++normalDiffCount;
  }

  require(diffCount > 20, "normal map should measurably alter the shading");
  require(normalDiffCount > 20, "normal map should measurably alter the normal buffer");
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    testNormalMapAffectsShading(*backend);
    std::cout << "normal_map_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "normal_map_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "normal_map_test FAILED: " << msg << std::endl;
    return 1;
  }
}
