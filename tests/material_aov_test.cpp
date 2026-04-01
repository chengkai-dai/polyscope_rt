#include <exception>
#include <iostream>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering_test_utils.h"
#include "test_helpers.h"

namespace {

void testMaterialAovsTrackSurfaceParameters(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  auto cfg = standardConfig();
  cfg.lighting.mainLightIntensity = 0.0f;
  cfg.groundPlane.height = -10.0f;

  auto renderAovQuad = [&](float metallic, float roughness, uint32_t hash) {
    rt::RTScene scene;
    rt::RTMesh quad = makeCameraFacingQuad(metallic > 0.5f ? "metal_aov_quad" : "dielectric_aov_quad", true,
                                           {0.8f, 0.6f, 0.2f, 1.0f});
    quad.metallicFactor = metallic;
    quad.roughnessFactor = roughness;
    scene.meshes.push_back(quad);
    scene.hash = hash;
    backend.setScene(scene);
    backend.resetAccumulation();
    backend.renderIteration(cfg);
    return backend.downloadRenderBuffer();
  };

  const rt::RenderBuffer dielectric = renderAovQuad(0.0f, 0.85f, 800);
  const rt::RenderBuffer metal = renderAovQuad(1.0f, 0.2f, 801);

  const size_t centerIdx = pixelIndex(cam.width, static_cast<int>(cam.width / 2), static_cast<int>(cam.height / 2));
  const glm::vec3 baseColor(0.8f, 0.6f, 0.2f);
  const glm::vec3 dielectricExpectedDiffuse = baseColor;
  const glm::vec3 dielectricExpectedSpecular(0.04f);
  const glm::vec3 metalExpectedDiffuse(0.0f);
  const glm::vec3 metalExpectedSpecular = baseColor;
  const glm::vec3 dielectricDiffuse = dielectric.diffuseAlbedo[centerIdx];
  const glm::vec3 metalDiffuse = metal.diffuseAlbedo[centerIdx];
  const glm::vec3 dielectricSpecular = dielectric.specularAlbedo[centerIdx];
  const glm::vec3 metalSpecular = metal.specularAlbedo[centerIdx];
  const float dielectricRoughness = dielectric.roughness[centerIdx];
  const float metalRoughness = metal.roughness[centerIdx];

  require(dielectric.objectId[centerIdx] != 0u && metal.objectId[centerIdx] != 0u,
          "AOV test quad should hit geometry at the center pixel");
  require(std::isfinite(dielectric.roughness[centerIdx]) && std::isfinite(metal.roughness[centerIdx]),
          "roughness AOV should remain finite at surface hits");

  require(glm::length(dielectricDiffuse - dielectricExpectedDiffuse) < 1e-3f,
          "dielectric diffuseAlbedo should equal baseColor * (1 - metallic)");
  require(glm::length(dielectricSpecular - dielectricExpectedSpecular) < 1e-3f,
          "dielectric specularAlbedo should equal mix(F0, baseColor, metallic)");
  requireNear(dielectricRoughness, 0.85f, 1e-4f,
              "dielectric roughness AOV should match the surfaced roughness");

  require(glm::length(metalDiffuse - metalExpectedDiffuse) < 1e-3f,
          "metal diffuseAlbedo should vanish for fully metallic surfaces");
  require(glm::length(metalSpecular - metalExpectedSpecular) < 1e-3f,
          "metal specularAlbedo should equal the baseColor for fully metallic surfaces");
  requireNear(metalRoughness, 0.2f, 1e-4f,
              "metal roughness AOV should match the surfaced roughness");
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    testMaterialAovsTrackSurfaceParameters(*backend);
    std::cout << "material_aov_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "material_aov_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "material_aov_test FAILED: " << msg << std::endl;
    return 1;
  }
}
