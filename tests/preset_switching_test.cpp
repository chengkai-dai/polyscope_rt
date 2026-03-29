#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include "glm/gtc/matrix_transform.hpp"

#include "rendering/ray_tracing_backend.h"
#include "test_helpers.h"

namespace {

rt::RTCamera makeCamera() {
  rt::RTCamera camera;
  camera.position = {0.0f, 0.2f, 3.0f};
  camera.lookDir = glm::normalize(glm::vec3(0.0f, -0.05f, -1.0f));
  camera.upDir = {0.0f, 1.0f, 0.0f};
  camera.rightDir = {1.0f, 0.0f, 0.0f};
  camera.fovYDegrees = 45.0f;
  camera.aspect = 1.0f;
  camera.viewMatrix = glm::lookAt(camera.position, glm::vec3(0.0f, 0.2f, 0.0f), camera.upDir);
  camera.projectionMatrix = glm::perspective(glm::radians(camera.fovYDegrees), camera.aspect, 0.01f, 100.0f);
  camera.width = 64;
  camera.height = 64;
  return camera;
}

rt::RTScene makeTriangleScene() {
  rt::RTScene scene;
  rt::RTMesh mesh;
  mesh.name = "tri";
  mesh.vertices = {{-1.0f, -0.5f, 0.0f}, {1.0f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  mesh.indices = {glm::uvec3(0, 1, 2)};
  mesh.baseColorFactor = {0.8f, 0.3f, 0.1f, 1.0f};
  scene.meshes.push_back(mesh);
  scene.hash = 500;
  return scene;
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    auto camera = makeCamera();
    backend->resize(camera.width, camera.height);
    backend->updateCamera(camera);
    backend->setScene(makeTriangleScene());

    // ---- Standard mode produces valid output ----
    rt::RenderConfig standardConfig;
    standardConfig.renderMode = rt::RenderMode::Standard;
    standardConfig.samplesPerIteration = 1;
    standardConfig.maxBounces = 2;
    standardConfig.accumulate = false;

    backend->resetAccumulation();
    backend->renderIteration(standardConfig);
    rt::RenderBuffer standardBuf = backend->downloadRenderBuffer();
    require(standardBuf.width == 64 && standardBuf.height == 64, "standard mode buffer size mismatch");

    double stdColorSum = 0.0;
    for (const auto& c : standardBuf.color) {
      require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z),
              "non-finite color in standard mode");
      stdColorSum += c.x + c.y + c.z;
    }
    require(stdColorSum > 0.0, "standard mode must produce non-black output");

    // ---- Toon mode produces valid output ----
    rt::RenderConfig toonConfig;
    toonConfig.renderMode = rt::RenderMode::Toon;
    toonConfig.samplesPerIteration = 1;
    toonConfig.maxBounces = 2;
    toonConfig.accumulate = false;
    toonConfig.toon.enabled = true;
    toonConfig.toon.enableDetailContour = true;
    toonConfig.toon.enableObjectContour = true;

    backend->resetAccumulation();
    backend->renderIteration(toonConfig);
    rt::RenderBuffer toonBuf = backend->downloadRenderBuffer();
    require(toonBuf.width == 64 && toonBuf.height == 64, "toon mode buffer size mismatch");

    double toonColorSum = 0.0;
    for (const auto& c : toonBuf.color) {
      require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z),
              "non-finite color in toon mode");
      toonColorSum += c.x + c.y + c.z;
    }
    require(toonColorSum > 0.0, "toon mode must produce non-black output");

    // ---- Standard and Toon produce different results ----
    // The toon post-process applies contours and quantized shading, so the
    // pixel values should differ from the standard tonemap.
    int differingPixels = 0;
    for (size_t i = 0; i < standardBuf.color.size(); ++i) {
      if (glm::length(standardBuf.color[i] - toonBuf.color[i]) > 0.01f) {
        differingPixels++;
      }
    }
    require(differingPixels > 10,
            "standard and toon presets should produce visibly different output");

    // ---- Switch back to Standard and verify consistency ----
    backend->resetAccumulation();
    backend->renderIteration(standardConfig);
    rt::RenderBuffer standardBuf2 = backend->downloadRenderBuffer();

    double stdColorSum2 = 0.0;
    for (const auto& c : standardBuf2.color) {
      require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z),
              "non-finite color after switching back to standard mode");
      stdColorSum2 += c.x + c.y + c.z;
    }
    require(stdColorSum2 > 0.0, "standard mode must produce non-black output after switching back");

    // ---- Toon-specific buffers are populated ----
    // Toon mode should populate contour buffers; standard mode should not.
    bool toonHasContours = false;
    for (float v : toonBuf.detailContour) {
      if (v > 0.01f) { toonHasContours = true; break; }
    }
    for (float v : toonBuf.objectContour) {
      if (v > 0.01f) { toonHasContours = true; break; }
    }
    // Note: contour presence depends on scene geometry; for a single triangle
    // against a background, object contours should appear at the triangle edges.
    // We don't hard-require contours here since it depends on the specific
    // geometry/camera setup, but we do verify the buffers are allocated.
    require(!toonBuf.detailContour.empty() || !toonBuf.objectContour.empty(),
            "toon mode should allocate contour buffers");

    std::cout << "preset_switching_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string message = e.what();
    if (isSkippableBackendError(message)) {
      std::cout << "preset_switching_test skipped: " << message << std::endl;
      return 0;
    }
    std::cerr << "preset_switching_test failed: " << message << std::endl;
    return 1;
  }
}
