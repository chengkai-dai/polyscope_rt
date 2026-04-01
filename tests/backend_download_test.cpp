#include <cmath>
#include <exception>
#include <iostream>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering_test_utils.h"
#include "test_helpers.h"

namespace {

void testBackendDownloadsStableAovs(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene scene;
  rt::RTMesh mesh;
  mesh.name = "tri";
  mesh.vertices = {{-1.0f, -0.5f, 0.0f}, {1.0f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  mesh.indices = {glm::uvec3(0, 1, 2)};
  mesh.baseColorFactor = {0.8f, 0.3f, 0.1f, 1.0f};
  scene.meshes.push_back(mesh);
  scene.hash = 42;
  backend.setScene(scene);

  rt::RenderConfig config;
  config.samplesPerIteration = 1;
  config.maxBounces = 2;
  config.accumulate = true;

  backend.renderIteration(config);
  const rt::RenderBuffer firstFrame = backend.downloadRenderBuffer();
  backend.renderIteration(config);
  const rt::RenderBuffer buffer = backend.downloadRenderBuffer();

  require(buffer.width == cam.width && buffer.height == cam.height, "unexpected render buffer size");
  require(buffer.accumulatedSamples == 2, "expected two accumulated samples");
  require(buffer.diffuseAlbedo.size() == buffer.color.size(), "unexpected diffuse AOV size");
  require(buffer.specularAlbedo.size() == buffer.color.size(), "unexpected specular AOV size");
  require(buffer.roughness.size() == buffer.color.size(), "unexpected roughness AOV size");

  double colorSum = 0.0;
  double linearDepthSum = 0.0;
  double normalSum = 0.0;
  double diffuseSum = 0.0;
  double specularSum = 0.0;
  double roughnessSum = 0.0;
  for (size_t i = 0; i < buffer.color.size(); ++i) {
    const auto& c = buffer.color[i];
    const auto& n = buffer.normal[i];
    const auto& d = buffer.diffuseAlbedo[i];
    const auto& s = buffer.specularAlbedo[i];
    require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z), "non-finite beauty value");
    require(std::isfinite(buffer.linearDepth[i]), "non-finite linear depth value");
    require(std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z), "non-finite normal value");
    require(std::isfinite(d.x) && std::isfinite(d.y) && std::isfinite(d.z), "non-finite diffuse albedo value");
    require(std::isfinite(s.x) && std::isfinite(s.y) && std::isfinite(s.z), "non-finite specular albedo value");
    require(std::isfinite(buffer.roughness[i]), "non-finite roughness value");
    colorSum += c.x + c.y + c.z;
    if (buffer.linearDepth[i] > 0.0f) linearDepthSum += buffer.linearDepth[i];
    normalSum += std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    diffuseSum += d.x + d.y + d.z;
    specularSum += s.x + s.y + s.z;
    roughnessSum += buffer.roughness[i];
  }
  require(colorSum > 0.0, "expected non-black backend output");
  require(linearDepthSum > 0.0, "expected positive hit depths");
  require(normalSum > 0.0, "expected non-zero normal output");
  require(diffuseSum > 0.0, "expected non-zero diffuse albedo output");
  require(specularSum > 0.0, "expected non-zero specular albedo output");
  require(roughnessSum > 0.0, "expected non-zero roughness output");
  require(firstFrame.depth == buffer.depth, "depth buffer should stay stable after first accumulated frame");
  require(firstFrame.linearDepth == buffer.linearDepth, "linear depth buffer should stay stable after first accumulated frame");
  require(firstFrame.objectId == buffer.objectId, "object ID buffer should stay stable after first accumulated frame");
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    testBackendDownloadsStableAovs(*backend);
    std::cout << "backend_download_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "backend_download_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "backend_download_test FAILED: " << msg << std::endl;
    return 1;
  }
}
