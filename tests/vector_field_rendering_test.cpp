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

rt::RenderConfig makeConfig() {
  rt::RenderConfig config;
  config.samplesPerIteration = 1;
  config.maxBounces = 1;
  config.accumulate = false;
  config.lighting.backgroundColor = {0.0f, 0.0f, 0.0f};
  config.lighting.environmentIntensity = 0.0f;
  config.lighting.mainLightIntensity = 1.0f;
  config.lighting.ambientFloor = 0.15f;
  return config;
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    auto camera = makeCamera();
    backend->resize(camera.width, camera.height);

    // ---- Single vector field arrow ----
    {
      rt::RTScene scene;
      rt::RTVectorField vf;
      vf.name = "test_vf";
      vf.roots = {{0.0f, 0.0f, 0.0f}};
      vf.directions = {{0.0f, 1.0f, 0.0f}};
      vf.color = {0.0f, 0.0f, 1.0f};
      vf.radius = 0.08f;
      scene.vectorFields.push_back(vf);
      scene.hash = 300;

      backend->setScene(scene);
      backend->updateCamera(camera);
      backend->resetAccumulation();
      backend->renderIteration(makeConfig());
      rt::RenderBuffer buf = backend->downloadRenderBuffer();

      require(buf.width == 64 && buf.height == 64, "vector field buffer size mismatch");

      double blueSum = 0.0;
      for (const auto& c : buf.color) {
        require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z),
                "non-finite color in vector field scene");
        blueSum += c.z;
      }
      require(blueSum > 0.0, "vector field arrow must produce visible blue output");
    }

    // ---- Vector field with multiple arrows ----
    {
      rt::RTScene scene;
      rt::RTVectorField vf;
      vf.name = "multi_vf";
      vf.roots = {{-0.5f, 0.0f, 0.0f}, {0.5f, 0.0f, 0.0f}};
      vf.directions = {{0.0f, 0.8f, 0.0f}, {0.0f, 0.8f, 0.0f}};
      vf.color = {1.0f, 0.0f, 0.0f};
      vf.radius = 0.08f;
      scene.vectorFields.push_back(vf);
      scene.hash = 301;

      backend->setScene(scene);
      backend->updateCamera(camera);
      backend->resetAccumulation();
      backend->renderIteration(makeConfig());
      rt::RenderBuffer buf = backend->downloadRenderBuffer();

      double redSum = 0.0;
      for (const auto& c : buf.color) redSum += c.x;
      require(redSum > 0.0, "multiple arrows must produce visible output");
    }

    // ---- Vector field coexisting with mesh ----
    {
      rt::RTScene scene;
      rt::RTMesh floor;
      floor.name = "floor";
      floor.vertices = {{-5, -0.5f, -5}, {5, -0.5f, -5}, {5, -0.5f, 5}, {-5, -0.5f, 5}};
      floor.indices = {glm::uvec3(0, 1, 2), glm::uvec3(0, 2, 3)};
      floor.normals = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
      floor.baseColorFactor = {0.5f, 0.5f, 0.5f, 1.0f};
      scene.meshes.push_back(floor);

      rt::RTVectorField vf;
      vf.name = "coexist_vf";
      vf.roots = {{0.0f, 0.0f, 0.0f}};
      vf.directions = {{0.0f, 1.0f, 0.0f}};
      vf.color = {0.0f, 1.0f, 0.0f};
      vf.radius = 0.1f;
      scene.vectorFields.push_back(vf);
      scene.hash = 302;

      backend->setScene(scene);
      backend->updateCamera(camera);
      backend->resetAccumulation();
      backend->renderIteration(makeConfig());
      rt::RenderBuffer buf = backend->downloadRenderBuffer();

      double greenSum = 0.0, colorSum = 0.0;
      for (const auto& c : buf.color) {
        greenSum += c.y;
        colorSum += c.x + c.y + c.z;
      }
      require(greenSum > 0.0, "vector field green arrow must be visible alongside mesh");
      require(colorSum > greenSum, "mesh should also contribute color in coexistence scene");
    }

    std::cout << "vector_field_rendering_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string message = e.what();
    if (isSkippableBackendError(message)) {
      std::cout << "vector_field_rendering_test skipped: " << message << std::endl;
      return 0;
    }
    std::cerr << "vector_field_rendering_test failed: " << message << std::endl;
    return 1;
  }
}
