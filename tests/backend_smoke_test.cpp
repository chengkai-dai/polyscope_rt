#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include "glm/gtc/matrix_transform.hpp"

#include "rendering/ray_tracing_backend.h"
#include "test_helpers.h"

namespace {

rt::RTMesh makeSphere(float radius, uint32_t latSegments, uint32_t lonSegments) {
  rt::RTMesh mesh;
  mesh.name = "glass_sphere";
  for (uint32_t y = 0; y <= latSegments; ++y) {
    float v = static_cast<float>(y) / static_cast<float>(latSegments);
    float theta = v * glm::pi<float>();
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    for (uint32_t x = 0; x <= lonSegments; ++x) {
      float u = static_cast<float>(x) / static_cast<float>(lonSegments);
      float phi = u * glm::two_pi<float>();
      float sinPhi = std::sin(phi);
      float cosPhi = std::cos(phi);
      glm::vec3 p(radius * sinTheta * cosPhi, radius * cosTheta, radius * sinTheta * sinPhi);
      mesh.vertices.push_back(p);
      mesh.normals.push_back(glm::normalize(p));
      mesh.texcoords.emplace_back(u, v);
    }
  }

  auto idx = [lonSegments](uint32_t y, uint32_t x) { return y * (lonSegments + 1u) + x; };
  for (uint32_t y = 0; y < latSegments; ++y) {
    for (uint32_t x = 0; x < lonSegments; ++x) {
      uint32_t i0 = idx(y, x);
      uint32_t i1 = idx(y, x + 1u);
      uint32_t i2 = idx(y + 1u, x + 1u);
      uint32_t i3 = idx(y + 1u, x);
      if (y != 0u) mesh.indices.emplace_back(i0, i3, i1);
      if (y + 1u != latSegments) mesh.indices.emplace_back(i1, i3, i2);
    }
  }
  return mesh;
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);

    rt::RTMesh mesh;
    mesh.name = "tri";
    mesh.vertices = {{-1.0f, -0.5f, 0.0f}, {1.0f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    mesh.indices = {glm::uvec3(0, 1, 2)};
    mesh.baseColorFactor = {0.8f, 0.3f, 0.1f, 1.0f};

    rt::RTScene scene;
    scene.meshes.push_back(mesh);
    scene.hash = 42;
    backend->setScene(scene);

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

    backend->resize(camera.width, camera.height);
    backend->updateCamera(camera);

    rt::RenderConfig config;
    config.samplesPerIteration = 1;
    config.maxBounces = 2;
    config.accumulate = true;

    backend->renderIteration(config);
    rt::RenderBuffer firstFrame = backend->downloadRenderBuffer();
    backend->renderIteration(config);

    rt::RenderBuffer buffer = backend->downloadRenderBuffer();
    require(buffer.width == 64 && buffer.height == 64, "unexpected buffer size");
    require(buffer.accumulatedSamples == 2, "expected two accumulated samples");
    require(buffer.color.size() == 64u * 64u, "unexpected color buffer size");
    require(buffer.depth.size() == 64u * 64u, "unexpected depth buffer size");
    require(buffer.linearDepth.size() == 64u * 64u, "unexpected linear depth buffer size");
    require(buffer.normal.size() == 64u * 64u, "unexpected normal buffer size");
    require(buffer.objectId.size() == 64u * 64u, "unexpected object id buffer size");

    double colorSum = 0.0;
    double normalSum = 0.0;
    double linearDepthSum = 0.0;
    for (const auto& c : buffer.color) {
      require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z), "non-finite color value");
      colorSum += c.x + c.y + c.z;
    }
    for (float d : buffer.linearDepth) {
      require(std::isfinite(d), "non-finite linear depth value");
      if (d > 0.0f) linearDepthSum += d;
    }
    for (const auto& n : buffer.normal) {
      require(std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z), "non-finite normal value");
      normalSum += std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    }
    require(colorSum > 0.0, "expected non-black output from backend");
    require(linearDepthSum > 0.0, "expected positive hit depths from backend");
    require(normalSum > 0.0, "expected non-zero normal output from backend");
    require(firstFrame.depth == buffer.depth, "depth buffer should stay stable after the first accumulated frame");
    require(firstFrame.linearDepth == buffer.linearDepth,
            "linear depth buffer should stay stable after the first accumulated frame");
    require(firstFrame.objectId == buffer.objectId, "object id buffer should stay stable after the first accumulated frame");

    backend->resetAccumulation();
    rt::RTScene glassScene;
    rt::RTMesh glassSphere = makeSphere(0.75f, 18u, 24u);
    glassSphere.baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
    glassSphere.roughnessFactor = 0.02f;
    glassSphere.metallicFactor = 0.0f;
    glassSphere.transmissionFactor = 1.0f;
    glassSphere.indexOfRefraction = 1.5f;
    glassScene.meshes.push_back(std::move(glassSphere));
    glassScene.hash = 43;
    backend->setScene(glassScene);

    rt::RenderConfig glassConfig;
    glassConfig.renderMode = rt::RenderMode::Standard;
    glassConfig.samplesPerIteration = 1;
    glassConfig.maxBounces = 2;
    glassConfig.accumulate = false;
    glassConfig.lighting.backgroundColor = {0.05f, 0.2f, 0.9f};
    glassConfig.lighting.environmentIntensity = 0.0f;
    glassConfig.lighting.ambientFloor = 0.0f;
    glassConfig.lighting.mainLightIntensity = 0.0f;
    glassConfig.lighting.enableAreaLight = false;
    backend->resetAccumulation();
    backend->renderIteration(glassConfig);
    rt::RenderBuffer glassBuffer = backend->downloadRenderBuffer();
    const glm::vec3 center = glassBuffer.color[(camera.height / 2) * camera.width + (camera.width / 2)];
    require(center.z > 0.15f, "glass material should transmit the blue background through the sphere");

    rt::RTScene opaqueScene;
    rt::RTMesh opaqueSphere = makeSphere(0.75f, 18u, 24u);
    opaqueSphere.baseColorFactor = {1.0f, 1.0f, 1.0f, 1.0f};
    opaqueSphere.roughnessFactor = 0.02f;
    opaqueSphere.metallicFactor = 0.0f;
    opaqueScene.meshes.push_back(std::move(opaqueSphere));
      opaqueScene.hash = 44;
      backend->setScene(opaqueScene);
    backend->resetAccumulation();
    backend->renderIteration(glassConfig);
    rt::RenderBuffer opaqueBuffer = backend->downloadRenderBuffer();
    const glm::vec3 opaqueCenter = opaqueBuffer.color[(camera.height / 2) * camera.width + (camera.width / 2)];
    require(center.z > opaqueCenter.z + 0.05f, "glass sphere should reveal more background than an opaque sphere");

    // ---- Curve primitive pipeline smoke test ----
    {
      rt::RTScene curveScene;
      rt::RTMesh floor;
      floor.name = "floor";
      floor.vertices = {{-5, 0, -5}, {5, 0, -5}, {5, 0, 5}, {-5, 0, 5}};
      floor.indices = {glm::uvec3(0, 1, 2), glm::uvec3(0, 2, 3)};
      floor.normals = {{0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}};
      floor.baseColorFactor = {0.5f, 0.5f, 0.5f, 1.0f};
      curveScene.meshes.push_back(floor);

      rt::RTCurveNetwork curveNet;
      curveNet.name = "test_curve";
      curveNet.baseColor = {1.0f, 0.0f, 0.0f, 1.0f};
      curveNet.roughness = 0.5f;

      rt::RTCurvePrimitive sphere;
      sphere.type = rt::RTCurvePrimitiveType::Sphere;
      sphere.p0 = {0.0f, 0.5f, 0.0f};
      sphere.radius = 0.3f;
      curveNet.primitives.push_back(sphere);

      rt::RTCurvePrimitive cylinder;
      cylinder.type = rt::RTCurvePrimitiveType::Cylinder;
      cylinder.p0 = {-1.0f, 0.3f, 0.0f};
      cylinder.p1 = {1.0f, 0.3f, 0.0f};
      cylinder.radius = 0.1f;
      curveNet.primitives.push_back(cylinder);

      curveScene.curveNetworks.push_back(curveNet);
      curveScene.hash = 100;

      backend->setScene(curveScene);
      backend->updateCamera(camera);
      backend->resetAccumulation();

      rt::RenderConfig curveConfig;
      curveConfig.samplesPerIteration = 1;
      curveConfig.maxBounces = 2;
      curveConfig.accumulate = false;

      backend->renderIteration(curveConfig);
      rt::RenderBuffer curveBuffer = backend->downloadRenderBuffer();
      require(curveBuffer.width == 64 && curveBuffer.height == 64, "curve scene buffer size mismatch");

      double curveColorSum = 0.0;
      for (const auto& c : curveBuffer.color) {
        require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z), "non-finite color in curve scene");
        curveColorSum += c.x + c.y + c.z;
      }
      require(curveColorSum > 0.0, "expected non-black output from curve scene");
    }

    // ---- Curves-only scene (no mesh triangles) ----
    {
      rt::RTScene curvesOnlyScene;
      rt::RTCurveNetwork curveNet;
      curveNet.name = "only_curve";
      curveNet.baseColor = {0.2f, 0.8f, 0.2f, 1.0f};
      rt::RTCurvePrimitive sphere;
      sphere.type = rt::RTCurvePrimitiveType::Sphere;
      sphere.p0 = {0.0f, 0.0f, 0.0f};
      sphere.radius = 0.5f;
      curveNet.primitives.push_back(sphere);
      curvesOnlyScene.curveNetworks.push_back(curveNet);
      curvesOnlyScene.hash = 101;

      backend->setScene(curvesOnlyScene);
      backend->updateCamera(camera);
      backend->resetAccumulation();

      rt::RenderConfig curvesOnlyConfig;
      curvesOnlyConfig.samplesPerIteration = 1;
      curvesOnlyConfig.maxBounces = 1;
      curvesOnlyConfig.accumulate = false;

      backend->renderIteration(curvesOnlyConfig);
      rt::RenderBuffer curvesOnlyBuffer = backend->downloadRenderBuffer();
      require(curvesOnlyBuffer.width == 64 && curvesOnlyBuffer.height == 64, "curves-only buffer size mismatch");

      double greenSum = 0.0;
      for (const auto& c : curvesOnlyBuffer.color) {
        require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z), "non-finite color in curves-only scene");
        greenSum += c.y;
      }
      require(greenSum > 0.0, "expected visible green curve primitives in curves-only scene");
    }

    // ---- Triangle-only regression (re-set original scene) ----
    {
      rt::RTScene triOnlyScene;
      triOnlyScene.meshes.push_back(mesh);
      triOnlyScene.hash = 102;

      backend->setScene(triOnlyScene);
      backend->updateCamera(camera);
      backend->resetAccumulation();

      rt::RenderConfig triConfig;
      triConfig.samplesPerIteration = 1;
      triConfig.maxBounces = 2;
      triConfig.accumulate = false;

      backend->renderIteration(triConfig);
      rt::RenderBuffer triBuffer = backend->downloadRenderBuffer();
      require(triBuffer.width == 64 && triBuffer.height == 64, "triangle regression buffer size mismatch");

      double triColorSum = 0.0;
      for (const auto& c : triBuffer.color) {
        require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z), "non-finite color in triangle regression");
        triColorSum += c.x + c.y + c.z;
      }
      require(triColorSum > 0.0, "expected non-black output from triangle-only regression");
    }

    // ---- Point-cloud-only scene ----
    // Verifies that the bounding-box BLAS and sphereIntersection IFT are wired up
    // correctly and produce visible output.
    {
      rt::RTScene pointScene;
      rt::RTPointCloud cloud;
      cloud.name = "red_cloud";
      cloud.centers = {{0.0f, 0.0f, 0.0f}, {0.4f, 0.1f, 0.0f}, {-0.4f, -0.1f, 0.0f}};
      cloud.radius = 0.35f;
      cloud.baseColor = {1.0f, 0.0f, 0.0f, 1.0f};
      pointScene.pointClouds.push_back(cloud);
      pointScene.hash = 200;

      backend->setScene(pointScene);
      backend->updateCamera(camera);
      backend->resetAccumulation();

      rt::RenderConfig pointConfig;
      pointConfig.samplesPerIteration = 1;
      pointConfig.maxBounces = 1;
      pointConfig.accumulate = false;
      pointConfig.lighting.backgroundColor = {0.0f, 0.0f, 0.0f};
      pointConfig.lighting.environmentIntensity = 0.0f;
      pointConfig.lighting.mainLightIntensity = 1.0f;
      pointConfig.lighting.ambientFloor = 0.15f;

      backend->renderIteration(pointConfig);
      rt::RenderBuffer pointBuffer = backend->downloadRenderBuffer();
      require(pointBuffer.width == 64 && pointBuffer.height == 64, "point cloud buffer size mismatch");

      double redSum = 0.0, blueSum = 0.0;
      for (const auto& c : pointBuffer.color) {
        require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z),
                "non-finite color in point-cloud-only scene");
        redSum  += c.x;
        blueSum += c.z;
      }
      require(redSum > 0.0, "point cloud must produce non-black output");
      require(redSum > blueSum + 0.1,
              "red point cloud should dominate in the red channel");
    }

    // ---- Per-point color (colormap path) ----
    // Verifies that RTPointCloud::colors (per-point overrides) flow from the CPU
    // buffer all the way to the rendered pixels.
    {
      rt::RTScene coloredScene;
      rt::RTPointCloud cloud;
      cloud.name = "blue_cloud";
      cloud.centers = {{0.0f, 0.0f, 0.0f}};
      cloud.radius = 0.55f;
      cloud.baseColor = {1.0f, 0.0f, 0.0f, 1.0f};        // base = red
      cloud.colors   = {{0.0f, 0.0f, 1.0f}};              // override = blue
      coloredScene.pointClouds.push_back(cloud);
      coloredScene.hash = 201;

      backend->setScene(coloredScene);
      backend->updateCamera(camera);
      backend->resetAccumulation();

      rt::RenderConfig coloredConfig;
      coloredConfig.samplesPerIteration = 1;
      coloredConfig.maxBounces = 1;
      coloredConfig.accumulate = false;
      coloredConfig.lighting.backgroundColor = {0.0f, 0.0f, 0.0f};
      coloredConfig.lighting.environmentIntensity = 0.0f;
      coloredConfig.lighting.mainLightIntensity = 1.0f;
      coloredConfig.lighting.ambientFloor = 0.15f;

      backend->renderIteration(coloredConfig);
      rt::RenderBuffer coloredBuffer = backend->downloadRenderBuffer();

      double redSum = 0.0, blueSum = 0.0;
      for (const auto& c : coloredBuffer.color) {
        require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z),
                "non-finite color in per-point-color scene");
        redSum  += c.x;
        blueSum += c.z;
      }
      require(blueSum > 0.0, "per-point blue color must produce visible output");
      require(blueSum > redSum + 0.05,
              "per-point color override (blue) should dominate over baseColor (red)");
    }

    // ---- Dual-IAS regression: curves and point clouds coexist ----
    // This is the regression test for the bug where adding a point cloud caused
    // Metal's built-in curve intersection to stop working.  Both the red point
    // cloud (left half) and the green curve sphere (right half) must be visible.
    {
      rt::RTScene mixedScene;

      rt::RTPointCloud cloud;
      cloud.name = "mixed_points";
      cloud.centers = {{-0.6f, 0.0f, 0.0f}};
      cloud.radius = 0.35f;
      cloud.baseColor = {1.0f, 0.0f, 0.0f, 1.0f};
      mixedScene.pointClouds.push_back(cloud);

      rt::RTCurveNetwork curveNet;
      curveNet.name = "mixed_curve";
      curveNet.baseColor = {0.0f, 1.0f, 0.0f, 1.0f};
      rt::RTCurvePrimitive curveSphere;
      curveSphere.type   = rt::RTCurvePrimitiveType::Sphere;
      curveSphere.p0     = {0.6f, 0.0f, 0.0f};
      curveSphere.radius = 0.35f;
      curveNet.primitives.push_back(curveSphere);
      mixedScene.curveNetworks.push_back(curveNet);

      mixedScene.hash = 202;

      backend->setScene(mixedScene);
      backend->updateCamera(camera);
      backend->resetAccumulation();

      rt::RenderConfig mixedConfig;
      mixedConfig.samplesPerIteration = 1;
      mixedConfig.maxBounces = 1;
      mixedConfig.accumulate = false;
      mixedConfig.lighting.backgroundColor = {0.0f, 0.0f, 0.0f};
      mixedConfig.lighting.environmentIntensity = 0.0f;
      mixedConfig.lighting.mainLightIntensity = 1.0f;
      mixedConfig.lighting.ambientFloor = 0.15f;

      backend->renderIteration(mixedConfig);
      rt::RenderBuffer mixedBuffer = backend->downloadRenderBuffer();

      double redSum = 0.0, greenSum = 0.0;
      for (const auto& c : mixedBuffer.color) {
        require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z),
                "non-finite color in mixed curve+point scene");
        redSum   += c.x;
        greenSum += c.y;
      }
      require(redSum > 0.0,
              "point cloud (red, left) must be visible in mixed curve+point scene");
      require(greenSum > 0.0,
              "curve network (green, right) must remain visible when a point cloud is "
              "also present — dual-IAS regression");
    }

    std::cout << "backend_smoke_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string message = e.what();
    if (isSkippableBackendError(message)) {
      std::cout << "backend_smoke_test skipped: " << message << std::endl;
      return 0;
    }
    std::cerr << "backend_smoke_test failed: " << message << std::endl;
    return 1;
  }
}
