#include <cmath>
#include <exception>
#include <iostream>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering_test_utils.h"
#include "test_helpers.h"

namespace {

void testCurveAndPointCloudPipelines(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  {
    rt::RTScene curveScene;
    rt::RTMesh floor = makeQuad("floor", {-5, 0, -5}, {5, 0, -5}, {5, 0, 5}, {-5, 0, 5}, {0, 1, 0});
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
    backend.setScene(curveScene);
    backend.resetAccumulation();
    backend.renderIteration(standardConfig());
    const rt::RenderBuffer buf = backend.downloadRenderBuffer();

    double colorSum = 0.0;
    for (const auto& c : buf.color) colorSum += c.x + c.y + c.z;
    require(colorSum > 0.0, "curve scene should produce visible output");
  }

  {
    rt::RTScene pointScene;
    rt::RTPointCloud cloud;
    cloud.name = "blue_cloud";
    cloud.centers = {{0.0f, 0.0f, 0.0f}};
    cloud.radius = 0.55f;
    cloud.baseColor = {1.0f, 0.0f, 0.0f, 1.0f};
    cloud.colors = {{0.0f, 0.0f, 1.0f}};
    pointScene.pointClouds.push_back(cloud);
    pointScene.hash = 201;
    backend.setScene(pointScene);
    backend.resetAccumulation();

    auto cfg = standardConfig();
    cfg.lighting.backgroundColor = {0.0f, 0.0f, 0.0f};
    cfg.lighting.environmentIntensity = 0.0f;
    cfg.lighting.mainLightIntensity = 1.0f;
    cfg.lighting.ambientFloor = 0.15f;
    backend.renderIteration(cfg);
    const rt::RenderBuffer buf = backend.downloadRenderBuffer();

    double redSum = 0.0, blueSum = 0.0;
    for (const auto& c : buf.color) {
      redSum += c.x;
      blueSum += c.z;
    }
    require(blueSum > redSum + 0.05, "per-point color override should dominate base color");
  }

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
    curveSphere.type = rt::RTCurvePrimitiveType::Sphere;
    curveSphere.p0 = {0.6f, 0.0f, 0.0f};
    curveSphere.radius = 0.35f;
    curveNet.primitives.push_back(curveSphere);
    mixedScene.curveNetworks.push_back(curveNet);
    mixedScene.hash = 202;

    backend.setScene(mixedScene);
    backend.resetAccumulation();

    auto cfg = standardConfig();
    cfg.lighting.backgroundColor = {0.0f, 0.0f, 0.0f};
    cfg.lighting.environmentIntensity = 0.0f;
    cfg.lighting.mainLightIntensity = 1.0f;
    cfg.lighting.ambientFloor = 0.15f;
    backend.renderIteration(cfg);
    const rt::RenderBuffer buf = backend.downloadRenderBuffer();

    double redSum = 0.0, greenSum = 0.0;
    for (const auto& c : buf.color) {
      redSum += c.x;
      greenSum += c.y;
    }
    require(redSum > 0.0, "point cloud should remain visible in mixed scene");
    require(greenSum > 0.0, "curve primitive should remain visible in mixed scene");
  }
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    testCurveAndPointCloudPipelines(*backend);
    std::cout << "geometry_primitives_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "geometry_primitives_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "geometry_primitives_test FAILED: " << msg << std::endl;
    return 1;
  }
}
