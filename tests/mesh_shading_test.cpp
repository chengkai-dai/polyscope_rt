#include <cmath>
#include <exception>
#include <iostream>
#include <set>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering/ray_tracing_types.h"
#include "rendering_test_utils.h"
#include "test_helpers.h"

namespace {

void testSmoothNormalsAffectShading(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  auto cfg = standardConfig();
  cfg.lighting.mainLightIntensity = 1.5f;

  {
    rt::RTScene scene;
    scene.meshes.push_back(makeSphere(0.9f, 12u, 16u, false, {0.8f, 0.8f, 0.8f, 1.0f}));
    scene.hash = 700;
    backend.setScene(scene);
    backend.resetAccumulation();
    backend.renderIteration(cfg);
  }
  const rt::RenderBuffer flatBuf = backend.downloadRenderBuffer();

  {
    rt::RTScene scene;
    scene.meshes.push_back(makeSphere(0.9f, 12u, 16u, true, {0.8f, 0.8f, 0.8f, 1.0f}));
    scene.hash = 701;
    backend.setScene(scene);
    backend.resetAccumulation();
    backend.renderIteration(cfg);
  }
  const rt::RenderBuffer smoothBuf = backend.downloadRenderBuffer();

  int diffCount = 0;
  for (size_t i = 0; i < flatBuf.color.size(); ++i) {
    if (glm::length(flatBuf.color[i] - smoothBuf.color[i]) > 0.005f) ++diffCount;
  }
  require(diffCount > 20, "smooth normals should produce measurably different shading from flat normals");
}

void testMeshBaseColorAffectsOutput(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);

  auto renderQuad = [&](glm::vec4 color, uint32_t hash) {
    rt::RTScene scene;
    rt::RTMesh quad;
    quad.name = "color_quad";
    quad.vertices = {{-1.2f, -0.8f, 0.0f}, {1.2f, -0.8f, 0.0f}, {1.2f, 0.8f, 0.0f}, {-1.2f, 0.8f, 0.0f}};
    quad.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    quad.indices = {glm::uvec3(0, 1, 2), glm::uvec3(0, 2, 3)};
    quad.baseColorFactor = color;
    scene.meshes.push_back(quad);
    scene.hash = hash;
    backend.setScene(scene);
    backend.updateCamera(cam);
    backend.resetAccumulation();
    backend.renderIteration(darkConfig());
    return backend.downloadRenderBuffer();
  };

  const rt::RenderBuffer redBuf = renderQuad({1.0f, 0.0f, 0.0f, 1.0f}, 710);
  const rt::RenderBuffer blueBuf = renderQuad({0.0f, 0.0f, 1.0f, 1.0f}, 711);

  double redR = 0.0, redB = 0.0, blueR = 0.0, blueB = 0.0;
  int diffCount = 0;
  for (size_t i = 0; i < redBuf.color.size(); ++i) {
    redR += redBuf.color[i].x;
    redB += redBuf.color[i].z;
    blueR += blueBuf.color[i].x;
    blueB += blueBuf.color[i].z;
    if (glm::length(redBuf.color[i] - blueBuf.color[i]) > 0.01f) ++diffCount;
  }
  require(diffCount > 10, "red and blue baseColor meshes should produce visibly different output");
  require(redR > redB, "red mesh should remain red-dominant");
  require(blueB > blueR, "blue mesh should remain blue-dominant");
}

void testWireframeMeshRenders(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene scene;
  rt::RTMesh mesh;
  mesh.name = "wf_mesh";
  mesh.vertices = {{-1, -0.8f, 0}, {1, -0.8f, 0}, {1, 0.8f, 0}, {-1, 0.8f, 0}};
  mesh.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  mesh.indices = {glm::uvec3(0, 1, 2), glm::uvec3(0, 2, 3)};
  mesh.baseColorFactor = {0.6f, 0.6f, 0.6f, 1.0f};
  mesh.wireframe = true;
  mesh.edgeColor = {1.0f, 0.0f, 0.0f};
  mesh.edgeWidth = 2.0f;
  scene.meshes.push_back(mesh);
  scene.hash = 720;
  backend.setScene(scene);
  backend.resetAccumulation();
  backend.renderIteration(darkConfig());
  const rt::RenderBuffer buf = backend.downloadRenderBuffer();

  double colorSum = 0.0;
  for (const auto& c : buf.color) colorSum += c.x + c.y + c.z;
  require(colorSum > 0.0, "wireframe mesh must produce non-black output");
}

void testPerPrimitiveCurveColorsVisible(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene scene;
  rt::RTCurveNetwork cn;
  cn.name = "colored_curve";
  cn.baseColor = {1.0f, 0.0f, 0.0f, 1.0f};
  rt::RTCurvePrimitive cyl;
  cyl.type = rt::RTCurvePrimitiveType::Cylinder;
  cyl.p0 = {-0.8f, 0.0f, 0.0f};
  cyl.p1 = {0.8f, 0.0f, 0.0f};
  cyl.radius = 0.3f;
  cn.primitives.push_back(cyl);
  cn.primitiveColors = {{0.0f, 0.0f, 1.0f}};

  scene.curveNetworks.push_back(cn);
  scene.hash = 730;
  backend.setScene(scene);
  backend.resetAccumulation();
  backend.renderIteration(darkConfig());
  const rt::RenderBuffer buf = backend.downloadRenderBuffer();

  double redSum = 0.0, blueSum = 0.0;
  for (const auto& c : buf.color) {
    redSum += c.x;
    blueSum += c.z;
  }
  require(blueSum > redSum + 0.05, "per-primitive curve color should dominate over base color");
}

void testMultipleMeshesCoexist(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene scene;
  rt::RTMesh left;
  left.name = "left";
  left.vertices = {{-2.0f, -0.8f, 0.0f}, {-0.2f, -0.8f, 0.0f}, {-0.2f, 0.8f, 0.0f}, {-2.0f, 0.8f, 0.0f}};
  left.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  left.indices = {glm::uvec3(0, 1, 2), glm::uvec3(0, 2, 3)};
  left.baseColorFactor = {1.0f, 0.0f, 0.0f, 1.0f};
  scene.meshes.push_back(left);

  rt::RTMesh right;
  right.name = "right";
  right.vertices = {{0.2f, -0.8f, 0.0f}, {2.0f, -0.8f, 0.0f}, {2.0f, 0.8f, 0.0f}, {0.2f, 0.8f, 0.0f}};
  right.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  right.indices = {glm::uvec3(0, 1, 2), glm::uvec3(0, 2, 3)};
  right.baseColorFactor = {0.0f, 0.0f, 1.0f, 1.0f};
  scene.meshes.push_back(right);

  scene.hash = 740;
  backend.setScene(scene);
  backend.resetAccumulation();
  backend.renderIteration(darkConfig());
  const rt::RenderBuffer buf = backend.downloadRenderBuffer();

  std::set<uint32_t> visibleIds;
  int leftHitCount = 0;
  int rightHitCount = 0;
  for (int y = 0; y < static_cast<int>(cam.height); ++y) {
    for (int x = 0; x < static_cast<int>(cam.width); ++x) {
      const uint32_t objectId = buf.objectId[pixelIndex(cam.width, x, y)];
      if (objectId == 0u) continue;
      visibleIds.insert(objectId);
      if (x < static_cast<int>(cam.width / 2)) ++leftHitCount;
      else ++rightHitCount;
    }
  }
  require(visibleIds.size() >= 2u, "two visible meshes should produce distinct object IDs");
  require(leftHitCount > 0 && rightHitCount > 0, "both meshes should occupy image space");
}

void testEmptySceneReturnsBackground(rt::IRayTracingBackend& backend) {
  const auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene empty;
  empty.hash = 760;
  backend.setScene(empty);
  backend.resetAccumulation();

  auto cfg = darkConfig();
  cfg.lighting.backgroundColor = {0.2f, 0.4f, 0.8f};
  backend.renderIteration(cfg);
  const rt::RenderBuffer buf = backend.downloadRenderBuffer();

  double redSum = 0.0, blueSum = 0.0;
  for (const auto& c : buf.color) {
    redSum += c.x;
    blueSum += c.z;
  }
  require(blueSum > redSum + 1.0, "empty scene should show the configured background color");
}

} // namespace

int main() {
  try {
    auto backend = rt::createBackend(rt::BackendType::Metal);
    testSmoothNormalsAffectShading(*backend);
    testMeshBaseColorAffectsOutput(*backend);
    testWireframeMeshRenders(*backend);
    testPerPrimitiveCurveColorsVisible(*backend);
    testMultipleMeshesCoexist(*backend);
    testEmptySceneReturnsBackground(*backend);
    std::cout << "mesh_shading_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "mesh_shading_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "mesh_shading_test FAILED: " << msg << std::endl;
    return 1;
  }
}
