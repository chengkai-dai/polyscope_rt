// GPU-level rendering quality tests.  These validate that data computed in
// polyscope_scene_snapshot / metal_scene_builder actually reaches the pixels
// rendered by the path tracer.
//
// Tests:
//  1.  Smooth vertex normals produce different shading from flat normals
//  2.  Mesh with per-vertex colors renders the correct hue
//  3.  Wireframe mesh renders without crash and produces non-black output
//  4.  Per-primitive curve colors (blue cylinder) are visible in output
//  5.  Multiple meshes in the same scene both contribute to the render
//  6.  Accumulation: second frame increases sample count but keeps depth stable
//  7.  Empty scene (no geometry) produces the background color

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "rendering/ray_tracing_backend.h"
#include "rendering/ray_tracing_types.h"
#include "test_helpers.h"

namespace {

// Camera looking straight at the origin from +Z.
rt::RTCamera makeCamera(uint32_t w = 64, uint32_t h = 64) {
  rt::RTCamera cam;
  cam.position   = {0.0f, 0.2f, 3.0f};
  cam.lookDir    = glm::normalize(glm::vec3(0.0f, -0.05f, -1.0f));
  cam.upDir      = {0.0f, 1.0f, 0.0f};
  cam.rightDir   = {1.0f, 0.0f, 0.0f};
  cam.fovYDegrees = 45.0f;
  cam.aspect     = static_cast<float>(w) / h;
  cam.viewMatrix = glm::lookAt(cam.position, glm::vec3(0.0f), cam.upDir);
  cam.projectionMatrix =
      glm::perspective(glm::radians(cam.fovYDegrees), cam.aspect, 0.01f, 100.0f);
  cam.width  = w;
  cam.height = h;
  return cam;
}

rt::RenderConfig darkConfig() {
  rt::RenderConfig cfg;
  cfg.renderMode = rt::RenderMode::Toon;
  cfg.samplesPerIteration = 1;
  cfg.maxBounces = 1;
  cfg.accumulate = false;
  // Toon settings: simple flat shading, no contours.
  cfg.toon.enabled = true;
  cfg.toon.enableDetailContour = false;
  cfg.toon.enableObjectContour = false;
  cfg.toon.tonemapExposure = 2.0f;
  cfg.toon.tonemapGamma = 2.2f;
  cfg.toon.bandCount = 0; // smooth shading (no quantization)
  cfg.lighting.backgroundColor     = {0.0f, 0.0f, 0.0f};
  cfg.lighting.environmentIntensity = 0.0f;
  cfg.lighting.mainLightIntensity   = 1.0f;
  cfg.lighting.ambientFloor         = 0.3f;
  return cfg;
}

// Build a tessellated unit sphere with the given lat/lon subdivision.
rt::RTMesh makeSphere(float r, uint32_t lat, uint32_t lon,
                      bool smoothNormals, const glm::vec4& color) {
  rt::RTMesh m;
  m.name = "sphere";
  for (uint32_t y = 0; y <= lat; ++y) {
    float v = static_cast<float>(y) / lat;
    float theta = v * glm::pi<float>();
    for (uint32_t x = 0; x <= lon; ++x) {
      float u = static_cast<float>(x) / lon;
      float phi = u * glm::two_pi<float>();
      glm::vec3 p(r * std::sin(theta) * std::cos(phi),
                  r * std::cos(theta),
                  r * std::sin(theta) * std::sin(phi));
      m.vertices.push_back(p);
      if (smoothNormals) m.normals.push_back(glm::normalize(p));
    }
  }
  auto idx = [&](uint32_t y, uint32_t x) { return y * (lon + 1) + x; };
  for (uint32_t y = 0; y < lat; ++y)
    for (uint32_t x = 0; x < lon; ++x) {
      if (y != 0)       m.indices.emplace_back(idx(y,x), idx(y+1,x), idx(y,x+1));
      if (y+1 != lat)   m.indices.emplace_back(idx(y,x+1), idx(y+1,x), idx(y+1,x+1));
    }
  m.baseColorFactor = color;
  m.roughnessFactor = 0.7f;
  return m;
}

// -------------------------------------------------------------------------
// Test 1: Smooth vertex normals yield visually different shading from flat.
// -------------------------------------------------------------------------
void testSmoothNormalsAffectShading(rt::IRayTracingBackend& backend) {
  auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  // Use a separate Standard-mode config for the shading test.
  rt::RenderConfig cfg;
  cfg.renderMode = rt::RenderMode::Standard;
  cfg.samplesPerIteration = 1;
  cfg.maxBounces = 1;
  cfg.accumulate = false;
  cfg.lighting.backgroundColor     = {0.0f, 0.0f, 0.0f};
  cfg.lighting.mainLightIntensity  = 1.5f;
  cfg.lighting.ambientFloor        = 0.0f;
  cfg.lighting.environmentIntensity = 0.0f;

  // Flat: no normals in RTMesh → shader computes geometric normal per triangle.
  {
    rt::RTScene s;
    auto sphere = makeSphere(0.9f, 12, 16, false, {0.8f, 0.8f, 0.8f, 1.0f});
    s.meshes.push_back(sphere);
    s.hash = 700;
    backend.setScene(s);
    backend.resetAccumulation();
    backend.renderIteration(cfg);
  }
  rt::RenderBuffer flatBuf = backend.downloadRenderBuffer();

  // Smooth: normals provided.
  {
    rt::RTScene s;
    auto sphere = makeSphere(0.9f, 12, 16, true, {0.8f, 0.8f, 0.8f, 1.0f});
    s.meshes.push_back(sphere);
    s.hash = 701;
    backend.setScene(s);
    backend.resetAccumulation();
    backend.renderIteration(cfg);
  }
  rt::RenderBuffer smoothBuf = backend.downloadRenderBuffer();

  // Both must produce non-black output.
  double flatSum = 0, smoothSum = 0;
  for (const auto& c : flatBuf.color)   flatSum   += c.x + c.y + c.z;
  for (const auto& c : smoothBuf.color) smoothSum += c.x + c.y + c.z;
  require(flatSum   > 0.0, "flat shaded sphere should produce non-black output");
  require(smoothSum > 0.0, "smooth shaded sphere should produce non-black output");

  // The two renders should differ (smooth shading changes normals at triangle edges).
  int diffCount = 0;
  for (size_t i = 0; i < flatBuf.color.size(); ++i)
    if (glm::length(flatBuf.color[i] - smoothBuf.color[i]) > 0.005f) ++diffCount;
  require(diffCount > 20,
          "smooth normals should produce measurably different shading from flat normals");
}

// -------------------------------------------------------------------------
// Test 2: Two meshes with different baseColorFactor produce different output.
//         This verifies the mesh color pipeline end-to-end without depending
//         on the exact behavior of the post-processing saturation curve.
// -------------------------------------------------------------------------
void testMeshBaseColorAffectsOutput(rt::IRayTracingBackend& backend) {
  auto cam = makeCamera();
  backend.resize(cam.width, cam.height);

  auto renderQuad = [&](glm::vec4 color, uint32_t hash) -> rt::RenderBuffer {
    rt::RTScene scene;
    rt::RTMesh m;
    m.name = "color_quad";
    m.vertices = {{-1.2f, -0.8f, 0.0f}, {1.2f, -0.8f, 0.0f},
                  {1.2f,  0.8f,  0.0f}, {-1.2f,  0.8f, 0.0f}};
    m.normals  = {{0,0,1},{0,0,1},{0,0,1},{0,0,1}};
    m.indices  = {glm::uvec3(0,1,2), glm::uvec3(0,2,3)};
    m.baseColorFactor = color;
    scene.meshes.push_back(m);
    scene.hash = hash;
    backend.setScene(scene);
    backend.updateCamera(cam);
    backend.resetAccumulation();
    backend.renderIteration(darkConfig());
    return backend.downloadRenderBuffer();
  };

  // Pure red mesh vs pure blue mesh — their renders must differ.
  rt::RenderBuffer redBuf  = renderQuad({1.0f, 0.0f, 0.0f, 1.0f}, 710);
  rt::RenderBuffer blueBuf = renderQuad({0.0f, 0.0f, 1.0f, 1.0f}, 711);

  // Both must produce non-black output.
  double redTotal = 0, blueTotal = 0;
  for (const auto& c : redBuf.color)  redTotal  += c.x + c.y + c.z;
  for (const auto& c : blueBuf.color) blueTotal += c.x + c.y + c.z;
  require(redTotal  > 0.0, "red mesh must produce non-black output");
  require(blueTotal > 0.0, "blue mesh must produce non-black output");

  // The two renders should differ — mesh color flows to the output.
  int diffCount = 0;
  for (size_t i = 0; i < redBuf.color.size(); ++i)
    if (glm::length(redBuf.color[i] - blueBuf.color[i]) > 0.01f) ++diffCount;
  require(diffCount > 10,
          "red and blue baseColorFactor meshes should produce visibly different output");

  // The red render should have more R relative to B; blue render should have more B relative to R.
  double redR = 0, redB = 0, blueR = 0, blueB = 0;
  for (const auto& c : redBuf.color)  { redR += c.x;  redB += c.z; }
  for (const auto& c : blueBuf.color) { blueR += c.x; blueB += c.z; }
  require(redR  > redB,  "red mesh: red channel should exceed blue channel in output");
  require(blueB > blueR, "blue mesh: blue channel should exceed red channel in output");
}

// -------------------------------------------------------------------------
// Test 3: Wireframe mesh renders without crash and produces non-black output.
// -------------------------------------------------------------------------
void testWireframeMeshRenders(rt::IRayTracingBackend& backend) {
  auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene scene;
  rt::RTMesh m;
  m.name = "wf_mesh";
  m.vertices = {{-1, -0.8f, 0}, {1, -0.8f, 0}, {1, 0.8f, 0}, {-1, 0.8f, 0}};
  m.normals  = {{0,0,1},{0,0,1},{0,0,1},{0,0,1}};
  m.indices  = {glm::uvec3(0,1,2), glm::uvec3(0,2,3)};
  m.baseColorFactor = {0.6f, 0.6f, 0.6f, 1.0f};
  m.wireframe = true;
  m.edgeColor = {1.0f, 0.0f, 0.0f};
  m.edgeWidth = 2.0f;
  scene.meshes.push_back(m);
  scene.hash = 720;
  backend.setScene(scene);
  backend.resetAccumulation();
  backend.renderIteration(darkConfig());
  rt::RenderBuffer buf = backend.downloadRenderBuffer();

  double colorSum = 0;
  for (const auto& c : buf.color) {
    require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z),
            "wireframe mesh must not produce non-finite colors");
    colorSum += c.x + c.y + c.z;
  }
  require(colorSum > 0.0, "wireframe mesh must produce non-black output");
}

// -------------------------------------------------------------------------
// Test 4: Per-primitive curve color (blue cylinder) is visible.
// -------------------------------------------------------------------------
void testPerPrimitiveCurveColorsVisible(rt::IRayTracingBackend& backend) {
  auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene scene;
  rt::RTCurveNetwork cn;
  cn.name      = "colored_curve";
  cn.baseColor = {1.0f, 0.0f, 0.0f, 1.0f};  // base = red

  rt::RTCurvePrimitive cyl;
  cyl.type   = rt::RTCurvePrimitiveType::Cylinder;
  cyl.p0     = {-0.8f, 0.0f, 0.0f};
  cyl.p1     = {0.8f, 0.0f, 0.0f};
  cyl.radius = 0.3f;
  cn.primitives.push_back(cyl);
  // Override with pure blue.
  cn.primitiveColors = {{0.0f, 0.0f, 1.0f}};

  scene.curveNetworks.push_back(cn);
  scene.hash = 730;
  backend.setScene(scene);
  backend.updateCamera(cam);
  backend.resetAccumulation();
  backend.renderIteration(darkConfig());
  rt::RenderBuffer buf = backend.downloadRenderBuffer();

  double redSum = 0, blueSum = 0;
  for (const auto& c : buf.color) { redSum += c.x; blueSum += c.z; }
  require(blueSum > 0.0, "per-primitive blue cylinder must produce blue output");
  require(blueSum > redSum + 0.05,
          "per-primitive color override (blue) must dominate over red baseColor");
}

// -------------------------------------------------------------------------
// Test 5: Two meshes in the same scene both contribute to the output.
// -------------------------------------------------------------------------
void testMultipleMeshesCoexist(rt::IRayTracingBackend& backend) {
  auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene scene;

  // Left red quad.
  rt::RTMesh left;
  left.name = "left";
  left.vertices = {{-2.0f,-0.8f,0.0f},{-0.2f,-0.8f,0.0f},
                   {-0.2f, 0.8f,0.0f},{-2.0f, 0.8f,0.0f}};
  left.normals  = {{0,0,1},{0,0,1},{0,0,1},{0,0,1}};
  left.indices  = {glm::uvec3(0,1,2),glm::uvec3(0,2,3)};
  left.baseColorFactor = {1.0f, 0.0f, 0.0f, 1.0f};
  scene.meshes.push_back(left);

  // Right blue quad.
  rt::RTMesh right;
  right.name = "right";
  right.vertices = {{0.2f,-0.8f,0.0f},{2.0f,-0.8f,0.0f},
                    {2.0f, 0.8f,0.0f},{0.2f, 0.8f,0.0f}};
  right.normals  = {{0,0,1},{0,0,1},{0,0,1},{0,0,1}};
  right.indices  = {glm::uvec3(0,1,2),glm::uvec3(0,2,3)};
  right.baseColorFactor = {0.0f, 0.0f, 1.0f, 1.0f};
  scene.meshes.push_back(right);

  scene.hash = 740;
  backend.setScene(scene);
  backend.updateCamera(cam);
  backend.resetAccumulation();
  backend.renderIteration(darkConfig());
  rt::RenderBuffer buf = backend.downloadRenderBuffer();

  double redSum = 0, blueSum = 0;
  for (const auto& c : buf.color) { redSum += c.x; blueSum += c.z; }
  require(redSum  > 0.5, "left red mesh must be visible");
  require(blueSum > 0.5, "right blue mesh must also be visible alongside the red one");
}

// -------------------------------------------------------------------------
// Test 6: Accumulation increases sample count while keeping depth stable.
// -------------------------------------------------------------------------
void testAccumulationKeepsDepthStable(rt::IRayTracingBackend& backend) {
  auto cam = makeCamera();
  backend.resize(cam.width, cam.height);

  rt::RTScene s;
  rt::RTMesh quad;
  quad.name = "acc_quad";
  quad.vertices = {{-1,-0.8f,0},{1,-0.8f,0},{1,0.8f,0},{-1,0.8f,0}};
  quad.normals  = {{0,0,1},{0,0,1},{0,0,1},{0,0,1}};
  quad.indices  = {glm::uvec3(0,1,2),glm::uvec3(0,2,3)};
  quad.baseColorFactor = {0.5f,0.5f,0.5f,1.0f};
  s.meshes.push_back(quad);
  s.hash = 750;
  backend.setScene(s);
  backend.updateCamera(cam);
  backend.resetAccumulation();

  auto cfg = darkConfig();
  cfg.accumulate = true;
  cfg.samplesPerIteration = 1;

  backend.renderIteration(cfg);
  rt::RenderBuffer frame1 = backend.downloadRenderBuffer();
  backend.renderIteration(cfg);
  rt::RenderBuffer frame2 = backend.downloadRenderBuffer();

  require(frame1.accumulatedSamples == 1, "first frame should have 1 accumulated sample");
  require(frame2.accumulatedSamples == 2, "second frame should have 2 accumulated samples");
  // Depth buffer is written on the first frame and held constant.
  require(frame1.depth == frame2.depth,
          "depth buffer must remain stable across accumulation frames");
  require(frame1.objectId == frame2.objectId,
          "object ID buffer must remain stable across accumulation frames");
}

// -------------------------------------------------------------------------
// Test 7: Empty scene (no geometry) produces approximately the background color.
// -------------------------------------------------------------------------
void testEmptySceneReturnsBackground(rt::IRayTracingBackend& backend) {
  auto cam = makeCamera();
  backend.resize(cam.width, cam.height);
  backend.updateCamera(cam);

  rt::RTScene empty;
  empty.hash = 760;
  backend.setScene(empty);
  backend.resetAccumulation();

  auto cfg = darkConfig();
  cfg.lighting.backgroundColor = {0.2f, 0.4f, 0.8f};  // custom background
  backend.renderIteration(cfg);
  rt::RenderBuffer buf = backend.downloadRenderBuffer();

  double blueSum = 0;
  for (const auto& c : buf.color) {
    require(std::isfinite(c.x) && std::isfinite(c.y) && std::isfinite(c.z),
            "empty scene must not produce non-finite colors");
    blueSum += c.z;
  }
  // Background is blue-dominant, so blue channel should dominate overall.
  double redSum = 0;
  for (const auto& c : buf.color) redSum += c.x;
  require(blueSum > redSum + 1.0,
          "empty scene should show background color (blue > red)");
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
    testAccumulationKeepsDepthStable(*backend);
    testEmptySceneReturnsBackground(*backend);

    std::cout << "rendering_quality_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    if (isSkippableBackendError(msg)) {
      std::cout << "rendering_quality_test skipped: " << msg << std::endl;
      return 0;
    }
    std::cerr << "rendering_quality_test FAILED: " << msg << std::endl;
    return 1;
  }
}
