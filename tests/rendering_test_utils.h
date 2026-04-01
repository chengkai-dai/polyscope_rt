#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "rendering/ray_tracing_types.h"

inline rt::RTCamera makeCamera(uint32_t w = 64, uint32_t h = 64) {
  rt::RTCamera cam;
  cam.position = {0.0f, 0.2f, 3.0f};
  cam.lookDir = glm::normalize(glm::vec3(0.0f, -0.05f, -1.0f));
  cam.upDir = {0.0f, 1.0f, 0.0f};
  cam.rightDir = {1.0f, 0.0f, 0.0f};
  cam.fovYDegrees = 45.0f;
  cam.aspect = static_cast<float>(w) / h;
  cam.viewMatrix = glm::lookAt(cam.position, glm::vec3(0.0f), cam.upDir);
  cam.projectionMatrix = glm::perspective(glm::radians(cam.fovYDegrees), cam.aspect, 0.01f, 100.0f);
  cam.width = w;
  cam.height = h;
  return cam;
}

inline rt::RenderConfig darkConfig() {
  rt::RenderConfig cfg;
  cfg.renderMode = rt::RenderMode::Toon;
  cfg.samplesPerIteration = 1;
  cfg.maxBounces = 1;
  cfg.accumulate = false;
  cfg.toon.enabled = true;
  cfg.toon.enableDetailContour = false;
  cfg.toon.enableObjectContour = false;
  cfg.toon.tonemapExposure = 2.0f;
  cfg.toon.tonemapGamma = 2.2f;
  cfg.toon.bandCount = 0;
  cfg.lighting.backgroundColor = {0.0f, 0.0f, 0.0f};
  cfg.lighting.environmentIntensity = 0.0f;
  cfg.lighting.mainLightIntensity = 1.0f;
  cfg.lighting.ambientFloor = 0.3f;
  return cfg;
}

inline rt::RenderConfig standardConfig() {
  rt::RenderConfig cfg;
  cfg.renderMode = rt::RenderMode::Standard;
  cfg.samplesPerIteration = 1;
  cfg.maxBounces = 1;
  cfg.accumulate = false;
  cfg.lighting.backgroundColor = {0.0f, 0.0f, 0.0f};
  cfg.lighting.environmentIntensity = 0.0f;
  cfg.lighting.ambientFloor = 0.0f;
  cfg.lighting.mainLightIntensity = 1.0f;
  return cfg;
}

inline size_t pixelIndex(uint32_t width, int x, int y) {
  return static_cast<size_t>(y) * width + static_cast<size_t>(x);
}

inline glm::vec3 averageCenterPatchVec3(const rt::RTCamera& cam, const std::vector<glm::vec3>& values, int halfExtent = 4) {
  glm::vec3 sum(0.0f);
  int count = 0;
  const int centerX = static_cast<int>(cam.width / 2);
  const int centerY = static_cast<int>(cam.height / 2);
  for (int y = std::max(0, centerY - halfExtent); y <= std::min(static_cast<int>(cam.height) - 1, centerY + halfExtent); ++y) {
    for (int x = std::max(0, centerX - halfExtent); x <= std::min(static_cast<int>(cam.width) - 1, centerX + halfExtent); ++x) {
      sum += values[pixelIndex(cam.width, x, y)];
      ++count;
    }
  }
  return count > 0 ? sum / static_cast<float>(count) : glm::vec3(0.0f);
}

inline float averageCenterPatchScalar(const rt::RTCamera& cam, const std::vector<float>& values, int halfExtent = 4) {
  float sum = 0.0f;
  int count = 0;
  const int centerX = static_cast<int>(cam.width / 2);
  const int centerY = static_cast<int>(cam.height / 2);
  for (int y = std::max(0, centerY - halfExtent); y <= std::min(static_cast<int>(cam.height) - 1, centerY + halfExtent); ++y) {
    for (int x = std::max(0, centerX - halfExtent); x <= std::min(static_cast<int>(cam.width) - 1, centerX + halfExtent); ++x) {
      sum += values[pixelIndex(cam.width, x, y)];
      ++count;
    }
  }
  return count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

inline double averageCenterPatchCoverage(const rt::RTCamera& cam, const std::vector<uint32_t>& values, int halfExtent = 4) {
  double sum = 0.0;
  int count = 0;
  const int centerX = static_cast<int>(cam.width / 2);
  const int centerY = static_cast<int>(cam.height / 2);
  for (int y = std::max(0, centerY - halfExtent); y <= std::min(static_cast<int>(cam.height) - 1, centerY + halfExtent); ++y) {
    for (int x = std::max(0, centerX - halfExtent); x <= std::min(static_cast<int>(cam.width) - 1, centerX + halfExtent); ++x) {
      sum += values[pixelIndex(cam.width, x, y)] != 0u ? 1.0 : 0.0;
      ++count;
    }
  }
  return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

inline double luminance(const glm::vec3& c) {
  return 0.2126 * static_cast<double>(c.r) +
         0.7152 * static_cast<double>(c.g) +
         0.0722 * static_cast<double>(c.b);
}

inline rt::RTMesh makeSphere(float radius, uint32_t latSegments, uint32_t lonSegments,
                             bool smoothNormals = true, const glm::vec4& color = {1.0f, 1.0f, 1.0f, 1.0f}) {
  rt::RTMesh mesh;
  mesh.name = "sphere";
  for (uint32_t y = 0; y <= latSegments; ++y) {
    const float v = static_cast<float>(y) / static_cast<float>(latSegments);
    const float theta = v * glm::pi<float>();
    const float sinTheta = std::sin(theta);
    const float cosTheta = std::cos(theta);
    for (uint32_t x = 0; x <= lonSegments; ++x) {
      const float u = static_cast<float>(x) / static_cast<float>(lonSegments);
      const float phi = u * glm::two_pi<float>();
      const float sinPhi = std::sin(phi);
      const float cosPhi = std::cos(phi);
      const glm::vec3 p(radius * sinTheta * cosPhi, radius * cosTheta, radius * sinTheta * sinPhi);
      mesh.vertices.push_back(p);
      if (smoothNormals) mesh.normals.push_back(glm::normalize(p));
      mesh.texcoords.emplace_back(u, v);
    }
  }

  auto idx = [lonSegments](uint32_t y, uint32_t x) { return y * (lonSegments + 1u) + x; };
  for (uint32_t y = 0; y < latSegments; ++y) {
    for (uint32_t x = 0; x < lonSegments; ++x) {
      const uint32_t i0 = idx(y, x);
      const uint32_t i1 = idx(y, x + 1u);
      const uint32_t i2 = idx(y + 1u, x + 1u);
      const uint32_t i3 = idx(y + 1u, x);
      if (y != 0u) mesh.indices.emplace_back(i0, i3, i1);
      if (y + 1u != latSegments) mesh.indices.emplace_back(i1, i3, i2);
    }
  }

  mesh.baseColorFactor = color;
  mesh.roughnessFactor = 0.7f;
  return mesh;
}

inline rt::RTMesh makeQuad(const char* name, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 normal) {
  rt::RTMesh mesh;
  mesh.name = name;
  mesh.vertices = {p0, p1, p2, p3};
  mesh.normals = {normal, normal, normal, normal};
  mesh.texcoords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  mesh.indices = {glm::uvec3(0, 1, 2), glm::uvec3(0, 2, 3)};
  return mesh;
}

inline rt::RTMesh makeCameraFacingQuad(const char* name, bool faceCamera, const glm::vec4& color) {
  rt::RTMesh mesh;
  mesh.name = name;
  mesh.vertices = {{-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}};
  mesh.texcoords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
  if (faceCamera) {
    mesh.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    mesh.indices = {glm::uvec3(0, 1, 2), glm::uvec3(0, 2, 3)};
  } else {
    mesh.normals = {{0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};
    mesh.indices = {glm::uvec3(0, 2, 1), glm::uvec3(0, 3, 2)};
  }
  mesh.baseColorFactor = color;
  mesh.roughnessFactor = 0.6f;
  return mesh;
}
