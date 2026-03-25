#include "rendering/toon_shading.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rt {
namespace {

size_t pixelIndex(uint32_t x, uint32_t y, uint32_t width) { return static_cast<size_t>(y) * width + x; }

bool validHit(const RenderBuffer& buffer, size_t index) {
  return index < buffer.objectId.size() && buffer.objectId[index] != 0u;
}

float objectContourContribution(const RenderBuffer& buffer, uint32_t x, uint32_t y, const ToonSettings& settings) {
  const size_t centerIndex = pixelIndex(x, y, buffer.width);
  const uint32_t centerObject = centerIndex < buffer.objectId.size() ? buffer.objectId[centerIndex] : 0u;

  int differentCorners = 0;
  int differentEdges = 0;
  const int offsets[8][2] = {{-1, 1}, {0, 1}, {1, 1}, {-1, 0}, {1, 0}, {-1, -1}, {0, -1}, {1, -1}};
  for (int i = 0; i < 8; ++i) {
    int nx = static_cast<int>(x) + offsets[i][0];
    int ny = static_cast<int>(y) + offsets[i][1];
    if (nx < 0 || ny < 0 || nx >= static_cast<int>(buffer.width) || ny >= static_cast<int>(buffer.height)) continue;
    const size_t neighborIndex = pixelIndex(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), buffer.width);
    const uint32_t neighborObject = neighborIndex < buffer.objectId.size() ? buffer.objectId[neighborIndex] : 0u;
    if (neighborObject == centerObject) continue;
    if (i == 0 || i == 2 || i == 5 || i == 7) {
      differentCorners++;
    } else {
      differentEdges++;
    }
  }

  float contour = differentCorners * (1.0f / 6.0f) + differentEdges * (1.0f / 3.0f);
  return std::clamp(contour * settings.objectThreshold, 0.0f, 1.0f);
}

float normalizedDepth(float depth, float minDepth, float maxDepth) {
  if (!(maxDepth > minDepth) || depth <= 0.0f) return 0.0f;
  return std::clamp((depth - minDepth) / (maxDepth - minDepth), 0.0f, 1.0f);
}

float detailContourContribution(const RenderBuffer& buffer, uint32_t x, uint32_t y, const ToonSettings& settings,
                                float minDepth, float maxDepth) {
  const size_t centerIndex = pixelIndex(x, y, buffer.width);
  if (!validHit(buffer, centerIndex)) return 0.0f;

  auto sampleNormal = [&](int ox, int oy) {
    int nx = std::clamp(static_cast<int>(x) + ox, 0, static_cast<int>(buffer.width) - 1);
    int ny = std::clamp(static_cast<int>(y) + oy, 0, static_cast<int>(buffer.height) - 1);
    size_t idx = pixelIndex(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), buffer.width);
    if (!validHit(buffer, idx)) return glm::vec3(0.0f);
    return glm::normalize(buffer.normal[idx]);
  };
  auto sampleDepth = [&](int ox, int oy) {
    int nx = std::clamp(static_cast<int>(x) + ox, 0, static_cast<int>(buffer.width) - 1);
    int ny = std::clamp(static_cast<int>(y) + oy, 0, static_cast<int>(buffer.height) - 1);
    size_t idx = pixelIndex(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny), buffer.width);
    if (idx >= buffer.linearDepth.size() || !validHit(buffer, idx)) return 0.0f;
    return normalizedDepth(buffer.linearDepth[idx], minDepth, maxDepth);
  };

  glm::vec3 A = sampleNormal(-1, 1);
  glm::vec3 B = sampleNormal(0, 1);
  glm::vec3 C = sampleNormal(1, 1);
  glm::vec3 D = sampleNormal(-1, 0);
  glm::vec3 E = sampleNormal(1, 0);
  glm::vec3 F = sampleNormal(-1, -1);
  glm::vec3 G = sampleNormal(0, -1);
  glm::vec3 H = sampleNormal(1, -1);

  const float k0 = 17.0f / 23.75f;
  const float k1 = 61.0f / 23.75f;
  glm::vec3 gradY = k0 * A + k1 * B + k0 * C - k0 * F - k1 * G - k0 * H;
  glm::vec3 gradX = k0 * C + k1 * E + k0 * H - k0 * A - k1 * D - k0 * F;
  float normalGradient = glm::length(gradX) + glm::length(gradY);
  float normalEdge = glm::smoothstep(2.0f, 3.0f, normalGradient * settings.normalThreshold);

  float Az = sampleDepth(-1, 1);
  float Bz = sampleDepth(0, 1);
  float Cz = sampleDepth(1, 1);
  float Dz = sampleDepth(-1, 0);
  float Ez = sampleDepth(1, 0);
  float Fz = sampleDepth(-1, -1);
  float Gz = sampleDepth(0, -1);
  float Hz = sampleDepth(1, -1);
  float Xz = sampleDepth(0, 0);
  float g = (std::abs(Az + 2.0f * Bz + Cz - Fz - 2.0f * Gz - Hz) +
             std::abs(Cz + 2.0f * Ez + Hz - Az - 2.0f * Dz - Fz)) /
            8.0f;
  float l = (8.0f * Xz - Az - Bz - Cz - Dz - Ez - Fz - Gz - Hz) / 3.0f;
  float depthEdge = glm::smoothstep(0.03f, 0.1f, (l + g) * settings.depthThreshold);

  return std::clamp(normalEdge + depthEdge, 0.0f, 1.0f);
}

} // namespace

std::vector<glm::vec3> applyToonShading(const RenderBuffer& buffer, const ToonSettings& settings) {
  if (!settings.enabled) return buffer.color;

  float minDepth = std::numeric_limits<float>::max();
  float maxDepth = 0.0f;
  for (size_t i = 0; i < buffer.linearDepth.size(); ++i) {
    if (!validHit(buffer, i)) continue;
    minDepth = std::min(minDepth, buffer.linearDepth[i]);
    maxDepth = std::max(maxDepth, buffer.linearDepth[i]);
  }
  if (!std::isfinite(minDepth)) {
    minDepth = 0.0f;
    maxDepth = 1.0f;
  }

  std::vector<glm::vec3> output(buffer.color.size(), glm::vec3(0.0f));
  for (uint32_t y = 0; y < buffer.height; ++y) {
    for (uint32_t x = 0; x < buffer.width; ++x) {
      const size_t index = pixelIndex(x, y, buffer.width);
      glm::vec3 color = validHit(buffer, index) ? buffer.color[index] : settings.backgroundColor;
      float objectEdge = objectContourContribution(buffer, x, y, settings);
      float detailEdge = detailContourContribution(buffer, x, y, settings, minDepth, maxDepth);
      float mixed = 0.0f;
      if (settings.enableDetailContour) {
        mixed = std::clamp(detailEdge * settings.detailContourStrength, 0.0f, 1.0f);
        color = glm::mix(color, settings.edgeColor, mixed);
      }
      if (settings.enableObjectContour) {
        mixed = std::clamp(objectEdge * settings.objectContourStrength, 0.0f, 1.0f);
        color = glm::mix(color, settings.edgeColor, mixed);
      }
      output[index] = color;
    }
  }
  return output;
}

} // namespace rt
