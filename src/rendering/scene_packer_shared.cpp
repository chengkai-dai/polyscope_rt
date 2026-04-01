#include "rendering/scene_packer_internal.h"

#include <algorithm>

namespace rt {
namespace scene_packer_detail {

float4 makeFloat4(const glm::vec3& v, float w) {
  return simd_make_float4(v.x, v.y, v.z, w);
}

std::string contourObjectKey(const std::string& meshName) {
  constexpr const char* primitiveTag = "/primitive_";
  const size_t pos = meshName.rfind(primitiveTag);
  if (pos == std::string::npos) return meshName;
  return meshName.substr(0, pos);
}

uint32_t registerTexture(SceneGpuAccumulator& acc, const RTTexture& tex) {
  auto existing = acc.textureLookup.find(tex.cacheKey);
  if (existing != acc.textureLookup.end()) return existing->second;

  const uint32_t idx = static_cast<uint32_t>(acc.textures.size());
  const uint32_t offset = static_cast<uint32_t>(acc.texturePixels.size());
  GPUTexture metadata{};
  metadata.data = simd_make_uint4(offset, tex.width, tex.height, 0u);
  acc.textures.push_back(metadata);
  acc.textureLookup.emplace(tex.cacheKey, idx);
  for (const glm::vec4& pixel : tex.pixels) {
    acc.texturePixels.push_back(simd_make_float4(pixel.r, pixel.g, pixel.b, pixel.a));
  }
  return idx;
}

PackedBoundingBox makePackedBoundingBox(const glm::vec3& center, float radius) {
  PackedBoundingBox bbox;
  bbox.min = float3{center.x - radius, center.y - radius, center.z - radius};
  bbox.max = float3{center.x + radius, center.y + radius, center.z + radius};
  return bbox;
}

float textureAverageLuminance(const RTTexture& texture) {
  if (texture.pixels.empty()) return 1.0f;

  double sum = 0.0;
  for (const glm::vec4& pixel : texture.pixels) {
    sum += 0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b;
  }

  return static_cast<float>(sum / static_cast<double>(texture.pixels.size()));
}

float emissiveMeshPowerEstimate(const RTMesh& mesh) {
  float power = 0.2126f * mesh.emissiveFactor.r +
                0.7152f * mesh.emissiveFactor.g +
                0.0722f * mesh.emissiveFactor.b;
  if (mesh.hasEmissiveTexture && !mesh.emissiveTexture.pixels.empty()) {
    power *= textureAverageLuminance(mesh.emissiveTexture);
  }
  return std::max(power, 0.0f);
}

} // namespace scene_packer_detail
} // namespace rt
