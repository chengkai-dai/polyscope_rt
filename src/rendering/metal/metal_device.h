#pragma once

#import <Metal/Metal.h>
#import <simd/simd.h>

#include <algorithm>
#include <string>
#include <vector>

#include "rendering/gpu_shared_types.h"
#include "rendering/ray_tracing_types.h"

namespace metal_rt {

std::string resolveShaderLibraryPath(const std::string& requestedPath);
std::string buildNSErrorMessage(NSError* error);

void dispatchThreads(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipelineState,
                     uint32_t width, uint32_t height);

std::string contourObjectKey(const std::string& meshName);

float haltonSequence(uint32_t index, uint32_t base);

simd_float4x4 makeFloat4x4(const glm::mat4& m);
simd_float4   makeFloat4(const glm::vec3& v, float w = 0.0f);

uint32_t registerTextureInAcc(SceneGpuAccumulator& acc, const rt::RTTexture& tex);

template <typename ConfigLike>
GPUToonUniforms makeToonShaderUniforms(const ConfigLike& config, uint32_t width, uint32_t height) {
  GPUToonUniforms toon;
  toon.width = width;
  toon.height = height;
  toon.contourMethod = 2u;
  toon.useFxaa = config.toon.useFxaa ? 1u : 0u;
  toon.detailContourStrength =
      config.toon.enabled && config.toon.enableDetailContour ? std::max(0.0f, config.toon.detailContourStrength) : 0.0f;
  toon.depthThreshold = std::max(0.0f, config.toon.depthThreshold);
  toon.normalThreshold = std::max(0.0f, config.toon.normalThreshold);
  toon.edgeThickness = std::max(1.0f, config.toon.edgeThickness);
  toon.exposure = config.renderMode == rt::RenderMode::Toon ? std::max(0.1f, config.toon.tonemapExposure)
                                                            : std::max(0.1f, config.lighting.standardExposure);
  toon.gamma = config.renderMode == rt::RenderMode::Toon ? std::max(0.1f, config.toon.tonemapGamma)
                                                         : std::max(0.1f, config.lighting.standardGamma);
  toon.saturation =
      config.renderMode == rt::RenderMode::Toon ? 1.0f : std::max(0.0f, config.lighting.standardSaturation);
  toon.objectContourStrength =
      config.toon.enabled && config.toon.enableObjectContour ? std::max(0.0f, config.toon.objectContourStrength) : 0.0f;
  toon.objectThreshold = std::max(0.0f, config.toon.objectThreshold);
  toon.enableDetailContour = config.toon.enabled && config.toon.enableDetailContour ? 1u : 0u;
  toon.enableObjectContour = config.toon.enabled && config.toon.enableObjectContour ? 1u : 0u;
  toon.enableNormalEdge = config.toon.enableNormalEdge ? 1u : 0u;
  toon.enableDepthEdge = config.toon.enableDepthEdge ? 1u : 0u;
  toon.backgroundColor = makeFloat4(config.lighting.backgroundColor, 1.0f);
  toon.edgeColor = makeFloat4(config.toon.edgeColor, 1.0f);
  return toon;
}

} // namespace metal_rt
