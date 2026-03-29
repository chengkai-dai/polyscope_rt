#pragma once

#include "gpu_shared_types.h"

constant float kBackgroundDepth = 0.999999f;

inline uint wangHash(uint x) {
  x = (x ^ 61u) ^ (x >> 16u);
  x *= 9u;
  x = x ^ (x >> 4u);
  x *= 0x27d4eb2du;
  x = x ^ (x >> 15u);
  return x;
}

inline float rand01(thread uint& state) {
  state = wangHash(state);
  return (float(state) + 0.5f) / 4294967296.0f;
}

constant unsigned int primes[] = {
    2,   3,  5,  7, 11, 13, 17, 19,
    23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89
};

inline float halton(unsigned int i, unsigned int d) {
    unsigned int b = primes[d];
    float f = 1.0f;
    float invB = 1.0f / b;
    float r = 0;
    while (i > 0) {
        f = f * invB;
        r = r + f * (i % b);
        i = i / b;
    }
    return r;
}

inline uint clampIndex(int value, uint limit) {
  if (limit == 0u) return 0u;
  return uint(clamp(value, 0, int(limit - 1u)));
}

inline uint pixelIndex(uint x, uint y, uint width) { return y * width + x; }

inline float4 loadPixel(device const float4* buffer, uint width, uint height, int x, int y) {
  return buffer[pixelIndex(clampIndex(x, width), clampIndex(y, height), width)];
}

inline float loadDepth(device const float* buffer, uint width, uint height, int x, int y) {
  return buffer[pixelIndex(clampIndex(x, width), clampIndex(y, height), width)];
}

inline uint loadObjectId(device const uint* buffer, uint width, uint height, int x, int y) {
  return buffer[pixelIndex(clampIndex(x, width), clampIndex(y, height), width)];
}

inline float4 sampleLinear(device const float4* buffer, uint width, uint height, float2 uv) {
  float px = clamp(uv.x * float(max(width, 1u)) - 0.5f, 0.0f, float(max(width, 1u) - 1u));
  float py = clamp(uv.y * float(max(height, 1u)) - 0.5f, 0.0f, float(max(height, 1u) - 1u));

  uint x0 = min(uint(floor(px)), max(width, 1u) - 1u);
  uint y0 = min(uint(floor(py)), max(height, 1u) - 1u);
  uint x1 = min(x0 + 1u, max(width, 1u) - 1u);
  uint y1 = min(y0 + 1u, max(height, 1u) - 1u);

  float tx = px - float(x0);
  float ty = py - float(y0);

  float4 c00 = buffer[pixelIndex(x0, y0, width)];
  float4 c10 = buffer[pixelIndex(x1, y0, width)];
  float4 c01 = buffer[pixelIndex(x0, y1, width)];
  float4 c11 = buffer[pixelIndex(x1, y1, width)];
  return mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
}

inline float4 sampleBaseColorTexture(device const GPUTexture* textures, device const float4* texturePixels, uint textureIndex,
                              float2 uv) {
  GPUTexture texture = textures[textureIndex];
  uint offset = texture.data.x;
  uint width = texture.data.y;
  uint height = texture.data.z;
  if (width == 0u || height == 0u) return float4(1.0f);

  float2 wrapped = uv - floor(uv);
  float x = wrapped.x * float(max(width - 1u, 1u));
  float y = wrapped.y * float(max(height - 1u, 1u));

  uint x0 = min(uint(floor(x)), width - 1u);
  uint y0 = min(uint(floor(y)), height - 1u);
  uint x1 = min(x0 + 1u, width - 1u);
  uint y1 = min(y0 + 1u, height - 1u);

  float tx = x - float(x0);
  float ty = y - float(y0);

  float4 c00 = texturePixels[offset + y0 * width + x0];
  float4 c10 = texturePixels[offset + y0 * width + x1];
  float4 c01 = texturePixels[offset + y1 * width + x0];
  float4 c11 = texturePixels[offset + y1 * width + x1];
  return mix(mix(c00, c10, tx), mix(c01, c11, tx), ty);
}

inline float spotAttenuation(float3 lightDir, float3 toLight, float innerConeAngle, float outerConeAngle) {
  float cosTheta = dot(normalize(-lightDir), normalize(toLight));
  float innerCos = cos(innerConeAngle);
  float outerCos = cos(outerConeAngle);
  if (cosTheta <= outerCos) return 0.0f;
  if (cosTheta >= innerCos) return 1.0f;
  return smoothstep(outerCos, innerCos, cosTheta);
}

inline float toonShading(float nDotL, uint bandCount) {
  if (bandCount == 0u) return max(0.0f, nDotL);
  float value = clamp(nDotL, 0.0f, 1.0f);
  float inv = 1.0f / float(max(bandCount, 1u));
  return floor(value * (1.0f + inv * 0.5f) * float(bandCount)) * inv;
}

inline float3 gammaCorrection(float3 color, float gamma) { return pow(color, float3(1.0f / max(gamma, 1e-4f))); }

inline float3 applySaturation(float3 color, float saturation) {
  float luma = dot(color, float3(0.299f, 0.587f, 0.114f));
  return mix(float3(luma), color, saturation);
}

inline float3 toneMapUncharted2Impl(float3 color) {
  const float A = 0.15f;
  const float B = 0.50f;
  const float C = 0.10f;
  const float D = 0.20f;
  const float E = 0.02f;
  const float F = 0.30f;
  return ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
}

inline float3 toneMapUncharted(float3 color, float gamma) {
  const float W = 11.2f;
  const float exposureBias = 2.0f;
  color = toneMapUncharted2Impl(color * exposureBias);
  float3 whiteScale = 1.0f / toneMapUncharted2Impl(float3(W));
  return gammaCorrection(color * whiteScale, gamma);
}

inline float normalizedInverseDepth(float depth, float zNear, float zFar) {
  if (depth <= 0.0f || zFar <= zNear) return 0.0f;
  float invDepth = 1.0f / depth;
  float invNear = 1.0f / zNear;
  float invFar = 1.0f / zFar;
  float denom = invFar - invNear;
  if (fabs(denom) < 1e-6f) return 0.0f;
  return fabs((invDepth - invNear) / denom);
}

inline float3 sampleContourNormal(device const float4* normals, device const uint* objectIds, uint width, uint height, int x, int y) {
  if (loadObjectId(objectIds, width, height, x, y) == 0u) return float3(0.0f);
  return normalize(loadPixel(normals, width, height, x, y).xyz);
}

inline float sampleContourDepth(device const float* linearDepth, device const uint* objectIds, uint width, uint height, int x, int y,
                         float zNear, float zFar) {
  if (loadObjectId(objectIds, width, height, x, y) == 0u) return 0.0f;
  float depth = loadDepth(linearDepth, width, height, x, y);
  return normalizedInverseDepth(depth, zNear, zFar);
}

inline float computeClipDepth(float3 worldPos, constant GPUCamera& camera) {
  float4 clip = camera.projectionMatrix * (camera.viewMatrix * float4(worldPos, 1.0f));
  float ndc = clip.z / max(clip.w, 1e-6f);
  return clamp(0.5f * ndc + 0.5f, 0.0f, 1.0f);
}
