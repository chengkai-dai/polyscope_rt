#include <metal_stdlib>
using namespace metal;
#include "gpu_shared_types.h"
#include "shader_common.h"

[[kernel]] void depthMinMaxKernel(device const float* linearDepth [[buffer(0)]],
                                  device atomic_uint* minmax [[buffer(1)]],
                                  constant GPUToonUniforms& toon [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= toon.width || gid.y >= toon.height) return;
  float depth = linearDepth[pixelIndex(gid.x, gid.y, toon.width)];
  if (depth > 0.0f) {
    uint bits = as_type<uint>(depth);
    atomic_fetch_min_explicit(&minmax[0], bits, memory_order_relaxed);
    atomic_fetch_max_explicit(&minmax[1], bits, memory_order_relaxed);
  }
}

[[kernel]] void detailContourKernel(device const float* linearDepth [[buffer(0)]],
                                    device const float4* normals [[buffer(1)]],
                                    device const uint* objectIds [[buffer(2)]],
                                    constant GPUToonUniforms& toon [[buffer(3)]],
                                    device atomic_uint* minmax [[buffer(4)]],
                                    device float4* output [[buffer(5)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= toon.width || gid.y >= toon.height) return;

  int radius = max(1, int(round(toon.edgeThickness)));
  int x = int(gid.x);
  int y = int(gid.y);
  uint centerObject = loadObjectId(objectIds, toon.width, toon.height, x, y);
  if (centerObject == 0u) {
    output[pixelIndex(gid.x, gid.y, toon.width)] = float4(0.0f);
    return;
  }

  float zNear = as_type<float>(atomic_load_explicit(&minmax[0], memory_order_relaxed));
  float zFar = as_type<float>(atomic_load_explicit(&minmax[1], memory_order_relaxed));

  float3 A = sampleContourNormal(normals, objectIds, toon.width, toon.height, x - radius, y + radius);
  float3 B = sampleContourNormal(normals, objectIds, toon.width, toon.height, x, y + radius);
  float3 C = sampleContourNormal(normals, objectIds, toon.width, toon.height, x + radius, y + radius);
  float3 D = sampleContourNormal(normals, objectIds, toon.width, toon.height, x - radius, y);
  float3 E = sampleContourNormal(normals, objectIds, toon.width, toon.height, x + radius, y);
  float3 F = sampleContourNormal(normals, objectIds, toon.width, toon.height, x - radius, y - radius);
  float3 G = sampleContourNormal(normals, objectIds, toon.width, toon.height, x, y - radius);
  float3 H = sampleContourNormal(normals, objectIds, toon.width, toon.height, x + radius, y - radius);

  const float k0 = 17.0f / 23.75f;
  const float k1 = 61.0f / 23.75f;
  float3 gradY = k0 * A + k1 * B + k0 * C - k0 * F - k1 * G - k0 * H;
  float3 gradX = k0 * C + k1 * E + k0 * H - k0 * A - k1 * D - k0 * F;
  float normalGradient = length(gradX) + length(gradY);
  float normalEdge = smoothstep(2.0f, 3.0f, normalGradient * toon.normalThreshold);
  if (toon.enableNormalEdge == 0u) normalEdge = 0.0f;

  // Depth edge requires a valid scene depth range; skip when zNear == zFar.
  float depthEdge = 0.0f;
  if (zFar > zNear) {
    float Az = sampleContourDepth(linearDepth, objectIds, toon.width, toon.height, x - radius, y + radius, zNear, zFar);
    float Bz = sampleContourDepth(linearDepth, objectIds, toon.width, toon.height, x, y + radius, zNear, zFar);
    float Cz = sampleContourDepth(linearDepth, objectIds, toon.width, toon.height, x + radius, y + radius, zNear, zFar);
    float Dz = sampleContourDepth(linearDepth, objectIds, toon.width, toon.height, x - radius, y, zNear, zFar);
    float Ez = sampleContourDepth(linearDepth, objectIds, toon.width, toon.height, x + radius, y, zNear, zFar);
    float Fz = sampleContourDepth(linearDepth, objectIds, toon.width, toon.height, x - radius, y - radius, zNear, zFar);
    float Gz = sampleContourDepth(linearDepth, objectIds, toon.width, toon.height, x, y - radius, zNear, zFar);
    float Hz = sampleContourDepth(linearDepth, objectIds, toon.width, toon.height, x + radius, y - radius, zNear, zFar);
    float Xz = sampleContourDepth(linearDepth, objectIds, toon.width, toon.height, x, y, zNear, zFar);
    float g = (fabs(Az + 2.0f * Bz + Cz - Fz - 2.0f * Gz - Hz) +
               fabs(Cz + 2.0f * Ez + Hz - Az - 2.0f * Dz - Fz)) /
              8.0f;
    float l = (8.0f * Xz - Az - Bz - Cz - Dz - Ez - Fz - Gz - Hz) / 3.0f;
    depthEdge = smoothstep(0.03f, 0.1f, (l + g) * toon.depthThreshold);
    if (toon.enableDepthEdge == 0u) depthEdge = 0.0f;
  }

  float edge = clamp(normalEdge + depthEdge, 0.0f, 1.0f);
  output[pixelIndex(gid.x, gid.y, toon.width)] = float4(edge, edge, edge, 1.0f);
}

[[kernel]] void objectContourKernel(device const uint* objectIds [[buffer(0)]],
                                    constant GPUToonUniforms& toon [[buffer(1)]],
                                    device float4* output [[buffer(2)]],
                                    uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= toon.width || gid.y >= toon.height) return;

  int radius = max(1, int(round(toon.edgeThickness)));
  int x = int(gid.x);
  int y = int(gid.y);
  uint center = loadObjectId(objectIds, toon.width, toon.height, x, y);
  float contour = 0.0f;

  if (toon.contourMethod == 2u) {
    contour = (center != loadObjectId(objectIds, toon.width, toon.height, x - radius, y + radius) ||
               center != loadObjectId(objectIds, toon.width, toon.height, x, y + radius) ||
               center != loadObjectId(objectIds, toon.width, toon.height, x + radius, y + radius) ||
               center != loadObjectId(objectIds, toon.width, toon.height, x - radius, y) ||
               center != loadObjectId(objectIds, toon.width, toon.height, x + radius, y) ||
               center != loadObjectId(objectIds, toon.width, toon.height, x - radius, y - radius) ||
               center != loadObjectId(objectIds, toon.width, toon.height, x, y - radius) ||
               center != loadObjectId(objectIds, toon.width, toon.height, x + radius, y - radius))
                  ? 1.0f
                  : 0.0f;
  } else {
    int differentCorners = 0;
    int differentEdges = 0;
    if (center != loadObjectId(objectIds, toon.width, toon.height, x - radius, y + radius)) differentCorners++;
    if (center != loadObjectId(objectIds, toon.width, toon.height, x + radius, y + radius)) differentCorners++;
    if (center != loadObjectId(objectIds, toon.width, toon.height, x - radius, y - radius)) differentCorners++;
    if (center != loadObjectId(objectIds, toon.width, toon.height, x + radius, y - radius)) differentCorners++;
    if (center != loadObjectId(objectIds, toon.width, toon.height, x, y + radius)) differentEdges++;
    if (center != loadObjectId(objectIds, toon.width, toon.height, x - radius, y)) differentEdges++;
    if (center != loadObjectId(objectIds, toon.width, toon.height, x + radius, y)) differentEdges++;
    if (center != loadObjectId(objectIds, toon.width, toon.height, x, y - radius)) differentEdges++;
    contour = differentCorners * (1.0f / 6.0f) + differentEdges * (1.0f / 3.0f);
  }

  contour = clamp(contour * toon.objectThreshold, 0.0f, 1.0f);
  output[pixelIndex(gid.x, gid.y, toon.width)] = float4(contour, contour, contour, 1.0f);
}

[[kernel]] void fxaaKernel(device const float4* input [[buffer(0)]],
                           constant GPUToonUniforms& toon [[buffer(1)]],
                           device float4* output [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= toon.width || gid.y >= toon.height) return;

  float2 resolution = float2(toon.width, toon.height);
  float2 fragCoord = float2(gid) + 0.5f;
  float2 inverseVP = 1.0f / resolution;
  float2 uvNW = (fragCoord + float2(-1.0f, -1.0f)) * inverseVP;
  float2 uvNE = (fragCoord + float2(1.0f, -1.0f)) * inverseVP;
  float2 uvSW = (fragCoord + float2(-1.0f, 1.0f)) * inverseVP;
  float2 uvSE = (fragCoord + float2(1.0f, 1.0f)) * inverseVP;
  float2 uvM = fragCoord * inverseVP;

  float3 rgbNW = sampleLinear(input, toon.width, toon.height, uvNW).xyz;
  float3 rgbNE = sampleLinear(input, toon.width, toon.height, uvNE).xyz;
  float3 rgbSW = sampleLinear(input, toon.width, toon.height, uvSW).xyz;
  float3 rgbSE = sampleLinear(input, toon.width, toon.height, uvSE).xyz;
  float4 texColor = sampleLinear(input, toon.width, toon.height, uvM);
  float3 rgbM = texColor.xyz;
  float3 lumaCoeff = float3(0.299f, 0.587f, 0.114f);

  float lumaNW = dot(rgbNW, lumaCoeff);
  float lumaNE = dot(rgbNE, lumaCoeff);
  float lumaSW = dot(rgbSW, lumaCoeff);
  float lumaSE = dot(rgbSE, lumaCoeff);
  float lumaM = dot(rgbM, lumaCoeff);
  float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

  float2 dir;
  dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
  dir.y = (lumaNW + lumaSW) - (lumaNE + lumaSE);

  const float fxaaReduceMin = 1.0f / 128.0f;
  const float fxaaReduceMul = 1.0f / 8.0f;
  const float fxaaSpanMax = 8.0f;
  float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25f * fxaaReduceMul), fxaaReduceMin);
  float rcpDirMin = 1.0f / (min(fabs(dir.x), fabs(dir.y)) + dirReduce);
  dir = clamp(dir * rcpDirMin, float2(-fxaaSpanMax), float2(fxaaSpanMax)) * inverseVP;

  float3 rgbA = 0.5f * (sampleLinear(input, toon.width, toon.height, uvM + dir * (1.0f / 3.0f - 0.5f)).xyz +
                        sampleLinear(input, toon.width, toon.height, uvM + dir * (2.0f / 3.0f - 0.5f)).xyz);
  float3 rgbB = rgbA * 0.5f +
                0.25f * (sampleLinear(input, toon.width, toon.height, uvM + dir * -0.5f).xyz +
                         sampleLinear(input, toon.width, toon.height, uvM + dir * 0.5f).xyz);

  float lumaB = dot(rgbB, lumaCoeff);
  float3 rgb = (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
  output[pixelIndex(gid.x, gid.y, toon.width)] = float4(rgb, texColor.w);
}

[[kernel]] void compositeKernel(device const float4* tonemapped [[buffer(0)]],
                                device const float4* detailContour [[buffer(1)]],
                                device const float4* objectContour [[buffer(2)]],
                                constant GPUToonUniforms& toon [[buffer(3)]],
                                device float4* output [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= toon.width || gid.y >= toon.height) return;
  uint index = pixelIndex(gid.x, gid.y, toon.width);
  float4 color = tonemapped[index];
  color.xyz = mix(toon.backgroundColor.xyz, color.xyz, color.w);
  float detailMask = toon.enableDetailContour != 0u ? clamp(detailContour[index].x * toon.detailContourStrength, 0.0f, 1.0f)
                                                    : 0.0f;
  float objectMask = toon.enableObjectContour != 0u ? clamp(objectContour[index].x * toon.objectContourStrength, 0.0f, 1.0f)
                                                    : 0.0f;
  color.xyz = mix(color.xyz, toon.edgeColor.xyz, detailMask);
  color.xyz = mix(color.xyz, toon.edgeColor.xyz, objectMask);
  output[index] = float4(color.xyz, 1.0f);
}
