#pragma once

// gpu_shared_types.h — Unified GPU struct definitions.
// This header is included by BOTH C++ (.mm) and Metal Shading Language (.metal)
// files.  On the Metal side, native types (float4, uint4, ...) are used.
// On the C++ side, the simd_ equivalents are aliased to the same names so that
// a single struct definition compiles on both sides with identical binary layout.

#ifdef __METAL_VERSION__
// Metal Shading Language: all vector / matrix types are built-in.
#include <metal_stdlib>
#else
// C++ side: pull in the SIMD types and alias Metal-style names.
#include <simd/simd.h>
#include <cstdint>

#include <string>
#include <unordered_map>
#include <vector>
#endif

// ---------------------------------------------------------------------------
// Type bridge: on Metal float4/uint4/float2/float4x4 are native;
//              on C++ we alias simd_float4 etc. to those names.
// ---------------------------------------------------------------------------
#ifndef __METAL_VERSION__
namespace gpu_types {
using float4   = simd_float4;
using uint4    = simd_uint4;
using float2   = simd_float2;
using float4x4 = simd_float4x4;
}
using gpu_types::float4;
using gpu_types::uint4;
using gpu_types::float2;
using gpu_types::float4x4;
#endif

// ---------------------------------------------------------------------------
// GPU structs — identical binary layout on Metal and C++ sides.
// ---------------------------------------------------------------------------

struct GPUCamera {
  float4    position;
  float4    lookDir;
  float4    upDir;
  float4    rightDir;
  float4    clipData;
  float4x4  viewMatrix;
  float4x4  projectionMatrix;
};

struct GPUFrameUniforms {
#ifdef __METAL_VERSION__
  uint      renderMode;
  uint      width;
  uint      height;
  uint      samplesPerIteration;
  uint      frameIndex;
  uint      maxBounces;
  uint      lightCount;
  uint      enableSceneLights;
  uint      enableAreaLight;
  uint      toonBandCount;
  float     ambientFloor;
  uint      rngFrameIndex;
#else
  uint32_t  renderMode         = 1;
  uint32_t  width              = 1;
  uint32_t  height             = 1;
  uint32_t  samplesPerIteration = 1;
  uint32_t  frameIndex         = 0;
  uint32_t  maxBounces         = 2;
  uint32_t  lightCount         = 0;
  uint32_t  enableSceneLights  = 1;
  uint32_t  enableAreaLight    = 0;
  uint32_t  toonBandCount      = 5;
  float     ambientFloor       = 0.1f;
  uint32_t  rngFrameIndex      = 0;
#endif
  float4    planeColorEnabled;
  float4    planeParams;
  float2    jitterOffset;
  float2    _pad1;
  float4x4  prevViewProj;
};

struct GPULighting {
  float4 backgroundColor;
  float4 mainLightDirection;
  float4 mainLightColorIntensity;
  float4 environmentTintIntensity;
  float4 areaLightCenterEnabled;
  float4 areaLightU;
  float4 areaLightV;
  float4 areaLightEmission;
};

struct GPUToonUniforms {
#ifdef __METAL_VERSION__
  uint  width;
  uint  height;
  uint  contourMethod;
  uint  useFxaa;
  float detailContourStrength;
  float depthThreshold;
  float normalThreshold;
  float edgeThickness;
  float exposure;
  float gamma;
  float saturation;
  float objectContourStrength;
  float objectThreshold;
  uint  enableDetailContour;
  uint  enableObjectContour;
  uint  enableNormalEdge;
  uint  enableDepthEdge;
#else
  uint32_t width                 = 1;
  uint32_t height                = 1;
  uint32_t contourMethod         = 2;
  uint32_t useFxaa               = 1;
  float    detailContourStrength = 1.0f;
  float    depthThreshold        = 1.0f;
  float    normalThreshold       = 0.5f;
  float    edgeThickness         = 1.0f;
  float    exposure              = 3.0f;
  float    gamma                 = 2.2f;
  float    saturation            = 1.0f;
  float    objectContourStrength = 1.0f;
  float    objectThreshold       = 1.0f;
  uint32_t enableDetailContour   = 1u;
  uint32_t enableObjectContour   = 1u;
  uint32_t enableNormalEdge      = 1u;
  uint32_t enableDepthEdge       = 1u;
#endif
  float4 backgroundColor;
  float4 edgeColor;
};

struct GPUTriangle {
  uint4 indicesMaterial;
  uint4 objectFlags;
};

struct GPUMaterial {
  float4 baseColorFactor;
  uint4  baseColorTextureData;
  float4 metallicRoughnessNormal;
  uint4  metallicRoughnessTextureData;
  float4 emissiveFactor;
  uint4  emissiveTextureData;
  uint4  normalTextureData;
  float4 transmissionIor;
  float4 wireframeEdgeData;
  // isoParams: x=style(0=off,1=stripe,2=contour), y=spacing(period), z=darkness, w=contourThickness
  float4 isoParams;
};

struct GPUTexture {
  uint4 data;
};

struct GPUPunctualLight {
  float4 positionRange;
  float4 directionType;
  float4 colorIntensity;
  float4 spotAngles;
};

struct GPUCurvePrimitive {
  float4 p0_radius;
  float4 p1_type;
  // Ghost control points for Catmull-Rom spline evaluation in the shader.
  // p_prev is the node before p0; p_next is the node after p1.
  // When no neighbour exists the ghost point is mirrored (2*p0 - p1 or 2*p1 - p0).
  float4 p_prev;
  float4 p_next;
  uint4  materialObjectId;
  // Per-primitive color override; w=0 means "use material baseColorFactor", w=1 means use xyz.
  float4 baseColor;
};

struct GPUPointPrimitive {
  float4 center_radius;
  float4 baseColor;
  uint4  materialObjectId;
};

// ---------------------------------------------------------------------------
// CPU-only helpers: not visible to Metal shaders.
// ---------------------------------------------------------------------------
#ifndef __METAL_VERSION__

struct PackedTriangleIndices {
  uint32_t i0;
  uint32_t i1;
  uint32_t i2;
};

struct SceneGpuAccumulator {
  std::vector<simd_float4>          positions;
  std::vector<simd_float4>          normals;
  std::vector<simd_float4>          vertexColors;
  std::vector<simd_float2>          texcoords;
  std::vector<float>                isoScalars;   // parallel to positions; 0.0 when isolines inactive
  std::vector<PackedTriangleIndices> accelIndices;
  std::vector<GPUTriangle>          shaderTriangles;
  std::vector<GPUMaterial>          materials;
  std::vector<GPUTexture>           textures;
  std::vector<simd_float4>          texturePixels;
  std::unordered_map<std::string, uint32_t> textureLookup;
  std::unordered_map<std::string, uint32_t> objectIdLookup;
  std::vector<GPUPunctualLight>     lights;
  std::vector<GPUCurvePrimitive>    curvePrimitives;
  uint32_t nextObjectId = 1u;
};

#endif
