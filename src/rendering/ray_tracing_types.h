#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "glm/glm.hpp"
#include "polyscope/rt/material_library.h"

namespace rt {

enum class RTPunctualLightType : uint32_t {
  Directional = 0,
  Point = 1,
  Spot = 2,
};

struct RTTexture {
  uint32_t width = 0;
  uint32_t height = 0;
  std::string cacheKey;
  std::vector<glm::vec4> pixels;
};

struct RTPunctualLight {
  RTPunctualLightType type = RTPunctualLightType::Directional;
  glm::vec3 color{1.0f, 1.0f, 1.0f};
  float intensity = 1.0f;
  glm::vec3 position{0.0f, 0.0f, 0.0f};
  float range = 0.0f;
  glm::vec3 direction{0.0f, -1.0f, 0.0f};
  float innerConeAngle = 0.0f;
  float outerConeAngle = 0.7853981634f;
};

struct RTMesh {
  std::string name;
  std::vector<glm::vec3> vertices;
  std::vector<glm::vec3> normals;
  std::vector<glm::uvec3> indices;
  std::vector<glm::vec2> texcoords;
  std::vector<glm::vec3> vertexColors;
  glm::mat4 transform{1.0f};
  glm::vec4 baseColorFactor{polyscope::rt::builtinDefaultMaterial().baseColor};
  bool hasBaseColorTexture = false;
  RTTexture baseColorTexture;
  float metallicFactor = polyscope::rt::builtinDefaultMaterial().metallic;
  float roughnessFactor = polyscope::rt::builtinDefaultMaterial().roughness;
  bool hasMetallicRoughnessTexture = false;
  RTTexture metallicRoughnessTexture;
  glm::vec3 emissiveFactor{polyscope::rt::builtinDefaultMaterial().emissive};
  bool hasEmissiveTexture = false;
  RTTexture emissiveTexture;
  bool hasNormalTexture = false;
  RTTexture normalTexture;
  float normalTextureScale = 1.0f;
  float transmissionFactor = polyscope::rt::builtinDefaultMaterial().transmission;
  float indexOfRefraction = polyscope::rt::builtinDefaultMaterial().ior;
  float opacity = polyscope::rt::builtinDefaultMaterial().opacity;
  bool doubleSided = false;
  bool unlit = polyscope::rt::builtinDefaultMaterial().unlit;

  bool wireframe = false;
  glm::vec3 edgeColor{0.0f, 0.0f, 0.0f};
  float edgeWidth = 1.0f;

  // Isoline stripe pattern (from active scalar quantity with isolines enabled).
  // isoScalars is per-vertex raw scalar values; empty means isolines are off.
  std::vector<float> isoScalars;
  float isoSpacing = 0.1f;          // period in raw data units (from getIsolinePeriod())
  float isoDarkness = 0.5f;         // dark band multiplier (from getIsolineDarkness())
  float isoContourThickness = 0.3f; // contour line width factor (from getIsolineContourThickness())
  int   isoStyle = 1;               // 1=stripe, 2=contour (from getIsolineStyle())
};

enum class RTCurvePrimitiveType : uint32_t {
  Sphere = 0,
  Cylinder = 1,
};

struct RTCurvePrimitive {
  RTCurvePrimitiveType type = RTCurvePrimitiveType::Sphere;
  glm::vec3 p0{0.0f};
  glm::vec3 p1{0.0f};
  float radius = 0.01f;
};

struct RTCurveNetwork {
  std::string name;
  glm::vec4 baseColor{0.8f, 0.8f, 0.8f, 1.0f};
  float metallic = 0.0f;
  float roughness = 0.5f;
  bool unlit = false;
  std::vector<RTCurvePrimitive> primitives;
  // Per-primitive color override (same length as primitives if populated, or empty for uniform color).
  // Layout mirrors primitives:
  //   [0 .. nSpheres-1]  colors for degree≠2 sphere nodes (endpoints & branch points)
  //   [nSpheres ..]      colors for cylinder edges
  std::vector<glm::vec3> primitiveColors;

  // Original node graph — used by the Metal backend to build Catmull-Rom
  // ghost points so adjacent segments blend smoothly at shared junctions.
  std::vector<glm::vec3> nodePositions; // world-space positions, one per node
  std::vector<uint32_t>  edgeTailInds;  // index into nodePositions for each edge tail
  std::vector<uint32_t>  edgeTipInds;   // index into nodePositions for each edge tip
};

struct RTPointCloud {
  std::string name;
  std::vector<glm::vec3> centers;
  // Per-point colors (same length as centers if populated, or empty to use baseColor uniformly).
  std::vector<glm::vec3> colors;
  float radius = 0.01f;
  glm::vec4 baseColor{0.8f, 0.8f, 0.8f, 1.0f};
  float metallic = 0.0f;
  float roughness = 0.8f;
  bool unlit = false;
};

// An enabled vector-field quantity from a SurfaceMesh.  Each arrow is a
// (root, direction) pair where direction is already scaled to world-space
// length (= raw_vector * lengthMult / lengthRange).
struct RTVectorField {
  std::string name;
  std::vector<glm::vec3> roots;       // base positions (world space)
  std::vector<glm::vec3> directions;  // scaled world-space vectors
  glm::vec3 color{0.6f, 0.6f, 0.9f};
  float radius   = 0.005f;            // shaft radius (world units)
  float metallic = 0.0f;
  float roughness = 0.4f;
};

struct RTScene {
  std::vector<RTMesh> meshes;
  std::vector<RTCurveNetwork> curveNetworks;
  std::vector<RTPointCloud> pointClouds;
  std::vector<RTVectorField> vectorFields;
  std::vector<RTPunctualLight> lights;
  uint64_t hash = 0;
};

struct RTCamera {
  glm::vec3 position{0.0f};
  glm::vec3 lookDir{0.0f, 0.0f, -1.0f};
  glm::vec3 upDir{0.0f, 1.0f, 0.0f};
  glm::vec3 rightDir{1.0f, 0.0f, 0.0f};
  float fovYDegrees = 45.0f;
  float aspect = 1.0f;
  float nearClip = 0.01f;
  float farClip = 100.0f;
  glm::mat4 viewMatrix{1.0f};
  glm::mat4 projectionMatrix{1.0f};
  uint32_t width = 1;
  uint32_t height = 1;
};

struct LightingSettings {
  glm::vec3 backgroundColor{1.0f, 1.0f, 1.0f};
  glm::vec3 mainLightDirection{-0.5f, -0.5f, 0.70710677f};
  glm::vec3 mainLightColor{1.0f, 1.0f, 1.0f};
  float mainLightIntensity = 0.0f;
  float ambientFloor = 0.1f;
  float standardExposure = 4.0f;
  float standardGamma = 2.2f;
  float standardSaturation = 1.6f;
  int toonBandCount = 5;
  glm::vec3 areaLightCenter{0.0f, 2.72f, -1.35f};
  glm::vec3 areaLightU{0.55f, 0.0f, 0.0f};
  glm::vec3 areaLightV{0.0f, 0.0f, -0.55f};
  glm::vec3 areaLightEmission{18.0f, 18.0f, 18.0f};
  float environmentIntensity = 0.3f;
  glm::vec3 environmentTint{1.0f, 1.0f, 1.0f};
  bool enableAreaLight = true;
};

struct ToonSettings {
  bool enabled = true;
  int bandCount = 5;
  float edgeThickness = 1.0f;
  float depthThreshold = 0.015f;
  float normalThreshold = 0.12f;
  float objectThreshold = 1.0f;
  bool enableDetailContour = true;
  bool enableObjectContour = true;
  bool enableNormalEdge = true;
  bool enableDepthEdge = true;
  float detailContourStrength = 1.0f;
  float objectContourStrength = 1.0f;
  glm::vec3 backgroundColor{1.0f, 1.0f, 1.0f};
  glm::vec3 edgeColor{0.3f, 0.3f, 0.3f};
  bool useFxaa = true;
  float tonemapExposure = 3.0f;
  float tonemapGamma = 2.2f;
};

enum class RenderMode : uint32_t {
  Standard = 0,
  Toon = 1,
};

struct MaterialOverride {
  std::optional<float> metallic;
  std::optional<float> roughness;
  std::optional<glm::vec4> baseColor;
  std::optional<glm::vec3> emissive;
  std::optional<float> transmission;
  std::optional<float> ior;
  std::optional<float> opacity;
  std::optional<bool> unlit;
};

struct InfinitePlaneSettings {
  float height = 0.0f;
  glm::vec3 color{230.0f / 255.0f, 230.0f / 255.0f, 230.0f / 255.0f};
  float metallic = 0.0f;
  float roughness = 1.0f;
  float reflectance = 0.0f;
};

struct AppearanceConfig {
  RenderMode mode = RenderMode::Standard;
  bool enableMetalFX = false;
  LightingSettings lighting;
  ToonSettings toon;
  InfinitePlaneSettings groundPlane;
};

struct RenderConfig {
  RenderMode renderMode = RenderMode::Standard;
  uint32_t samplesPerIteration = 1;
  uint32_t maxBounces = 2;
  bool accumulate = true;
  bool enableMetalFX = false;
  uint32_t metalFXOutputWidth = 0;
  uint32_t metalFXOutputHeight = 0;
  LightingSettings lighting;
  ToonSettings toon;
  InfinitePlaneSettings groundPlane;
};

struct RenderBuffer {
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<glm::vec3> color;
  std::vector<float> depth;
  std::vector<float> linearDepth;
  std::vector<glm::vec3> normal;
  std::vector<uint32_t> objectId;
  std::vector<float> detailContour;
  std::vector<float> detailContourRaw;
  std::vector<float> objectContour;
  std::vector<float> objectContourRaw;
  uint32_t accumulatedSamples = 0;
};

} // namespace rt
