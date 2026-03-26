#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

constant float kBackgroundDepth = 0.999999f;
struct CameraData {
  float4 position;
  float4 lookDir;
  float4 upDir;
  float4 rightDir;
  float4 clipData;
  float4x4 viewMatrix;
  float4x4 projectionMatrix;
};

struct FrameUniforms {
  uint renderMode;
  uint width;
  uint height;
  uint samplesPerIteration;
  uint frameIndex;
  uint maxBounces;
  uint lightCount;
  uint enableSceneLights;  // reserved, unused
  uint enableAreaLight;
  uint toonBandCount;
  float ambientFloor;
  uint rngFrameIndex;
  float4 planeColorEnabled;
  float4 planeParams;
  float2 jitterOffset;
  float2 _pad1;
  float4x4 prevViewProj;
};

struct LightingData {
  float4 backgroundColor;
  float4 mainLightDirection;
  float4 mainLightColorIntensity;
  float4 environmentTintIntensity;
  float4 areaLightCenterEnabled;
  float4 areaLightU;
  float4 areaLightV;
  float4 areaLightEmission;
};

struct ToonUniforms {
  uint width;
  uint height;
  uint contourMethod;
  uint useFxaa;
  float detailContourStrength;
  float depthThreshold;
  float normalThreshold;
  float edgeThickness;
  float exposure;
  float gamma;
  float saturation;
  float objectContourStrength;
  float objectThreshold;
  uint enableDetailContour;
  uint enableObjectContour;
  uint enableNormalEdge;
  uint enableDepthEdge;
  float4 backgroundColor;
  float4 edgeColor;
};

struct TriangleData {
  uint4 indicesMaterial;
  uint4 objectFlags;
};

struct MaterialData {
  float4 baseColorFactor;
  uint4 baseColorTextureData;
  float4 metallicRoughnessNormal;
  uint4 metallicRoughnessTextureData;
  float4 emissiveFactor;
  uint4 emissiveTextureData;
  uint4 normalTextureData;
  float4 transmissionIor;
  float4 wireframeEdgeData;  // xyz = edge color, w = barycentric threshold (0 = disabled)
};

struct TextureData {
  uint4 data;
};

struct PunctualLightData {
  float4 positionRange;
  float4 directionType;
  float4 colorIntensity;
  float4 spotAngles;
};

struct CurvePrimitiveGPU {
  float4 p0_radius;
  float4 p1_type;
  uint4 materialObjectId;
};

// One bounding-box sphere primitive.  center_radius.w = radius.
// materialObjectId.x = material index, .y = object id.
struct PointPrimitiveGPU {
  float4 center_radius;
  float4 baseColor;         // xyz = per-point color, w = unused
  uint4  materialObjectId;
};


uint wangHash(uint x) {
  x = (x ^ 61u) ^ (x >> 16u);
  x *= 9u;
  x = x ^ (x >> 4u);
  x *= 0x27d4eb2du;
  x = x ^ (x >> 15u);
  return x;
}

float rand01(thread uint& state) {
  state = wangHash(state);
  return (float(state) + 0.5f) / 4294967296.0f;
}

constant unsigned int primes[] = {
    2,   3,  5,  7, 11, 13, 17, 19,
    23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89
};

float halton(unsigned int i, unsigned int d) {
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

uint clampIndex(int value, uint limit) {
  if (limit == 0u) return 0u;
  return uint(clamp(value, 0, int(limit - 1u)));
}

uint pixelIndex(uint x, uint y, uint width) { return y * width + x; }

float4 loadPixel(device const float4* buffer, uint width, uint height, int x, int y) {
  return buffer[pixelIndex(clampIndex(x, width), clampIndex(y, height), width)];
}

float loadDepth(device const float* buffer, uint width, uint height, int x, int y) {
  return buffer[pixelIndex(clampIndex(x, width), clampIndex(y, height), width)];
}

uint loadObjectId(device const uint* buffer, uint width, uint height, int x, int y) {
  return buffer[pixelIndex(clampIndex(x, width), clampIndex(y, height), width)];
}

float4 sampleLinear(device const float4* buffer, uint width, uint height, float2 uv) {
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

float4 sampleBaseColorTexture(device const TextureData* textures, device const float4* texturePixels, uint textureIndex,
                              float2 uv) {
  TextureData texture = textures[textureIndex];
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

float spotAttenuation(float3 lightDir, float3 toLight, float innerConeAngle, float outerConeAngle) {
  float cosTheta = dot(normalize(-lightDir), normalize(toLight));
  float innerCos = cos(innerConeAngle);
  float outerCos = cos(outerConeAngle);
  if (cosTheta <= outerCos) return 0.0f;
  if (cosTheta >= innerCos) return 1.0f;
  return smoothstep(outerCos, innerCos, cosTheta);
}

float toonShading(float nDotL, uint bandCount) {
  if (bandCount == 0u) return max(0.0f, nDotL);
  float value = clamp(nDotL, 0.0f, 1.0f);
  float inv = 1.0f / float(max(bandCount, 1u));
  return floor(value * (1.0f + inv * 0.5f) * float(bandCount)) * inv;
}

float3 gammaCorrection(float3 color, float gamma) { return pow(color, float3(1.0f / max(gamma, 1e-4f))); }

float3 applySaturation(float3 color, float saturation) {
  float luma = dot(color, float3(0.299f, 0.587f, 0.114f));
  return mix(float3(luma), color, saturation);
}

float3 toneMapUncharted2Impl(float3 color) {
  const float A = 0.15f;
  const float B = 0.50f;
  const float C = 0.10f;
  const float D = 0.20f;
  const float E = 0.02f;
  const float F = 0.30f;
  return ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
}

float3 toneMapUncharted(float3 color, float gamma) {
  const float W = 11.2f;
  const float exposureBias = 2.0f;
  color = toneMapUncharted2Impl(color * exposureBias);
  float3 whiteScale = 1.0f / toneMapUncharted2Impl(float3(W));
  return gammaCorrection(color * whiteScale, gamma);
}

float normalizedInverseDepth(float depth, float zNear, float zFar) {
  if (depth <= 0.0f || zFar <= zNear) return 0.0f;
  float invDepth = 1.0f / depth;
  float invNear = 1.0f / zNear;
  float invFar = 1.0f / zFar;
  float denom = invFar - invNear;
  if (fabs(denom) < 1e-6f) return 0.0f;
  return fabs((invDepth - invNear) / denom);
}

float3 sampleContourNormal(device const float4* normals, device const uint* objectIds, uint width, uint height, int x, int y) {
  if (loadObjectId(objectIds, width, height, x, y) == 0u) return float3(0.0f);
  return normalize(loadPixel(normals, width, height, x, y).xyz);
}

float sampleContourDepth(device const float* linearDepth, device const uint* objectIds, uint width, uint height, int x, int y,
                         float zNear, float zFar) {
  if (loadObjectId(objectIds, width, height, x, y) == 0u) return 0.0f;
  float depth = loadDepth(linearDepth, width, height, x, y);
  return normalizedInverseDepth(depth, zNear, zFar);
}

[[kernel]] void depthMinMaxKernel(device const float* linearDepth [[buffer(0)]],
                                  device atomic_uint* minmax [[buffer(1)]],
                                  constant ToonUniforms& toon [[buffer(2)]],
                                  uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= toon.width || gid.y >= toon.height) return;
  float depth = linearDepth[pixelIndex(gid.x, gid.y, toon.width)];
  if (depth > 0.0f) {
    uint bits = as_type<uint>(depth);
    atomic_fetch_min_explicit(&minmax[0], bits, memory_order_relaxed);
    atomic_fetch_max_explicit(&minmax[1], bits, memory_order_relaxed);
  }
}

float computeClipDepth(float3 worldPos, constant CameraData& camera) {
  float4 clip = camera.projectionMatrix * (camera.viewMatrix * float4(worldPos, 1.0f));
  float ndc = clip.z / max(clip.w, 1e-6f);
  return clamp(0.5f * ndc + 0.5f, 0.0f, 1.0f);
}

float evaluateDirectionalLight(float3 lightDir, float lightIntensity, float3 hitPos, float3 normal,
                               intersector<curve_data, triangle_data, instancing> isector,
                               instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                               intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 toLight = normalize(-lightDir);
  float nDotL = max(dot(normal, toLight), 0.0f);
  if (nDotL <= 0.0f) return 0.0f;

  ray shadowRay;
  shadowRay.origin = hitPos;
  shadowRay.direction = toLight;
  shadowRay.min_distance = 5e-3f;
  shadowRay.max_distance = 1e6f;
  isector.accept_any_intersection(true);
  auto mcShadow = isector.intersect(shadowRay, scene, 0xFFu);
  auto ptShadow = isector.intersect(shadowRay, pointScene, 0xFFu, ftable);
  isector.accept_any_intersection(false);
  if (mcShadow.type != intersection_type::none || ptShadow.type != intersection_type::none) return 0.0f;
  return nDotL * lightIntensity;
}

float shadowVisibility(float3 hitPos, float3 toLight, float maxDistance,
                       device const TriangleData* triangles, device const MaterialData* materials,
                       thread uint& rng, intersector<curve_data, triangle_data, instancing> isector,
                       instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                       intersection_function_table<curve_data, triangle_data, instancing> ftable);

bool sampleAreaLight(float3 hitPos, float3 normal, constant LightingData& lighting,
                     device const TriangleData* triangles, device const MaterialData* materials,
                     thread uint& rng, thread float3& toLight,
                     thread float3& radiance, intersector<curve_data, triangle_data, instancing> isector,
                     instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                     intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  if (lighting.areaLightCenterEnabled.w < 0.5f) return false;

  float su = rand01(rng) * 2.0f - 1.0f;
  float sv = rand01(rng) * 2.0f - 1.0f;
  float3 lightPoint = lighting.areaLightCenterEnabled.xyz + su * lighting.areaLightU.xyz + sv * lighting.areaLightV.xyz;
  float3 lightNormal = normalize(-cross(lighting.areaLightU.xyz, lighting.areaLightV.xyz));
  toLight = lightPoint - hitPos;
  float distance2 = max(dot(toLight, toLight), 1e-5f);
  float distance = sqrt(distance2);
  toLight /= distance;

  float nDotL = max(dot(normal, toLight), 0.0f);
  float lightCos = max(dot(lightNormal, -toLight), 0.0f);
  if (nDotL <= 0.0f || lightCos <= 0.0f) return false;

  float visibility = shadowVisibility(hitPos, toLight, max(distance - 2e-2f, 1e-3f),
                                      triangles, materials, rng, isector, scene, pointScene, ftable);
  if (visibility <= 0.0f) return false;

  float area = 4.0f * length(cross(lighting.areaLightU.xyz, lighting.areaLightV.xyz));
  radiance = lighting.areaLightEmission.xyz * (lightCos * area / distance2);
  return true;
}

float3 fresnelSchlick(float cosTheta, float3 F0) {
  return F0 + (1.0f - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

float3 fresnelSchlick(float cosTheta, float3 F0, float grazingMax) {
  return F0 + (float3(grazingMax) - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}

float distributionGGX(float3 N, float3 H, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0f);
  float NdotH2 = NdotH * NdotH;
  float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
  return a2 / max(3.14159265f * denom * denom, 1e-5f);
}

float geometrySchlickGGX(float NdotV, float roughness) {
  float r = roughness + 1.0f;
  float k = (r * r) / 8.0f;
  return NdotV / max(NdotV * (1.0f - k) + k, 1e-5f);
}

float geometrySmith(float3 N, float3 V, float3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0f);
  float NdotL = max(dot(N, L), 0.0f);
  return geometrySchlickGGX(NdotV, roughness) * geometrySchlickGGX(NdotL, roughness);
}

float3 evalSimpleSky(float3 dir, constant LightingData& lighting) {
  float3 sunDir = normalize(-lighting.mainLightDirection.xyz);
  float sunIntensity = lighting.mainLightColorIntensity.w;
  float3 sunColor = lighting.mainLightColorIntensity.xyz;
  // Always use the live UI background color (LightingData from frame).
  float3 bgColor = lighting.backgroundColor.xyz;

  float y = dir.y;

  // Sky gradient is anchored to backgroundColor: horizon matches it, zenith is a relative shift (not a fixed RGB sky).
  float3 horizonColor = bgColor;
  float3 zenithColor = mix(bgColor, bgColor * float3(0.35f, 0.55f, 1.05f), 0.75f);
  float3 groundColor = bgColor * float3(0.08f, 0.08f, 0.09f);

  float3 sky;
  if (y > 0.0f) {
    // Smaller exponent => t rises faster with elevation => shorter horizon-colored band (narrower "strip").
    float t = pow(y, 0.29f);
    sky = mix(horizonColor, zenithColor, t);
  } else {
    float t = pow(clamp(-y, 0.0f, 1.0f), 0.92f);
    sky = mix(horizonColor * 0.65f, groundColor, t);
  }

  if (sunIntensity > 1e-5f) {
    float cosAngle = dot(dir, sunDir);
    float sunRadius = 0.02f;
    float disc = smoothstep(1.0f - sunRadius, 1.0f - sunRadius * 0.1f, cosAngle);
    sky += sunColor * sunIntensity * disc * 2.0f;
    float glow = pow(max(cosAngle, 0.0f), 32.0f);
    sky += sunColor * glow * sunIntensity * 0.15f;
    float horizonGlow = pow(max(cosAngle, 0.0f), 4.0f) * max(1.0f - abs(y) * 2.0f, 0.0f);
    sky += sunColor * horizonGlow * sunIntensity * 0.08f;
  }

  return sky;
}

float3 sampleEnvironment(float3 dir, constant LightingData& lighting) {
  float hemi = saturate(dir.y * 0.5f + 0.5f);
  float3 sky = mix(float3(0.04f, 0.04f, 0.05f), lighting.environmentTintIntensity.xyz, hemi);
  return sky * lighting.environmentTintIntensity.w;
}

float3 cosineWeightedHemisphere(float3 normal, thread uint& rng) {
  float u1 = rand01(rng);
  float u2 = rand01(rng);
  float r = sqrt(u1);
  float theta = 2.0f * 3.14159265f * u2;
  float z = sqrt(max(0.0f, 1.0f - u1));
  float3 up = abs(normal.y) < 0.999f ? float3(0,1,0) : float3(1,0,0);
  float3 tangent = normalize(cross(up, normal));
  float3 bitangent = cross(normal, tangent);
  return normalize(tangent * (r * cos(theta)) + bitangent * (r * sin(theta)) + normal * z);
}

// GGX Visible Normal Distribution Function (VNDF) importance sampling
// Reference: "Sampling the GGX Distribution of Visible Normals" (Heitz 2018)
float3 sampleGGXVNDF(float3 Ve, float alpha, float u1, float u2) {
  float3 Vh = normalize(float3(alpha * Ve.x, alpha * Ve.y, Ve.z));
  float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
  float3 T1 = lensq > 0.0f ? float3(-Vh.y, Vh.x, 0.0f) * rsqrt(lensq) : float3(1.0f, 0.0f, 0.0f);
  float3 T2 = cross(Vh, T1);
  float r = sqrt(u1);
  float phi = 2.0f * 3.14159265f * u2;
  float t1 = r * cos(phi);
  float t2 = r * sin(phi);
  float s = 0.5f * (1.0f + Vh.z);
  t2 = (1.0f - s) * sqrt(max(0.0f, 1.0f - t1 * t1)) + s * t2;
  float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
  return normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));
}

float3 sampleGGXBounce(float3 N, float3 V, float roughness, thread uint& rng) {
  float alpha = max(roughness * roughness, 0.001f);
  float3 up = abs(N.y) < 0.999f ? float3(0,1,0) : float3(1,0,0);
  float3 T = normalize(cross(up, N));
  float3 B = cross(N, T);
  float3 Ve = float3(dot(V, T), dot(V, B), dot(V, N));
  float3 H_local = sampleGGXVNDF(Ve, alpha, rand01(rng), rand01(rng));
  float3 H = T * H_local.x + B * H_local.y + N * H_local.z;
  float3 L = reflect(-V, H);
  return L;
}

float3 applyNormalMap(float3 geomNormal, float3 shadingNormal, float3 p0, float3 p1, float3 p2, float2 uv0, float2 uv1, float2 uv2,
                      float2 uv, device const TextureData* textures, device const float4* texturePixels, uint4 normalTextureData,
                      float normalScale) {
  if (normalTextureData.y == 0u) return shadingNormal;

  float3 tangentSample = sampleBaseColorTexture(textures, texturePixels, normalTextureData.x, uv).xyz * 2.0f - 1.0f;
  tangentSample.xy *= normalScale;
  tangentSample = normalize(tangentSample);

  float3 e1 = p1 - p0;
  float3 e2 = p2 - p0;
  float2 duv1 = uv1 - uv0;
  float2 duv2 = uv2 - uv0;
  float det = duv1.x * duv2.y - duv1.y * duv2.x;
  if (fabs(det) < 1e-8f) return shadingNormal;

  float invDet = 1.0f / det;
  float3 tangent = normalize((e1 * duv2.y - e2 * duv1.y) * invDet);
  tangent = normalize(tangent - shadingNormal * dot(shadingNormal, tangent));
  float3 bitangent = normalize(cross(shadingNormal, tangent));
  if (dot(bitangent, (e2 * duv1.x - e1 * duv2.x) * invDet) < 0.0f) {
    bitangent = -bitangent;
  }

  float3 mapped = normalize(tangent * tangentSample.x + bitangent * tangentSample.y + shadingNormal * tangentSample.z);
  if (dot(mapped, geomNormal) <= 0.0f) return shadingNormal;
  return mapped;
}

float shadowVisibility(float3 hitPos, float3 toLight, float maxDistance,
                       device const TriangleData* triangles, device const MaterialData* materials,
                       thread uint& rng, intersector<curve_data, triangle_data, instancing> isector,
                       instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                       intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  ray shadowRay;
  shadowRay.origin = hitPos;
  shadowRay.direction = toLight;
  shadowRay.min_distance = 5e-3f;
  shadowRay.max_distance = maxDistance;

  for (uint i = 0u; i < 8u; ++i) {
    auto mcHit = isector.intersect(shadowRay, scene, 0xFFu);
    auto ptHit = isector.intersect(shadowRay, pointScene, 0xFFu, ftable);
    if (mcHit.type == intersection_type::none && ptHit.type == intersection_type::none) return 1.0f;

    // Sphere (point cloud) hit → always opaque
    bool ptCloser = ptHit.type != intersection_type::none &&
                    (mcHit.type == intersection_type::none || ptHit.distance < mcHit.distance);
    if (ptCloser) return 0.0f;

    if (mcHit.type == intersection_type::curve) return 0.0f;

    uint objectId = triangles[mcHit.primitive_id].objectFlags.x;
    float opacity = clamp(materials[objectId].transmissionIor.w, 0.0f, 1.0f);
    float transmission = clamp(materials[objectId].transmissionIor.x, 0.0f, 1.0f);

    if (opacity < 1e-5f && transmission < 1e-5f) {
      return 0.0f;
    }

    bool fullyOpaque = (opacity >= 0.999f) && (transmission <= 0.001f);
    if (fullyOpaque) return 0.0f;

    if (opacity < 1.0f - 1e-4f && transmission <= 0.5f) {
      if (rand01(rng) < opacity) return 0.0f;
    }

    float traveled = mcHit.distance + 2e-3f;
    shadowRay.origin = shadowRay.origin + toLight * traveled;
    shadowRay.min_distance = 1e-3f;
    shadowRay.max_distance = maxDistance - traveled;
    if (shadowRay.max_distance <= 0.0f) return 1.0f;
  }
  return 0.0f;
}

float evaluatePunctualLight(device const PunctualLightData* lights, uint lightIndex, float3 hitPos, float3 normal,
                            intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                            intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  PunctualLightData light = lights[lightIndex];
  uint lightType = uint(light.directionType.w);
  float3 lightColor = light.colorIntensity.xyz * light.colorIntensity.w;
  if (all(lightColor == float3(0.0f))) return 0.0f;

  float3 toLight = float3(0.0f);
  float maxDistance = 1e6f;
  float attenuation = 1.0f;

  if (lightType == 0u) {
    toLight = normalize(-light.directionType.xyz);
  } else {
    float3 lightPos = light.positionRange.xyz;
    toLight = lightPos - hitPos;
    float dist2 = max(dot(toLight, toLight), 1e-6f);
    maxDistance = sqrt(dist2);
    toLight /= maxDistance;
    attenuation = 1.0f / dist2;

    float range = light.positionRange.w;
    if (range > 0.0f) {
      if (maxDistance >= range) return 0.0f;
      float falloff = clamp(1.0f - (maxDistance / range), 0.0f, 1.0f);
      attenuation *= falloff * falloff;
    }

    if (lightType == 2u) {
      attenuation *= spotAttenuation(light.directionType.xyz, toLight, light.spotAngles.x, light.spotAngles.y);
      if (attenuation <= 0.0f) return 0.0f;
    }
  }

  float nDotL = max(dot(normal, toLight), 0.0f);
  if (nDotL <= 0.0f) return 0.0f;

  ray shadowRay;
  shadowRay.origin = hitPos;
  shadowRay.direction = toLight;
  shadowRay.min_distance = 5e-3f;
  shadowRay.max_distance = maxDistance;
  isector.accept_any_intersection(true);
  auto mcShadow = isector.intersect(shadowRay, scene, 0xFFu);
  auto ptShadow = isector.intersect(shadowRay, pointScene, 0xFFu, ftable);
  isector.accept_any_intersection(false);
  if (mcShadow.type != intersection_type::none || ptShadow.type != intersection_type::none) return 0.0f;

  float luminance = dot(lightColor, float3(0.2126f, 0.7152f, 0.0722f));
  return luminance * attenuation * nDotL;
}

bool samplePunctualLight(device const PunctualLightData* lights, uint lightIndex, float3 hitPos, thread float3& toLight,
                         thread float3& radiance, thread float& maxDistance) {
  PunctualLightData light = lights[lightIndex];
  uint lightType = uint(light.directionType.w);
  float3 lightColor = light.colorIntensity.xyz * light.colorIntensity.w;
  if (all(lightColor == float3(0.0f))) return false;

  float attenuation = 1.0f;
  maxDistance = 1e6f;

  if (lightType == 0u) {
    toLight = normalize(-light.directionType.xyz);
  } else {
    float3 lightPos = light.positionRange.xyz;
    toLight = lightPos - hitPos;
    float dist2 = max(dot(toLight, toLight), 1e-6f);
    maxDistance = sqrt(dist2);
    toLight /= maxDistance;
    attenuation = 1.0f / dist2;

    float range = light.positionRange.w;
    if (range > 0.0f) {
      if (maxDistance >= range) return false;
      float falloff = clamp(1.0f - (maxDistance / range), 0.0f, 1.0f);
      attenuation *= falloff * falloff;
    }

    if (lightType == 2u) {
      attenuation *= spotAttenuation(light.directionType.xyz, toLight, light.spotAngles.x, light.spotAngles.y);
      if (attenuation <= 0.0f) return false;
    }
  }

  radiance = lightColor * attenuation;
  return true;
}

struct SurfaceHitInfo {
  uint hit = 0u;
  uint objectId = 0u;
  float distance = 0.0f;
  float3 hitPos = float3(0.0f);
  float3 geomNormal = float3(0.0f, 1.0f, 0.0f);
  float3 normal = float3(0.0f, 1.0f, 0.0f);
  float3 baseColor = float3(1.0f);
  float3 emissive = float3(0.0f);
  float metallic = 0.0f;
  float roughness = 1.0f;
  float transmission = 0.0f;
  float ior = 1.5f;
  float dielectricF0 = 0.04f;
  float opacity = 1.0f;
  bool unlit = false;
  bool isInfinitePlane = false;
};

SurfaceHitInfo intersectGroundPlane(ray r, constant FrameUniforms& frame) {
  SurfaceHitInfo out;
  if (frame.planeColorEnabled.w < 0.5f) return out;

  float3 normal = float3(0, 1, 0);
  float planeHeight = frame.planeParams.x;

  // Only report intersection if ray origin is above the plane
  if (r.origin.y <= planeHeight) return out;

  float denom = dot(r.direction, normal);
  if (fabs(denom) < 1e-6f) return out;

  float t = (planeHeight - dot(r.origin, normal)) / denom;
  if (t < r.min_distance || t > r.max_distance) return out;

  out.hit = 1u;
  out.objectId = 0xFFFFFFFEu;
  out.distance = t;
  out.hitPos = r.origin + r.direction * t;
  out.geomNormal = normal;
  out.normal = normal;
  out.baseColor = frame.planeColorEnabled.xyz;
  out.metallic = frame.planeParams.y;
  out.roughness = clamp(frame.planeParams.z, 0.045f, 1.0f);
  out.dielectricF0 = frame.planeParams.w;
  out.isInfinitePlane = true;
  return out;
}

float3 traceSpecularPath(ray currentRay, uint remainingBounces, device const float4* positions, device const float4* normals,
                         device const float2* texcoords, device const float4* vertexColors,
                         device const TriangleData* triangles,
                         device const MaterialData* materials, device const TextureData* textures,
                         device const float4* texturePixels, device const PunctualLightData* sceneLights,
                         device const CurvePrimitiveGPU* curvePrimitives,
                         device const PointPrimitiveGPU* pointPrimitives,
                         constant FrameUniforms& frame, constant LightingData& lighting, thread uint& rng,
                         intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                         intersection_function_table<curve_data, triangle_data, instancing> ftable);

float3 sampleMissRadiance(float3 dir, constant LightingData& lighting) {
  return evalSimpleSky(dir, lighting);
}

SurfaceHitInfo shadeCurveHit(ray currentRay, float hitDistance, float curveParam, uint segmentId,
                             device const CurvePrimitiveGPU* curvePrimitives,
                             device const MaterialData* materials) {
  SurfaceHitInfo out;
  CurvePrimitiveGPU prim = curvePrimitives[segmentId];
  float3 hitPos = currentRay.origin + currentRay.direction * hitDistance;
  float3 p0 = prim.p0_radius.xyz;
  float3 p1 = prim.p1_type.xyz;
  uint matIdx = prim.materialObjectId.x;
  uint objId = prim.materialObjectId.y;

  float3 axis = normalize(p1 - p0);
  float3 normal;
  if (curveParam < 0.01f || curveParam > 0.99f) {
    normal = (curveParam < 0.5f) ? -axis : axis;
  } else {
    float3 pointOnAxis = mix(p0, p1, curveParam);
    normal = normalize(hitPos - pointOnAxis);
  }
  if (dot(normal, -currentRay.direction) < 0.0f) {
    normal = -normal;
  }

  MaterialData material = materials[matIdx];

  out.hit = 1u;
  out.objectId = objId;
  out.distance = hitDistance;
  out.hitPos = hitPos;
  out.geomNormal = normal;
  out.normal = normal;
  out.baseColor = clamp(material.baseColorFactor.xyz, 0.0f, 1.0f);
  out.emissive = material.emissiveFactor.xyz;
  out.metallic = material.metallicRoughnessNormal.x;
  out.roughness = clamp(material.metallicRoughnessNormal.y, 0.045f, 1.0f);
  out.transmission = 0.0f;
  out.ior = 1.5f;
  out.opacity = 1.0f;
  out.unlit = (material.transmissionIor.z > 0.5f);
  out.isInfinitePlane = false;
  return out;
}

// Return type for custom bounding-box intersection functions.
struct BoundingBoxIntersectionResult {
  bool  accept   [[accept_intersection]];
  float distance [[distance]];
};

// Custom bounding-box intersection function for analytic sphere ray-tracing.
// [[visible]] is required so the function is accessible by name from the host
// and can be inserted into an MTLIntersectionFunctionTable.
[[intersection(bounding_box, instancing)]]
BoundingBoxIntersectionResult sphereIntersection(float3 rayOrigin         [[origin]],
                                                  float3 rayDirection      [[direction]],
                                                  float  rayMinDistance    [[min_distance]],
                                                  float  rayMaxDistance    [[max_distance]],
                                                  uint   primitiveIndex    [[primitive_id]],
                                                  device const PointPrimitiveGPU* points [[buffer(25)]]) {
  BoundingBoxIntersectionResult result;
  result.accept = false;

  PointPrimitiveGPU pt = points[primitiveIndex];
  float3 center = pt.center_radius.xyz;
  float  radius = pt.center_radius.w;

  float3 oc = rayOrigin - center;
  float  b  = dot(oc, rayDirection);
  float  c  = dot(oc, oc) - radius * radius;
  float  disc = b * b - c;
  if (disc < 0.0f) return result;

  float sqrtDisc = sqrt(disc);
  float t = -b - sqrtDisc;
  if (t < rayMinDistance || t > rayMaxDistance) {
    t = -b + sqrtDisc;
    if (t < rayMinDistance || t > rayMaxDistance) return result;
  }

  result.accept   = true;
  result.distance = t;
  return result;
}

SurfaceHitInfo shadePointHit(ray currentRay, float hitDistance, uint primitiveIndex,
                              device const PointPrimitiveGPU* points,
                              device const MaterialData* materials) {
  SurfaceHitInfo out;
  PointPrimitiveGPU pt = points[primitiveIndex];
  float3 center  = pt.center_radius.xyz;
  float3 hitPos  = currentRay.origin + currentRay.direction * hitDistance;
  float3 normal  = normalize(hitPos - center);
  if (dot(normal, -currentRay.direction) < 0.0f) normal = -normal;

  uint matIdx = pt.materialObjectId.x;
  uint objId  = pt.materialObjectId.y;
  MaterialData material = materials[matIdx];

  out.hit        = 1u;
  out.objectId   = objId;
  out.distance   = hitDistance;
  out.hitPos     = hitPos;
  out.geomNormal = normal;
  out.normal     = normal;
  // Per-point color overrides the shared material base color when populated.
  out.baseColor  = clamp(pt.baseColor.xyz, 0.0f, 1.0f);
  out.emissive   = material.emissiveFactor.xyz;
  out.metallic   = material.metallicRoughnessNormal.x;
  out.roughness  = clamp(material.metallicRoughnessNormal.y, 0.045f, 1.0f);
  out.transmission = 0.0f;
  out.ior        = 1.5f;
  out.opacity    = 1.0f;
  out.unlit      = (material.transmissionIor.z > 0.5f);
  out.isInfinitePlane = false;
  return out;
}

SurfaceHitInfo intersectSurface(ray currentRay, device const float4* positions, device const float4* normals,
                                device const float2* texcoords, device const float4* vertexColors,
                                device const TriangleData* triangles,
                                device const MaterialData* materials, device const TextureData* textures,
                                device const float4* texturePixels,
                                device const CurvePrimitiveGPU* curvePrimitives,
                                device const PointPrimitiveGPU* pointPrimitives,
                                intersector<curve_data, triangle_data, instancing> isector,
                                instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                                intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  SurfaceHitInfo out;
  auto mcHit = isector.intersect(currentRay, scene, 0xFFu);
  auto ptHit = isector.intersect(currentRay, pointScene, 0xFFu, ftable);
  if (mcHit.type == intersection_type::none && ptHit.type == intersection_type::none) return out;

  bool ptCloser = ptHit.type != intersection_type::none &&
                  (mcHit.type == intersection_type::none || ptHit.distance < mcHit.distance);
  if (ptCloser) {
    return shadePointHit(currentRay, ptHit.distance, ptHit.primitive_id, pointPrimitives, materials);
  }

  if (mcHit.type == intersection_type::curve) {
    return shadeCurveHit(currentRay, mcHit.distance, mcHit.curve_parameter, mcHit.primitive_id, curvePrimitives, materials);
  }
  uint triangleIndex = mcHit.primitive_id;
  TriangleData tri = triangles[triangleIndex];
  float3 p0 = positions[tri.indicesMaterial.x].xyz;
  float3 p1 = positions[tri.indicesMaterial.y].xyz;
  float3 p2 = positions[tri.indicesMaterial.z].xyz;
  float3 hitPos = currentRay.origin + currentRay.direction * mcHit.distance;

  float2 bary = mcHit.triangle_barycentric_coord;
  float w0 = 1.0f - bary.x - bary.y;
  float w1 = bary.x;
  float w2 = bary.y;

  float3 geomNormal = normalize(cross(p1 - p0, p2 - p0));
  float3 normal = geomNormal;
  if (tri.objectFlags.y != 0u) {
    float3 n0 = normals[tri.indicesMaterial.x].xyz;
    float3 n1 = normals[tri.indicesMaterial.y].xyz;
    float3 n2 = normals[tri.indicesMaterial.z].xyz;
    normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
  }
  if (dot(normal, geomNormal) <= 0.0f) normal = -normal;

  float3 vc0 = vertexColors[tri.indicesMaterial.x].xyz;
  float3 vc1 = vertexColors[tri.indicesMaterial.y].xyz;
  float3 vc2 = vertexColors[tri.indicesMaterial.z].xyz;
  float3 interpVertexColor = vc0 * w0 + vc1 * w1 + vc2 * w2;

  MaterialData material = materials[tri.indicesMaterial.w];
  float4 baseColor = material.baseColorFactor * float4(interpVertexColor, 1.0f);
  float metallic = material.metallicRoughnessNormal.x;
  float roughness = material.metallicRoughnessNormal.y;
  float normalScale = material.metallicRoughnessNormal.z;
  float3 emissive = material.emissiveFactor.xyz;
  float2 uv0 = texcoords[tri.indicesMaterial.x];
  float2 uv1 = texcoords[tri.indicesMaterial.y];
  float2 uv2 = texcoords[tri.indicesMaterial.z];
  float2 uv = uv0 * w0 + uv1 * w1 + uv2 * w2;

  if (material.baseColorTextureData.y != 0u) {
    baseColor *= sampleBaseColorTexture(textures, texturePixels, material.baseColorTextureData.x, uv);
    if (material.emissiveTextureData.y != 0u) {
      emissive *= sampleBaseColorTexture(textures, texturePixels, material.emissiveTextureData.x, uv).xyz;
    }
  } else if (material.emissiveTextureData.y != 0u) {
    emissive *= sampleBaseColorTexture(textures, texturePixels, material.emissiveTextureData.x, uv).xyz;
  }

  if (material.metallicRoughnessTextureData.y != 0u) {
    float4 mr = sampleBaseColorTexture(textures, texturePixels, material.metallicRoughnessTextureData.x, uv);
    roughness *= mr.y;
    metallic *= mr.z;
  }
  roughness = clamp(roughness, 0.045f, 1.0f);
  metallic = saturate(metallic);
  normal = applyNormalMap(geomNormal, normal, p0, p1, p2, uv0, uv1, uv2, uv, textures, texturePixels,
                          material.normalTextureData, normalScale);

  out.hit = 1u;
  out.objectId = tri.objectFlags.x;
  out.distance = mcHit.distance;
  out.hitPos = hitPos;
  out.geomNormal = geomNormal;
  out.normal = normal;
  out.baseColor = clamp(baseColor.xyz, 0.0f, 1.0f);
  out.emissive = emissive;
  out.metallic = metallic;
  out.roughness = roughness;
  out.transmission = clamp(material.transmissionIor.x, 0.0f, 1.0f);
  out.ior = max(material.transmissionIor.y, 1.0f);
  out.opacity = clamp(material.transmissionIor.w, 0.0f, 1.0f);
  out.unlit = (material.transmissionIor.z > 0.5f);
  out.isInfinitePlane = false;

  // Wireframe edge overlay: if enabled, override to edge color when close to any triangle edge.
  if (tri.objectFlags.z != 0u) {
    float edgeThreshold = material.wireframeEdgeData.w;
    float minBary = min(w0, min(w1, w2));
    if (minBary < edgeThreshold) {
      float3 edgeCol = material.wireframeEdgeData.xyz;
      out.baseColor   = edgeCol;
      out.emissive    = edgeCol;
      out.metallic    = 0.0f;
      out.roughness   = 1.0f;
      out.transmission = 0.0f;
      out.opacity     = 1.0f;
      out.unlit       = true;
    }
  }

  return out;
}

float3 evaluateDirectLightingPBR(SurfaceHitInfo surf, float3 viewDir, device const PunctualLightData* sceneLights,
                                 device const TriangleData* triangles, device const MaterialData* materials,
                                 constant FrameUniforms& frame, constant LightingData& lighting, thread uint& rng,
                                 intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                                 intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 N = surf.normal;
  float3 V = viewDir;
  float3 albedo = surf.baseColor;
  float metallic = surf.metallic;
  float roughness = surf.roughness;
  float3 F0 = mix(float3(surf.dielectricF0), albedo, metallic);
  float grazingMax = mix(clamp(surf.dielectricF0 * 25.0f, 0.0f, 1.0f), 1.0f, metallic);
  float3 directLighting = float3(0.0f);
  // Infinite ground plane: use Lambert-only direct light. GGX + area lights on a huge plane produces a long grazing
  // highlight strip (denominator ~ NdotV, foreshortened light footprint) unrelated to the sky gradient.
  const bool isInfinitePlane = surf.isInfinitePlane;
  float3 diffuseAlbedo = albedo * (1.0f - metallic);

  // Main directional light (always evaluate if intensity > 0)
  float mainIntensity = lighting.mainLightColorIntensity.w;
  if (mainIntensity > 1e-5f) {
    float3 L = normalize(-lighting.mainLightDirection.xyz);
    float visibility = shadowVisibility(surf.hitPos, L, 1e6f, triangles, materials, rng, isector, scene, pointScene, ftable);
    float3 radiance = lighting.mainLightColorIntensity.xyz * mainIntensity;
    float NdotL = max(dot(N, L), 0.0f);
    if (NdotL > 0.0f && visibility > 0.0f) {
      if (isInfinitePlane) {
        directLighting += (diffuseAlbedo / 3.14159265f) * radiance * NdotL * visibility;
      } else {
        float3 H = normalize(V + L);
        float NDF = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0, grazingMax);
        float3 specular = (NDF * G * F) / max(4.0f * max(dot(N, V), 0.0f) * NdotL, 1e-4f);
        float3 kS = F;
        float3 kD = (1.0f - kS) * (1.0f - metallic);
        directLighting += ((kD * albedo / 3.14159265f) + specular) * radiance * NdotL * visibility;
      }
    }
  }

  // Punctual scene lights (always evaluate if enabled and count > 0)
  if (frame.lightCount > 0u) {
    for (uint lightIndex = 0u; lightIndex < frame.lightCount; ++lightIndex) {
      float3 L;
      float3 radiance;
      float maxDistance = 1e6f;
      if (!samplePunctualLight(sceneLights, lightIndex, surf.hitPos, L, radiance, maxDistance)) continue;
      float visibility = shadowVisibility(surf.hitPos, L, maxDistance, triangles, materials, rng, isector, scene, pointScene, ftable);
      float NdotL = max(dot(N, L), 0.0f);
      if (NdotL <= 0.0f || visibility <= 0.0f || all(radiance <= 0.0f)) continue;
      if (isInfinitePlane) {
        directLighting += (diffuseAlbedo / 3.14159265f) * radiance * NdotL * visibility;
      } else {
        float3 H = normalize(V + L);
        float NDF = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0, grazingMax);
        float3 specular = (NDF * G * F) / max(4.0f * max(dot(N, V), 0.0f) * NdotL, 1e-4f);
        float3 kS = F;
        float3 kD = (1.0f - kS) * (1.0f - metallic);
        directLighting += ((kD * albedo / 3.14159265f) + specular) * radiance * NdotL * visibility;
      }
    }
  }

  // Area light (NEE only; full two-way MIS would also check BRDF bounce hits on the light)
  if (frame.enableAreaLight != 0u) {
    float3 L;
    float3 radiance;
    if (sampleAreaLight(surf.hitPos, N, lighting, triangles, materials, rng, L, radiance, isector, scene, pointScene, ftable)) {
      float NdotL = max(dot(N, L), 0.0f);
      if (isInfinitePlane) {
        directLighting += (diffuseAlbedo / 3.14159265f) * radiance * NdotL;
      } else {
        float3 H = normalize(V + L);
        float NDF = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0, grazingMax);
        float3 specular = (NDF * G * F) / max(4.0f * max(dot(N, V), 0.0f) * NdotL, 1e-4f);
        float3 kS = F;
        float3 kD = (1.0f - kS) * (1.0f - metallic);
        directLighting += ((kD * albedo / 3.14159265f) + specular) * radiance * NdotL;
      }
    }
  }

  return directLighting;
}

float3 shadeStandardTransmissionSurface(SurfaceHitInfo surf, float3 viewDir, device const PunctualLightData* sceneLights,
                                        device const TriangleData* triangles, device const MaterialData* materials,
                                        constant FrameUniforms& frame, constant LightingData& lighting, thread uint& rng,
                                        intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                                        intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 N = surf.normal;
  float3 V = viewDir;
  float roughness = clamp(surf.roughness, 0.01f, 1.0f);
  float f0Scalar = pow((surf.ior - 1.0f) / (surf.ior + 1.0f), 2.0f);
  float3 F0 = float3(f0Scalar);
  float3 specularLighting = float3(0.0f);

  // Main directional light
  float mainIntensity = lighting.mainLightColorIntensity.w;
  if (mainIntensity > 1e-5f) {
    float3 L = normalize(-lighting.mainLightDirection.xyz);
    float3 radiance = lighting.mainLightColorIntensity.xyz * mainIntensity;
    float visibility = shadowVisibility(surf.hitPos, L, 1e6f, triangles, materials, rng, isector, scene, pointScene, ftable);
    float NdotL = max(dot(N, L), 0.0f);
    if (NdotL > 0.0f && visibility > 0.0f) {
      float3 H = normalize(V + L);
      float NDF = distributionGGX(N, H, roughness);
      float G = geometrySmith(N, V, L, roughness);
      float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);
      float3 specular = (NDF * G * F) / max(4.0f * max(dot(N, V), 0.0f) * NdotL, 1e-4f);
      specularLighting += specular * radiance * NdotL * visibility;
    }
  }

  // Punctual scene lights
  if (frame.lightCount > 0u) {
    for (uint lightIndex = 0u; lightIndex < frame.lightCount; ++lightIndex) {
      float3 L;
      float3 radiance;
      float maxDistance = 1e6f;
      if (!samplePunctualLight(sceneLights, lightIndex, surf.hitPos, L, radiance, maxDistance)) continue;
      float visibility = shadowVisibility(surf.hitPos, L, maxDistance, triangles, materials, rng, isector, scene, pointScene, ftable);
      float NdotL = max(dot(N, L), 0.0f);
      if (NdotL <= 0.0f || visibility <= 0.0f || all(radiance <= 0.0f)) continue;
      float3 H = normalize(V + L);
      float NDF = distributionGGX(N, H, roughness);
      float G = geometrySmith(N, V, L, roughness);
      float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);
      float3 specular = (NDF * G * F) / max(4.0f * max(dot(N, V), 0.0f) * NdotL, 1e-4f);
      specularLighting += specular * radiance * NdotL * visibility;
    }
  }

  // Area light
  if (frame.enableAreaLight != 0u) {
    float3 L;
    float3 radiance;
    if (sampleAreaLight(surf.hitPos, N, lighting, triangles, materials, rng, L, radiance, isector, scene, pointScene, ftable)) {
      float NdotL = max(dot(N, L), 0.0f);
      if (NdotL > 0.0f && any(radiance > 0.0f)) {
        float3 H = normalize(V + L);
        float NDF = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        float3 F = fresnelSchlick(max(dot(H, V), 0.0f), F0);
        float3 specular = (NDF * G * F) / max(4.0f * max(dot(N, V), 0.0f) * NdotL, 1e-4f);
        specularLighting += specular * radiance * NdotL;
      }
    }
  }

  float3 ambientSpec = F0 * sampleEnvironment(N, lighting) * 0.15f;
  return specularLighting + ambientSpec + surf.emissive;
}

float3 traceStandardPath(ray currentRay, float3 viewDir, SurfaceHitInfo firstHit, device const float4* positions, device const float4* normals,
                         device const float2* texcoords, device const float4* vertexColors,
                         device const TriangleData* triangles,
                         device const MaterialData* materials, device const TextureData* textures,
                         device const float4* texturePixels, device const PunctualLightData* sceneLights,
                         device const CurvePrimitiveGPU* curvePrimitives,
                         device const PointPrimitiveGPU* pointPrimitives,
                         constant FrameUniforms& frame, constant LightingData& lighting, thread uint& rng,
                         intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                         intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 throughput = float3(1.0f);
  float3 radiance = float3(0.0f);
  float3 currentViewDir = viewDir;
  bool prevWasInfinitePlane = false;

  for (uint bounce = 0u; bounce < max(frame.maxBounces, 1u); ++bounce) {
    SurfaceHitInfo surf;
    if (bounce == 0u && firstHit.hit != 0u) {
      surf = firstHit;
    } else {
      surf = intersectSurface(currentRay, positions, normals, texcoords, vertexColors, triangles, materials, textures, texturePixels,
                              curvePrimitives, pointPrimitives, isector, scene, pointScene, ftable);
      SurfaceHitInfo planeSurf = intersectGroundPlane(currentRay, frame);
      if (planeSurf.hit != 0u && (surf.hit == 0u || planeSurf.distance < surf.distance)) {
        surf = planeSurf;
      }
    }
    if (surf.hit == 0u) {
      float3 miss = sampleMissRadiance(currentRay.direction, lighting);
      if (prevWasInfinitePlane) {
        miss = mix(miss, lighting.backgroundColor.xyz, 0.92f);
      }
      radiance += throughput * miss;
      break;
    }

    if (surf.opacity < 1.0f - 1e-4f) {
      if (surf.transmission > 0.5f) {
        float f0 = pow((surf.ior - 1.0f) / (surf.ior + 1.0f), 2.0f);
        float cosTheta = abs(dot(currentRay.direction, surf.normal));
        float fresnel = f0 + (1.0f - f0) * pow(1.0f - cosTheta, 5.0f);
        if (rand01(rng) < fresnel) {
          bool front = dot(currentRay.direction, surf.normal) < 0.0f;
          float3 rN = front ? surf.normal : -surf.normal;
          currentRay.origin = surf.hitPos + rN * 2e-3f;
          currentRay.direction = reflect(currentRay.direction, rN);
          currentRay.min_distance = 1e-3f;
          currentRay.max_distance = 1e6f;
          currentViewDir = -currentRay.direction;
          prevWasInfinitePlane = false;
          continue;
        }
      }
      if (rand01(rng) > surf.opacity) {
        float pushSign = dot(currentRay.direction, surf.normal) > 0.0f ? 1.0f : -1.0f;
        currentRay.origin = surf.hitPos + pushSign * surf.normal * 2e-3f;
        currentRay.min_distance = 1e-3f;
        currentRay.max_distance = 1e6f;
        prevWasInfinitePlane = false;
        continue;
      }
    }

    if (surf.transmission > 1e-4f && surf.opacity > 1.0f - 1e-4f) {
      {
        SurfaceHitInfo glassSurf = surf;
        glassSurf.roughness = max(glassSurf.roughness, 0.08f);
        radiance += throughput * shadeStandardTransmissionSurface(glassSurf, currentViewDir, sceneLights, triangles, materials,
                                                                  frame, lighting, rng, isector, scene, pointScene, ftable);
      }
      float3 N = surf.normal;
      bool frontFace = dot(currentRay.direction, N) < 0.0f;
      float3 orientedN = frontFace ? N : -N;
      float eta = frontFace ? (1.0f / surf.ior) : surf.ior;
      float cosTheta = clamp(dot(-currentRay.direction, orientedN), 0.0f, 1.0f);
      float sin2Theta = max(0.0f, 1.0f - cosTheta * cosTheta);
      bool cannotRefract = eta * eta * sin2Theta > 1.0f;
      float f0 = pow((surf.ior - 1.0f) / (surf.ior + 1.0f), 2.0f);
      float fresnel = f0 + (1.0f - f0) * pow(1.0f - cosTheta, 5.0f);
      if (cannotRefract) {
        currentRay.origin = surf.hitPos + orientedN * 1e-2f;
        currentRay.direction = reflect(currentRay.direction, orientedN);
        currentRay.min_distance = 1e-3f;
        currentRay.max_distance = 1e6f;
        currentViewDir = -currentRay.direction;
        prevWasInfinitePlane = surf.isInfinitePlane;
        continue;
      }

      ray reflectedRay;
      reflectedRay.origin = surf.hitPos + orientedN * 1e-2f;
      reflectedRay.direction = reflect(currentRay.direction, orientedN);
      reflectedRay.min_distance = 1e-3f;
      reflectedRay.max_distance = 1e6f;
      radiance += throughput *
                  traceSpecularPath(reflectedRay, max(frame.maxBounces - bounce - 1u, 1u), positions, normals, texcoords, vertexColors,
                                    triangles, materials, textures, texturePixels, sceneLights, curvePrimitives, pointPrimitives,
                                    frame, lighting, rng, isector, scene, pointScene, ftable) *
                  fresnel;

      float3 tint = mix(float3(1.0f), surf.baseColor, 0.08f);
      throughput *= tint * surf.transmission * max(1.0f - fresnel, 1e-3f);
      currentRay.origin = surf.hitPos - orientedN * 1e-2f;
      currentRay.direction = normalize(refract(currentRay.direction, orientedN, eta));
      currentRay.min_distance = 1e-3f;
      currentRay.max_distance = 1e6f;
      currentViewDir = -currentRay.direction;
      prevWasInfinitePlane = surf.isInfinitePlane;
      continue;
    }

    if (surf.unlit) {
      radiance += throughput * surf.baseColor;
      break;
    }

    if (surf.isInfinitePlane) {
      float3 directLighting = evaluateDirectLightingPBR(surf, currentViewDir, sceneLights, triangles, materials, frame, lighting, rng, isector, scene, pointScene, ftable);
      radiance += throughput * (directLighting + surf.emissive);
      throughput *= surf.baseColor * (1.0f - surf.metallic);
      currentRay.direction = cosineWeightedHemisphere(surf.normal, rng);
      if (max(max(throughput.x, throughput.y), throughput.z) < 1e-5f) break;
      if (bounce >= 2u) {
        float rrProb = clamp(max(max(throughput.x, throughput.y), throughput.z) + 0.001f, 0.05f, 0.95f);
        if (rand01(rng) >= rrProb) break;
        throughput /= rrProb;
      }
      currentRay.origin = surf.hitPos + surf.normal * 1e-3f;
      currentRay.min_distance = 1e-3f;
      currentRay.max_distance = 1e6f;
      currentViewDir = -currentRay.direction;
      prevWasInfinitePlane = true;
      continue;
    }

    // Opaque surface — NEE direct lighting + emissive (no ambient, indirect comes from bounces)
    float3 directLighting = evaluateDirectLightingPBR(surf, currentViewDir, sceneLights, triangles, materials, frame, lighting, rng, isector, scene, pointScene, ftable);
    float3 F0 = mix(float3(surf.dielectricF0), surf.baseColor, surf.metallic);
    float grazingMax = mix(clamp(surf.dielectricF0 * 25.0f, 0.0f, 1.0f), 1.0f, surf.metallic);
    float NdotV = max(dot(surf.normal, currentViewDir), 0.0f);
    float3 F = fresnelSchlick(NdotV, F0, grazingMax);
    radiance += throughput * (directLighting + surf.emissive);

    // Choose specular vs diffuse bounce via Fresnel
    float specProb = clamp((F.x + F.y + F.z) / 3.0f + surf.metallic * 0.5f, 0.04f, 0.96f);

    if (rand01(rng) < specProb) {
      // Specular bounce: GGX VNDF importance sampling
      float3 bounceDir = sampleGGXBounce(surf.normal, currentViewDir, surf.roughness, rng);
      if (dot(bounceDir, surf.normal) <= 0.0f) {
        bounceDir = reflect(-currentViewDir, surf.normal);
      }
      throughput *= F / specProb;
      currentRay.direction = bounceDir;
    } else {
      // Diffuse bounce
      float3 kD = (1.0f - F) * (1.0f - surf.metallic);
      throughput *= surf.baseColor * kD / (1.0f - specProb);
      currentRay.direction = cosineWeightedHemisphere(surf.normal, rng);
    }

    // Early exit if throughput is negligible
    if (max(max(throughput.x, throughput.y), throughput.z) < 1e-5f) break;

    // Russian roulette from bounce 2+ (guarantee first 2 indirect bounces)
    if (bounce >= 2u) {
      float rrProb = clamp(max(max(throughput.x, throughput.y), throughput.z) + 0.001f, 0.05f, 0.95f);
      if (rand01(rng) >= rrProb) break;
      throughput /= rrProb;
    }

    currentRay.origin = surf.hitPos + surf.normal * 1e-3f;
    currentRay.min_distance = 1e-3f;
    currentRay.max_distance = 1e6f;
    currentViewDir = -currentRay.direction;
    prevWasInfinitePlane = surf.isInfinitePlane;
  }

  return radiance;
}

float3 traceSpecularPath(ray currentRay, uint remainingBounces, device const float4* positions, device const float4* normals,
                         device const float2* texcoords, device const float4* vertexColors,
                         device const TriangleData* triangles,
                         device const MaterialData* materials, device const TextureData* textures,
                         device const float4* texturePixels, device const PunctualLightData* sceneLights,
                         device const CurvePrimitiveGPU* curvePrimitives,
                         device const PointPrimitiveGPU* pointPrimitives,
                         constant FrameUniforms& frame, constant LightingData& lighting, thread uint& rng,
                         intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                         intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 throughput = float3(1.0f);
  float3 radiance = float3(0.0f);
  float3 currentViewDir = -currentRay.direction;
  bool prevWasInfinitePlane = false;

  for (uint bounce = 0u; bounce < max(remainingBounces, 1u); ++bounce) {
    SurfaceHitInfo surf = intersectSurface(currentRay, positions, normals, texcoords, vertexColors, triangles, materials, textures, texturePixels,
                                           curvePrimitives, pointPrimitives, isector, scene, pointScene, ftable);
    SurfaceHitInfo planeSurf = intersectGroundPlane(currentRay, frame);
    if (planeSurf.hit != 0u && (surf.hit == 0u || planeSurf.distance < surf.distance)) {
      surf = planeSurf;
    }
    if (surf.hit == 0u) {
      float3 miss = sampleMissRadiance(currentRay.direction, lighting);
      if (prevWasInfinitePlane) {
        miss = mix(miss, lighting.backgroundColor.xyz, 0.92f);
      }
      radiance += throughput * miss;
      break;
    }

    if (surf.unlit) {
      radiance += throughput * surf.baseColor;
      break;
    }

    if (surf.opacity < 1.0f - 1e-4f) {
      if (surf.transmission > 0.5f) {
        float f0 = pow((surf.ior - 1.0f) / (surf.ior + 1.0f), 2.0f);
        float cosTheta = abs(dot(currentRay.direction, surf.normal));
        float fresnel = f0 + (1.0f - f0) * pow(1.0f - cosTheta, 5.0f);
        if (rand01(rng) < fresnel) {
          bool front = dot(currentRay.direction, surf.normal) < 0.0f;
          float3 rN = front ? surf.normal : -surf.normal;
          currentRay.origin = surf.hitPos + rN * 2e-3f;
          currentRay.direction = reflect(currentRay.direction, rN);
          currentRay.min_distance = 1e-3f;
          currentRay.max_distance = 1e6f;
          currentViewDir = -currentRay.direction;
          prevWasInfinitePlane = false;
          continue;
        }
      }
      if (rand01(rng) > surf.opacity) {
        float pushSign = dot(currentRay.direction, surf.normal) > 0.0f ? 1.0f : -1.0f;
        currentRay.origin = surf.hitPos + pushSign * surf.normal * 2e-3f;
        currentRay.min_distance = 1e-3f;
        currentRay.max_distance = 1e6f;
        prevWasInfinitePlane = false;
        continue;
      }
    }

    if (surf.transmission > 1e-4f && surf.opacity > 1.0f - 1e-4f) {
      float3 N = surf.normal;
      bool frontFace = dot(currentRay.direction, N) < 0.0f;
      float3 orientedN = frontFace ? N : -N;
      float eta = frontFace ? (1.0f / surf.ior) : surf.ior;
      float cosTheta = clamp(dot(-currentRay.direction, orientedN), 0.0f, 1.0f);
      float sin2Theta = max(0.0f, 1.0f - cosTheta * cosTheta);
      bool cannotRefract = eta * eta * sin2Theta > 1.0f;
      float f0 = pow((surf.ior - 1.0f) / (surf.ior + 1.0f), 2.0f);
      float fresnel = f0 + (1.0f - f0) * pow(1.0f - cosTheta, 5.0f);

      if (cannotRefract || rand01(rng) < fresnel) {
        currentRay.origin = surf.hitPos + orientedN * 1e-2f;
        currentRay.direction = reflect(currentRay.direction, orientedN);
      } else {
        float3 tint = mix(float3(1.0f), surf.baseColor, 0.08f);
        throughput *= tint * surf.transmission;
        currentRay.origin = surf.hitPos - orientedN * 1e-2f;
        currentRay.direction = normalize(refract(currentRay.direction, orientedN, eta));
      }

      currentRay.min_distance = 1e-3f;
      currentRay.max_distance = 1e6f;
      currentViewDir = -currentRay.direction;
      prevWasInfinitePlane = surf.isInfinitePlane;
      continue;
    }

    if (surf.isInfinitePlane) {
      float3 directLighting = evaluateDirectLightingPBR(surf, currentViewDir, sceneLights, triangles, materials, frame, lighting, rng, isector, scene, pointScene, ftable);
      radiance += throughput * (directLighting + surf.emissive);
      throughput *= surf.baseColor * (1.0f - surf.metallic);
      currentRay.direction = cosineWeightedHemisphere(surf.normal, rng);
      if (max(max(throughput.x, throughput.y), throughput.z) < 1e-5f) break;
      if (bounce >= 2u) {
        float rrProb = clamp(max(max(throughput.x, throughput.y), throughput.z) + 0.001f, 0.05f, 0.95f);
        if (rand01(rng) >= rrProb) break;
        throughput /= rrProb;
      }
      currentRay.origin = surf.hitPos + surf.normal * 1e-3f;
      currentRay.min_distance = 1e-3f;
      currentRay.max_distance = 1e6f;
      currentViewDir = -currentRay.direction;
      prevWasInfinitePlane = true;
      continue;
    }

    // Full bounce for opaque surfaces (same quality as traceStandardPath)
    float3 directLighting = evaluateDirectLightingPBR(surf, currentViewDir, sceneLights, triangles, materials, frame, lighting, rng, isector, scene, pointScene, ftable);
    float3 F0 = mix(float3(surf.dielectricF0), surf.baseColor, surf.metallic);
    float grazingMax = mix(clamp(surf.dielectricF0 * 25.0f, 0.0f, 1.0f), 1.0f, surf.metallic);
    float NdotV = max(dot(surf.normal, currentViewDir), 0.0f);
    float3 F = fresnelSchlick(NdotV, F0, grazingMax);
    radiance += throughput * (directLighting + surf.emissive);

    float specProb = clamp((F.x + F.y + F.z) / 3.0f + surf.metallic * 0.5f, 0.04f, 0.96f);
    if (rand01(rng) < specProb) {
      float3 bounceDir = sampleGGXBounce(surf.normal, currentViewDir, surf.roughness, rng);
      if (dot(bounceDir, surf.normal) <= 0.0f) bounceDir = reflect(-currentViewDir, surf.normal);
      throughput *= F / specProb;
      currentRay.direction = bounceDir;
    } else {
      float3 kD = (1.0f - F) * (1.0f - surf.metallic);
      throughput *= surf.baseColor * kD / (1.0f - specProb);
      currentRay.direction = cosineWeightedHemisphere(surf.normal, rng);
    }

    if (max(max(throughput.x, throughput.y), throughput.z) < 1e-5f) break;
    if (bounce >= 2u) {
      float rrProb = clamp(max(max(throughput.x, throughput.y), throughput.z) + 0.001f, 0.05f, 0.95f);
      if (rand01(rng) >= rrProb) break;
      throughput /= rrProb;
    }

    currentRay.origin = surf.hitPos + surf.normal * 1e-3f;
    currentRay.min_distance = 1e-3f;
    currentRay.max_distance = 1e6f;
    currentViewDir = -currentRay.direction;
    prevWasInfinitePlane = surf.isInfinitePlane;
  }

  return radiance;
}

[[kernel]] void pathTraceKernel(device const float4* positions [[buffer(0)]],
                                device const float4* normals [[buffer(1)]],
                                device const float2* texcoords [[buffer(2)]],
                                device const TriangleData* triangles [[buffer(3)]],
                                device const MaterialData* materials [[buffer(4)]],
                                device const TextureData* textures [[buffer(5)]],
                                device const float4* texturePixels [[buffer(6)]],
                                device const PunctualLightData* sceneLights [[buffer(7)]],
                                constant CameraData& camera [[buffer(8)]],
                                constant FrameUniforms& frame [[buffer(9)]],
                                constant LightingData& lighting [[buffer(10)]],
                                device float4* accumulation [[buffer(11)]],
                                device float4* output [[buffer(12)]],
                                device float* depthBuffer [[buffer(13)]],
                                device float* linearDepthBuffer [[buffer(14)]],
                                device float4* normalBuffer [[buffer(15)]],
                                device uint* objectIdBuffer [[buffer(16)]],
                                instance_acceleration_structure scene [[buffer(17)]],
                                instance_acceleration_structure pointScene [[buffer(26)]],
                                device float4* diffuseAlbedoBuffer [[buffer(18)]],
                                device float4* specularAlbedoBuffer [[buffer(19)]],
                                device float* roughnessBuffer [[buffer(20)]],
                                device float4* motionVectorBuffer [[buffer(21)]],
                                device const float4* vertexColors [[buffer(22)]],
                                device const CurvePrimitiveGPU* curvePrimitives [[buffer(23)]],
                                intersection_function_table<curve_data, triangle_data, instancing> ftable [[buffer(24)]],
                                device const PointPrimitiveGPU* pointPrimitives [[buffer(25)]],
                                uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= frame.width || gid.y >= frame.height) return;

  uint pixelIndex = gid.y * frame.width + gid.x;
  intersector<curve_data, triangle_data, instancing> isector;
  isector.assume_geometry_type(geometry_type::bounding_box | geometry_type::curve | geometry_type::triangle);

  float3 colorSum = float3(0.0f);
  float alphaSum = 0.0f;
  bool writeDataBuffer = frame.frameIndex == 0u;
  float firstDepth = 1.0f;
  float firstLinearDepth = -1.0f;
  float3 firstNormal = float3(0.0f, 0.0f, 0.0f);
  uint firstObjectId = 0u;

  for (uint sample = 0; sample < frame.samplesPerIteration; ++sample) {
    uint haltonIndex = pixelIndex * 131u + frame.rngFrameIndex * frame.samplesPerIteration + sample;
    uint rng = wangHash(pixelIndex ^ (frame.rngFrameIndex * 9781u) ^ (sample * 6271u));

    bool firstFrameSample = frame.frameIndex == 0u && sample == 0u;
    uint cpSeed = wangHash(pixelIndex ^ 0x7a3c5e91u);
    float cpX = float(cpSeed & 0xFFFFu) / 65536.0f;
    float cpY = float((cpSeed >> 16u) & 0xFFFFu) / 65536.0f;
    float jitterX = firstFrameSample ? 0.5f : fract(halton(haltonIndex, 0) + cpX);
    float jitterY = firstFrameSample ? 0.5f : fract(halton(haltonIndex, 1) + cpY);
    jitterX += frame.jitterOffset.x;
    jitterY += frame.jitterOffset.y;
    float ndcX = (2.0f * ((float(gid.x) + jitterX) / float(frame.width)) - 1.0f) *
                 camera.clipData.y * tan(0.5f * camera.clipData.x);
    float ndcY = (1.0f - 2.0f * ((float(gid.y) + jitterY) / float(frame.height))) * tan(0.5f * camera.clipData.x);

    float3 direction = normalize(camera.lookDir.xyz + ndcX * camera.rightDir.xyz + ndcY * camera.upDir.xyz);

    ray currentRay;
    currentRay.origin = camera.position.xyz;
    currentRay.direction = direction;
    currentRay.min_distance = 1e-3f;
    currentRay.max_distance = 1e6f;

    float3 sampleColor = float3(0.0f);
    float sampleAlpha = 1.0f;

    SurfaceHitInfo surf = intersectSurface(currentRay, positions, normals, texcoords, vertexColors, triangles, materials, textures, texturePixels,
                                           curvePrimitives, pointPrimitives, isector, scene, pointScene, ftable);
    SurfaceHitInfo planeSurf = intersectGroundPlane(currentRay, frame);
    if (planeSurf.hit != 0u && (surf.hit == 0u || planeSurf.distance < surf.distance)) {
      surf = planeSurf;
    }
    if (surf.hit != 0u) {
      if (frame.renderMode == 0u) {
        sampleColor =
            traceStandardPath(currentRay, normalize(camera.position.xyz - surf.hitPos), surf, positions, normals, texcoords, vertexColors,
                              triangles, materials, textures, texturePixels, sceneLights, curvePrimitives, pointPrimitives,
                              frame, lighting, rng, isector, scene, pointScene, ftable);
        float lum = dot(sampleColor, float3(0.212671f, 0.715160f, 0.072169f));
        if (lum > 5.0f) {
          sampleColor *= 5.0f / lum;
        }
      } else {
        if (surf.unlit) {
          sampleColor = surf.baseColor;
        } else {
          float direct = 0.0f;
          if (lighting.mainLightColorIntensity.w > 1e-5f) {
            direct += evaluateDirectionalLight(lighting.mainLightDirection.xyz, lighting.mainLightColorIntensity.w, surf.hitPos,
                                              surf.normal, isector, scene, pointScene, ftable);
          }
          if (frame.lightCount > 0u) {
            for (uint lightIndex = 0u; lightIndex < frame.lightCount; ++lightIndex) {
              direct += evaluatePunctualLight(sceneLights, lightIndex, surf.hitPos, surf.normal, isector, scene, pointScene, ftable);
            }
          }
          if (frame.enableAreaLight != 0u) {
            float3 areaL;
            float3 areaRadiance;
            if (sampleAreaLight(surf.hitPos, surf.normal, lighting, triangles, materials, rng, areaL, areaRadiance, isector, scene, pointScene, ftable)) {
              direct += dot(areaRadiance, float3(0.2126f, 0.7152f, 0.0722f)) * max(dot(surf.normal, areaL), 0.0f);
            }
          }
          float intensity = max(frame.ambientFloor, toonShading(min(direct, 1.0f), frame.toonBandCount));
          sampleColor = surf.baseColor * intensity + surf.emissive;
        }
      }
      sampleAlpha = 1.0f;

      if (firstFrameSample) {
        firstDepth = computeClipDepth(surf.hitPos, camera);
        firstLinearDepth = surf.distance;
        firstNormal = surf.normal;
        firstObjectId = surf.objectId;
      }
    } else {
      sampleColor = sampleMissRadiance(direction, lighting);
      if (firstFrameSample) {
        firstDepth = kBackgroundDepth;
        firstLinearDepth = -1.0f;
        firstNormal = float3(0.0f, 0.0f, 0.0f);
        firstObjectId = 0u;
      }
    }

    colorSum += sampleColor;
    alphaSum += sampleAlpha;
    if (writeDataBuffer && firstFrameSample) {
      depthBuffer[pixelIndex] = firstDepth;
      linearDepthBuffer[pixelIndex] = firstLinearDepth;
      normalBuffer[pixelIndex] = length_squared(firstNormal) > 0.0f ? float4(normalize(firstNormal), 0.0f) : float4(0.0f);
      objectIdBuffer[pixelIndex] = firstObjectId;

      if (surf.hit != 0u) {
        float3 diffAlbedo = surf.baseColor * (1.0f - surf.metallic);
        float3 specAlbedo = mix(float3(surf.dielectricF0), surf.baseColor, surf.metallic);
        diffuseAlbedoBuffer[pixelIndex] = float4(diffAlbedo, 1.0f);
        specularAlbedoBuffer[pixelIndex] = float4(specAlbedo, 1.0f);
        roughnessBuffer[pixelIndex] = surf.roughness;

        float4 worldPos4 = float4(surf.hitPos, 1.0f);
        float4 prevClip = frame.prevViewProj * worldPos4;
        float2 prevUV = prevClip.xy / prevClip.w * 0.5f + 0.5f;
        prevUV.y = 1.0f - prevUV.y;
        float2 currUV = (float2(gid) + 0.5f + frame.jitterOffset) / float2(frame.width, frame.height);
        float2 motion = prevUV - currUV;
        motionVectorBuffer[pixelIndex] = float4(motion, 0.0f, 0.0f);
      } else {
        diffuseAlbedoBuffer[pixelIndex] = float4(0.0f);
        specularAlbedoBuffer[pixelIndex] = float4(0.0f);
        roughnessBuffer[pixelIndex] = 0.0f;
        motionVectorBuffer[pixelIndex] = float4(0.0f);
      }
    }
  }

  float3 iterationColor = colorSum / max(1u, frame.samplesPerIteration);
  float iterationAlpha = alphaSum / max(1u, frame.samplesPerIteration);
  if (frame.frameIndex == 0u) {
    accumulation[pixelIndex] = float4(iterationColor, iterationAlpha);
  } else {
    float a = 1.0f / float(frame.frameIndex + 1u);
    float4 previous = accumulation[pixelIndex];
    accumulation[pixelIndex] = mix(previous, float4(iterationColor, iterationAlpha), a);
  }

  output[pixelIndex] = accumulation[pixelIndex];
}

[[kernel]] void tonemapKernel(device const float4* input [[buffer(0)]],
                              constant ToonUniforms& toon [[buffer(1)]],
                              device float4* output [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= toon.width || gid.y >= toon.height) return;
  uint index = pixelIndex(gid.x, gid.y, toon.width);
  float4 color = input[index];
  float3 mapped = toneMapUncharted(color.xyz * toon.exposure, toon.gamma);
  mapped = max(mapped, float3(0.0f));
  mapped = applySaturation(mapped, toon.saturation);
  output[index] = float4(mapped, color.w);
}

[[kernel]] void objectContourKernel(device const uint* objectIds [[buffer(0)]],
                                    constant ToonUniforms& toon [[buffer(1)]],
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

[[kernel]] void detailContourKernel(device const float* linearDepth [[buffer(0)]],
                                    device const float4* normals [[buffer(1)]],
                                    device const uint* objectIds [[buffer(2)]],
                                    constant ToonUniforms& toon [[buffer(3)]],
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
  if (!(zFar > zNear)) {
    output[pixelIndex(gid.x, gid.y, toon.width)] = float4(0.0f);
    return;
  }

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
  float depthEdge = smoothstep(0.03f, 0.1f, (l + g) * toon.depthThreshold);

  if (toon.enableNormalEdge == 0u) normalEdge = 0.0f;
  if (toon.enableDepthEdge == 0u) depthEdge = 0.0f;
  float edge = clamp(normalEdge + depthEdge, 0.0f, 1.0f);
  output[pixelIndex(gid.x, gid.y, toon.width)] = float4(edge, edge, edge, 1.0f);
}

[[kernel]] void fxaaKernel(device const float4* input [[buffer(0)]],
                           constant ToonUniforms& toon [[buffer(1)]],
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
                                constant ToonUniforms& toon [[buffer(3)]],
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

[[kernel]] void bufferToTextureKernel(device const float4* input [[buffer(0)]],
                                      texture2d<half, access::write> output [[texture(0)]],
                                      uint2 gid [[thread_position_in_grid]]) {
  uint w = output.get_width();
  uint h = output.get_height();
  if (gid.x >= w || gid.y >= h) return;
  output.write(half4(input[gid.y * w + gid.x]), gid);
}

[[kernel]] void textureToBufferKernel(texture2d<half, access::read> input [[texture(0)]],
                                      device float4* output [[buffer(0)]],
                                      uint2 gid [[thread_position_in_grid]]) {
  uint w = input.get_width();
  uint h = input.get_height();
  if (gid.x >= w || gid.y >= h) return;
  output[gid.y * w + gid.x] = float4(input.read(gid));
}

[[kernel]] void depthToTextureKernel(device const float* input [[buffer(0)]],
                                     texture2d<float, access::write> output [[texture(0)]],
                                     uint2 gid [[thread_position_in_grid]]) {
  uint w = output.get_width();
  uint h = output.get_height();
  if (gid.x >= w || gid.y >= h) return;
  output.write(float4(input[gid.y * w + gid.x], 0, 0, 0), gid);
}

[[kernel]] void roughnessToTextureKernel(device const float* input [[buffer(0)]],
                                          texture2d<half, access::write> output [[texture(0)]],
                                          uint2 gid [[thread_position_in_grid]]) {
  uint w = output.get_width();
  uint h = output.get_height();
  if (gid.x >= w || gid.y >= h) return;
  output.write(half4(half(input[gid.y * w + gid.x]), 0, 0, 0), gid);
}

[[kernel]] void motionToTextureKernel(device const float4* input [[buffer(0)]],
                                       texture2d<half, access::write> output [[texture(0)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  uint w = output.get_width();
  uint h = output.get_height();
  if (gid.x >= w || gid.y >= h) return;
  float4 v = input[gid.y * w + gid.x];
  output.write(half4(v.x, v.y, 0, 0), gid);
}
