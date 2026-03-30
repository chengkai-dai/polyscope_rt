#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;
#include "gpu_shared_types.h"
#include "shader_common.h"

// Forward declaration so evaluateDirectionalLight can call shadowVisibility.
float shadowVisibility(float3 hitPos, float3 toLight, float maxDistance,
                       device const GPUTriangle* triangles, device const GPUMaterial* materials,
                       thread uint& rng, intersector<curve_data, triangle_data, instancing> isector,
                       instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                       intersection_function_table<curve_data, triangle_data, instancing> ftable);

float evaluateDirectionalLight(float3 lightDir, float lightIntensity, float3 hitPos, float3 normal,
                               device const GPUTriangle* triangles, device const GPUMaterial* materials,
                               thread uint& rng,
                               intersector<curve_data, triangle_data, instancing> isector,
                               instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                               intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 toLight = normalize(-lightDir);
  float nDotL = max(dot(normal, toLight), 0.0f);
  if (nDotL <= 0.0f) return 0.0f;

  float visibility = shadowVisibility(hitPos, toLight, 1e6f, triangles, materials, rng, isector, scene, pointScene, ftable);
  return nDotL * lightIntensity * visibility;
}

bool sampleAreaLight(float3 hitPos, float3 normal, constant GPULighting& lighting,
                     device const GPUTriangle* triangles, device const GPUMaterial* materials,
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

float3 evalSimpleSky(float3 dir, constant GPULighting& lighting) {
  float3 sunDir = normalize(-lighting.mainLightDirection.xyz);
  float sunIntensity = lighting.mainLightColorIntensity.w;
  float3 sunColor = lighting.mainLightColorIntensity.xyz;
  float3 bgColor = lighting.backgroundColor.xyz;

  float y = dir.y;

  // Sky gradient is anchored to backgroundColor: horizon matches it, zenith is a relative shift.
  float3 horizonColor = bgColor;
  float3 zenithColor = mix(bgColor, bgColor * float3(0.35f, 0.55f, 1.05f), 0.75f);
  float3 groundColor = bgColor * float3(0.08f, 0.08f, 0.09f);

  float3 sky;
  if (y > 0.0f) {
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

float3 sampleEnvironment(float3 dir, constant GPULighting& lighting) {
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
                      float2 uv, device const GPUTexture* textures, device const float4* texturePixels, uint4 normalTextureData,
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
                       device const GPUTriangle* triangles, device const GPUMaterial* materials,
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

    bool ptCloser = ptHit.type != intersection_type::none &&
                    (mcHit.type == intersection_type::none || ptHit.distance < mcHit.distance);
    if (ptCloser) return 0.0f;

    if (mcHit.type == intersection_type::curve) return 0.0f;

    uint objectId = triangles[mcHit.primitive_id].objectFlags.x;
    float opacity = clamp(materials[objectId].transmissionIor.w, 0.0f, 1.0f);
    float transmission = clamp(materials[objectId].transmissionIor.x, 0.0f, 1.0f);

    if (opacity < 1e-5f && transmission < 1e-5f) {
      // Fully invisible surface: advance shadow ray through without blocking.
      float traveled = mcHit.distance + 2e-3f;
      shadowRay.origin = shadowRay.origin + toLight * traveled;
      shadowRay.min_distance = 1e-3f;
      shadowRay.max_distance = maxDistance - traveled;
      if (shadowRay.max_distance <= 0.0f) return 1.0f;
      continue;
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

float evaluatePunctualLight(device const GPUPunctualLight* lights, uint lightIndex, float3 hitPos, float3 normal,
                            intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                            intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  GPUPunctualLight light = lights[lightIndex];
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

bool samplePunctualLight(device const GPUPunctualLight* lights, uint lightIndex, float3 hitPos, thread float3& toLight,
                         thread float3& radiance, thread float& maxDistance) {
  GPUPunctualLight light = lights[lightIndex];
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
  // Isoline data used for contour post-processing (style 2).
  // Populated by intersectSurface when a contour-style iso material is hit.
  float4 isoParams = float4(0.0f);  // x=style, y=period, z=darkness, w=thickness
  float  isoScalarVal = 0.0f;
  float3 isoGradWorld = float3(0.0f);
};

SurfaceHitInfo intersectGroundPlane(ray r, constant GPUFrameUniforms& frame) {
  SurfaceHitInfo out;
  if (frame.planeColorEnabled.w < 0.5f) return out;

  float3 normal = float3(0, 1, 0);
  float planeHeight = frame.planeParams.x;

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
                         device const GPUTriangle* triangles,
                         device const GPUMaterial* materials, device const GPUTexture* textures,
                         device const float4* texturePixels, device const GPUPunctualLight* sceneLights,
                         device const GPUCurvePrimitive* curvePrimitives,
                         device const GPUPointPrimitive* pointPrimitives,
                         constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
                         intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                         intersection_function_table<curve_data, triangle_data, instancing> ftable,
                         device const float* isoScalars, float tanHalfFov);

float3 sampleMissRadiance(float3 dir, constant GPULighting& lighting) {
  return evalSimpleSky(dir, lighting);
}

SurfaceHitInfo shadeCurveHit(ray currentRay, float hitDistance, float curveParam, uint segmentId,
                             device const GPUCurvePrimitive* curvePrimitives,
                             device const GPUMaterial* materials) {
  SurfaceHitInfo out;
  GPUCurvePrimitive prim = curvePrimitives[segmentId];
  float3 hitPos = currentRay.origin + currentRay.direction * hitDistance;
  float3 p0 = prim.p0_radius.xyz;
  float3 p1 = prim.p1_type.xyz;
  uint matIdx = prim.materialObjectId.x;
  uint objId = prim.materialObjectId.y;

  // Evaluate the Catmull-Rom curve axis position at the hit parameter t.
  // Control points: [p_prev, p0, p1, p_next]  (curve runs p0→p1 for t∈[0,1]).
  float t  = clamp(curveParam, 0.0f, 1.0f);
  float t2 = t * t, t3 = t2 * t;
  float3 pm1 = prim.p_prev.xyz;
  float3 p2  = prim.p_next.xyz;
  float3 axisPoint = 0.5f * ((2.0f * p0)
                            + (-pm1 + p1)                          * t
                            + (2.0f*pm1 - 5.0f*p0 + 4.0f*p1 - p2) * t2
                            + (-pm1 + 3.0f*p0 - 3.0f*p1 + p2)     * t3);

  float3 normal = normalize(hitPos - axisPoint);
  if (dot(normal, -currentRay.direction) < 0.0f) {
    normal = -normal;
  }

  GPUMaterial material = materials[matIdx];

  out.hit = 1u;
  out.objectId = objId;
  out.distance = hitDistance;
  out.hitPos = hitPos;
  out.geomNormal = normal;
  out.normal = normal;
  // Use per-primitive color gradient if set (baseColor.w == 1):
  // linearly interpolate from baseColor (tail, p0) to baseColor1 (tip, p1) along t.
  float3 baseCol = (prim.baseColor.w > 0.5f)
      ? mix(prim.baseColor.xyz, prim.baseColor1.xyz, t)
      : material.baseColorFactor.xyz;
  out.baseColor = clamp(baseCol, 0.0f, 1.0f);
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

SurfaceHitInfo shadePointHit(ray currentRay, float hitDistance, uint primitiveIndex,
                              device const GPUPointPrimitive* points,
                              device const GPUMaterial* materials) {
  SurfaceHitInfo out;
  GPUPointPrimitive pt = points[primitiveIndex];
  float3 center  = pt.center_radius.xyz;
  float3 hitPos  = currentRay.origin + currentRay.direction * hitDistance;
  float3 normal  = normalize(hitPos - center);
  if (dot(normal, -currentRay.direction) < 0.0f) normal = -normal;

  uint matIdx = pt.materialObjectId.x;
  uint objId  = pt.materialObjectId.y;
  GPUMaterial material = materials[matIdx];

  out.hit        = 1u;
  out.objectId   = objId;
  out.distance   = hitDistance;
  out.hitPos     = hitPos;
  out.geomNormal = normal;
  out.normal     = normal;
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

// Apply isoline contour lines (mirrors Polyscope CONTOUR_VALUECOLOR).
// Must be called after intersectSurface when surf.isoParams.x is in [1.5, 2.5].
// ray_dir    — direction of the ray that produced this hit.
// tan_hfov   — tan(0.5 * vertical_fov_radians).
// img_height — render target height in pixels.
inline void applyContour(thread SurfaceHitInfo& surf, float3 ray_dir,
                         float tan_hfov, uint img_height) {
  // Only active for contour style (isoParams.x ≈ 2).
  if (surf.isoParams.x < 1.5f) return;

  float period    = surf.isoParams.y;
  float darkness  = surf.isoParams.z;
  float thickness = surf.isoParams.w;
  if (period < 1e-8f || thickness < 1e-8f) return;

  // World-space size of one pixel at the hit distance (vertical direction).
  // pixel_world_size = [world_unit / pixel]
  float pixel_world_size = 2.0f * surf.distance * tan_hfov / float(img_height);
  if (pixel_world_size < 1e-12f) return;

  // Project the world-space gradient onto the plane perpendicular to the ray.
  // grad_perp is in [scalar_value / world_unit].
  float3 grad_perp = surf.isoGradWorld - dot(surf.isoGradWorld, ray_dir) * ray_dir;

  // Screen-space gradient magnitude: [scalar_value/world_unit] * [world_unit/pixel]
  // = [scalar_value / pixel]  — same units as GLSL dFdx(shadeValue).
  float grad_screen = length(grad_perp) * pixel_world_size;
  if (grad_screen < 1e-12f) return;

  // Polyscope CONTOUR_VALUECOLOR formula:
  //   w  = 1 / (10 / period * thickness * length(gradF))
  //   s  = darkness * exp( -pow( w*(fract(|val/period|)-0.5), 8 ) )
  //   color *= 1 - s
  // Use abs() on the base to avoid undefined Metal pow() behaviour for negative x.
  float w     = 1.0f / (10.0f / period * thickness * grad_screen);
  float phase = fract(abs(surf.isoScalarVal / period)) - 0.5f;
  float s     = darkness * exp(-pow(abs(w * phase), 8.0f));
  surf.baseColor *= (1.0f - s);
}

SurfaceHitInfo intersectSurface(ray currentRay, device const float4* positions, device const float4* normals,
                                device const float2* texcoords, device const float4* vertexColors,
                                device const GPUTriangle* triangles,
                                device const GPUMaterial* materials, device const GPUTexture* textures,
                                device const float4* texturePixels,
                                device const GPUCurvePrimitive* curvePrimitives,
                                device const GPUPointPrimitive* pointPrimitives,
                                intersector<curve_data, triangle_data, instancing> isector,
                                instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                                intersection_function_table<curve_data, triangle_data, instancing> ftable,
                                device const float* isoScalars) {
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
  GPUTriangle tri = triangles[triangleIndex];
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

  GPUMaterial material = materials[tri.indicesMaterial.w];
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

  // Isoline effect — mirrors Polyscope's stripe/contour rendering.
  // isoParams.x: 0=off, 1=stripe (ISOLINE_STRIPE_VALUECOLOR), 2=contour (CONTOUR_VALUECOLOR).
  if (material.isoParams.x > 0.5f) {
    float s0 = isoScalars[tri.indicesMaterial.x];
    float s1 = isoScalars[tri.indicesMaterial.y];
    float s2 = isoScalars[tri.indicesMaterial.z];
    float scalarVal = s0 * w0 + s1 * w1 + s2 * w2;
    float period    = material.isoParams.y;
    float darkness  = material.isoParams.z;

    // Store for caller-side contour post-processing.
    out.isoParams     = material.isoParams;
    out.isoScalarVal  = scalarVal;

    uint style = (uint)round(material.isoParams.x);  // 1=stripe, 2=contour
    if (style == 1u && period > 1e-8f) {
      // Stripe: periodic dark bands, applied immediately.
      float modVal = fmod(scalarVal, 2.0f * period);
      if (modVal < 0.0f) modVal += 2.0f * period;
      if (modVal > period) {
        out.baseColor *= darkness;
      }
    } else if (style == 2u) {
      // Contour: compute world-space gradient; caller applies screen-space formula.
      float3 N  = cross(p1 - p0, p2 - p0);  // unnormalized normal, |N|² = (2A)²
      float A2sq = dot(N, N);
      if (A2sq > 1e-20f) {
        out.isoGradWorld = (s0 * cross(p2 - p1, N) +
                            s1 * cross(p0 - p2, N) +
                            s2 * cross(p1 - p0, N)) / A2sq;
      }
    }
  }

  return out;
}

float3 evaluateDirectLightingPBR(SurfaceHitInfo surf, float3 viewDir, device const GPUPunctualLight* sceneLights,
                                 device const GPUTriangle* triangles, device const GPUMaterial* materials,
                                 constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
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
  const bool isInfinitePlane = surf.isInfinitePlane;
  float3 diffuseAlbedo = albedo * (1.0f - metallic);

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

  // Environment ambient (IBL approximation).
  // Metal surfaces have no diffuse term, so without this they stay black everywhere
  // the GGX specular lobe doesn't directly face a light.  The path-bounce already
  // picks up the sky stochastically; this term ensures the first convergence isn't
  // pitch-black.  Scale by 0.3 so near-white metals (Silver, Mirror) don't blow out.
  float3 envAmbient   = sampleEnvironment(N, lighting);
  float3 ambientSpec  = F0 * envAmbient * 0.3f;
  float3 kDa          = (float3(1.0f) - F0) * (1.0f - metallic);
  float3 ambientDiff  = kDa * albedo * envAmbient * (0.3f / 3.14159265f);
  directLighting += ambientSpec + ambientDiff;

  return directLighting;
}

float3 shadeStandardTransmissionSurface(SurfaceHitInfo surf, float3 viewDir, device const GPUPunctualLight* sceneLights,
                                        device const GPUTriangle* triangles, device const GPUMaterial* materials,
                                        constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
                                        intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                                        intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 N = surf.normal;
  float3 V = viewDir;
  float roughness = clamp(surf.roughness, 0.01f, 1.0f);
  float f0Scalar = pow((surf.ior - 1.0f) / (surf.ior + 1.0f), 2.0f);
  float3 F0 = float3(f0Scalar);
  float3 specularLighting = float3(0.0f);

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
                         device const GPUTriangle* triangles,
                         device const GPUMaterial* materials, device const GPUTexture* textures,
                         device const float4* texturePixels, device const GPUPunctualLight* sceneLights,
                         device const GPUCurvePrimitive* curvePrimitives,
                         device const GPUPointPrimitive* pointPrimitives,
                         constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
                         intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                         intersection_function_table<curve_data, triangle_data, instancing> ftable,
                         device const float* isoScalars, float tanHalfFov) {
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
                              curvePrimitives, pointPrimitives, isector, scene, pointScene, ftable, isoScalars);
      applyContour(surf, currentRay.direction, tanHalfFov, frame.height);
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
                                    frame, lighting, rng, isector, scene, pointScene, ftable, isoScalars, tanHalfFov) *
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

    float3 directLighting = evaluateDirectLightingPBR(surf, currentViewDir, sceneLights, triangles, materials, frame, lighting, rng, isector, scene, pointScene, ftable);
    float3 F0 = mix(float3(surf.dielectricF0), surf.baseColor, surf.metallic);
    float grazingMax = mix(clamp(surf.dielectricF0 * 25.0f, 0.0f, 1.0f), 1.0f, surf.metallic);
    float NdotV = max(dot(surf.normal, currentViewDir), 0.0f);
    float3 F = fresnelSchlick(NdotV, F0, grazingMax);
    radiance += throughput * (directLighting + surf.emissive);

    float specProb = clamp((F.x + F.y + F.z) / 3.0f + surf.metallic * 0.5f, 0.04f, 0.96f);

    if (rand01(rng) < specProb) {
      float3 bounceDir = sampleGGXBounce(surf.normal, currentViewDir, surf.roughness, rng);
      if (dot(bounceDir, surf.normal) <= 0.0f) {
        bounceDir = reflect(-currentViewDir, surf.normal);
      }
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

float3 traceSpecularPath(ray currentRay, uint remainingBounces, device const float4* positions, device const float4* normals,
                         device const float2* texcoords, device const float4* vertexColors,
                         device const GPUTriangle* triangles,
                         device const GPUMaterial* materials, device const GPUTexture* textures,
                         device const float4* texturePixels, device const GPUPunctualLight* sceneLights,
                         device const GPUCurvePrimitive* curvePrimitives,
                         device const GPUPointPrimitive* pointPrimitives,
                         constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
                         intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                         intersection_function_table<curve_data, triangle_data, instancing> ftable,
                         device const float* isoScalars, float tanHalfFov) {
  float3 throughput = float3(1.0f);
  float3 radiance = float3(0.0f);
  float3 currentViewDir = -currentRay.direction;
  bool prevWasInfinitePlane = false;

  for (uint bounce = 0u; bounce < max(remainingBounces, 1u); ++bounce) {
    SurfaceHitInfo surf = intersectSurface(currentRay, positions, normals, texcoords, vertexColors, triangles, materials, textures, texturePixels,
                                           curvePrimitives, pointPrimitives, isector, scene, pointScene, ftable, isoScalars);
    applyContour(surf, currentRay.direction, tanHalfFov, frame.height);
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
                                device const GPUTriangle* triangles [[buffer(3)]],
                                device const GPUMaterial* materials [[buffer(4)]],
                                device const GPUTexture* textures [[buffer(5)]],
                                device const float4* texturePixels [[buffer(6)]],
                                device const GPUPunctualLight* sceneLights [[buffer(7)]],
                                constant GPUCamera& camera [[buffer(8)]],
                                constant GPUFrameUniforms& frame [[buffer(9)]],
                                constant GPULighting& lighting [[buffer(10)]],
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
                                device const GPUCurvePrimitive* curvePrimitives [[buffer(23)]],
                                intersection_function_table<curve_data, triangle_data, instancing> ftable [[buffer(24)]],
                                device const GPUPointPrimitive* pointPrimitives [[buffer(25)]],
                                device const float* isoScalars [[buffer(27)]],
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

    // Pre-compute tan(fov/2) for contour screen-space gradient approximation.
    float tanHalfFov = tan(0.5f * camera.clipData.x);

    SurfaceHitInfo surf = intersectSurface(currentRay, positions, normals, texcoords, vertexColors, triangles, materials, textures, texturePixels,
                                           curvePrimitives, pointPrimitives, isector, scene, pointScene, ftable, isoScalars);
    applyContour(surf, currentRay.direction, tanHalfFov, frame.height);
    SurfaceHitInfo planeSurf = intersectGroundPlane(currentRay, frame);
    if (planeSurf.hit != 0u && (surf.hit == 0u || planeSurf.distance < surf.distance)) {
      surf = planeSurf;
    }
    if (surf.hit != 0u) {
      if (frame.renderMode == 0u) {
        sampleColor =
            traceStandardPath(currentRay, normalize(camera.position.xyz - surf.hitPos), surf, positions, normals, texcoords, vertexColors,
                              triangles, materials, textures, texturePixels, sceneLights, curvePrimitives, pointPrimitives,
                              frame, lighting, rng, isector, scene, pointScene, ftable, isoScalars, tanHalfFov);
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
                                              surf.normal, triangles, materials, rng, isector, scene, pointScene, ftable);
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
