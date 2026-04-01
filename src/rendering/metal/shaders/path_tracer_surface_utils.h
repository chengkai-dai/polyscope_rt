#pragma once

struct BsdfEvalResult {
  float3 value = float3(0.0f);
  float3 diffuse = float3(0.0f);
  float3 specular = float3(0.0f);
  float3 fresnel = float3(0.0f);
  float diffusePdf = 0.0f;
  float specularPdf = 0.0f;
  float pdf = 0.0f;
  float NdotL = 0.0f;
};

struct BsdfSampleResult {
  bool valid = false;
  bool skipLightHitMIS = false;
  float3 direction = float3(0.0f);
  float3 throughput = float3(0.0f);
  float pdf = 0.0f;
};

struct OpaqueBsdfSamplingWeights {
  float diffuseWeight = 0.0f;
  float specularWeight = 0.0f;
  float diffuseProb = 0.0f;
  float specularProb = 1.0f;
};

struct PathLightState {
  float bsdfPdf = 1.0f;
  bool skipLightHitMIS = true;
};

struct VisibleLightHit {
  float3 radiance = float3(0.0f);
  float lightPdf = 0.0f;
};

float3 orientedGeomNormal(float3 geomNormal, float3 rayDir) {
  float3 ng = normalize(geomNormal);
  return dot(ng, rayDir) >= 0.0f ? ng : -ng;
}

float3 offsetRayOrigin(float3 hitPos, float3 geomNormal, float3 rayDir) {
  float3 dir = normalize(rayDir);
  float3 offsetN = orientedGeomNormal(geomNormal, dir);
  return hitPos + offsetN * 2e-3f + dir * 5e-4f;
}

void initializePathRay(thread ray& pathRay, float3 origin, float3 direction) {
  pathRay.origin = origin;
  pathRay.direction = normalize(direction);
  pathRay.min_distance = kRayMinDistance;
  pathRay.max_distance = kRayMaxDistance;
}

void spawnPathRay(thread ray& pathRay, float3 hitPos, float3 geomNormal, float3 direction) {
  initializePathRay(pathRay, offsetRayOrigin(hitPos, geomNormal, direction), direction);
}

PathLightState deltaPathLightState() {
  return PathLightState();
}

PathLightState makePathLightState(float bsdfPdf, bool skipLightHitMIS) {
  PathLightState state;
  state.bsdfPdf = bsdfPdf;
  state.skipLightHitMIS = skipLightHitMIS;
  return state;
}

void makeBasis(float3 axis, thread float3& tangent, thread float3& bitangent) {
  float3 up = abs(axis.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
  tangent = normalize(cross(up, axis));
  bitangent = cross(axis, tangent);
}

float3 faceForwardToRay(float3 normal, float3 rayDir) {
  return dot(normal, rayDir) < 0.0f ? normal : -normal;
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
