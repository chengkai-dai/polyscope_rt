#pragma once

bool sampleSunLight(constant GPULighting& lighting, thread uint& rng,
                    thread float3& toLight, thread float3& radiance, thread float& pdf) {
  float intensity = lighting.mainLightColorIntensity.w;
  if (intensity <= 1e-5f) return false;

  float3 sunAxis = normalize(-lighting.mainLightDirection.xyz);
  float angularRadius = max(lighting.mainLightDirection.w, 0.0f);
  if (angularRadius <= 1e-4f) {
    toLight = sunAxis;
    radiance = lighting.mainLightColorIntensity.xyz * intensity;
    pdf = 1.0f;
    return true;
  }

  float cosThetaMax = cos(min(angularRadius, 1.55334306f));
  float solidAngle = max(2.0f * kPi * (1.0f - cosThetaMax), 1e-6f);
  float cosTheta = mix(cosThetaMax, 1.0f, rand01(rng));
  float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
  float phi = 2.0f * kPi * rand01(rng);

  float3 tangent;
  float3 bitangent;
  makeBasis(sunAxis, tangent, bitangent);
  toLight = normalize(tangent * (cos(phi) * sinTheta) +
                      bitangent * (sin(phi) * sinTheta) +
                      sunAxis * cosTheta);
  pdf = 1.0f / solidAngle;
  radiance = lighting.mainLightColorIntensity.xyz * (intensity / solidAngle);
  return true;
}

float evaluateSunLight(constant GPULighting& lighting, float3 hitPos, float3 geomNormal, float3 normal,
                       device const GPUTriangle* triangles, device const GPUMaterial* materials,
                       thread uint& rng,
                       intersector<curve_data, triangle_data, instancing> isector,
                       instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                       intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 toLight;
  float3 radiance;
  float pdf = 0.0f;
  if (!sampleSunLight(lighting, rng, toLight, radiance, pdf)) return 0.0f;

  float nDotL = max(dot(normal, toLight), 0.0f);
  if (nDotL <= 0.0f) return 0.0f;

  float visibility = shadowVisibility(hitPos, geomNormal, toLight, kRayMaxDistance,
                                      triangles, materials, rng, isector, scene, pointScene, ftable);
  if (visibility <= 0.0f || pdf <= 0.0f) return 0.0f;
  return dot(radiance, float3(0.2126f, 0.7152f, 0.0722f)) * nDotL * visibility / pdf;
}

bool sampleAreaLight(float3 hitPos, float3 geomNormal, float3 normal, constant GPULighting& lighting,
                     device const GPUTriangle* triangles, device const GPUMaterial* materials,
                     thread uint& rng, thread float3& toLight,
                     thread float3& radiance, thread float& pdf,
                     intersector<curve_data, triangle_data, instancing> isector,
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
  float lightCos = abs(dot(lightNormal, -toLight));
  if (nDotL <= 0.0f || lightCos <= 0.0f) return false;

  float visibility = shadowVisibility(hitPos, geomNormal, toLight, max(distance - 2e-2f, 1e-3f),
                                      triangles, materials, rng, isector, scene, pointScene, ftable);
  if (visibility <= 0.0f) return false;

  float area = 4.0f * length(cross(lighting.areaLightU.xyz, lighting.areaLightV.xyz));
  pdf = distance2 / max(lightCos * area, 1e-5f);
  if (pdf <= 0.0f) return false;
  radiance = lighting.areaLightEmission.xyz;
  return true;
}

bool intersectVisibleAreaLight(ray currentRay, constant GPULighting& lighting,
                               thread float3& radiance, thread float& pdf) {
  if (lighting.areaLightCenterEnabled.w < 0.5f) return false;

  float3 center = lighting.areaLightCenterEnabled.xyz;
  float3 U = lighting.areaLightU.xyz;
  float3 V = lighting.areaLightV.xyz;
  float3 lightNormal = normalize(-cross(U, V));
  float denom = dot(currentRay.direction, lightNormal);
  if (abs(denom) <= 1e-6f) return false;

  float t = dot(center - currentRay.origin, lightNormal) / denom;
  if (t <= max(currentRay.min_distance, 1e-5f) || t > currentRay.max_distance) return false;

  float3 hitPoint = currentRay.origin + currentRay.direction * t;
  float3 rel = hitPoint - center;
  float uLen2 = max(dot(U, U), 1e-6f);
  float vLen2 = max(dot(V, V), 1e-6f);
  float su = dot(rel, U) / uLen2;
  float sv = dot(rel, V) / vLen2;
  if (abs(su) > 1.0f || abs(sv) > 1.0f) return false;

  float lightCos = abs(dot(lightNormal, -currentRay.direction));
  float area = 4.0f * length(cross(U, V));
  if (lightCos <= 0.0f || area <= 1e-6f) return false;

  radiance = lighting.areaLightEmission.xyz;
  pdf = (t * t) / max(lightCos * area, 1e-5f);
  return pdf > 0.0f;
}

bool sampleEmissiveTriangleLight(float3 hitPos, float3 geomNormal, float3 normal,
                                 device const float4* positions, device const float2* texcoords,
                                 device const GPUTriangle* triangles, device const GPUMaterial* materials,
                                 device const GPUTexture* textures, device const float4* texturePixels,
                                 device const GPUEmissiveTriangle* emissiveTriangles, uint emissiveTriangleCount,
                                 uint excludeTriangleIndex, thread uint& rng,
                                 thread float3& toLight, thread float3& radiance, thread float& pdf,
                                 intersector<curve_data, triangle_data, instancing> isector,
                                 instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                                 intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  if (emissiveTriangleCount == 0u) return false;

  for (uint attempt = 0u; attempt < min(emissiveTriangleCount, 4u); ++attempt) {
    float uSelect = rand01(rng);
    uint lo = 0u;
    uint hi = emissiveTriangleCount - 1u;
    while (lo < hi) {
      uint mid = (lo + hi) >> 1u;
      if (uSelect <= emissiveTriangles[mid].params.z) hi = mid;
      else lo = mid + 1u;
    }

    GPUEmissiveTriangle emissive = emissiveTriangles[lo];
    uint triangleIndex = emissive.data.x;
    if (triangleIndex == excludeTriangleIndex || emissive.params.y <= 0.0f || emissive.params.x <= 1e-7f) continue;

    GPUTriangle tri = triangles[triangleIndex];
    GPUMaterial material = materials[tri.indicesMaterial.w];
    float3 p0 = positions[tri.indicesMaterial.x].xyz;
    float3 p1 = positions[tri.indicesMaterial.y].xyz;
    float3 p2 = positions[tri.indicesMaterial.z].xyz;
    float3 triCross = cross(p1 - p0, p2 - p0);
    float triCrossLen = length(triCross);
    if (triCrossLen <= 1e-7f) continue;

    float u = rand01(rng);
    float v = rand01(rng);
    float su = sqrt(u);
    float b0 = 1.0f - su;
    float b1 = v * su;
    float b2 = 1.0f - b0 - b1;

    float3 lightPoint = p0 * b0 + p1 * b1 + p2 * b2;
    float3 lightNormal = triCross / triCrossLen;

    float3 emission = material.emissiveFactor.xyz;
    if (material.emissiveTextureData.y != 0u) {
      float2 uv0 = texcoords[tri.indicesMaterial.x];
      float2 uv1 = texcoords[tri.indicesMaterial.y];
      float2 uv2 = texcoords[tri.indicesMaterial.z];
      float2 uv = uv0 * b0 + uv1 * b1 + uv2 * b2;
      emission *= sampleBaseColorTexture(textures, texturePixels, material.emissiveTextureData.x, uv).xyz;
    }
    if (all(emission <= 0.0f)) continue;

    toLight = lightPoint - hitPos;
    float distance2 = max(dot(toLight, toLight), 1e-5f);
    float distance = sqrt(distance2);
    if (distance <= 2e-3f) continue;
    toLight /= distance;

    float nDotL = max(dot(normal, toLight), 0.0f);
    bool lightDoubleSided = material.materialFlags.y != 0u;
    float lightCos = lightDoubleSided ? abs(dot(lightNormal, -toLight))
                                      : max(dot(lightNormal, -toLight), 0.0f);
    if (nDotL <= 0.0f || lightCos <= 0.0f) continue;

    float visibility = shadowVisibility(hitPos, geomNormal, toLight, max(distance - 2e-2f, 1e-3f),
                                        triangles, materials, rng, isector, scene, pointScene, ftable);
    if (visibility <= 0.0f) continue;

    pdf = emissive.params.y * distance2 / max(lightCos * emissive.params.x, 1e-5f);
    if (pdf <= 0.0f) continue;

    radiance = emission;
    return true;
  }
  return false;
}

bool sampleEnvironmentLight(float3 hitPos, float3 geomNormal, float3 normal,
                            device const GPUTriangle* triangles, device const GPUMaterial* materials,
                            device const GPUEnvironmentSampleCell* environmentCells,
                            constant GPULighting& lighting, thread uint& rng,
                            thread float3& toLight, thread float3& radiance, thread float& pdf,
                            intersector<curve_data, triangle_data, instancing> isector,
                            instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                            intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float uSelect = rand01(rng);
  uint lo = 0u;
  uint hi = kEnvironmentSampleWidth * kEnvironmentSampleHeight - 1u;
  while (lo < hi) {
    uint mid = (lo + hi) >> 1u;
    if (uSelect <= environmentCells[mid].data.y) hi = mid;
    else lo = mid + 1u;
  }

  uint idx = lo;
  float pmf = environmentCells[idx].data.x;
  if (pmf <= 0.0f) return false;

  uint row = idx / kEnvironmentSampleWidth;
  uint col = idx % kEnvironmentSampleWidth;
  float phi0 = 2.0f * kPi * float(col) / float(kEnvironmentSampleWidth);
  float phi1 = 2.0f * kPi * float(col + 1u) / float(kEnvironmentSampleWidth);
  float theta0 = kPi * float(row) / float(kEnvironmentSampleHeight);
  float theta1 = kPi * float(row + 1u) / float(kEnvironmentSampleHeight);

  float cosTheta0 = cos(theta0);
  float cosTheta1 = cos(theta1);
  float cosTheta = mix(cosTheta0, cosTheta1, rand01(rng));
  float sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
  float phi = mix(phi0, phi1, rand01(rng));

  toLight = float3(cos(phi) * sinTheta, cosTheta, sin(phi) * sinTheta);
  float nDotL = max(dot(normal, toLight), 0.0f);
  if (nDotL <= 0.0f) return false;

  float visibility = shadowVisibility(hitPos, geomNormal, toLight, 1e6f, triangles, materials, rng, isector, scene, pointScene, ftable);
  if (visibility <= 0.0f) return false;

  radiance = evaluateEnvironmentRadiance(toLight, lighting);
  if (all(radiance <= 0.0f)) return false;

  pdf = pmf / environmentBinSolidAngle(row);
  return pdf > 0.0f;
}

float shadowVisibility(float3 hitPos, float3 geomNormal, float3 toLight, float maxDistance,
                       device const GPUTriangle* triangles, device const GPUMaterial* materials,
                       thread uint& rng, intersector<curve_data, triangle_data, instancing> isector,
                       instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                       intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  ray shadowRay;
  shadowRay.origin = offsetRayOrigin(hitPos, geomNormal, toLight);
  shadowRay.direction = toLight;
  shadowRay.min_distance = 0.0f;
  shadowRay.max_distance = maxDistance;

  for (uint i = 0u; i < 8u; ++i) {
    auto mcHit = isector.intersect(shadowRay, scene, 0xFFu);
    auto ptHit = isector.intersect(shadowRay, pointScene, 0xFFu, ftable);
    if (mcHit.type == intersection_type::none && ptHit.type == intersection_type::none) return 1.0f;

    bool ptCloser = ptHit.type != intersection_type::none &&
                    (mcHit.type == intersection_type::none || ptHit.distance < mcHit.distance);
    if (ptCloser) return 0.0f;

    if (mcHit.type == intersection_type::curve) return 0.0f;

    uint materialIndex = triangles[mcHit.primitive_id].indicesMaterial.w;
    float opacity = clamp(materials[materialIndex].transmissionIor.w, 0.0f, 1.0f);
    float transmission = clamp(materials[materialIndex].transmissionIor.x, 0.0f, 1.0f);

    if (opacity < 1e-5f && transmission < 1e-5f) {
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

float evaluatePunctualLight(device const GPUPunctualLight* lights, uint lightIndex, float3 hitPos, float3 geomNormal, float3 normal,
                            device const GPUTriangle* triangles, device const GPUMaterial* materials,
                            thread uint& rng,
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

  float visibility = shadowVisibility(hitPos, geomNormal, toLight, maxDistance, triangles, materials, rng, isector, scene, pointScene, ftable);
  if (visibility <= 0.0f) return 0.0f;

  float luminance = dot(lightColor, float3(0.2126f, 0.7152f, 0.0722f));
  return luminance * attenuation * nDotL * visibility;
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
