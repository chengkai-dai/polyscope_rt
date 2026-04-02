#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;
#include "environment_model_shared.h"
#include "gpu_shared_types.h"
#include "shader_common.h"

// Forward declaration so lighting helpers can query shadow visibility.
float shadowVisibility(float3 hitPos, float3 geomNormal, float3 toLight, float maxDistance,
                       device const GPUTriangle* triangles, device const GPUMaterial* materials,
                       thread uint& rng, intersector<curve_data, triangle_data, instancing> isector,
                       instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                       intersection_function_table<curve_data, triangle_data, instancing> ftable);

constant float kPi = 3.14159265f;
constant uint kInvalidTriangleIndex = 0xFFFFFFFFu;
constant uint kInvalidLightIndex = 0xFFFFFFFFu;
constant uint kEnvironmentSampleWidth = RT_ENVIRONMENT_SAMPLE_WIDTH;
constant uint kEnvironmentSampleHeight = RT_ENVIRONMENT_SAMPLE_HEIGHT;
constant float kRayMinDistance = 1e-3f;
constant float kRayMaxDistance = 1e6f;

#include "path_tracer_helpers.h"
struct SurfaceHitInfo {
  uint hit = 0u;
  uint objectId = 0u;
  uint triangleIndex = kInvalidTriangleIndex;
  uint emissiveLightIndex = kInvalidLightIndex;
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
  bool frontFacing = true;
  bool doubleSided = false;
  bool isInfinitePlane = false;
  // Isoline data used for contour post-processing (style 2).
  // Populated by intersectSurface when a contour-style iso material is hit.
  float4 isoParams = float4(0.0f);  // x=style, y=period, z=darkness, w=thickness
  float  isoScalarVal = 0.0f;
  float3 isoGradWorld = float3(0.0f);
};

float3 opaqueBsdfNormal(SurfaceHitInfo surf, float3 rayDir) {
  return surf.doubleSided ? faceForwardToRay(surf.normal, rayDir) : surf.normal;
}

SurfaceHitInfo intersectGroundPlane(ray r, constant GPUFrameUniforms& frame, constant GPULighting& lighting) {
  SurfaceHitInfo out;
  if (frame.planeColorEnabled.w < 0.5f) return out;

  float3 normal = normalize(lighting.sceneUpDir.xyz);
  float planeHeight = frame.planeParams.x;
  float3 axisMask = abs(normal);
  float originCoord = axisMask.x > 0.5f ? r.origin.x : (axisMask.y > 0.5f ? r.origin.y : r.origin.z);
  float dirCoord = axisMask.x > 0.5f ? r.direction.x : (axisMask.y > 0.5f ? r.direction.y : r.direction.z);
  float normalSign = axisMask.x > 0.5f ? normal.x : (axisMask.y > 0.5f ? normal.y : normal.z);

  if ((originCoord - planeHeight) * normalSign <= 0.0f) return out;

  if (fabs(dirCoord) < 1e-6f) return out;

  float t = (planeHeight - originCoord) / dirCoord;
  if (t < r.min_distance || t > r.max_distance) return out;

  out.hit = 1u;
  out.objectId = 0xFFFFFFFEu;
  out.distance = t;
  out.hitPos = r.origin + r.direction * t;
  out.geomNormal = normal;
  out.normal = normal;
  out.frontFacing = dot(normal, -r.direction) >= 0.0f;
  out.baseColor = frame.planeColorEnabled.xyz;
  out.metallic = frame.planeParams.y;
  out.roughness = clamp(frame.planeParams.z, 0.001f, 1.0f);
  out.dielectricF0 = frame.planeParams.w;
  out.isInfinitePlane = true;
  return out;
}

float3 traceSpecularPath(ray currentRay, uint remainingBounces, device const float4* positions, device const float4* normals,
                         device const float2* texcoords, device const float4* vertexColors,
                         device const GPUTriangle* triangles,
                         device const GPUMaterial* materials, device const GPUTexture* textures,
                         device const float4* texturePixels, device const GPUPunctualLight* sceneLights,
                         device const GPUEmissiveTriangle* emissiveTriangles,
                         device const GPUEnvironmentSampleCell* environmentCells,
                         device const GPUCurvePrimitive* curvePrimitives,
                         device const GPUPointPrimitive* pointPrimitives,
                         constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
                         intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                         intersection_function_table<curve_data, triangle_data, instancing> ftable,
                         device const float* isoScalars, float tanHalfFov);

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
  out.roughness = clamp(material.metallicRoughnessNormal.y, 0.001f, 1.0f);
  out.transmission = 0.0f;
  out.ior = 1.5f;
  out.opacity = 1.0f;
  out.unlit = material.materialFlags.x != 0u;
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
  out.roughness  = clamp(material.metallicRoughnessNormal.y, 0.001f, 1.0f);
  out.transmission = 0.0f;
  out.ior        = 1.5f;
  out.opacity    = 1.0f;
  out.unlit      = material.materialFlags.x != 0u;
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
  float w1 = bary.x;
  float w2 = bary.y;
  float w0 = 1.0f - w1 - w2;

  float3 geomNormal = normalize(cross(p1 - p0, p2 - p0));
  float3 normal = geomNormal;
  if (tri.objectFlags.y != 0u) {
    float3 n0 = normals[tri.indicesMaterial.x].xyz;
    float3 n1 = normals[tri.indicesMaterial.y].xyz;
    float3 n2 = normals[tri.indicesMaterial.z].xyz;
    normal = normalize(n0 * w0 + n1 * w1 + n2 * w2);
  }

  float3 vc0 = vertexColors[tri.indicesMaterial.x].xyz;
  float3 vc1 = vertexColors[tri.indicesMaterial.y].xyz;
  float3 vc2 = vertexColors[tri.indicesMaterial.z].xyz;
  float3 interpVertexColor = vc0 * w0 + vc1 * w1 + vc2 * w2;

  GPUMaterial material = materials[tri.indicesMaterial.w];
  bool doubleSided = material.materialFlags.y != 0u;
  bool frontFacing = dot(geomNormal, -currentRay.direction) >= 0.0f;
  if (doubleSided) {
    geomNormal = faceForwardToRay(geomNormal, currentRay.direction);
    normal = faceForwardToRay(normal, currentRay.direction);
  } else if (dot(normal, geomNormal) <= 0.0f) {
    normal = -normal;
  }
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
  roughness = clamp(roughness, 0.001f, 1.0f);
  metallic = saturate(metallic);
  normal = applyNormalMap(geomNormal, normal, p0, p1, p2, uv0, uv1, uv2, uv, textures, texturePixels,
                          material.normalTextureData, normalScale);

  out.hit = 1u;
  out.objectId = tri.objectFlags.x;
  out.triangleIndex = triangleIndex;
  out.emissiveLightIndex = tri.objectFlags.w != 0u ? (tri.objectFlags.w - 1u) : kInvalidLightIndex;
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
  out.unlit = material.materialFlags.x != 0u;
  out.frontFacing = frontFacing;
  out.doubleSided = doubleSided;
  if (!doubleSided && !frontFacing) {
    out.emissive = float3(0.0f);
  }
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

float emissiveSurfaceLightPdf(SurfaceHitInfo surf, float3 incomingDir,
                              device const GPUEmissiveTriangle* emissiveTriangles) {
  if (surf.emissiveLightIndex == kInvalidLightIndex) return 0.0f;

  GPUEmissiveTriangle emissive = emissiveTriangles[surf.emissiveLightIndex];
  float area = emissive.params.x;
  float selectionPdf = emissive.params.y;
  float lightCos = surf.doubleSided ? abs(dot(surf.geomNormal, -incomingDir))
                                    : max(dot(surf.geomNormal, -incomingDir), 0.0f);
  if (area <= 1e-7f || selectionPdf <= 0.0f || lightCos <= 0.0f) return 0.0f;
  return selectionPdf * max(surf.distance * surf.distance, 1e-6f) / max(lightCos * area, 1e-5f);
}

float lightHitMISWeight(PathLightState state, float lightPdf) {
  if (state.skipLightHitMIS || lightPdf <= 0.0f) return 1.0f;
  return powerHeuristic(state.bsdfPdf, lightPdf);
}

float emissiveSurfaceMISWeight(SurfaceHitInfo surf, float3 incomingDir, PathLightState state,
                               device const GPUEmissiveTriangle* emissiveTriangles) {
  if (surf.emissiveLightIndex == kInvalidLightIndex) return 1.0f;
  float lightPdf = emissiveSurfaceLightPdf(surf, incomingDir, emissiveTriangles);
  if (lightPdf <= 0.0f) return 1.0f;
  return lightHitMISWeight(state, lightPdf);
}

VisibleLightHit sampleVisibleMissLight(ray currentRay, constant GPULighting& lighting,
                                       device const GPUEnvironmentSampleCell* environmentCells) {
  VisibleLightHit out;
  // Only finite visible emitters and the environment participate here.
  // Punctual lights remain analytic direct-lighting terms and do not become
  // visible geometry in camera / reflection rays.
  if (intersectVisibleAreaLight(currentRay, lighting, out.radiance, out.lightPdf)) {
    return out;
  }

  out.radiance = evaluateEnvironmentRadiance(currentRay.direction, lighting);
  out.lightPdf = environmentDirectionalPdf(currentRay.direction, environmentCells);
  return out;
}

float3 evaluateVisibleMissLight(ray currentRay, PathLightState state, constant GPULighting& lighting,
                                device const GPUEnvironmentSampleCell* environmentCells) {
  VisibleLightHit visibleLight = sampleVisibleMissLight(currentRay, lighting, environmentCells);
  return visibleLight.radiance * lightHitMISWeight(state, visibleLight.lightPdf);
}

float3 evaluateDirectLightingPBR(SurfaceHitInfo surf, float3 normal, float3 viewDir,
                                 device const float4* positions, device const float2* texcoords,
                                 device const GPUTriangle* triangles, device const GPUMaterial* materials,
                                 device const GPUTexture* textures, device const float4* texturePixels,
                                 device const GPUPunctualLight* sceneLights, device const GPUEmissiveTriangle* emissiveTriangles,
                                 device const GPUEnvironmentSampleCell* environmentCells,
                                 constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
                                 intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                                 intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 directLighting = float3(0.0f);
  float3 N = normal;
  float3 V = viewDir;

  float mainIntensity = lighting.mainLightColorIntensity.w;
  if (mainIntensity > 1e-5f) {
    float3 L;
    float3 radiance;
    float lightPdf = 0.0f;
    if (sampleSunLight(lighting, rng, L, radiance, lightPdf)) {
      float visibility = shadowVisibility(surf.hitPos, surf.geomNormal, L, kRayMaxDistance,
                                          triangles, materials, rng, isector, scene, pointScene, ftable);
      if (visibility > 0.0f) {
        BsdfEvalResult eval = evaluateOpaqueBsdf(N, V, L, surf.baseColor, surf.metallic, surf.roughness, surf.dielectricF0);
        if (eval.NdotL > 0.0f && eval.pdf > 0.0f && lightPdf > 0.0f) {
          float weight = powerHeuristic(lightPdf, eval.pdf);
          directLighting += eval.value * radiance * eval.NdotL * visibility * weight / lightPdf;
        }
      }
    }
  }

  if (frame.lightCount > 0u) {
    for (uint lightIndex = 0u; lightIndex < frame.lightCount; ++lightIndex) {
      float3 L;
      float3 radiance;
      float maxDistance = 1e6f;
      if (!samplePunctualLight(sceneLights, lightIndex, surf.hitPos, L, radiance, maxDistance)) continue;
      float visibility = shadowVisibility(surf.hitPos, surf.geomNormal, L, maxDistance, triangles, materials, rng, isector, scene, pointScene, ftable);
      if (visibility <= 0.0f || all(radiance <= 0.0f)) continue;
      BsdfEvalResult eval = evaluateOpaqueBsdf(N, V, L, surf.baseColor, surf.metallic, surf.roughness, surf.dielectricF0);
      if (eval.NdotL <= 0.0f) continue;
      directLighting += eval.value * radiance * eval.NdotL * visibility;
    }
  }

  if (frame.emissiveTriangleCount > 0u) {
    float3 L;
    float3 radiance;
    float lightPdf = 0.0f;
    if (sampleEmissiveTriangleLight(surf.hitPos, surf.geomNormal, N, positions, texcoords, triangles, materials, textures, texturePixels,
                                    emissiveTriangles, frame.emissiveTriangleCount, surf.triangleIndex, rng,
                                    L, radiance, lightPdf, isector, scene, pointScene, ftable)) {
      BsdfEvalResult eval = evaluateOpaqueBsdf(N, V, L, surf.baseColor, surf.metallic, surf.roughness, surf.dielectricF0);
      if (eval.NdotL > 0.0f && eval.pdf > 0.0f) {
        float weight = powerHeuristic(lightPdf, eval.pdf);
        directLighting += eval.value * radiance * eval.NdotL * weight / lightPdf;
      }
    }
  }

  if (frame.enableAreaLight != 0u) {
    float3 L;
    float3 radiance;
    float lightPdf = 0.0f;
    if (sampleAreaLight(surf.hitPos, surf.geomNormal, N, lighting, triangles, materials, rng, L, radiance, lightPdf, isector, scene, pointScene, ftable)) {
      BsdfEvalResult eval = evaluateOpaqueBsdf(N, V, L, surf.baseColor, surf.metallic, surf.roughness, surf.dielectricF0);
      if (eval.NdotL > 0.0f && eval.pdf > 0.0f) {
        float weight = powerHeuristic(lightPdf, eval.pdf);
        directLighting += eval.value * radiance * eval.NdotL * weight / lightPdf;
      }
    }
  }

  float3 L;
  float3 radiance;
  float lightPdf = 0.0f;
  if (sampleEnvironmentLight(surf.hitPos, surf.geomNormal, N, triangles, materials, environmentCells, lighting, rng,
                             L, radiance, lightPdf, isector, scene, pointScene, ftable)) {
    BsdfEvalResult eval = evaluateOpaqueBsdf(N, V, L, surf.baseColor, surf.metallic, surf.roughness, surf.dielectricF0);
    if (eval.NdotL > 0.0f && eval.pdf > 0.0f) {
      float weight = powerHeuristic(lightPdf, eval.pdf);
      directLighting += eval.value * radiance * eval.NdotL * weight / lightPdf;
    }
  }

  return directLighting;
}

float3 shadeStandardTransmissionSurface(SurfaceHitInfo surf, float3 normal, float3 viewDir,
                                        device const float4* positions, device const float2* texcoords,
                                        device const GPUTriangle* triangles, device const GPUMaterial* materials,
                                        device const GPUTexture* textures, device const float4* texturePixels,
                                        device const GPUPunctualLight* sceneLights, device const GPUEmissiveTriangle* emissiveTriangles,
                                        device const GPUEnvironmentSampleCell* environmentCells,
                                        constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
                                        intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                                        intersection_function_table<curve_data, triangle_data, instancing> ftable) {
  float3 specularLighting = float3(0.0f);
  float3 N = normal;
  float3 V = viewDir;
  float roughness = clamp(surf.roughness, 0.001f, 1.0f);

  float mainIntensity = lighting.mainLightColorIntensity.w;
  if (mainIntensity > 1e-5f) {
    float3 L;
    float3 radiance;
    float lightPdf = 0.0f;
    if (sampleSunLight(lighting, rng, L, radiance, lightPdf)) {
      float visibility = shadowVisibility(surf.hitPos, surf.geomNormal, L, kRayMaxDistance,
                                          triangles, materials, rng, isector, scene, pointScene, ftable);
      if (visibility > 0.0f) {
        BsdfEvalResult eval = evaluateDielectricSpecularBsdf(N, V, L, roughness, surf.ior);
        if (eval.NdotL > 0.0f && eval.pdf > 0.0f && lightPdf > 0.0f) {
          float weight = powerHeuristic(lightPdf, eval.pdf);
          specularLighting += eval.value * radiance * eval.NdotL * visibility * weight / lightPdf;
        }
      }
    }
  }

  if (frame.lightCount > 0u) {
    for (uint lightIndex = 0u; lightIndex < frame.lightCount; ++lightIndex) {
      float3 L;
      float3 radiance;
      float maxDistance = 1e6f;
      if (!samplePunctualLight(sceneLights, lightIndex, surf.hitPos, L, radiance, maxDistance)) continue;
      float visibility = shadowVisibility(surf.hitPos, surf.geomNormal, L, maxDistance, triangles, materials, rng, isector, scene, pointScene, ftable);
      if (visibility <= 0.0f || all(radiance <= 0.0f)) continue;
      BsdfEvalResult eval = evaluateDielectricSpecularBsdf(N, V, L, roughness, surf.ior);
      if (eval.NdotL <= 0.0f) continue;
      specularLighting += eval.value * radiance * eval.NdotL * visibility;
    }
  }

  if (frame.emissiveTriangleCount > 0u) {
    float3 L;
    float3 radiance;
    float lightPdf = 0.0f;
    if (sampleEmissiveTriangleLight(surf.hitPos, surf.geomNormal, N, positions, texcoords, triangles, materials, textures, texturePixels,
                                    emissiveTriangles, frame.emissiveTriangleCount, surf.triangleIndex, rng,
                                    L, radiance, lightPdf, isector, scene, pointScene, ftable)) {
      BsdfEvalResult eval = evaluateDielectricSpecularBsdf(N, V, L, roughness, surf.ior);
      if (eval.NdotL > 0.0f && eval.pdf > 0.0f) {
        float weight = powerHeuristic(lightPdf, eval.pdf);
        specularLighting += eval.value * radiance * eval.NdotL * weight / lightPdf;
      }
    }
  }

  if (frame.enableAreaLight != 0u) {
    float3 L;
    float3 radiance;
    float lightPdf = 0.0f;
    if (sampleAreaLight(surf.hitPos, surf.geomNormal, N, lighting, triangles, materials, rng, L, radiance, lightPdf, isector, scene, pointScene, ftable)) {
      BsdfEvalResult eval = evaluateDielectricSpecularBsdf(N, V, L, roughness, surf.ior);
      if (eval.NdotL > 0.0f && eval.pdf > 0.0f) {
        float weight = powerHeuristic(lightPdf, eval.pdf);
        specularLighting += eval.value * radiance * eval.NdotL * weight / lightPdf;
      }
    }
  }

  float3 L;
  float3 radiance;
  float lightPdf = 0.0f;
  if (sampleEnvironmentLight(surf.hitPos, surf.geomNormal, N, triangles, materials, environmentCells, lighting, rng,
                             L, radiance, lightPdf, isector, scene, pointScene, ftable)) {
    BsdfEvalResult eval = evaluateDielectricSpecularBsdf(N, V, L, roughness, surf.ior);
    if (eval.NdotL > 0.0f && eval.pdf > 0.0f) {
      float weight = powerHeuristic(lightPdf, eval.pdf);
      specularLighting += eval.value * radiance * eval.NdotL * weight / lightPdf;
    }
  }

  return specularLighting;
}

SurfaceHitInfo intersectPathSurface(ray currentRay, device const float4* positions, device const float4* normals,
                                    device const float2* texcoords, device const float4* vertexColors,
                                    device const GPUTriangle* triangles,
                                    device const GPUMaterial* materials, device const GPUTexture* textures,
                                    device const float4* texturePixels,
                                    device const GPUCurvePrimitive* curvePrimitives,
                                    device const GPUPointPrimitive* pointPrimitives,
                                    constant GPUFrameUniforms& frame, constant GPULighting& lighting,
                                    intersector<curve_data, triangle_data, instancing> isector,
                                    instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                                    intersection_function_table<curve_data, triangle_data, instancing> ftable,
                                    device const float* isoScalars, float tanHalfFov) {
  SurfaceHitInfo surf = intersectSurface(currentRay, positions, normals, texcoords, vertexColors, triangles, materials, textures, texturePixels,
                                         curvePrimitives, pointPrimitives, isector, scene, pointScene, ftable, isoScalars);
  applyContour(surf, currentRay.direction, tanHalfFov, frame.height);
  SurfaceHitInfo planeSurf = intersectGroundPlane(currentRay, frame, lighting);
  if (planeSurf.hit != 0u && (surf.hit == 0u || planeSurf.distance < surf.distance)) {
    surf = planeSurf;
  }
  return surf;
}

float3 traceStandardPath(ray currentRay, float3 viewDir, SurfaceHitInfo firstHit, device const float4* positions, device const float4* normals,
                         device const float2* texcoords, device const float4* vertexColors,
                         device const GPUTriangle* triangles,
                         device const GPUMaterial* materials, device const GPUTexture* textures,
                         device const float4* texturePixels, device const GPUPunctualLight* sceneLights,
                         device const GPUEmissiveTriangle* emissiveTriangles,
                         device const GPUEnvironmentSampleCell* environmentCells,
                         device const GPUCurvePrimitive* curvePrimitives,
                         device const GPUPointPrimitive* pointPrimitives,
                         constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
                         intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                         intersection_function_table<curve_data, triangle_data, instancing> ftable,
                         device const float* isoScalars, float tanHalfFov) {
  float3 throughput = float3(1.0f);
  float3 radiance = float3(0.0f);
  float3 currentViewDir = viewDir;
  PathLightState prevLightState = deltaPathLightState();

  for (uint bounce = 0u; bounce < max(frame.maxBounces, 1u); ++bounce) {
    SurfaceHitInfo surf;
    if (bounce == 0u && firstHit.hit != 0u) {
      surf = firstHit;
    } else {
      surf = intersectPathSurface(currentRay, positions, normals, texcoords, vertexColors, triangles, materials, textures, texturePixels,
                                  curvePrimitives, pointPrimitives, frame, lighting, isector, scene, pointScene, ftable, isoScalars, tanHalfFov);
    }
    if (surf.hit == 0u) {
      radiance += throughput * evaluateVisibleMissLight(currentRay, prevLightState, lighting, environmentCells);
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
          currentRay.direction = reflect(currentRay.direction, rN);
          spawnPathRay(currentRay, surf.hitPos, surf.geomNormal, currentRay.direction);
          currentViewDir = -currentRay.direction;
          prevLightState = deltaPathLightState();
          continue;
        }
      }
      if (rand01(rng) > surf.opacity) {
        spawnPathRay(currentRay, surf.hitPos, surf.geomNormal, currentRay.direction);
        continue;
      }
    }

    if (surf.unlit) {
      radiance += throughput * surf.baseColor;
      break;
    }

    if (any(surf.emissive > 0.0f)) {
      float emissionWeight = emissiveSurfaceMISWeight(surf, currentRay.direction, prevLightState, emissiveTriangles);
      radiance += throughput * surf.emissive * emissionWeight;
    }

    if (surf.transmission > 1e-4f && surf.opacity > 1.0f - 1e-4f) {
      float3 N = surf.normal;
      bool frontFace = dot(currentRay.direction, N) < 0.0f;
      float3 orientedN = frontFace ? N : -N;
      radiance += throughput * shadeStandardTransmissionSurface(surf, orientedN, currentViewDir, positions, texcoords, triangles, materials,
                                                                textures, texturePixels, sceneLights, emissiveTriangles, environmentCells,
                                                                frame, lighting, rng, isector, scene, pointScene, ftable);

      float eta = frontFace ? (1.0f / surf.ior) : surf.ior;
      float cosTheta = clamp(dot(-currentRay.direction, orientedN), 0.0f, 1.0f);
      float sin2Theta = max(0.0f, 1.0f - cosTheta * cosTheta);
      bool cannotRefract = eta * eta * sin2Theta > 1.0f;
      float f0 = pow((surf.ior - 1.0f) / (surf.ior + 1.0f), 2.0f);
      float fresnel = f0 + (1.0f - f0) * pow(1.0f - cosTheta, 5.0f);
      if (cannotRefract) {
        currentRay.direction = reflect(currentRay.direction, orientedN);
        spawnPathRay(currentRay, surf.hitPos, surf.geomNormal, currentRay.direction);
        currentViewDir = -currentRay.direction;
        prevLightState = deltaPathLightState();
        continue;
      }

      ray reflectedRay;
      reflectedRay.direction = reflect(currentRay.direction, orientedN);
      spawnPathRay(reflectedRay, surf.hitPos, surf.geomNormal, reflectedRay.direction);
      radiance += throughput *
                  traceSpecularPath(reflectedRay, max(frame.maxBounces - bounce - 1u, 1u), positions, normals, texcoords, vertexColors,
                                    triangles, materials, textures, texturePixels, sceneLights, emissiveTriangles, environmentCells, curvePrimitives, pointPrimitives,
                                    frame, lighting, rng, isector, scene, pointScene, ftable, isoScalars, tanHalfFov) *
                  fresnel;

      float3 tint = mix(float3(1.0f), surf.baseColor, 0.08f);
      throughput *= tint * surf.transmission * max(1.0f - fresnel, 1e-3f);
      currentRay.direction = normalize(refract(currentRay.direction, orientedN, eta));
      spawnPathRay(currentRay, surf.hitPos, surf.geomNormal, currentRay.direction);
      currentViewDir = -currentRay.direction;
      prevLightState = deltaPathLightState();
      continue;
    }

    float3 bsdfNormal = opaqueBsdfNormal(surf, currentRay.direction);
    float3 directLighting = evaluateDirectLightingPBR(surf, bsdfNormal, currentViewDir, positions, texcoords, triangles, materials, textures, texturePixels,
                                                      sceneLights, emissiveTriangles, environmentCells, frame, lighting, rng, isector, scene, pointScene, ftable);
    radiance += throughput * directLighting;

    BsdfSampleResult bsdfSample = sampleOpaqueBsdf(bsdfNormal, currentViewDir, surf.baseColor, surf.metallic,
                                                   surf.roughness, surf.dielectricF0, rng);
    if (!bsdfSample.valid) break;

    throughput *= bsdfSample.throughput;
    currentRay.direction = bsdfSample.direction;
    prevLightState = makePathLightState(bsdfSample.pdf, bsdfSample.skipLightHitMIS);

    if (max(max(throughput.x, throughput.y), throughput.z) < 1e-5f) break;

    if (bounce >= 2u) {
      float rrProb = clamp(max(max(throughput.x, throughput.y), throughput.z) + 0.001f, 0.05f, 0.95f);
      if (rand01(rng) >= rrProb) break;
      throughput /= rrProb;
    }

    spawnPathRay(currentRay, surf.hitPos, surf.geomNormal, currentRay.direction);
    currentViewDir = -currentRay.direction;
  }

  return radiance;
}

float3 traceSpecularPath(ray currentRay, uint remainingBounces, device const float4* positions, device const float4* normals,
                         device const float2* texcoords, device const float4* vertexColors,
                         device const GPUTriangle* triangles,
                         device const GPUMaterial* materials, device const GPUTexture* textures,
                         device const float4* texturePixels, device const GPUPunctualLight* sceneLights,
                         device const GPUEmissiveTriangle* emissiveTriangles,
                         device const GPUEnvironmentSampleCell* environmentCells,
                         device const GPUCurvePrimitive* curvePrimitives,
                         device const GPUPointPrimitive* pointPrimitives,
                         constant GPUFrameUniforms& frame, constant GPULighting& lighting, thread uint& rng,
                         intersector<curve_data, triangle_data, instancing> isector, instance_acceleration_structure scene, instance_acceleration_structure pointScene,
                         intersection_function_table<curve_data, triangle_data, instancing> ftable,
                         device const float* isoScalars, float tanHalfFov) {
  float3 throughput = float3(1.0f);
  float3 radiance = float3(0.0f);
  float3 currentViewDir = -currentRay.direction;
  PathLightState prevLightState = deltaPathLightState();

  for (uint bounce = 0u; bounce < max(remainingBounces, 1u); ++bounce) {
    SurfaceHitInfo surf = intersectPathSurface(currentRay, positions, normals, texcoords, vertexColors, triangles, materials, textures, texturePixels,
                                               curvePrimitives, pointPrimitives, frame, lighting, isector, scene, pointScene, ftable, isoScalars, tanHalfFov);
    if (surf.hit == 0u) {
      radiance += throughput * evaluateVisibleMissLight(currentRay, prevLightState, lighting, environmentCells);
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
          currentRay.direction = reflect(currentRay.direction, rN);
          spawnPathRay(currentRay, surf.hitPos, surf.geomNormal, currentRay.direction);
          currentViewDir = -currentRay.direction;
          prevLightState = deltaPathLightState();
          continue;
        }
      }
      if (rand01(rng) > surf.opacity) {
        spawnPathRay(currentRay, surf.hitPos, surf.geomNormal, currentRay.direction);
        continue;
      }
    }

    if (surf.unlit) {
      radiance += throughput * surf.baseColor;
      break;
    }

    if (any(surf.emissive > 0.0f)) {
      float emissionWeight = emissiveSurfaceMISWeight(surf, currentRay.direction, prevLightState, emissiveTriangles);
      radiance += throughput * surf.emissive * emissionWeight;
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
        currentRay.direction = reflect(currentRay.direction, orientedN);
      } else {
        float3 tint = mix(float3(1.0f), surf.baseColor, 0.08f);
        throughput *= tint * surf.transmission;
        currentRay.direction = normalize(refract(currentRay.direction, orientedN, eta));
      }

      spawnPathRay(currentRay, surf.hitPos, surf.geomNormal, currentRay.direction);
      currentViewDir = -currentRay.direction;
      prevLightState = deltaPathLightState();
      continue;
    }

    float3 bsdfNormal = opaqueBsdfNormal(surf, currentRay.direction);
    float3 directLighting = evaluateDirectLightingPBR(surf, bsdfNormal, currentViewDir, positions, texcoords, triangles, materials, textures, texturePixels,
                                                      sceneLights, emissiveTriangles, environmentCells, frame, lighting, rng, isector, scene, pointScene, ftable);
    radiance += throughput * directLighting;

    BsdfSampleResult bsdfSample = sampleOpaqueBsdf(bsdfNormal, currentViewDir, surf.baseColor, surf.metallic,
                                                   surf.roughness, surf.dielectricF0, rng);
    if (!bsdfSample.valid) break;

    throughput *= bsdfSample.throughput;
    currentRay.direction = bsdfSample.direction;
    prevLightState = makePathLightState(bsdfSample.pdf, bsdfSample.skipLightHitMIS);

    if (max(max(throughput.x, throughput.y), throughput.z) < 1e-5f) break;
    if (bounce >= 2u) {
      float rrProb = clamp(max(max(throughput.x, throughput.y), throughput.z) + 0.001f, 0.05f, 0.95f);
      if (rand01(rng) >= rrProb) break;
      throughput /= rrProb;
    }

    spawnPathRay(currentRay, surf.hitPos, surf.geomNormal, currentRay.direction);
    currentViewDir = -currentRay.direction;
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
                                device const GPUEmissiveTriangle* emissiveTriangles [[buffer(28)]],
                                device const GPUEnvironmentSampleCell* environmentCells [[buffer(29)]],
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
    initializePathRay(currentRay, camera.position.xyz, direction);

    float3 sampleColor = float3(0.0f);
    float sampleAlpha = 1.0f;

    // Pre-compute tan(fov/2) for contour screen-space gradient approximation.
    float tanHalfFov = tan(0.5f * camera.clipData.x);

    SurfaceHitInfo surf = intersectPathSurface(currentRay, positions, normals, texcoords, vertexColors, triangles, materials, textures, texturePixels,
                                               curvePrimitives, pointPrimitives, frame, lighting, isector, scene, pointScene, ftable, isoScalars, tanHalfFov);
    if (surf.hit != 0u) {
      if (frame.renderMode == 0u) {
        sampleColor =
            traceStandardPath(currentRay, normalize(camera.position.xyz - surf.hitPos), surf, positions, normals, texcoords, vertexColors,
                              triangles, materials, textures, texturePixels, sceneLights, emissiveTriangles, environmentCells, curvePrimitives, pointPrimitives,
                              frame, lighting, rng, isector, scene, pointScene, ftable, isoScalars, tanHalfFov);
      } else {
        if (surf.unlit) {
          sampleColor = surf.baseColor;
        } else {
          float direct = 0.0f;
          if (lighting.mainLightColorIntensity.w > 1e-5f) {
            direct += evaluateSunLight(lighting, surf.hitPos, surf.geomNormal, surf.normal,
                                       triangles, materials, rng, isector, scene, pointScene, ftable);
          }
          if (frame.lightCount > 0u) {
            for (uint lightIndex = 0u; lightIndex < frame.lightCount; ++lightIndex) {
              direct += evaluatePunctualLight(sceneLights, lightIndex, surf.hitPos, surf.geomNormal, surf.normal,
                                             triangles, materials, rng, isector, scene, pointScene, ftable);
            }
          }
          if (frame.enableAreaLight != 0u) {
            float3 areaL;
            float3 areaRadiance;
            float areaPdf = 0.0f;
            if (sampleAreaLight(surf.hitPos, surf.geomNormal, surf.normal, lighting, triangles, materials, rng, areaL, areaRadiance, areaPdf,
                                isector, scene, pointScene, ftable)) {
              direct += dot(areaRadiance, float3(0.2126f, 0.7152f, 0.0722f)) *
                        max(dot(surf.normal, areaL), 0.0f) / max(areaPdf, 1e-6f);
            }
          }
          if (frame.emissiveTriangleCount > 0u) {
            float3 emissiveL;
            float3 emissiveRadiance;
            float emissivePdf = 0.0f;
            if (sampleEmissiveTriangleLight(surf.hitPos, surf.geomNormal, surf.normal, positions, texcoords, triangles, materials, textures, texturePixels,
                                            emissiveTriangles, frame.emissiveTriangleCount, surf.triangleIndex, rng,
                                            emissiveL, emissiveRadiance, emissivePdf,
                                            isector, scene, pointScene, ftable)) {
              direct += dot(emissiveRadiance, float3(0.2126f, 0.7152f, 0.0722f)) *
                        max(dot(surf.normal, emissiveL), 0.0f) / max(emissivePdf, 1e-6f);
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
      sampleColor = evaluateVisibleMissLight(currentRay, deltaPathLightState(), lighting, environmentCells);
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
