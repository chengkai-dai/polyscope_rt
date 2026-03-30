#include "rendering/metal/metal_scene_builder.h"
#include "rendering/metal/metal_device.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_inverse.hpp"

namespace metal_rt {

void gatherMeshGpuData(SceneGpuAccumulator& acc, const rt::RTScene& scene) {
  for (const rt::RTMesh& mesh : scene.meshes) {
    if (mesh.vertices.empty() || mesh.indices.empty()) continue;

    const uint32_t baseVertex = static_cast<uint32_t>(acc.positions.size());
    const uint32_t materialIndex = static_cast<uint32_t>(acc.materials.size());
    const std::string objectKey = contourObjectKey(mesh.name);
    auto objectIt = acc.objectIdLookup.find(objectKey);
    uint32_t meshObjectId = 0u;
    if (objectIt != acc.objectIdLookup.end()) {
      meshObjectId = objectIt->second;
    } else {
      meshObjectId = acc.nextObjectId++;
      acc.objectIdLookup.emplace(objectKey, meshObjectId);
    }

    const bool hasVertexNormals = mesh.normals.size() == mesh.vertices.size();
    GPUMaterial material;
    material.baseColorFactor =
        simd_make_float4(mesh.baseColorFactor.r, mesh.baseColorFactor.g, mesh.baseColorFactor.b, mesh.baseColorFactor.a);
    material.baseColorTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
    material.metallicRoughnessNormal =
        simd_make_float4(mesh.metallicFactor, mesh.roughnessFactor, mesh.normalTextureScale, 0.0f);
    material.metallicRoughnessTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
    material.emissiveFactor =
        simd_make_float4(mesh.emissiveFactor.r, mesh.emissiveFactor.g, mesh.emissiveFactor.b, 1.0f);
    material.emissiveTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
    material.normalTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
    float opacityPacked = mesh.opacity;
    float transmissionPacked = mesh.transmissionFactor;
    if (opacityPacked < 1e-5f && transmissionPacked < 1e-5f) opacityPacked = 1.0f;
    material.transmissionIor =
        simd_make_float4(transmissionPacked, mesh.indexOfRefraction, mesh.unlit ? 1.0f : 0.0f, opacityPacked);
    const float edgeBaryThreshold = mesh.wireframe ? (mesh.edgeWidth / 100.0f) : 0.0f;
    material.wireframeEdgeData =
        simd_make_float4(mesh.edgeColor.r, mesh.edgeColor.g, mesh.edgeColor.b, edgeBaryThreshold);
    const bool hasIsolines = !mesh.isoScalars.empty();
    // isoParams.x encodes style: 0=off, 1=stripe, 2=contour
    float isoStyleF = hasIsolines ? static_cast<float>(mesh.isoStyle) : 0.0f;
    material.isoParams = simd_make_float4(isoStyleF, mesh.isoSpacing,
                                          mesh.isoDarkness, mesh.isoContourThickness);

    if (mesh.hasBaseColorTexture && !mesh.baseColorTexture.pixels.empty()) {
      material.baseColorTextureData = simd_make_uint4(registerTextureInAcc(acc, mesh.baseColorTexture), 1u, 0u, 0u);
    }
    if (mesh.hasEmissiveTexture && !mesh.emissiveTexture.pixels.empty()) {
      material.emissiveTextureData = simd_make_uint4(registerTextureInAcc(acc, mesh.emissiveTexture), 1u, 0u, 0u);
    }
    if (mesh.hasMetallicRoughnessTexture && !mesh.metallicRoughnessTexture.pixels.empty()) {
      material.metallicRoughnessTextureData =
          simd_make_uint4(registerTextureInAcc(acc, mesh.metallicRoughnessTexture), 1u, 0u, 0u);
    }
    if (mesh.hasNormalTexture && !mesh.normalTexture.pixels.empty()) {
      material.normalTextureData = simd_make_uint4(registerTextureInAcc(acc, mesh.normalTexture), 1u, 0u, 0u);
    }
    acc.materials.push_back(material);

    std::vector<glm::vec3> worldVertices(mesh.vertices.size());
    std::vector<glm::vec3> worldNormals(mesh.vertices.size(), glm::vec3(0.0f));
    glm::mat3 normalTransform = glm::transpose(glm::inverse(glm::mat3(mesh.transform)));
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
      glm::vec4 worldPos = mesh.transform * glm::vec4(mesh.vertices[i], 1.0f);
      worldVertices[i] = glm::vec3(worldPos);
      if (hasVertexNormals) worldNormals[i] = glm::normalize(normalTransform * mesh.normals[i]);
    }

    const bool hasVertexColors = mesh.vertexColors.size() == mesh.vertices.size();
    for (size_t i = 0; i < worldVertices.size(); ++i) {
      acc.positions.push_back(makeFloat4(worldVertices[i], 1.0f));
      acc.normals.push_back(makeFloat4(worldNormals[i], 0.0f));
      if (hasVertexColors) {
        acc.vertexColors.push_back(
            simd_make_float4(mesh.vertexColors[i].r, mesh.vertexColors[i].g, mesh.vertexColors[i].b, 1.0f));
      } else {
        acc.vertexColors.push_back(simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f));
      }
      glm::vec2 uv = i < mesh.texcoords.size() ? mesh.texcoords[i] : glm::vec2(0.0f);
      acc.texcoords.push_back(simd_make_float2(uv.x, uv.y));
      float isoVal = (hasIsolines && i < mesh.isoScalars.size()) ? mesh.isoScalars[i] : 0.0f;
      acc.isoScalars.push_back(isoVal);
    }

    for (const glm::uvec3& tri : mesh.indices) {
      acc.accelIndices.push_back({baseVertex + tri.x, baseVertex + tri.y, baseVertex + tri.z});
      GPUTriangle triangle{};
      triangle.indicesMaterial =
          simd_make_uint4(baseVertex + tri.x, baseVertex + tri.y, baseVertex + tri.z, materialIndex);
      triangle.objectFlags = simd_make_uint4(meshObjectId, hasVertexNormals ? 1u : 0u, mesh.wireframe ? 1u : 0u, 0u);
      acc.shaderTriangles.push_back(triangle);
    }
  }
}

void gatherVectorFieldGpuData(SceneGpuAccumulator& acc, const rt::RTScene& scene) {
  constexpr int   kSides        = 8;
  constexpr float kTipLenFrac   = 0.2f;
  constexpr float kTipRadMult   = 1.0f / 0.6f;

  for (const rt::RTVectorField& vf : scene.vectorFields) {
    if (vf.roots.empty()) continue;

    const uint32_t materialIndex = static_cast<uint32_t>(acc.materials.size());
    const uint32_t objectId      = acc.nextObjectId++;

    GPUMaterial mat{};
    mat.baseColorFactor              = simd_make_float4(vf.color.r, vf.color.g, vf.color.b, 1.0f);
    mat.baseColorTextureData         = simd_make_uint4(0u, 0u, 0u, 0u);
    mat.metallicRoughnessNormal      = simd_make_float4(vf.metallic, vf.roughness, 1.0f, 0.0f);
    mat.metallicRoughnessTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
    mat.emissiveFactor               = simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    mat.emissiveTextureData          = simd_make_uint4(0u, 0u, 0u, 0u);
    mat.normalTextureData            = simd_make_uint4(0u, 0u, 0u, 0u);
    mat.transmissionIor              = simd_make_float4(0.0f, 1.5f, 0.0f, 1.0f);
    mat.wireframeEdgeData            = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    acc.materials.push_back(mat);

    for (size_t ai = 0; ai < vf.roots.size(); ++ai) {
      const glm::vec3& root = vf.roots[ai];
      const glm::vec3& dir  = vf.directions[ai];

      const float arrowLen = glm::length(dir);
      if (arrowLen < 1e-7f) continue;

      const glm::vec3 axis = dir / arrowLen;
      const float shaftLen = arrowLen * (1.0f - kTipLenFrac);
      const float tipLen   = arrowLen * kTipLenFrac;
      const float shaftR   = vf.radius;
      const float coneBaseR = shaftR * kTipRadMult;

      glm::vec3 tang;
      if (std::abs(axis.x) < 0.9f) tang = glm::normalize(glm::cross(axis, glm::vec3(1,0,0)));
      else                          tang = glm::normalize(glm::cross(axis, glm::vec3(0,1,0)));
      const glm::vec3 bitang = glm::normalize(glm::cross(axis, tang));

      const glm::vec3 shaftEnd = root + axis * shaftLen;
      const glm::vec3 tipEnd   = root + dir;

      const uint32_t baseV = static_cast<uint32_t>(acc.positions.size());

      // Cylinder shaft: two rings
      for (int side = 0; side < kSides; ++side) {
        const float theta = static_cast<float>(side) / kSides * 6.2831853f;
        const float c = std::cos(theta), s = std::sin(theta);
        const glm::vec3 n = glm::normalize(tang * c + bitang * s);
        const glm::vec3 r0 = root     + n * shaftR;
        const glm::vec3 r1 = shaftEnd + n * shaftR;

        acc.positions.push_back(simd_make_float4(r0.x, r0.y, r0.z, 1.0f));
        acc.normals.push_back  (simd_make_float4(n.x,  n.y,  n.z,  0.0f));
        acc.vertexColors.push_back(simd_make_float4(1,1,1,1));
        acc.texcoords.push_back(simd_make_float2(0,0));

        acc.positions.push_back(simd_make_float4(r1.x, r1.y, r1.z, 1.0f));
        acc.normals.push_back  (simd_make_float4(n.x,  n.y,  n.z,  0.0f));
        acc.vertexColors.push_back(simd_make_float4(1,1,1,1));
        acc.texcoords.push_back(simd_make_float2(0,0));
      }
      for (int side = 0; side < kSides; ++side) {
        const uint32_t a = baseV + static_cast<uint32_t>(side * 2);
        const uint32_t b = baseV + static_cast<uint32_t>(side * 2 + 1);
        const uint32_t c = baseV + static_cast<uint32_t>(((side + 1) % kSides) * 2);
        const uint32_t d = baseV + static_cast<uint32_t>(((side + 1) % kSides) * 2 + 1);
        const uint32_t m = materialIndex;
        GPUTriangle t1{};
        t1.indicesMaterial = simd_make_uint4(a, c, b, m);
        t1.objectFlags     = simd_make_uint4(objectId, 1u, 0u, 0u);
        acc.shaderTriangles.push_back(t1);
        acc.accelIndices.push_back({a, c, b});
        GPUTriangle t2{};
        t2.indicesMaterial = simd_make_uint4(b, c, d, m);
        t2.objectFlags     = simd_make_uint4(objectId, 1u, 0u, 0u);
        acc.shaderTriangles.push_back(t2);
        acc.accelIndices.push_back({b, c, d});
      }

      // Cone tip
      const uint32_t coneBase = static_cast<uint32_t>(acc.positions.size());
      const float slopeAngle = std::atan2(coneBaseR, tipLen);
      for (int side = 0; side < kSides; ++side) {
        const float theta = static_cast<float>(side) / kSides * 6.2831853f;
        const float cs = std::cos(theta), sn = std::sin(theta);
        const glm::vec3 radial  = tang * cs + bitang * sn;
        const glm::vec3 rp = shaftEnd + radial * coneBaseR;
        const glm::vec3 cn = glm::normalize(radial * std::cos(slopeAngle) +
                                             axis   * std::sin(slopeAngle));
        acc.positions.push_back(simd_make_float4(rp.x, rp.y, rp.z, 1.0f));
        acc.normals.push_back  (simd_make_float4(cn.x, cn.y, cn.z, 0.0f));
        acc.vertexColors.push_back(simd_make_float4(1,1,1,1));
        acc.texcoords.push_back(simd_make_float2(0,0));
      }
      const uint32_t coneApexBase = static_cast<uint32_t>(acc.positions.size());
      for (int side = 0; side < kSides; ++side) {
        const float thetaMid = (static_cast<float>(side) + 0.5f) / kSides * 6.2831853f;
        const glm::vec3 radialMid = glm::normalize(tang * std::cos(thetaMid) +
                                                   bitang * std::sin(thetaMid));
        const glm::vec3 apexN = glm::normalize(radialMid * std::cos(slopeAngle) +
                                                axis      * std::sin(slopeAngle));
        acc.positions.push_back(simd_make_float4(tipEnd.x, tipEnd.y, tipEnd.z, 1.0f));
        acc.normals.push_back  (simd_make_float4(apexN.x, apexN.y, apexN.z, 0.0f));
        acc.vertexColors.push_back(simd_make_float4(1,1,1,1));
        acc.texcoords.push_back(simd_make_float2(0,0));
      }

      for (int side = 0; side < kSides; ++side) {
        const uint32_t a     = coneBase     + static_cast<uint32_t>(side);
        const uint32_t b     = coneBase     + static_cast<uint32_t>((side + 1) % kSides);
        const uint32_t apexI = coneApexBase + static_cast<uint32_t>(side);
        const uint32_t m = materialIndex;
        GPUTriangle t{};
        t.indicesMaterial = simd_make_uint4(a, b, apexI, m);
        t.objectFlags     = simd_make_uint4(objectId, 1u, 0u, 0u);
        acc.shaderTriangles.push_back(t);
        acc.accelIndices.push_back({a, b, apexI});
      }
    }
  }
}

void gatherCurveGpuData(SceneGpuAccumulator& acc, const rt::RTScene& scene,
                        std::vector<simd_float3>& curveControlPoints,
                        std::vector<float>& curveRadii,
                        std::vector<GPUPointPrimitive>& pointPrimitives,
                        std::vector<MTLAxisAlignedBoundingBox>& pointBboxData) {
  curveControlPoints.clear();
  curveRadii.clear();
  pointPrimitives.clear();
  pointBboxData.clear();

  for (const rt::RTCurveNetwork& curveNet : scene.curveNetworks) {
    const std::string objectKey = curveNet.name;
    auto objectIt = acc.objectIdLookup.find(objectKey);
    uint32_t curveObjectId = 0u;
    if (objectIt != acc.objectIdLookup.end()) {
      curveObjectId = objectIt->second;
    } else {
      curveObjectId = acc.nextObjectId++;
      acc.objectIdLookup.emplace(objectKey, curveObjectId);
    }

    uint32_t curveMaterialIndex = static_cast<uint32_t>(acc.materials.size());
    GPUMaterial curveMaterial{};
    curveMaterial.baseColorFactor =
        simd_make_float4(curveNet.baseColor.r, curveNet.baseColor.g, curveNet.baseColor.b, curveNet.baseColor.a);
    curveMaterial.metallicRoughnessNormal = simd_make_float4(curveNet.metallic, curveNet.roughness, 1.0f, 0.0f);
    curveMaterial.transmissionIor = simd_make_float4(0.0f, 1.5f, curveNet.unlit ? 1.0f : 0.0f, 1.0f);
    acc.materials.push_back(curveMaterial);

    for (const rt::RTCurvePrimitive& prim : curveNet.primitives) {
      if (prim.type != rt::RTCurvePrimitiveType::Cylinder) continue;

      GPUCurvePrimitive shaderPrim;
      shaderPrim.p0_radius = simd_make_float4(prim.p0.x, prim.p0.y, prim.p0.z, prim.radius);
      shaderPrim.p1_type = simd_make_float4(prim.p1.x, prim.p1.y, prim.p1.z, 1.0f);
      shaderPrim.materialObjectId = simd_make_uint4(curveMaterialIndex, curveObjectId, 0u, 0u);
      acc.curvePrimitives.push_back(shaderPrim);

      curveControlPoints.push_back(simd_make_float3(prim.p0.x, prim.p0.y, prim.p0.z));
      curveControlPoints.push_back(simd_make_float3(prim.p1.x, prim.p1.y, prim.p1.z));
      curveRadii.push_back(prim.radius);
      curveRadii.push_back(prim.radius);
    }

    for (const rt::RTCurvePrimitive& prim : curveNet.primitives) {
      if (prim.type != rt::RTCurvePrimitiveType::Sphere) continue;
      GPUPointPrimitive gpuPt;
      gpuPt.center_radius = simd_make_float4(prim.p0.x, prim.p0.y, prim.p0.z, prim.radius);
      gpuPt.baseColor =
          simd_make_float4(curveNet.baseColor.r, curveNet.baseColor.g, curveNet.baseColor.b, 1.0f);
      gpuPt.materialObjectId = simd_make_uint4(curveMaterialIndex, curveObjectId, 0u, 0u);
      pointPrimitives.push_back(gpuPt);

      MTLAxisAlignedBoundingBox bb;
      bb.min = {prim.p0.x - prim.radius, prim.p0.y - prim.radius, prim.p0.z - prim.radius};
      bb.max = {prim.p0.x + prim.radius, prim.p0.y + prim.radius, prim.p0.z + prim.radius};
      pointBboxData.push_back(bb);
    }
  }
}

void gatherPointBboxData(SceneGpuAccumulator& acc, const rt::RTScene& scene,
                         std::vector<GPUPointPrimitive>& pointPrimitives,
                         std::vector<MTLAxisAlignedBoundingBox>& pointBboxData) {
  for (const rt::RTPointCloud& pc : scene.pointClouds) {
    const std::string objectKey = pc.name;
    auto objectIt = acc.objectIdLookup.find(objectKey);
    uint32_t pointObjectId = 0u;
    if (objectIt != acc.objectIdLookup.end()) {
      pointObjectId = objectIt->second;
    } else {
      pointObjectId = acc.nextObjectId++;
      acc.objectIdLookup.emplace(objectKey, pointObjectId);
    }

    uint32_t pointMaterialIndex = static_cast<uint32_t>(acc.materials.size());
    GPUMaterial ptMat{};
    ptMat.baseColorFactor = simd_make_float4(pc.baseColor.r, pc.baseColor.g, pc.baseColor.b, pc.baseColor.a);
    ptMat.metallicRoughnessNormal = simd_make_float4(pc.metallic, pc.roughness, 1.0f, 0.0f);
    ptMat.transmissionIor = simd_make_float4(0.0f, 1.5f, pc.unlit ? 1.0f : 0.0f, 1.0f);
    acc.materials.push_back(ptMat);

    for (size_t ci = 0; ci < pc.centers.size(); ++ci) {
      const glm::vec3& center = pc.centers[ci];
      glm::vec3 ptColor = pc.colors.empty() ? glm::vec3(pc.baseColor)
                                            : pc.colors[std::min(ci, pc.colors.size() - 1)];
      GPUPointPrimitive gpuPt;
      gpuPt.center_radius = simd_make_float4(center.x, center.y, center.z, pc.radius);
      gpuPt.baseColor = simd_make_float4(ptColor.r, ptColor.g, ptColor.b, 1.0f);
      gpuPt.materialObjectId = simd_make_uint4(pointMaterialIndex, pointObjectId, 0u, 0u);
      pointPrimitives.push_back(gpuPt);

      MTLAxisAlignedBoundingBox bb;
      bb.min = {center.x - pc.radius, center.y - pc.radius, center.z - pc.radius};
      bb.max = {center.x + pc.radius, center.y + pc.radius, center.z + pc.radius};
      pointBboxData.push_back(bb);
    }
  }
}

void gatherLightData(SceneGpuAccumulator& acc, const rt::RTScene& scene) {
  for (const rt::RTPunctualLight& light : scene.lights) {
    GPUPunctualLight shaderLight;
    shaderLight.positionRange =
        simd_make_float4(light.position.x, light.position.y, light.position.z, light.range);
    shaderLight.directionType = simd_make_float4(light.direction.x, light.direction.y, light.direction.z,
                                                 static_cast<float>(static_cast<uint32_t>(light.type)));
    shaderLight.colorIntensity =
        simd_make_float4(light.color.x, light.color.y, light.color.z, light.intensity);
    shaderLight.spotAngles = simd_make_float4(light.innerConeAngle, light.outerConeAngle, 0.0f, 0.0f);
    acc.lights.push_back(shaderLight);
  }
}

} // namespace metal_rt
