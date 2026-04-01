#include "rendering/scene_packer_internal.h"

#include <algorithm>
#include <cmath>

#include "glm/gtc/matrix_inverse.hpp"

namespace rt {
namespace scene_packer_detail {

namespace {

void appendVectorFieldVertex(SceneGpuAccumulator& acc, const glm::vec3& position, const glm::vec3& normal) {
  acc.positions.push_back(simd_make_float4(position.x, position.y, position.z, 1.0f));
  acc.normals.push_back(simd_make_float4(normal.x, normal.y, normal.z, 0.0f));
  acc.vertexColors.push_back(simd_make_float4(1, 1, 1, 1));
  acc.texcoords.push_back(simd_make_float2(0, 0));
}

} // namespace

void gatherMeshGpuData(SceneGpuAccumulator& acc, const RTScene& scene) {
  for (const RTMesh& mesh : scene.meshes) {
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
    GPUMaterial material{};
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
    material.materialFlags = simd_make_uint4(mesh.unlit ? 1u : 0u, mesh.doubleSided ? 1u : 0u, 0u, 0u);
    material.transmissionIor =
        simd_make_float4(mesh.transmissionFactor, mesh.indexOfRefraction, 0.0f, mesh.opacity);
    const float edgeBaryThreshold = mesh.wireframe ? (mesh.edgeWidth / 100.0f) : 0.0f;
    material.wireframeEdgeData =
        simd_make_float4(mesh.edgeColor.r, mesh.edgeColor.g, mesh.edgeColor.b, edgeBaryThreshold);
    const bool hasIsolines = !mesh.isoScalars.empty();
    const float isoStyleF = hasIsolines ? static_cast<float>(mesh.isoStyle) : 0.0f;
    material.isoParams =
        simd_make_float4(isoStyleF, mesh.isoSpacing, mesh.isoDarkness, mesh.isoContourThickness);

    if (mesh.hasBaseColorTexture && !mesh.baseColorTexture.pixels.empty()) {
      material.baseColorTextureData = simd_make_uint4(registerTexture(acc, mesh.baseColorTexture), 1u, 0u, 0u);
    }
    if (mesh.hasEmissiveTexture && !mesh.emissiveTexture.pixels.empty()) {
      material.emissiveTextureData = simd_make_uint4(registerTexture(acc, mesh.emissiveTexture), 1u, 0u, 0u);
    }
    if (mesh.hasMetallicRoughnessTexture && !mesh.metallicRoughnessTexture.pixels.empty()) {
      material.metallicRoughnessTextureData =
          simd_make_uint4(registerTexture(acc, mesh.metallicRoughnessTexture), 1u, 0u, 0u);
    }
    if (mesh.hasNormalTexture && !mesh.normalTexture.pixels.empty()) {
      material.normalTextureData = simd_make_uint4(registerTexture(acc, mesh.normalTexture), 1u, 0u, 0u);
    }
    acc.materials.push_back(material);
    const bool emitsLight =
        glm::length(mesh.emissiveFactor) > 1e-6f ||
        (mesh.hasEmissiveTexture && !mesh.emissiveTexture.pixels.empty());
    const float emissivePowerEstimate = emissiveMeshPowerEstimate(mesh);

    std::vector<glm::vec3> worldVertices(mesh.vertices.size());
    std::vector<glm::vec3> worldNormals(mesh.vertices.size(), glm::vec3(0.0f));
    const glm::mat3 normalTransform = glm::transpose(glm::inverse(glm::mat3(mesh.transform)));
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
      const glm::vec4 worldPos = mesh.transform * glm::vec4(mesh.vertices[i], 1.0f);
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
      const glm::vec2 uv = i < mesh.texcoords.size() ? mesh.texcoords[i] : glm::vec2(0.0f);
      acc.texcoords.push_back(simd_make_float2(uv.x, uv.y));
      const float isoVal = (hasIsolines && i < mesh.isoScalars.size()) ? mesh.isoScalars[i] : 0.0f;
      acc.isoScalars.push_back(isoVal);
    }

    for (const glm::uvec3& tri : mesh.indices) {
      glm::uvec3 packedTri = tri;
      if (hasVertexNormals) {
        const glm::vec3 averagedNormal =
            glm::normalize(worldNormals[tri.x] + worldNormals[tri.y] + worldNormals[tri.z]);
        const glm::vec3 triCross =
            glm::cross(worldVertices[tri.y] - worldVertices[tri.x], worldVertices[tri.z] - worldVertices[tri.x]);
        if (glm::dot(averagedNormal, averagedNormal) > 1e-10f && glm::dot(triCross, triCross) > 1e-10f &&
            glm::dot(averagedNormal, triCross) < 0.0f) {
          packedTri = glm::uvec3(tri.x, tri.z, tri.y);
        }
      }

      acc.accelIndices.push_back({baseVertex + packedTri.x, baseVertex + packedTri.y, baseVertex + packedTri.z});
      GPUTriangle triangle{};
      triangle.indicesMaterial =
          simd_make_uint4(baseVertex + packedTri.x, baseVertex + packedTri.y, baseVertex + packedTri.z, materialIndex);
      triangle.objectFlags =
          simd_make_uint4(meshObjectId, hasVertexNormals ? 1u : 0u, mesh.wireframe ? 1u : 0u, 0u);
      const uint32_t triangleIndex = static_cast<uint32_t>(acc.shaderTriangles.size());
      acc.shaderTriangles.push_back(triangle);
      if (emitsLight) {
        const glm::vec3& wp0 = worldVertices[packedTri.x];
        const glm::vec3& wp1 = worldVertices[packedTri.y];
        const glm::vec3& wp2 = worldVertices[packedTri.z];
        const float triangleArea = 0.5f * glm::length(glm::cross(wp1 - wp0, wp2 - wp0));
        const float selectionWeight = triangleArea * emissivePowerEstimate;
        if (triangleArea <= 1e-8f || selectionWeight <= 1e-8f) continue;

        GPUEmissiveTriangle emissiveTriangle{};
        emissiveTriangle.data = simd_make_uint4(triangleIndex, 0u, 0u, 0u);
        emissiveTriangle.params = simd_make_float4(triangleArea, selectionWeight, 0.0f, 0.0f);
        const uint32_t emissiveIndex = static_cast<uint32_t>(acc.emissiveTriangles.size());
        acc.emissiveTriangles.push_back(emissiveTriangle);
        acc.shaderTriangles[triangleIndex].objectFlags.w = emissiveIndex + 1u;
      }
    }
  }
}

void gatherVectorFieldGpuData(SceneGpuAccumulator& acc, const RTScene& scene) {
  constexpr int kSides = 8;
  constexpr float kTipLenFrac = 0.2f;
  constexpr float kTipRadMult = 1.0f / 0.6f;

  for (const RTVectorField& vf : scene.vectorFields) {
    if (vf.roots.empty()) continue;

    const uint32_t materialIndex = static_cast<uint32_t>(acc.materials.size());
    const uint32_t objectId = acc.nextObjectId++;

    GPUMaterial mat{};
    mat.baseColorFactor = simd_make_float4(vf.color.r, vf.color.g, vf.color.b, 1.0f);
    mat.baseColorTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
    mat.metallicRoughnessNormal = simd_make_float4(vf.metallic, vf.roughness, 1.0f, 0.0f);
    mat.metallicRoughnessTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
    mat.emissiveFactor = simd_make_float4(0.0f, 0.0f, 0.0f, 1.0f);
    mat.emissiveTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
    mat.normalTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
    mat.materialFlags = simd_make_uint4(0u, 0u, 0u, 0u);
    mat.transmissionIor = simd_make_float4(0.0f, 1.5f, 0.0f, 1.0f);
    mat.wireframeEdgeData = simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    acc.materials.push_back(mat);

    for (size_t ai = 0; ai < vf.roots.size(); ++ai) {
      const glm::vec3& root = vf.roots[ai];
      const glm::vec3& dir = vf.directions[ai];

      const float arrowLen = glm::length(dir);
      if (arrowLen < 1e-7f) continue;

      const glm::vec3 axis = dir / arrowLen;
      const float shaftLen = arrowLen * (1.0f - kTipLenFrac);
      const float tipLen = arrowLen * kTipLenFrac;
      const float shaftR = vf.radius;
      const float coneBaseR = shaftR * kTipRadMult;

      glm::vec3 tang;
      if (std::abs(axis.x) < 0.9f) tang = glm::normalize(glm::cross(axis, glm::vec3(1, 0, 0)));
      else tang = glm::normalize(glm::cross(axis, glm::vec3(0, 1, 0)));
      const glm::vec3 bitang = glm::normalize(glm::cross(axis, tang));

      const glm::vec3 shaftEnd = root + axis * shaftLen;
      const glm::vec3 tipEnd = root + dir;

      const uint32_t baseV = static_cast<uint32_t>(acc.positions.size());

      for (int side = 0; side < kSides; ++side) {
        const float theta = static_cast<float>(side) / kSides * 6.2831853f;
        const float c = std::cos(theta);
        const float s = std::sin(theta);
        const glm::vec3 n = glm::normalize(tang * c + bitang * s);
        const glm::vec3 r0 = root + n * shaftR;
        const glm::vec3 r1 = shaftEnd + n * shaftR;

        appendVectorFieldVertex(acc, r0, n);
        appendVectorFieldVertex(acc, r1, n);
      }
      for (int side = 0; side < kSides; ++side) {
        const uint32_t a = baseV + static_cast<uint32_t>(side * 2);
        const uint32_t b = baseV + static_cast<uint32_t>(side * 2 + 1);
        const uint32_t c = baseV + static_cast<uint32_t>(((side + 1) % kSides) * 2);
        const uint32_t d = baseV + static_cast<uint32_t>(((side + 1) % kSides) * 2 + 1);
        GPUTriangle t1{};
        t1.indicesMaterial = simd_make_uint4(a, c, b, materialIndex);
        t1.objectFlags = simd_make_uint4(objectId, 1u, 0u, 0u);
        acc.shaderTriangles.push_back(t1);
        acc.accelIndices.push_back({a, c, b});
        GPUTriangle t2{};
        t2.indicesMaterial = simd_make_uint4(b, c, d, materialIndex);
        t2.objectFlags = simd_make_uint4(objectId, 1u, 0u, 0u);
        acc.shaderTriangles.push_back(t2);
        acc.accelIndices.push_back({b, c, d});
      }

      const uint32_t rootCapBase = static_cast<uint32_t>(acc.positions.size());
      for (int side = 0; side < kSides; ++side) {
        const float theta = static_cast<float>(side) / kSides * 6.2831853f;
        const glm::vec3 radial = tang * std::cos(theta) + bitang * std::sin(theta);
        appendVectorFieldVertex(acc, root + radial * shaftR, -axis);
      }
      const uint32_t rootCapCenter = static_cast<uint32_t>(acc.positions.size());
      appendVectorFieldVertex(acc, root, -axis);
      for (int side = 0; side < kSides; ++side) {
        const uint32_t a = rootCapBase + static_cast<uint32_t>(side);
        const uint32_t b = rootCapBase + static_cast<uint32_t>((side + 1) % kSides);
        GPUTriangle t{};
        t.indicesMaterial = simd_make_uint4(rootCapCenter, b, a, materialIndex);
        t.objectFlags = simd_make_uint4(objectId, 1u, 0u, 0u);
        acc.shaderTriangles.push_back(t);
        acc.accelIndices.push_back({rootCapCenter, b, a});
      }

      const uint32_t coneBase = static_cast<uint32_t>(acc.positions.size());
      const float slopeAngle = std::atan2(coneBaseR, tipLen);
      for (int side = 0; side < kSides; ++side) {
        const float theta = static_cast<float>(side) / kSides * 6.2831853f;
        const float cs = std::cos(theta);
        const float sn = std::sin(theta);
        const glm::vec3 radial = tang * cs + bitang * sn;
        const glm::vec3 rp = shaftEnd + radial * coneBaseR;
        const glm::vec3 cn = glm::normalize(radial * std::cos(slopeAngle) + axis * std::sin(slopeAngle));
        appendVectorFieldVertex(acc, rp, cn);
      }
      const uint32_t coneApexBase = static_cast<uint32_t>(acc.positions.size());
      for (int side = 0; side < kSides; ++side) {
        const float thetaMid = (static_cast<float>(side) + 0.5f) / kSides * 6.2831853f;
        const glm::vec3 radialMid =
            glm::normalize(tang * std::cos(thetaMid) + bitang * std::sin(thetaMid));
        const glm::vec3 apexN =
            glm::normalize(radialMid * std::cos(slopeAngle) + axis * std::sin(slopeAngle));
        appendVectorFieldVertex(acc, tipEnd, apexN);
      }

      for (int side = 0; side < kSides; ++side) {
        const uint32_t a = coneBase + static_cast<uint32_t>(side);
        const uint32_t b = coneBase + static_cast<uint32_t>((side + 1) % kSides);
        const uint32_t apexI = coneApexBase + static_cast<uint32_t>(side);
        GPUTriangle t{};
        t.indicesMaterial = simd_make_uint4(a, b, apexI, materialIndex);
        t.objectFlags = simd_make_uint4(objectId, 1u, 0u, 0u);
        acc.shaderTriangles.push_back(t);
        acc.accelIndices.push_back({a, b, apexI});
      }

      const uint32_t coneCapBase = static_cast<uint32_t>(acc.positions.size());
      for (int side = 0; side < kSides; ++side) {
        const float theta = static_cast<float>(side) / kSides * 6.2831853f;
        const glm::vec3 radial = tang * std::cos(theta) + bitang * std::sin(theta);
        appendVectorFieldVertex(acc, shaftEnd + radial * coneBaseR, -axis);
      }
      const uint32_t coneCapCenter = static_cast<uint32_t>(acc.positions.size());
      appendVectorFieldVertex(acc, shaftEnd, -axis);
      for (int side = 0; side < kSides; ++side) {
        const uint32_t a = coneCapBase + static_cast<uint32_t>(side);
        const uint32_t b = coneCapBase + static_cast<uint32_t>((side + 1) % kSides);
        GPUTriangle t{};
        t.indicesMaterial = simd_make_uint4(coneCapCenter, b, a, materialIndex);
        t.objectFlags = simd_make_uint4(objectId, 1u, 0u, 0u);
        acc.shaderTriangles.push_back(t);
        acc.accelIndices.push_back({coneCapCenter, b, a});
      }
    }
  }
}

} // namespace scene_packer_detail
} // namespace rt
