#include "rendering/scene_packer_internal.h"

#include <algorithm>

namespace rt {
namespace scene_packer_detail {

void gatherCurveGpuData(SceneGpuAccumulator& acc, const RTScene& scene,
                        std::vector<float3>& curveControlPoints, std::vector<float>& curveRadii,
                        std::vector<GPUPointPrimitive>& pointPrimitives,
                        std::vector<PackedBoundingBox>& pointBoundingBoxes) {
  curveControlPoints.clear();
  curveRadii.clear();
  pointPrimitives.clear();
  pointBoundingBoxes.clear();

  for (const RTCurveNetwork& curveNet : scene.curveNetworks) {
    const std::string objectKey = curveNet.name;
    auto objectIt = acc.objectIdLookup.find(objectKey);
    uint32_t curveObjectId = 0u;
    if (objectIt != acc.objectIdLookup.end()) {
      curveObjectId = objectIt->second;
    } else {
      curveObjectId = acc.nextObjectId++;
      acc.objectIdLookup.emplace(objectKey, curveObjectId);
    }

    const uint32_t curveMaterialIndex = static_cast<uint32_t>(acc.materials.size());
    GPUMaterial curveMaterial{};
    curveMaterial.baseColorFactor =
        simd_make_float4(curveNet.baseColor.r, curveNet.baseColor.g, curveNet.baseColor.b, curveNet.baseColor.a);
    curveMaterial.metallicRoughnessNormal = simd_make_float4(curveNet.metallic, curveNet.roughness, 1.0f, 0.0f);
    curveMaterial.materialFlags = simd_make_uint4(curveNet.unlit ? 1u : 0u, 0u, 0u, 0u);
    curveMaterial.transmissionIor = simd_make_float4(0.0f, 1.5f, 0.0f, 1.0f);
    acc.materials.push_back(curveMaterial);

    const bool hasPrimColors = curveNet.primitiveColors.size() == curveNet.primitives.size();
    const bool hasPrimColors1 = curveNet.primitiveColors1.size() == curveNet.primitives.size();

    const size_t nNodes = curveNet.nodePositions.size();
    const size_t nEdges = curveNet.edgeTailInds.size();
    std::vector<std::vector<uint32_t>> nodeAdj(nNodes);
    for (size_t ei = 0; ei < nEdges; ++ei) {
      const uint32_t a = curveNet.edgeTailInds[ei];
      const uint32_t b = curveNet.edgeTipInds[ei];
      if (a < nNodes) nodeAdj[a].push_back(b);
      if (b < nNodes) nodeAdj[b].push_back(a);
    }

    auto ghostPoint = [&](uint32_t fromIdx, uint32_t toIdx) -> glm::vec3 {
      if (fromIdx < nNodes) {
        for (uint32_t nb : nodeAdj[fromIdx]) {
          if (nb != toIdx) return curveNet.nodePositions[nb];
        }
      }
      return curveNet.nodePositions[fromIdx] * 2.0f - curveNet.nodePositions[toIdx];
    };

    size_t sphereCount = 0;
    for (const RTCurvePrimitive& p : curveNet.primitives) {
      if (p.type == RTCurvePrimitiveType::Sphere) ++sphereCount;
    }
    size_t spherePrimIdx = 0;
    size_t cylPrimIdx = sphereCount;
    size_t cylinderEdgeSlot = 0;

    for (const RTCurvePrimitive& prim : curveNet.primitives) {
      if (prim.type != RTCurvePrimitiveType::Cylinder) {
        ++spherePrimIdx;
        continue;
      }

      const glm::vec3 col0 = hasPrimColors ? curveNet.primitiveColors[cylPrimIdx]
                                           : glm::vec3(curveNet.baseColor);
      const glm::vec3 col1 = hasPrimColors1 ? curveNet.primitiveColors1[cylPrimIdx] : col0;

      glm::vec3 pPrev = prim.p0;
      glm::vec3 pNext = prim.p1;
      if (cylinderEdgeSlot < nEdges) {
        const uint32_t tail = curveNet.edgeTailInds[cylinderEdgeSlot];
        const uint32_t tip = curveNet.edgeTipInds[cylinderEdgeSlot];
        pPrev = ghostPoint(tail, tip);
        pNext = ghostPoint(tip, tail);
      }

      GPUCurvePrimitive shaderPrim{};
      shaderPrim.p0_radius = simd_make_float4(prim.p0.x, prim.p0.y, prim.p0.z, prim.radius);
      shaderPrim.p1_type = simd_make_float4(prim.p1.x, prim.p1.y, prim.p1.z, 1.0f);
      shaderPrim.p_prev = simd_make_float4(pPrev.x, pPrev.y, pPrev.z, 0.0f);
      shaderPrim.p_next = simd_make_float4(pNext.x, pNext.y, pNext.z, 0.0f);
      shaderPrim.materialObjectId = simd_make_uint4(curveMaterialIndex, curveObjectId, 0u, 0u);
      shaderPrim.baseColor = simd_make_float4(col0.r, col0.g, col0.b, hasPrimColors ? 1.0f : 0.0f);
      shaderPrim.baseColor1 = simd_make_float4(col1.r, col1.g, col1.b, 1.0f);
      acc.curvePrimitives.push_back(shaderPrim);

      curveControlPoints.push_back(float3{pPrev.x, pPrev.y, pPrev.z});
      curveControlPoints.push_back(float3{prim.p0.x, prim.p0.y, prim.p0.z});
      curveControlPoints.push_back(float3{prim.p1.x, prim.p1.y, prim.p1.z});
      curveControlPoints.push_back(float3{pNext.x, pNext.y, pNext.z});
      curveRadii.push_back(prim.radius);
      curveRadii.push_back(prim.radius);
      curveRadii.push_back(prim.radius);
      curveRadii.push_back(prim.radius);

      ++cylPrimIdx;
      ++cylinderEdgeSlot;
    }

    spherePrimIdx = 0;
    for (const RTCurvePrimitive& prim : curveNet.primitives) {
      if (prim.type != RTCurvePrimitiveType::Sphere) continue;

      const glm::vec3 col = hasPrimColors ? curveNet.primitiveColors[spherePrimIdx]
                                          : glm::vec3(curveNet.baseColor);
      GPUPointPrimitive gpuPt{};
      gpuPt.center_radius = simd_make_float4(prim.p0.x, prim.p0.y, prim.p0.z, prim.radius);
      gpuPt.baseColor = simd_make_float4(col.r, col.g, col.b, 1.0f);
      gpuPt.materialObjectId = simd_make_uint4(curveMaterialIndex, curveObjectId, 0u, 0u);
      pointPrimitives.push_back(gpuPt);
      pointBoundingBoxes.push_back(makePackedBoundingBox(prim.p0, prim.radius));
      ++spherePrimIdx;
    }
  }
}

void gatherPointBboxData(SceneGpuAccumulator& acc, const RTScene& scene,
                         std::vector<GPUPointPrimitive>& pointPrimitives,
                         std::vector<PackedBoundingBox>& pointBoundingBoxes) {
  for (const RTPointCloud& pc : scene.pointClouds) {
    const std::string objectKey = pc.name;
    auto objectIt = acc.objectIdLookup.find(objectKey);
    uint32_t pointObjectId = 0u;
    if (objectIt != acc.objectIdLookup.end()) {
      pointObjectId = objectIt->second;
    } else {
      pointObjectId = acc.nextObjectId++;
      acc.objectIdLookup.emplace(objectKey, pointObjectId);
    }

    const uint32_t pointMaterialIndex = static_cast<uint32_t>(acc.materials.size());
    GPUMaterial ptMat{};
    ptMat.baseColorFactor = simd_make_float4(pc.baseColor.r, pc.baseColor.g, pc.baseColor.b, pc.baseColor.a);
    ptMat.metallicRoughnessNormal = simd_make_float4(pc.metallic, pc.roughness, 1.0f, 0.0f);
    ptMat.materialFlags = simd_make_uint4(pc.unlit ? 1u : 0u, 0u, 0u, 0u);
    ptMat.transmissionIor = simd_make_float4(0.0f, 1.5f, 0.0f, 1.0f);
    acc.materials.push_back(ptMat);

    for (size_t ci = 0; ci < pc.centers.size(); ++ci) {
      const glm::vec3& center = pc.centers[ci];
      const glm::vec3 ptColor = pc.colors.empty() ? glm::vec3(pc.baseColor)
                                                  : pc.colors[std::min(ci, pc.colors.size() - 1)];
      GPUPointPrimitive gpuPt{};
      gpuPt.center_radius = simd_make_float4(center.x, center.y, center.z, pc.radius);
      gpuPt.baseColor = simd_make_float4(ptColor.r, ptColor.g, ptColor.b, 1.0f);
      gpuPt.materialObjectId = simd_make_uint4(pointMaterialIndex, pointObjectId, 0u, 0u);
      pointPrimitives.push_back(gpuPt);
      pointBoundingBoxes.push_back(makePackedBoundingBox(center, pc.radius));
    }
  }
}

void gatherLightData(SceneGpuAccumulator& acc, const RTScene& scene) {
  for (const RTPunctualLight& light : scene.lights) {
    GPUPunctualLight shaderLight{};
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

} // namespace scene_packer_detail
} // namespace rt
