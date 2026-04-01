#pragma once

#include <string>
#include <vector>

#include "glm/glm.hpp"

#include "rendering/scene_packer.h"

namespace rt {
namespace scene_packer_detail {

float4 makeFloat4(const glm::vec3& v, float w = 0.0f);
std::string contourObjectKey(const std::string& meshName);
uint32_t registerTexture(SceneGpuAccumulator& acc, const RTTexture& tex);
PackedBoundingBox makePackedBoundingBox(const glm::vec3& center, float radius);
float textureAverageLuminance(const RTTexture& texture);
float emissiveMeshPowerEstimate(const RTMesh& mesh);

void gatherMeshGpuData(SceneGpuAccumulator& acc, const RTScene& scene);
void gatherVectorFieldGpuData(SceneGpuAccumulator& acc, const RTScene& scene);
void gatherCurveGpuData(SceneGpuAccumulator& acc, const RTScene& scene,
                        std::vector<float3>& curveControlPoints, std::vector<float>& curveRadii,
                        std::vector<GPUPointPrimitive>& pointPrimitives,
                        std::vector<PackedBoundingBox>& pointBoundingBoxes);
void gatherPointBboxData(SceneGpuAccumulator& acc, const RTScene& scene,
                         std::vector<GPUPointPrimitive>& pointPrimitives,
                         std::vector<PackedBoundingBox>& pointBoundingBoxes);
void gatherLightData(SceneGpuAccumulator& acc, const RTScene& scene);

} // namespace scene_packer_detail
} // namespace rt
