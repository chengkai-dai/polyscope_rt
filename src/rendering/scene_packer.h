#pragma once

#include <vector>

#include "rendering/gpu_shared_types.h"
#include "rendering/ray_tracing_types.h"

namespace rt {

struct PackedBoundingBox {
  float3 min = float3(0.0f);
  float3 max = float3(0.0f);
};

struct PackedSceneData {
  SceneGpuAccumulator acc;
  std::vector<float3> curveControlPoints;
  std::vector<float> curveRadii;
  std::vector<GPUPointPrimitive> pointPrimitives;
  std::vector<PackedBoundingBox> pointBoundingBoxes;
};

PackedSceneData packScene(const RTScene& scene);

} // namespace rt
