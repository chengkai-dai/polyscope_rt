#pragma once

#import <Metal/Metal.h>
#import <simd/simd.h>

#include <vector>

#include "rendering/gpu_shared_types.h"
#include "rendering/ray_tracing_types.h"

namespace metal_rt {

void gatherMeshGpuData(SceneGpuAccumulator& acc, const rt::RTScene& scene);

void gatherVectorFieldGpuData(SceneGpuAccumulator& acc, const rt::RTScene& scene);

void gatherCurveGpuData(SceneGpuAccumulator& acc, const rt::RTScene& scene,
                        std::vector<simd_float3>& curveControlPoints,
                        std::vector<float>& curveRadii,
                        std::vector<GPUPointPrimitive>& pointPrimitives,
                        std::vector<MTLAxisAlignedBoundingBox>& pointBboxData);

void gatherPointBboxData(SceneGpuAccumulator& acc, const rt::RTScene& scene,
                         std::vector<GPUPointPrimitive>& pointPrimitives,
                         std::vector<MTLAxisAlignedBoundingBox>& pointBboxData);

void gatherLightData(SceneGpuAccumulator& acc, const rt::RTScene& scene);

} // namespace metal_rt
