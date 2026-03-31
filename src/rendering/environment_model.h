#pragma once

#include <vector>

#include "rendering/gpu_shared_types.h"
#include "rendering/ray_tracing_types.h"

namespace rt {

std::vector<GPUEnvironmentSampleCell> buildEnvironmentSampleCells(const LightingSettings& lighting);

} // namespace rt
