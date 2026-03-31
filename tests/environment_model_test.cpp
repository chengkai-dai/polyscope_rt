#include <cmath>
#include <iostream>

#include "rendering/environment_model.h"
#include "rendering/environment_model_shared.h"
#include "rendering/ray_tracing_types.h"
#include "test_helpers.h"

namespace {

void testEnvironmentPdfIsNormalized() {
  rt::LightingSettings lighting;
  std::vector<GPUEnvironmentSampleCell> cells = rt::buildEnvironmentSampleCells(lighting);
  require(cells.size() == RT_ENVIRONMENT_SAMPLE_WIDTH * RT_ENVIRONMENT_SAMPLE_HEIGHT,
          "environment sampler should emit the full discretized hemisphere grid");

  float prevCdf = 0.0f;
  for (const GPUEnvironmentSampleCell& cell : cells) {
    require(std::isfinite(cell.data.x) && std::isfinite(cell.data.y),
            "environment sample CDF must not contain NaNs");
    require(cell.data.x >= 0.0f, "environment PMF must stay non-negative");
    require(cell.data.y + 1e-6f >= prevCdf, "environment CDF must be monotonic");
    prevCdf = cell.data.y;
  }
  requireNear(cells.back().data.y, 1.0f, 1e-4f, "environment CDF should terminate at 1");
}

void testEnvironmentRadianceProfileIsShared() {
  rt::LightingSettings lighting;
  const float3 background = float3{lighting.backgroundColor.x, lighting.backgroundColor.y, lighting.backgroundColor.z};
  const float3 tint = float3{lighting.environmentTint.x, lighting.environmentTint.y, lighting.environmentTint.z};
  const float envIntensity = lighting.environmentIntensity;

  const float3 zenith = rtEvaluateEnvironmentRadiance(background, tint, envIntensity, float3{0.0f, 1.0f, 0.0f});
  const float3 horizon = rtEvaluateEnvironmentRadiance(background, tint, envIntensity, float3{1.0f, 0.0f, 0.0f});
  const float3 ground = rtEvaluateEnvironmentRadiance(background, tint, envIntensity, float3{0.0f, -1.0f, 0.0f});

  const float zenithLum = rtEnvironmentLuminance(zenith);
  const float horizonLum = rtEnvironmentLuminance(horizon);
  const float groundLum = rtEnvironmentLuminance(ground);

  require(zenithLum > groundLum, "shared environment model should keep the sky brighter than the ground");
  require(horizonLum > groundLum, "shared environment model should keep the horizon above the ground bounce");
}

} // namespace

int main() {
  try {
    testEnvironmentPdfIsNormalized();
    testEnvironmentRadianceProfileIsShared();
    std::cout << "environment_model_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "environment_model_test failed: " << e.what() << std::endl;
    return 1;
  }
}
