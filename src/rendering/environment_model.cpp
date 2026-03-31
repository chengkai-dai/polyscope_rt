#include "rendering/environment_model.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "rendering/environment_model_shared.h"

namespace {

constexpr float kPi = 3.14159265358979323846f;

} // namespace

namespace rt {

std::vector<GPUEnvironmentSampleCell> buildEnvironmentSampleCells(const LightingSettings& lighting) {
  const uint32_t sampleWidth = RT_ENVIRONMENT_SAMPLE_WIDTH;
  const uint32_t sampleHeight = RT_ENVIRONMENT_SAMPLE_HEIGHT;
  const uint32_t cellCount = sampleWidth * sampleHeight;

  std::vector<GPUEnvironmentSampleCell> cells(cellCount);
  std::vector<float> weights(cellCount, 0.0f);
  double totalWeight = 0.0;

  float3 backgroundColor = float3{lighting.backgroundColor.x, lighting.backgroundColor.y, lighting.backgroundColor.z};
  float3 environmentTint = float3{lighting.environmentTint.x, lighting.environmentTint.y, lighting.environmentTint.z};
  float environmentIntensity = std::max(lighting.environmentIntensity, 0.0f);

  for (uint32_t row = 0; row < sampleHeight; ++row) {
    const float theta0 = kPi * static_cast<float>(row) / static_cast<float>(sampleHeight);
    const float theta1 = kPi * static_cast<float>(row + 1u) / static_cast<float>(sampleHeight);
    const float theta = 0.5f * (theta0 + theta1);
    const float sinTheta = std::sin(theta);
    const float cosTheta = std::cos(theta);
    const float solidAngle =
        (2.0f * kPi / static_cast<float>(sampleWidth)) *
        std::max(std::cos(theta0) - std::cos(theta1), 1e-6f);

    for (uint32_t col = 0; col < sampleWidth; ++col) {
      const uint32_t idx = row * sampleWidth + col;
      const float phi = 2.0f * kPi * (static_cast<float>(col) + 0.5f) / static_cast<float>(sampleWidth);
      const float3 dir = float3{std::cos(phi) * sinTheta, cosTheta, std::sin(phi) * sinTheta};
      const float3 radiance = rtEvaluateEnvironmentRadiance(backgroundColor, environmentTint, environmentIntensity, dir);
      const float weight = std::max(rtEnvironmentLuminance(radiance), 0.0f) * solidAngle;
      weights[idx] = weight;
      totalWeight += static_cast<double>(weight);
    }
  }

  if (totalWeight <= 1e-8) {
    for (uint32_t idx = 0; idx < cellCount; ++idx) {
      cells[idx].data = simd_make_float4(0.0f, idx == 0u ? 1.0f : 0.0f, 0.0f, 0.0f);
    }
    return cells;
  }

  float cumulative = 0.0f;
  for (uint32_t idx = 0; idx < cellCount; ++idx) {
    const float pmf = static_cast<float>(weights[idx] / totalWeight);
    cumulative += pmf;
    cells[idx].data = simd_make_float4(pmf, cumulative, 0.0f, 0.0f);
  }
  cells.back().data.y = 1.0f;
  return cells;
}

} // namespace rt
