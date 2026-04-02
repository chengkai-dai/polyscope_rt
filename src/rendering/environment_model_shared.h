#pragma once

#include "gpu_shared_types.h"

#ifdef __METAL_VERSION__
inline float rtEnvironmentClamp(float value, float lo, float hi) {
  return clamp(value, lo, hi);
}

inline float3 rtEnvironmentMax(float3 a, float3 b) {
  return max(a, b);
}
#else
#include <algorithm>
#include <cmath>

inline float rtEnvironmentClamp(float value, float lo, float hi) {
  return std::clamp(value, lo, hi);
}

inline float3 rtEnvironmentMax(float3 a, float3 b) {
  return simd_max(a, b);
}
#endif

#define RT_ENVIRONMENT_SAMPLE_WIDTH 64u
#define RT_ENVIRONMENT_SAMPLE_HEIGHT 32u

inline float3 rtLerp3(float3 a, float3 b, float t) {
  return a + (b - a) * t;
}

inline float rtEnvironmentDot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline float3 rtEnvironmentNormalize(float3 v) {
  float len2 = rtEnvironmentDot(v, v);
  if (len2 <= 1e-12f) return float3{0.0f, 1.0f, 0.0f};
#ifdef __METAL_VERSION__
  return v * rsqrt(len2);
#else
  return v * (1.0f / std::sqrt(len2));
#endif
}

inline float rtEnvironmentLuminance(float3 color) {
  return 0.2126f * color.x + 0.7152f * color.y + 0.0722f * color.z;
}

inline float3 rtEvaluateEnvironmentRadiance(float3 backgroundColor, float3 environmentTint,
                                            float environmentIntensity, float3 sceneUpDir, float3 dir) {
  float3 up = rtEnvironmentNormalize(sceneUpDir);
  float3 sampleDir = rtEnvironmentNormalize(dir);
  float y = rtEnvironmentClamp(rtEnvironmentDot(sampleDir, up), -1.0f, 1.0f);

  float3 horizonColor = backgroundColor;
  float3 zenithColor = rtLerp3(backgroundColor, backgroundColor * 0.82f, 0.55f);
  float3 groundColor = backgroundColor * 0.09f;

  float3 sky;
  if (y > 0.0f) {
    float t = pow(y, 0.29f);
    sky = rtLerp3(horizonColor, zenithColor, t);
  } else {
    float t = pow(rtEnvironmentClamp(-y, 0.0f, 1.0f), 0.92f);
    sky = rtLerp3(horizonColor * 0.62f, groundColor, t);
  }

  float hemi = rtEnvironmentClamp(y * 0.5f + 0.5f, 0.0f, 1.0f);
  float3 tintHorizon = backgroundColor * 0.48f;
  float3 envTint = rtLerp3(tintHorizon, environmentTint, hemi) * environmentIntensity;
  return rtEnvironmentMax(sky + envTint, float3{0.0f, 0.0f, 0.0f});
}
