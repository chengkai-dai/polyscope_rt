#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "glm/glm.hpp"

#include "rendering/ray_tracing_backend.h"
#include "test_helpers.h"

namespace {

bool isSkippableBackendError(const std::string& message) {
  return message.find("Metal is unavailable") != std::string::npos ||
         message.find("does not report ray tracing support") != std::string::npos;
}

size_t pixelIndex(uint32_t x, uint32_t y, uint32_t width) {
  return static_cast<size_t>(y) * width + x;
}

bool nearlyEqual(float a, float b, float eps = 1e-4f) {
  return std::abs(a - b) <= eps;
}

glm::vec3 gammaCorrection(const glm::vec3& color, float gamma) {
  return glm::pow(color, glm::vec3(1.0f / gamma));
}

glm::vec3 toneMapUncharted2Impl(const glm::vec3& color) {
  constexpr float A = 0.15f;
  constexpr float B = 0.50f;
  constexpr float C = 0.10f;
  constexpr float D = 0.20f;
  constexpr float E = 0.02f;
  constexpr float F = 0.30f;
  return ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
}

glm::vec3 toneMapUnchartedReference(const glm::vec3& color, float gamma, float exposure) {
  constexpr float W = 11.2f;
  constexpr float exposureBias = 2.0f;
  glm::vec3 mapped = toneMapUncharted2Impl(color * exposure * exposureBias);
  glm::vec3 whiteScale = glm::vec3(1.0f) / toneMapUncharted2Impl(glm::vec3(W));
  return gammaCorrection(mapped * whiteScale, gamma);
}

rt::MetalPostprocessTestInput makeInput(uint32_t width, uint32_t height) {
  const size_t count = static_cast<size_t>(width) * height;
  rt::MetalPostprocessTestInput input;
  input.renderMode = rt::RenderMode::Toon;
  input.width = width;
  input.height = height;
  input.rawColor.assign(count, glm::vec4(0.0f));
  input.linearDepth.assign(count, -1.0f);
  input.normal.assign(count, glm::vec3(0.0f, 0.0f, 1.0f));
  input.objectId.assign(count, 0u);
  input.lighting.backgroundColor = glm::vec3(1.0f);
  input.toon.edgeColor = glm::vec3(0.3f);
  input.toon.depthThreshold = 1.0f;
  input.toon.normalThreshold = 0.5f;
  input.toon.enableDetailContour = true;
  input.toon.enableObjectContour = true;
  input.toon.enableNormalEdge = true;
  input.toon.enableDepthEdge = true;
  input.toon.detailContourStrength = 1.0f;
  input.toon.objectContourStrength = 1.0f;
  input.toon.edgeThickness = 1.0f;
  input.toon.objectThreshold = 1.0f;
  input.toon.useFxaa = true;
  input.toon.tonemapExposure = 3.0f;
  input.toon.tonemapGamma = 2.2f;
  return input;
}

void testTonemapMatchesReference(rt::IMetalShaderTestHarness& harness) {
  auto input = makeInput(1, 1);
  input.rawColor[0] = glm::vec4(0.18f, 0.18f, 0.18f, 1.0f);
  input.linearDepth[0] = 2.0f;
  input.objectId[0] = 1u;

  const auto output = harness.runPostprocess(input);
  const glm::vec3 expected = toneMapUnchartedReference(glm::vec3(0.18f), input.toon.tonemapGamma, input.toon.tonemapExposure);
  const glm::vec3 got = glm::vec3(output.tonemapped[0]);

  require(glm::length(got - expected) < 1e-3f, "tonemap kernel diverged from reference formula");
}

void testStandardTonemapUsesLightingControls(rt::IMetalShaderTestHarness& harness) {
  auto input = makeInput(1, 1);
  input.renderMode = rt::RenderMode::Standard;
  input.rawColor[0] = glm::vec4(0.3f, 0.2f, 0.1f, 1.0f);
  input.linearDepth[0] = 2.0f;
  input.objectId[0] = 1u;
  input.toon.tonemapExposure = 1.0f;
  input.toon.tonemapGamma = 1.0f;
  input.lighting.standardExposure = 4.0f;
  input.lighting.standardGamma = 2.2f;
  input.lighting.standardSaturation = 1.35f;

  const auto output = harness.runPostprocess(input);
  const glm::vec3 expectedBase =
      toneMapUnchartedReference(glm::vec3(0.3f, 0.2f, 0.1f), input.lighting.standardGamma, input.lighting.standardExposure);
  const float luma = glm::dot(expectedBase, glm::vec3(0.299f, 0.587f, 0.114f));
  const glm::vec3 expected = glm::mix(glm::vec3(luma), expectedBase, input.lighting.standardSaturation);
  const glm::vec3 got = glm::vec3(output.tonemapped[0]);

  require(glm::length(got - expected) < 1e-3f, "standard tonemap should use lighting exposure/gamma/saturation");
}

void testDepthMinMaxIgnoresBackground(rt::IMetalShaderTestHarness& harness) {
  auto input = makeInput(4, 4);
  input.objectId[pixelIndex(1, 1, input.width)] = 1u;
  input.objectId[pixelIndex(2, 2, input.width)] = 1u;
  input.linearDepth[pixelIndex(1, 1, input.width)] = 2.5f;
  input.linearDepth[pixelIndex(2, 2, input.width)] = 6.5f;

  const auto output = harness.runPostprocess(input);
  require(nearlyEqual(output.minDepth, 2.5f, 1e-4f), "depth min/max kernel computed the wrong minimum");
  require(nearlyEqual(output.maxDepth, 6.5f, 1e-4f), "depth min/max kernel computed the wrong maximum");
}

void testObjectContourDetectsBoundary(rt::IMetalShaderTestHarness& harness) {
  auto input = makeInput(5, 5);
  for (uint32_t y = 0; y < input.height; ++y) {
    for (uint32_t x = 0; x < input.width; ++x) {
      size_t idx = pixelIndex(x, y, input.width);
      input.objectId[idx] = x < 2 ? 1u : 2u;
      input.linearDepth[idx] = 2.0f;
      input.rawColor[idx] = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);
    }
  }

  const auto output = harness.runPostprocess(input);
  require(output.objectContour[pixelIndex(0, 0, input.width)] < 0.01f, "object contour should stay dark in a flat region");
  require(output.objectContour[pixelIndex(1, 2, input.width)] > 0.9f, "object contour should light up on object boundaries");
}

void testDetailContourDetectsNormalDiscontinuity(rt::IMetalShaderTestHarness& harness) {
  auto input = makeInput(5, 5);
  for (uint32_t y = 0; y < input.height; ++y) {
    for (uint32_t x = 0; x < input.width; ++x) {
      size_t idx = pixelIndex(x, y, input.width);
      input.objectId[idx] = 1u;
      input.linearDepth[idx] = 2.0f;
      input.rawColor[idx] = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
      input.normal[idx] = x < 2 ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f));
    }
  }
  input.toon.normalThreshold = 1.0f;
  input.toon.depthThreshold = 1.0f;

  const auto output = harness.runPostprocess(input);
  require(output.detailContour[pixelIndex(0, 0, input.width)] < 0.01f, "detail contour should stay dark in smooth regions");
  require(output.detailContour[pixelIndex(1, 2, input.width)] > 0.2f,
          "detail contour should detect the normal discontinuity");
}

void testDepthContourStaysLowOnFlatSurface(rt::IMetalShaderTestHarness& harness) {
  auto input = makeInput(5, 5);
  input.toon.enableNormalEdge = false;
  input.toon.enableDepthEdge = true;
  input.toon.depthThreshold = 1.0f;
  for (uint32_t y = 0; y < input.height; ++y) {
    for (uint32_t x = 0; x < input.width; ++x) {
      size_t idx = pixelIndex(x, y, input.width);
      input.objectId[idx] = 1u;
      input.linearDepth[idx] = 4.0f;
      input.rawColor[idx] = glm::vec4(0.4f, 0.4f, 0.4f, 1.0f);
      input.normal[idx] = glm::vec3(0.0f, 0.0f, 1.0f);
    }
  }

  const auto output = harness.runPostprocess(input);
  require(output.detailContour[pixelIndex(2, 2, input.width)] < 0.01f,
          "depth contour should stay dark on a flat constant-depth surface");
}

void testDepthContourDoesNotWashOutSmoothSurface(rt::IMetalShaderTestHarness& harness) {
  auto input = makeInput(17, 17);
  input.toon.enableNormalEdge = false;
  input.toon.enableDepthEdge = true;
  input.toon.useFxaa = true;
  input.toon.depthThreshold = 0.015f;
  for (uint32_t y = 0; y < input.height; ++y) {
    for (uint32_t x = 0; x < input.width; ++x) {
      size_t idx = pixelIndex(x, y, input.width);
      float fx = (static_cast<float>(x) - 8.0f) / 8.0f;
      float fy = (static_cast<float>(y) - 8.0f) / 8.0f;
      input.objectId[idx] = 1u;
      input.linearDepth[idx] = 3.5f + 0.35f * (fx * fx + fy * fy);
      input.rawColor[idx] = glm::vec4(0.6f, 0.6f, 0.6f, 1.0f);
      input.normal[idx] = glm::vec3(0.0f, 0.0f, 1.0f);
    }
  }

  const auto output = harness.runPostprocess(input);
  double rawMean = 0.0;
  double fxaaMean = 0.0;
  for (float v : output.detailContour) rawMean += v;
  for (float v : output.detailContourFxaa) fxaaMean += v;
  rawMean /= static_cast<double>(output.detailContour.size());
  fxaaMean /= static_cast<double>(output.detailContourFxaa.size());

  require(rawMean < 0.35, "raw depth contour should not wash out a smooth depth surface");
  require(fxaaMean < 0.45, "FXAA depth contour should not wash out a smooth depth surface");
}

void testFxaaAndCompositeBackground(rt::IMetalShaderTestHarness& harness) {
  auto input = makeInput(8, 8);
  input.lighting.backgroundColor = glm::vec3(0.9f, 0.95f, 1.0f);
  for (uint32_t y = 0; y < input.height; ++y) {
    for (uint32_t x = 0; x < input.width; ++x) {
      size_t idx = pixelIndex(x, y, input.width);
      if (x >= 3 && x <= 4) {
        input.objectId[idx] = 1u;
        input.linearDepth[idx] = 3.0f;
        input.rawColor[idx] = glm::vec4(0.2f, 0.2f, 0.2f, 1.0f);
      } else {
        input.rawColor[idx] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
      }
    }
  }

  const auto output = harness.runPostprocess(input);
  const glm::vec3 bg = output.finalColor[pixelIndex(0, 0, input.width)];
  require(glm::length(bg - input.lighting.backgroundColor) < 1e-3f,
          "composite kernel should restore the configured background color");

  bool foundIntermediateFxaa = false;
  for (float value : output.objectContourFxaa) {
    if (value > 0.05f && value < 0.95f) {
      foundIntermediateFxaa = true;
      break;
    }
  }
  require(foundIntermediateFxaa, "FXAA should introduce intermediate contour coverage values");
}

} // namespace

int main() {
  try {
    auto harness = rt::createMetalShaderTestHarness();
  testTonemapMatchesReference(*harness);
  testStandardTonemapUsesLightingControls(*harness);
  testDepthMinMaxIgnoresBackground(*harness);
    testObjectContourDetectsBoundary(*harness);
    testDetailContourDetectsNormalDiscontinuity(*harness);
    testDepthContourStaysLowOnFlatSurface(*harness);
    testDepthContourDoesNotWashOutSmoothSurface(*harness);
    testFxaaAndCompositeBackground(*harness);
    std::cout << "metal_shader_unit_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    const std::string message = e.what();
    if (isSkippableBackendError(message)) {
      std::cout << "metal_shader_unit_test skipped: " << message << std::endl;
      return 0;
    }
    std::cerr << "metal_shader_unit_test failed: " << message << std::endl;
    return 1;
  }
}
