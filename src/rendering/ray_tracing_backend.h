#pragma once

#include <memory>
#include <string>

#include "rendering/ray_tracing_types.h"

namespace rt {

class IRayTracingBackend {
public:
  virtual ~IRayTracingBackend() = default;

  virtual std::string name() const = 0;
  virtual void setScene(const RTScene& scene) = 0;
  virtual void updateCamera(const RTCamera& camera) = 0;
  virtual void resize(uint32_t width, uint32_t height) = 0;
  virtual void resetAccumulation() = 0;
  virtual void renderIteration(const RenderConfig& config) = 0;
  virtual RenderBuffer downloadRenderBuffer() const = 0;
};

struct MetalPostprocessTestInput {
  RenderMode renderMode = RenderMode::Toon;
  uint32_t width = 0;
  uint32_t height = 0;
  std::vector<glm::vec4> rawColor;
  std::vector<float> linearDepth;
  std::vector<glm::vec3> normal;
  std::vector<uint32_t> objectId;
  LightingSettings lighting;
  ToonSettings toon;
};

struct MetalPostprocessTestOutput {
  uint32_t width = 0;
  uint32_t height = 0;
  float minDepth = 0.0f;
  float maxDepth = 0.0f;
  std::vector<glm::vec4> tonemapped;
  std::vector<float> detailContour;
  std::vector<float> objectContour;
  std::vector<float> detailContourFxaa;
  std::vector<float> objectContourFxaa;
  std::vector<glm::vec3> finalColor;
};

class IMetalShaderTestHarness {
public:
  virtual ~IMetalShaderTestHarness() = default;
  virtual MetalPostprocessTestOutput runPostprocess(const MetalPostprocessTestInput& input) = 0;
};

std::unique_ptr<IRayTracingBackend> createMetalPathTracerBackend(const std::string& shaderLibraryPath = {});
std::unique_ptr<IMetalShaderTestHarness> createMetalShaderTestHarness(const std::string& shaderLibraryPath = {});

} // namespace rt
