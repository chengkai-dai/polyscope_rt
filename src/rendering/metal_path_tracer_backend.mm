#include "rendering/ray_tracing_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/QuartzCore.h>
#import <mach-o/dyld.h>
#import <simd/simd.h>

#include "glm/glm.hpp"
#include "glm/gtc/matrix_inverse.hpp"

namespace {

constexpr const char* kShaderLibraryEnvVar = "POLYSCOPE_RT_SHADER_LIBRARY";
constexpr const char* kLegacyShaderLibraryEnvVar = "POLYSCOPE_RT_METALLIB_PATH";

std::filesystem::path executableDirectory() {
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  if (size == 0) return {};

  std::string buffer(size, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) return {};

  std::error_code ec;
  std::filesystem::path executablePath = std::filesystem::weakly_canonical(std::filesystem::path(buffer.c_str()), ec);
  if (ec) executablePath = std::filesystem::path(buffer.c_str());
  return executablePath.parent_path();
}

std::vector<std::filesystem::path> shaderLibraryCandidates(const std::string& requestedPath) {
  std::vector<std::filesystem::path> candidates;

  auto appendCandidate = [&candidates](const std::filesystem::path& path) {
    if (path.empty()) return;
    if (std::find(candidates.begin(), candidates.end(), path) == candidates.end()) {
      candidates.push_back(path);
    }
  };

  if (!requestedPath.empty()) {
    appendCandidate(requestedPath);
  }
  if (const char* envPath = std::getenv(kShaderLibraryEnvVar)) {
    appendCandidate(envPath);
  }
  if (const char* envPath = std::getenv(kLegacyShaderLibraryEnvVar)) {
    appendCandidate(envPath);
  }

  const std::filesystem::path executableDir = executableDirectory();
  if (!executableDir.empty()) {
    appendCandidate(executableDir / POLYSCOPE_RT_METALLIB_FILENAME);
    appendCandidate(executableDir / "polyscope_rt" / POLYSCOPE_RT_METALLIB_FILENAME);
    appendCandidate(executableDir / ".." / "lib" / "polyscope_rt" / POLYSCOPE_RT_METALLIB_FILENAME);
    appendCandidate(executableDir / ".." / "share" / "polyscope_rt" / POLYSCOPE_RT_METALLIB_FILENAME);
  }

  appendCandidate(POLYSCOPE_RT_BUILD_METALLIB_PATH);
  return candidates;
}

std::string resolveShaderLibraryPath(const std::string& requestedPath) {
  std::error_code ec;
  std::vector<std::filesystem::path> candidates = shaderLibraryCandidates(requestedPath);
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate, ec) && !ec) {
      return std::filesystem::absolute(candidate, ec).string();
    }
    ec.clear();
  }

  std::string message = "Failed to locate Metal shader library. Checked:";
  for (const auto& candidate : candidates) {
    message += "\n - " + candidate.string();
  }
  throw std::runtime_error(message);
}

struct CameraUniforms {
  simd_float4 position;
  simd_float4 lookDir;
  simd_float4 upDir;
  simd_float4 rightDir;
  simd_float4 clipData;
  simd_float4x4 viewMatrix;
  simd_float4x4 projectionMatrix;
};

struct FrameUniforms {
  uint32_t renderMode = 1;
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t samplesPerIteration = 1;
  uint32_t frameIndex = 0;
  uint32_t maxBounces = 2;
  uint32_t lightCount = 0;
  uint32_t enableSceneLights = 1;  // always enabled; controlled per-frame via lightCount
  uint32_t enableAreaLight = 0;
  uint32_t toonBandCount = 5;
  float ambientFloor = 0.1f;
  uint32_t rngFrameIndex = 0;
  simd_float4 planeColorEnabled;
  simd_float4 planeParams;
  simd_float2 jitterOffset = simd_make_float2(0.0f, 0.0f);
  simd_float2 _pad1 = simd_make_float2(0.0f, 0.0f);
  simd_float4x4 prevViewProj;
};

struct LightingUniforms {
  simd_float4 backgroundColor;
  simd_float4 mainLightDirection;
  simd_float4 mainLightColorIntensity;
  simd_float4 environmentTintIntensity;
  simd_float4 areaLightCenterEnabled;
  simd_float4 areaLightU;
  simd_float4 areaLightV;
  simd_float4 areaLightEmission;
};

struct ToonShaderUniforms {
  uint32_t width = 1;
  uint32_t height = 1;
  uint32_t contourMethod = 2;
  uint32_t useFxaa = 1;
  float detailContourStrength = 1.0f;
  float depthThreshold = 1.0f;
  float normalThreshold = 0.5f;
  float edgeThickness = 1.0f;
  float exposure = 3.0f;
  float gamma = 2.2f;
  float saturation = 1.0f;
  float objectContourStrength = 1.0f;
  float objectThreshold = 1.0f;
  uint32_t enableDetailContour = 1u;
  uint32_t enableObjectContour = 1u;
  uint32_t enableNormalEdge = 1u;
  uint32_t enableDepthEdge = 1u;
  simd_float4 backgroundColor;
  simd_float4 edgeColor;
};

struct PackedTriangleIndices {
  uint32_t i0;
  uint32_t i1;
  uint32_t i2;
};

struct TriangleShaderData {
  simd_uint4 indicesMaterial;
  simd_uint4 objectFlags;
};

struct MaterialShaderData {
  simd_float4 baseColorFactor;
  simd_uint4 baseColorTextureData;
  simd_float4 metallicRoughnessNormal;
  simd_uint4 metallicRoughnessTextureData;
  simd_float4 emissiveFactor;
  simd_uint4 emissiveTextureData;
  simd_uint4 normalTextureData;
  simd_float4 transmissionIor;
};

struct TextureShaderData {
  simd_uint4 data;
};

struct PunctualLightShaderData {
  simd_float4 positionRange;
  simd_float4 directionType;
  simd_float4 colorIntensity;
  simd_float4 spotAngles;
};

struct CurvePrimitiveShaderData {
  simd_float4 p0_radius;
  simd_float4 p1_type;
  simd_uint4 materialObjectId;
};

simd_float4 makeFloat4(const glm::vec3& v, float w = 0.0f) { return simd_make_float4(v.x, v.y, v.z, w); }

std::string buildNSErrorMessage(NSError* error) {
  if (error == nil) return "unknown error";

  std::string message;
  auto appendPart = [&](NSString* value) {
    if (value == nil || value.length == 0) return;
    if (!message.empty()) message += " | ";
    message += value.UTF8String;
  };

  appendPart(error.localizedDescription);
  appendPart(error.localizedFailureReason);
  appendPart(error.localizedRecoverySuggestion);

  for (id value in error.userInfo.allValues) {
    if ([value isKindOfClass:[NSError class]]) {
      NSError* nested = (NSError*)value;
      appendPart(nested.localizedDescription);
      appendPart(nested.localizedFailureReason);
      continue;
    }
    if ([value isKindOfClass:[NSArray class]]) {
      for (id nestedValue in (NSArray*)value) {
        if (![nestedValue isKindOfClass:[NSError class]]) continue;
        NSError* nested = (NSError*)nestedValue;
        appendPart(nested.localizedDescription);
        appendPart(nested.localizedFailureReason);
      }
    }
  }

  return message.empty() ? "unknown error" : message;
}

std::string contourObjectKey(const std::string& meshName) {
  constexpr const char* primitiveTag = "/primitive_";
  const size_t pos = meshName.rfind(primitiveTag);
  if (pos == std::string::npos) return meshName;
  return meshName.substr(0, pos);
}

float haltonSequence(uint32_t index, uint32_t base) {
  float f = 1.0f, r = 0.0f;
  float invBase = 1.0f / static_cast<float>(base);
  while (index > 0) {
    f *= invBase;
    r += f * static_cast<float>(index % base);
    index /= base;
  }
  return r;
}

simd_float4x4 makeFloat4x4(const glm::mat4& m) {
  simd_float4x4 result;
  result.columns[0] = simd_make_float4(m[0][0], m[0][1], m[0][2], m[0][3]);
  result.columns[1] = simd_make_float4(m[1][0], m[1][1], m[1][2], m[1][3]);
  result.columns[2] = simd_make_float4(m[2][0], m[2][1], m[2][2], m[2][3]);
  result.columns[3] = simd_make_float4(m[3][0], m[3][1], m[3][2], m[3][3]);
  return result;
}

class MetalPathTracerBackend final : public rt::IRayTracingBackend {
public:
  explicit MetalPathTracerBackend(const std::string& shaderLibraryPath) {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (device_ == nil) {
        throw std::runtime_error("Metal is unavailable on this machine.");
      }
      if (@available(macOS 11.0, *)) {
        if (![device_ supportsRaytracing]) {
          throw std::runtime_error("This Metal device does not report ray tracing support.");
        }
      }

      commandQueue_ = [device_ newCommandQueue];
      if (commandQueue_ == nil) {
        throw std::runtime_error("Failed to create the Metal command queue.");
      }

      NSError* error = nil;
      std::string resolvedShaderLibraryPath = resolveShaderLibraryPath(shaderLibraryPath);
      NSString* metallibPath = [NSString stringWithUTF8String:resolvedShaderLibraryPath.c_str()];
      library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:metallibPath] error:&error];
      if (library_ == nil) {
        throw std::runtime_error(std::string("Failed to load Metal shader library: ") + buildNSErrorMessage(error));
      }

      id<MTLFunction> pathTraceKernel = [library_ newFunctionWithName:@"pathTraceKernel"];

      MTLLinkedFunctions* linkedFunctions = [[MTLLinkedFunctions alloc] init];

      MTLComputePipelineDescriptor* pipelineDesc = [[MTLComputePipelineDescriptor alloc] init];
      pipelineDesc.computeFunction = pathTraceKernel;
      pipelineDesc.linkedFunctions = linkedFunctions;

      pathTracePipelineState_ = [device_ newComputePipelineStateWithDescriptor:pipelineDesc
                                                                       options:0
                                                                    reflection:nil
                                                                         error:&error];
      if (pathTracePipelineState_ == nil) {
        throw std::runtime_error(std::string("Failed to create the Metal compute pipeline: ") +
                                 buildNSErrorMessage(error));
      }

      {
        MTLIntersectionFunctionTableDescriptor* ftDesc = [[MTLIntersectionFunctionTableDescriptor alloc] init];
        ftDesc.functionCount = 0;
        intersectionFunctionTable_ = [pathTracePipelineState_ newIntersectionFunctionTableWithDescriptor:ftDesc];
      }
      id<MTLFunction> tonemapKernel = [library_ newFunctionWithName:@"tonemapKernel"];
      tonemapPipelineState_ = [device_ newComputePipelineStateWithFunction:tonemapKernel error:&error];
      if (tonemapPipelineState_ == nil) {
        throw std::runtime_error(std::string("Failed to create the Metal tonemap pipeline: ") +
                                 buildNSErrorMessage(error));
      }
      id<MTLFunction> objectContourKernel = [library_ newFunctionWithName:@"objectContourKernel"];
      objectContourPipelineState_ = [device_ newComputePipelineStateWithFunction:objectContourKernel error:&error];
      if (objectContourPipelineState_ == nil) {
        throw std::runtime_error(std::string("Failed to create the Metal object contour pipeline: ") +
                                 buildNSErrorMessage(error));
      }
      id<MTLFunction> detailContourKernel = [library_ newFunctionWithName:@"detailContourKernel"];
      detailContourPipelineState_ = [device_ newComputePipelineStateWithFunction:detailContourKernel error:&error];
      if (detailContourPipelineState_ == nil) {
        throw std::runtime_error(std::string("Failed to create the Metal detail contour pipeline: ") +
                                 buildNSErrorMessage(error));
      }
      id<MTLFunction> depthMinMaxKernel = [library_ newFunctionWithName:@"depthMinMaxKernel"];
      depthMinMaxPipelineState_ = [device_ newComputePipelineStateWithFunction:depthMinMaxKernel error:&error];
      if (depthMinMaxPipelineState_ == nil) {
        throw std::runtime_error(std::string("Failed to create the Metal depth min/max pipeline: ") +
                                 buildNSErrorMessage(error));
      }
      id<MTLFunction> fxaaKernel = [library_ newFunctionWithName:@"fxaaKernel"];
      fxaaPipelineState_ = [device_ newComputePipelineStateWithFunction:fxaaKernel error:&error];
      if (fxaaPipelineState_ == nil) {
        throw std::runtime_error(std::string("Failed to create the Metal FXAA pipeline: ") +
                                 buildNSErrorMessage(error));
      }
      id<MTLFunction> compositeKernel = [library_ newFunctionWithName:@"compositeKernel"];
      compositePipelineState_ = [device_ newComputePipelineStateWithFunction:compositeKernel error:&error];
      if (compositePipelineState_ == nil) {
        throw std::runtime_error(std::string("Failed to create the Metal compositing pipeline: ") +
                                 buildNSErrorMessage(error));
      }

      auto tryCreatePipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [library_ newFunctionWithName:name];
        if (fn == nil) return nil;
        return [device_ newComputePipelineStateWithFunction:fn error:&error];
      };
      bufferToTexturePipelineState_ = tryCreatePipeline(@"bufferToTextureKernel");
      textureToBufferPipelineState_ = tryCreatePipeline(@"textureToBufferKernel");
      depthToTexturePipelineState_ = tryCreatePipeline(@"depthToTextureKernel");
      roughnessToTexturePipelineState_ = tryCreatePipeline(@"roughnessToTextureKernel");
      motionToTexturePipelineState_ = tryCreatePipeline(@"motionToTextureKernel");

      cameraBuffer_ = [device_ newBufferWithLength:sizeof(CameraUniforms) options:MTLResourceStorageModeShared];
      frameBuffer_ = [device_ newBufferWithLength:sizeof(FrameUniforms) options:MTLResourceStorageModeShared];
      lightingBuffer_ = [device_ newBufferWithLength:sizeof(LightingUniforms) options:MTLResourceStorageModeShared];
      toonBuffer_ = [device_ newBufferWithLength:sizeof(ToonShaderUniforms) options:MTLResourceStorageModeShared];
    }
  }

  std::string name() const override { return "Metal Path Tracer"; }

  void setScene(const rt::RTScene& scene) override {
    scene_ = scene;
    buildSceneBuffers();
    resetAccumulation();
  }

  void updateCamera(const rt::RTCamera& camera) override {
    camera_ = camera;

    CameraUniforms uniforms;
    uniforms.position = makeFloat4(camera.position, 1.0f);
    uniforms.lookDir = makeFloat4(glm::normalize(camera.lookDir));
    uniforms.upDir = makeFloat4(glm::normalize(camera.upDir));
    uniforms.rightDir = makeFloat4(glm::normalize(camera.rightDir));
    uniforms.clipData = simd_make_float4(glm::radians(camera.fovYDegrees), camera.aspect, camera.nearClip, camera.farClip);
    uniforms.viewMatrix = makeFloat4x4(camera.viewMatrix);
    uniforms.projectionMatrix = makeFloat4x4(camera.projectionMatrix);

    std::memcpy(cameraBuffer_.contents, &uniforms, sizeof(CameraUniforms));
  }

  void resize(uint32_t width, uint32_t height) override {
    width_ = std::max<uint32_t>(1, width);
    height_ = std::max<uint32_t>(1, height);

    const NSUInteger pixelCount = static_cast<NSUInteger>(width_) * static_cast<NSUInteger>(height_);
    accumulationBuffer_ =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    rawColorBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    tonemappedBuffer_ =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    detailContourBuffer_ =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    objectContourBuffer_ =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    detailContourFxaaBuffer_ =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    objectContourFxaaBuffer_ =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    depthMinMaxBuffer_ = [device_ newBufferWithLength:2 * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    outputBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    depthBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
    linearDepthBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
    normalBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    objectIdBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    diffuseAlbedoBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    specularAlbedoBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    roughnessAuxBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
    motionVectorBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    latestBuffer_.width = width_;
    latestBuffer_.height = height_;
    latestBuffer_.color.assign(pixelCount, glm::vec3(0.0f));
    latestBuffer_.depth.assign(pixelCount, 1.0f);
    latestBuffer_.linearDepth.assign(pixelCount, -1.0f);
    latestBuffer_.normal.assign(pixelCount, glm::vec3(0.0f, 1.0f, 0.0f));
    latestBuffer_.objectId.assign(pixelCount, 0u);
    latestBuffer_.detailContour.assign(pixelCount, 0.0f);
    latestBuffer_.detailContourRaw.assign(pixelCount, 0.0f);
    latestBuffer_.objectContour.assign(pixelCount, 0.0f);
    latestBuffer_.objectContourRaw.assign(pixelCount, 0.0f);
    resetAccumulation();
  }

  void resetAccumulation() override {
    frameIndex_ = 0;
    latestBuffer_.accumulatedSamples = 0;

    if (accumulationBuffer_ != nil) {
      std::memset(accumulationBuffer_.contents, 0, accumulationBuffer_.length);
    }
    if (rawColorBuffer_ != nil) {
      std::memset(rawColorBuffer_.contents, 0, rawColorBuffer_.length);
    }
    if (tonemappedBuffer_ != nil) {
      std::memset(tonemappedBuffer_.contents, 0, tonemappedBuffer_.length);
    }
    if (detailContourBuffer_ != nil) {
      std::memset(detailContourBuffer_.contents, 0, detailContourBuffer_.length);
    }
    if (objectContourBuffer_ != nil) {
      std::memset(objectContourBuffer_.contents, 0, objectContourBuffer_.length);
    }
    if (detailContourFxaaBuffer_ != nil) {
      std::memset(detailContourFxaaBuffer_.contents, 0, detailContourFxaaBuffer_.length);
    }
    if (objectContourFxaaBuffer_ != nil) {
      std::memset(objectContourFxaaBuffer_.contents, 0, objectContourFxaaBuffer_.length);
    }
    if (outputBuffer_ != nil) {
      std::memset(outputBuffer_.contents, 0, outputBuffer_.length);
    }
    if (depthBuffer_ != nil) {
      float* depth = static_cast<float*>(depthBuffer_.contents);
      const size_t count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
      std::fill(depth, depth + count, 1.0f);
    }
    if (linearDepthBuffer_ != nil) {
      float* linearDepth = static_cast<float*>(linearDepthBuffer_.contents);
      const size_t count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
      std::fill(linearDepth, linearDepth + count, -1.0f);
    }
    if (normalBuffer_ != nil) {
      simd_float4* normal = static_cast<simd_float4*>(normalBuffer_.contents);
      const size_t count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
      std::fill(normal, normal + count, simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f));
    }
    if (objectIdBuffer_ != nil) {
      uint32_t* objectId = static_cast<uint32_t*>(objectIdBuffer_.contents);
      const size_t count = static_cast<size_t>(width_) * static_cast<size_t>(height_);
      std::fill(objectId, objectId + count, 0u);
    }
    if (diffuseAlbedoBuffer_ != nil) std::memset(diffuseAlbedoBuffer_.contents, 0, diffuseAlbedoBuffer_.length);
    if (specularAlbedoBuffer_ != nil) std::memset(specularAlbedoBuffer_.contents, 0, specularAlbedoBuffer_.length);
    if (roughnessAuxBuffer_ != nil) std::memset(roughnessAuxBuffer_.contents, 0, roughnessAuxBuffer_.length);
    if (motionVectorBuffer_ != nil) std::memset(motionVectorBuffer_.contents, 0, motionVectorBuffer_.length);
    hasPrevViewProj_ = false;
  }

  void renderIteration(const rt::RenderConfig& config) override {
    if (width_ == 0 || height_ == 0) {
      throw std::runtime_error("The render target size is invalid.");
    }
    if (scene_.meshes.empty() && scene_.curveNetworks.empty()) {
      throw std::runtime_error("No geometry was provided to the Metal ray tracer.");
    }
    if (pathTracePipelineState_ == nil || tonemapPipelineState_ == nil || objectContourPipelineState_ == nil ||
        detailContourPipelineState_ == nil || depthMinMaxPipelineState_ == nil || fxaaPipelineState_ == nil ||
        compositePipelineState_ == nil || accelerationStructure_ == nil) {
      throw std::runtime_error("The Metal ray tracing pipeline is not initialized.");
    }

    FrameUniforms frame;
    frame.renderMode = static_cast<uint32_t>(config.renderMode);
    frame.width = width_;
    frame.height = height_;
    frame.samplesPerIteration = std::max<uint32_t>(1, config.samplesPerIteration);
    frame.frameIndex = (config.enableMetalFX && !config.accumulate) ? 0u : frameIndex_;
    frame.rngFrameIndex = frameIndex_;
    frame.maxBounces = std::max<uint32_t>(1, config.maxBounces);
    if (sceneContainsTransmission()) {
      frame.maxBounces = std::max<uint32_t>(frame.maxBounces, 4u);
    }
    frame.lightCount = static_cast<uint32_t>(scene_.lights.size());
    frame.enableAreaLight = config.lighting.enableAreaLight ? 1u : 0u;
    frame.toonBandCount = static_cast<uint32_t>(std::max(0, config.lighting.toonBandCount));
    frame.ambientFloor = std::max(0.0f, config.lighting.ambientFloor);
    frame.planeColorEnabled = makeFloat4(config.groundPlane.color, 1.0f);
    frame.planeParams = simd_make_float4(config.groundPlane.height, config.groundPlane.metallic, config.groundPlane.roughness, config.groundPlane.reflectance);
    if (config.enableMetalFX) {
      float jx = haltonSequence(frameIndex_ + 1, 2) - 0.5f;
      float jy = haltonSequence(frameIndex_ + 1, 3) - 0.5f;
      frame.jitterOffset = simd_make_float2(jx, jy);
    }
    if (hasPrevViewProj_) {
      frame.prevViewProj = prevViewProj_;
    } else {
      CameraUniforms* cam = static_cast<CameraUniforms*>(cameraBuffer_.contents);
      frame.prevViewProj = simd_mul(cam->projectionMatrix, cam->viewMatrix);
    }
    std::memcpy(frameBuffer_.contents, &frame, sizeof(FrameUniforms));

    LightingUniforms lighting;
    lighting.backgroundColor = makeFloat4(config.lighting.backgroundColor, 1.0f);
    lighting.mainLightDirection = makeFloat4(glm::normalize(config.lighting.mainLightDirection), 0.0f);
    lighting.mainLightColorIntensity =
        makeFloat4(config.lighting.mainLightColor, std::max(config.lighting.mainLightIntensity, 0.0f));
    lighting.environmentTintIntensity = makeFloat4(config.lighting.environmentTint, config.lighting.environmentIntensity);
    lighting.areaLightCenterEnabled = makeFloat4(config.lighting.areaLightCenter, config.lighting.enableAreaLight ? 1.0f : 0.0f);
    lighting.areaLightU = makeFloat4(config.lighting.areaLightU, 0.0f);
    lighting.areaLightV = makeFloat4(config.lighting.areaLightV, 0.0f);
    lighting.areaLightEmission = makeFloat4(config.lighting.areaLightEmission, 0.0f);
    std::memcpy(lightingBuffer_.contents, &lighting, sizeof(LightingUniforms));

    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

      [encoder setComputePipelineState:pathTracePipelineState_];
      [encoder setBuffer:positionBuffer_ offset:0 atIndex:0];
      [encoder setBuffer:normalVertexBuffer_ offset:0 atIndex:1];
      [encoder setBuffer:texcoordBuffer_ offset:0 atIndex:2];
      [encoder setBuffer:triangleBuffer_ offset:0 atIndex:3];
      [encoder setBuffer:materialBuffer_ offset:0 atIndex:4];
      [encoder setBuffer:textureMetadataBuffer_ offset:0 atIndex:5];
      [encoder setBuffer:texturePixelBuffer_ offset:0 atIndex:6];
      [encoder setBuffer:lightBuffer_ offset:0 atIndex:7];
      [encoder setBuffer:cameraBuffer_ offset:0 atIndex:8];
      [encoder setBuffer:frameBuffer_ offset:0 atIndex:9];
      [encoder setBuffer:lightingBuffer_ offset:0 atIndex:10];
      [encoder setBuffer:accumulationBuffer_ offset:0 atIndex:11];
      [encoder setBuffer:rawColorBuffer_ offset:0 atIndex:12];
      [encoder setBuffer:depthBuffer_ offset:0 atIndex:13];
      [encoder setBuffer:linearDepthBuffer_ offset:0 atIndex:14];
      [encoder setBuffer:normalBuffer_ offset:0 atIndex:15];
      [encoder setBuffer:objectIdBuffer_ offset:0 atIndex:16];
      [encoder setAccelerationStructure:accelerationStructure_ atBufferIndex:17];
      [encoder useResource:accelerationStructure_ usage:MTLResourceUsageRead];
      [encoder setBuffer:diffuseAlbedoBuffer_ offset:0 atIndex:18];
      [encoder setBuffer:specularAlbedoBuffer_ offset:0 atIndex:19];
      [encoder setBuffer:roughnessAuxBuffer_ offset:0 atIndex:20];
      [encoder setBuffer:motionVectorBuffer_ offset:0 atIndex:21];
      [encoder setBuffer:vertexColorBuffer_ offset:0 atIndex:22];
      [encoder setBuffer:curvePrimitiveBuffer_ offset:0 atIndex:23];
      if (intersectionFunctionTable_ != nil) {
        [encoder setIntersectionFunctionTable:intersectionFunctionTable_ atBufferIndex:24];
      }
      if (triangleBLAS_ != nil && triangleBLAS_ != accelerationStructure_) {
        [encoder useResource:triangleBLAS_ usage:MTLResourceUsageRead];
      }
      if (curveBLAS_ != nil) {
        [encoder useResource:curveBLAS_ usage:MTLResourceUsageRead];
      }
      if (curvePrimitiveBuffer_ != nil) {
        [encoder useResource:curvePrimitiveBuffer_ usage:MTLResourceUsageRead];
      }

      dispatchThreads(encoder, pathTracePipelineState_);
      [encoder endEncoding];

      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
    }

    ToonShaderUniforms toon;
    toon.width = width_;
    toon.height = height_;
    toon.contourMethod = 2u;
    toon.useFxaa = config.toon.useFxaa ? 1u : 0u;
    toon.detailContourStrength =
        config.toon.enabled && config.toon.enableDetailContour ? std::max(0.0f, config.toon.detailContourStrength) : 0.0f;
    toon.depthThreshold = std::max(0.0f, config.toon.depthThreshold);
    toon.normalThreshold = std::max(0.0f, config.toon.normalThreshold);
    toon.edgeThickness = std::max(1.0f, config.toon.edgeThickness);
    toon.exposure = config.renderMode == rt::RenderMode::Toon ? std::max(0.1f, config.toon.tonemapExposure)
                                                                  : std::max(0.1f, config.lighting.standardExposure);
    toon.gamma = config.renderMode == rt::RenderMode::Toon ? std::max(0.1f, config.toon.tonemapGamma)
                                                               : std::max(0.1f, config.lighting.standardGamma);
    toon.saturation =
        config.renderMode == rt::RenderMode::Toon ? 1.0f : std::max(0.0f, config.lighting.standardSaturation);
    toon.objectContourStrength =
        config.toon.enabled && config.toon.enableObjectContour ? std::max(0.0f, config.toon.objectContourStrength) : 0.0f;
    toon.objectThreshold = std::max(0.0f, config.toon.objectThreshold);
    toon.enableDetailContour = config.toon.enabled && config.toon.enableDetailContour ? 1u : 0u;
    toon.enableObjectContour = config.toon.enabled && config.toon.enableObjectContour ? 1u : 0u;
    toon.enableNormalEdge = config.toon.enableNormalEdge ? 1u : 0u;
    toon.enableDepthEdge = config.toon.enableDepthEdge ? 1u : 0u;
    toon.backgroundColor = makeFloat4(config.lighting.backgroundColor, 1.0f);
    toon.edgeColor = makeFloat4(config.toon.edgeColor, 1.0f);
    std::memcpy(toonBuffer_.contents, &toon, sizeof(ToonShaderUniforms));

    bool needContours = config.toon.enabled &&
                        (config.toon.enableDetailContour || config.toon.enableObjectContour);

    if (needContours) {
      uint32_t init[2];
      float big = 10000.0f;
      std::memcpy(&init[0], &big, sizeof(float));
      init[1] = 0u;
      std::memcpy(depthMinMaxBuffer_.contents, init, sizeof(init));
    }

    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];

      id<MTLComputeCommandEncoder> tonemapEncoder = [commandBuffer computeCommandEncoder];
      [tonemapEncoder setComputePipelineState:tonemapPipelineState_];
      [tonemapEncoder setBuffer:rawColorBuffer_ offset:0 atIndex:0];
      [tonemapEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
      [tonemapEncoder setBuffer:tonemappedBuffer_ offset:0 atIndex:2];
      dispatchThreads(tonemapEncoder, tonemapPipelineState_);
      [tonemapEncoder endEncoding];

      if (needContours) {
      id<MTLComputeCommandEncoder> depthMinMaxEncoder = [commandBuffer computeCommandEncoder];
      [depthMinMaxEncoder setComputePipelineState:depthMinMaxPipelineState_];
      [depthMinMaxEncoder setBuffer:linearDepthBuffer_ offset:0 atIndex:0];
      [depthMinMaxEncoder setBuffer:depthMinMaxBuffer_ offset:0 atIndex:1];
      [depthMinMaxEncoder setBuffer:toonBuffer_ offset:0 atIndex:2];
      dispatchThreads(depthMinMaxEncoder, depthMinMaxPipelineState_);
      [depthMinMaxEncoder endEncoding];

      id<MTLComputeCommandEncoder> detailEncoder = [commandBuffer computeCommandEncoder];
      [detailEncoder setComputePipelineState:detailContourPipelineState_];
      [detailEncoder setBuffer:linearDepthBuffer_ offset:0 atIndex:0];
      [detailEncoder setBuffer:normalBuffer_ offset:0 atIndex:1];
      [detailEncoder setBuffer:objectIdBuffer_ offset:0 atIndex:2];
      [detailEncoder setBuffer:toonBuffer_ offset:0 atIndex:3];
      [detailEncoder setBuffer:depthMinMaxBuffer_ offset:0 atIndex:4];
      [detailEncoder setBuffer:detailContourBuffer_ offset:0 atIndex:5];
      dispatchThreads(detailEncoder, detailContourPipelineState_);
      [detailEncoder endEncoding];

      id<MTLComputeCommandEncoder> objectEncoder = [commandBuffer computeCommandEncoder];
      [objectEncoder setComputePipelineState:objectContourPipelineState_];
      [objectEncoder setBuffer:objectIdBuffer_ offset:0 atIndex:0];
      [objectEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
      [objectEncoder setBuffer:objectContourBuffer_ offset:0 atIndex:2];
      dispatchThreads(objectEncoder, objectContourPipelineState_);
      [objectEncoder endEncoding];
      } // needContours

      if (needContours) {
        id<MTLBuffer> detailInput = detailContourBuffer_;
        id<MTLBuffer> objectInput = objectContourBuffer_;
        if (config.toon.useFxaa) {
          id<MTLComputeCommandEncoder> detailFxaaEncoder = [commandBuffer computeCommandEncoder];
          [detailFxaaEncoder setComputePipelineState:fxaaPipelineState_];
          [detailFxaaEncoder setBuffer:detailContourBuffer_ offset:0 atIndex:0];
          [detailFxaaEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
          [detailFxaaEncoder setBuffer:detailContourFxaaBuffer_ offset:0 atIndex:2];
          dispatchThreads(detailFxaaEncoder, fxaaPipelineState_);
          [detailFxaaEncoder endEncoding];

          id<MTLComputeCommandEncoder> objectFxaaEncoder = [commandBuffer computeCommandEncoder];
          [objectFxaaEncoder setComputePipelineState:fxaaPipelineState_];
          [objectFxaaEncoder setBuffer:objectContourBuffer_ offset:0 atIndex:0];
          [objectFxaaEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
          [objectFxaaEncoder setBuffer:objectContourFxaaBuffer_ offset:0 atIndex:2];
          dispatchThreads(objectFxaaEncoder, fxaaPipelineState_);
          [objectFxaaEncoder endEncoding];

          detailInput = detailContourFxaaBuffer_;
          objectInput = objectContourFxaaBuffer_;
        }

        id<MTLComputeCommandEncoder> compositeEncoder = [commandBuffer computeCommandEncoder];
        [compositeEncoder setComputePipelineState:compositePipelineState_];
        [compositeEncoder setBuffer:tonemappedBuffer_ offset:0 atIndex:0];
        [compositeEncoder setBuffer:detailInput offset:0 atIndex:1];
        [compositeEncoder setBuffer:objectInput offset:0 atIndex:2];
        [compositeEncoder setBuffer:toonBuffer_ offset:0 atIndex:3];
        [compositeEncoder setBuffer:outputBuffer_ offset:0 atIndex:4];
        dispatchThreads(compositeEncoder, compositePipelineState_);
        [compositeEncoder endEncoding];
      } else {
        // Standard mode: tonemap output is final, copy to outputBuffer
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        [blit copyFromBuffer:tonemappedBuffer_ sourceOffset:0
                    toBuffer:outputBuffer_ destinationOffset:0
                        size:tonemappedBuffer_.length];
        [blit endEncoding];
      }

      if (config.enableMetalFX && config.metalFXOutputWidth > 0 && config.metalFXOutputHeight > 0 &&
          bufferToTexturePipelineState_ != nil && textureToBufferPipelineState_ != nil) {
        ensureMetalFXResources(config.metalFXOutputWidth, config.metalFXOutputHeight);

        if (metalFXDenoisedScaler_ != nil) {
          auto encodeBufferToHalf4Texture = [&](id<MTLBuffer> buf, id<MTLTexture> tex) {
            id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
            [enc setComputePipelineState:bufferToTexturePipelineState_];
            [enc setBuffer:buf offset:0 atIndex:0];
            [enc setTexture:tex atIndex:0];
            dispatchThreads(enc, bufferToTexturePipelineState_);
            [enc endEncoding];
          };

          // Raw HDR color → RGBA16Float
          encodeBufferToHalf4Texture(rawColorBuffer_, metalFXInputTexture_);
          // Normals → RGBA16Float
          encodeBufferToHalf4Texture(normalBuffer_, metalFXNormalTexture_);
          // Diffuse albedo → RGBA16Float
          encodeBufferToHalf4Texture(diffuseAlbedoBuffer_, metalFXDiffuseAlbedoTexture_);
          // Specular albedo → RGBA16Float
          encodeBufferToHalf4Texture(specularAlbedoBuffer_, metalFXSpecularAlbedoTexture_);

          // Depth → R32Float
          if (depthToTexturePipelineState_ != nil) {
            id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
            [enc setComputePipelineState:depthToTexturePipelineState_];
            [enc setBuffer:depthBuffer_ offset:0 atIndex:0];
            [enc setTexture:metalFXDepthTexture_ atIndex:0];
            dispatchThreads(enc, depthToTexturePipelineState_);
            [enc endEncoding];
          }
          // Motion vectors → RG16Float
          if (motionToTexturePipelineState_ != nil) {
            id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
            [enc setComputePipelineState:motionToTexturePipelineState_];
            [enc setBuffer:motionVectorBuffer_ offset:0 atIndex:0];
            [enc setTexture:metalFXMotionTexture_ atIndex:0];
            dispatchThreads(enc, motionToTexturePipelineState_);
            [enc endEncoding];
          }
          // Roughness → R16Float
          if (roughnessToTexturePipelineState_ != nil) {
            id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
            [enc setComputePipelineState:roughnessToTexturePipelineState_];
            [enc setBuffer:roughnessAuxBuffer_ offset:0 atIndex:0];
            [enc setTexture:metalFXRoughnessTexture_ atIndex:0];
            dispatchThreads(enc, roughnessToTexturePipelineState_);
            [enc endEncoding];
          }

          CameraUniforms* cam = static_cast<CameraUniforms*>(cameraBuffer_.contents);

          metalFXDenoisedScaler_.colorTexture = metalFXInputTexture_;
          metalFXDenoisedScaler_.depthTexture = metalFXDepthTexture_;
          metalFXDenoisedScaler_.motionTexture = metalFXMotionTexture_;
          metalFXDenoisedScaler_.normalTexture = metalFXNormalTexture_;
          metalFXDenoisedScaler_.diffuseAlbedoTexture = metalFXDiffuseAlbedoTexture_;
          metalFXDenoisedScaler_.specularAlbedoTexture = metalFXSpecularAlbedoTexture_;
          metalFXDenoisedScaler_.roughnessTexture = metalFXRoughnessTexture_;
          metalFXDenoisedScaler_.outputTexture = metalFXOutputTexture_;
          metalFXDenoisedScaler_.jitterOffsetX = frame.jitterOffset.x;
          metalFXDenoisedScaler_.jitterOffsetY = frame.jitterOffset.y;
          metalFXDenoisedScaler_.shouldResetHistory = !hasPrevViewProj_;
          metalFXDenoisedScaler_.worldToViewMatrix = cam->viewMatrix;
          metalFXDenoisedScaler_.viewToClipMatrix = cam->projectionMatrix;
          [metalFXDenoisedScaler_ encodeToCommandBuffer:commandBuffer];

          // Denoised HDR output → buffer
          {
            id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
            [enc setComputePipelineState:textureToBufferPipelineState_];
            [enc setTexture:metalFXOutputTexture_ atIndex:0];
            [enc setBuffer:metalFXOutputBuffer_ offset:0 atIndex:0];
            dispatchThreads(enc, textureToBufferPipelineState_, metalFXOutputWidth_, metalFXOutputHeight_);
            [enc endEncoding];
          }

          // GPU tonemap at upscaled resolution
          {
            ToonShaderUniforms mfxToon{};
            mfxToon.width = metalFXOutputWidth_;
            mfxToon.height = metalFXOutputHeight_;
            mfxToon.exposure = config.renderMode == rt::RenderMode::Toon
                                   ? std::max(0.1f, config.toon.tonemapExposure)
                                   : std::max(0.1f, config.lighting.standardExposure);
            mfxToon.gamma = config.renderMode == rt::RenderMode::Toon
                                ? std::max(0.1f, config.toon.tonemapGamma)
                                : std::max(0.1f, config.lighting.standardGamma);
            mfxToon.saturation = config.renderMode == rt::RenderMode::Toon
                                     ? 1.0f
                                     : std::max(0.0f, config.lighting.standardSaturation);
            std::memcpy(metalFXToonBuffer_.contents, &mfxToon, sizeof(ToonShaderUniforms));

            id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
            [enc setComputePipelineState:tonemapPipelineState_];
            [enc setBuffer:metalFXOutputBuffer_ offset:0 atIndex:0];
            [enc setBuffer:metalFXToonBuffer_ offset:0 atIndex:1];
            [enc setBuffer:metalFXTonemappedBuffer_ offset:0 atIndex:2];
            dispatchThreads(enc, tonemapPipelineState_, metalFXOutputWidth_, metalFXOutputHeight_);
            [enc endEncoding];
          }
        }
      } else if (lastEnableMetalFX_) {
        teardownMetalFXResources();
      }

      [commandBuffer commit];
      lastCommandBuffer_ = commandBuffer;
    }

    latestBuffer_.accumulatedSamples += frame.samplesPerIteration;
    frameIndex_ += 1;
    lastUseFxaa_ = config.toon.useFxaa;
    lastEnableMetalFX_ = config.enableMetalFX && metalFXDenoisedScaler_ != nil;

    CameraUniforms* cam = static_cast<CameraUniforms*>(cameraBuffer_.contents);
    prevViewProj_ = simd_mul(cam->projectionMatrix, cam->viewMatrix);
    hasPrevViewProj_ = true;
  }

  rt::RenderBuffer downloadRenderBuffer() const override {
    if (lastCommandBuffer_ != nil) {
      [lastCommandBuffer_ waitUntilCompleted];
      lastCommandBuffer_ = nil;
    }

    const auto* depth = static_cast<const float*>(depthBuffer_.contents);
    const auto* linearDepth = static_cast<const float*>(linearDepthBuffer_.contents);
    const auto* normal = static_cast<const simd_float4*>(normalBuffer_.contents);
    const auto* objectId = static_cast<const uint32_t*>(objectIdBuffer_.contents);
    const auto* detailContourRaw = static_cast<const simd_float4*>(detailContourBuffer_.contents);
    const auto* objectContourRaw = static_cast<const simd_float4*>(objectContourBuffer_.contents);
    const auto* detailContour = static_cast<const simd_float4*>((lastUseFxaa_ ? detailContourFxaaBuffer_ : detailContourBuffer_).contents);
    const auto* objectContour = static_cast<const simd_float4*>((lastUseFxaa_ ? objectContourFxaaBuffer_ : objectContourBuffer_).contents);

    if (lastEnableMetalFX_ && metalFXTonemappedBuffer_ != nil) {
      const uint32_t outW = metalFXOutputWidth_;
      const uint32_t outH = metalFXOutputHeight_;
      const size_t outPixels = static_cast<size_t>(outW) * static_cast<size_t>(outH);
      const auto* mfxOutput = static_cast<const simd_float4*>(metalFXTonemappedBuffer_.contents);

      latestBuffer_.width = outW;
      latestBuffer_.height = outH;
      latestBuffer_.color.resize(outPixels);
      latestBuffer_.depth.resize(outPixels);
      latestBuffer_.linearDepth.resize(outPixels);
      latestBuffer_.normal.resize(outPixels);
      latestBuffer_.objectId.resize(outPixels);
      latestBuffer_.detailContour.resize(outPixels);
      latestBuffer_.detailContourRaw.resize(outPixels);
      latestBuffer_.objectContour.resize(outPixels);
      latestBuffer_.objectContourRaw.resize(outPixels);

      for (size_t y = 0; y < outH; ++y) {
        for (size_t x = 0; x < outW; ++x) {
          const size_t outIdx = y * outW + x;
          const size_t srcX = x * width_ / outW;
          const size_t srcY = y * height_ / outH;
          const size_t srcIdx = srcY * width_ + srcX;

          latestBuffer_.color[outIdx] = glm::vec3(mfxOutput[outIdx].x, mfxOutput[outIdx].y, mfxOutput[outIdx].z);
          latestBuffer_.depth[outIdx] = depth[srcIdx];
          latestBuffer_.linearDepth[outIdx] = linearDepth[srcIdx];
          latestBuffer_.normal[outIdx] = glm::vec3(normal[srcIdx].x, normal[srcIdx].y, normal[srcIdx].z);
          latestBuffer_.objectId[outIdx] = objectId[srcIdx];
          latestBuffer_.detailContourRaw[outIdx] = detailContourRaw[srcIdx].x;
          latestBuffer_.detailContour[outIdx] = detailContour[srcIdx].x;
          latestBuffer_.objectContourRaw[outIdx] = objectContourRaw[srcIdx].x;
          latestBuffer_.objectContour[outIdx] = objectContour[srcIdx].x;
        }
      }
    } else {
      const auto* output = static_cast<const simd_float4*>(outputBuffer_.contents);
      const size_t pixelCount = static_cast<size_t>(width_) * static_cast<size_t>(height_);
      latestBuffer_.width = width_;
      latestBuffer_.height = height_;
      latestBuffer_.color.resize(pixelCount);
      latestBuffer_.depth.resize(pixelCount);
      latestBuffer_.linearDepth.resize(pixelCount);
      latestBuffer_.normal.resize(pixelCount);
      latestBuffer_.objectId.resize(pixelCount);
      latestBuffer_.detailContour.resize(pixelCount);
      latestBuffer_.detailContourRaw.resize(pixelCount);
      latestBuffer_.objectContour.resize(pixelCount);
      latestBuffer_.objectContourRaw.resize(pixelCount);
      for (size_t i = 0; i < pixelCount; ++i) {
        latestBuffer_.color[i] = glm::vec3(output[i].x, output[i].y, output[i].z);
        latestBuffer_.depth[i] = depth[i];
        latestBuffer_.linearDepth[i] = linearDepth[i];
        latestBuffer_.normal[i] = glm::vec3(normal[i].x, normal[i].y, normal[i].z);
        latestBuffer_.objectId[i] = objectId[i];
        latestBuffer_.detailContourRaw[i] = detailContourRaw[i].x;
        latestBuffer_.detailContour[i] = detailContour[i].x;
        latestBuffer_.objectContourRaw[i] = objectContourRaw[i].x;
        latestBuffer_.objectContour[i] = objectContour[i].x;
      }
    }

    return latestBuffer_;
  }

private:
  static void dispatchThreads(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipelineState,
                              uint32_t width, uint32_t height) {
    const NSUInteger threadWidth = pipelineState.threadExecutionWidth;
    const NSUInteger threadHeight = std::max<NSUInteger>(1, pipelineState.maxTotalThreadsPerThreadgroup / threadWidth);
    MTLSize threadsPerGroup = MTLSizeMake(threadWidth, threadHeight, 1);
    MTLSize threadsPerGrid = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
  }

  void dispatchThreads(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipelineState) const {
    dispatchThreads(encoder, pipelineState, width_, height_);
  }

  id<MTLTexture> createPrivateTexture(MTLPixelFormat format, uint32_t w, uint32_t h) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModePrivate;
    return [device_ newTextureWithDescriptor:desc];
  }

  void ensureMetalFXResources(uint32_t outputWidth, uint32_t outputHeight) {
    if (metalFXDenoisedScaler_ != nil && metalFXOutputWidth_ == outputWidth && metalFXOutputHeight_ == outputHeight &&
        metalFXInputTexture_.width == width_ && metalFXInputTexture_.height == height_) {
      return;
    }

    metalFXOutputWidth_ = outputWidth;
    metalFXOutputHeight_ = outputHeight;

    metalFXInputTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, width_, height_);
    metalFXOutputTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, outputWidth, outputHeight);
    metalFXDepthTexture_ = createPrivateTexture(MTLPixelFormatR32Float, width_, height_);
    metalFXMotionTexture_ = createPrivateTexture(MTLPixelFormatRG16Float, width_, height_);
    metalFXNormalTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, width_, height_);
    metalFXDiffuseAlbedoTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, width_, height_);
    metalFXSpecularAlbedoTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, width_, height_);
    metalFXRoughnessTexture_ = createPrivateTexture(MTLPixelFormatR16Float, width_, height_);

    const NSUInteger outputPixels = static_cast<NSUInteger>(outputWidth) * static_cast<NSUInteger>(outputHeight);
    metalFXOutputBuffer_ = [device_ newBufferWithLength:outputPixels * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    metalFXTonemappedBuffer_ = [device_ newBufferWithLength:outputPixels * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    metalFXToonBuffer_ = [device_ newBufferWithLength:sizeof(ToonShaderUniforms) options:MTLResourceStorageModeShared];

    MTLFXTemporalDenoisedScalerDescriptor* desc = [[MTLFXTemporalDenoisedScalerDescriptor alloc] init];
    desc.inputWidth = width_;
    desc.inputHeight = height_;
    desc.outputWidth = outputWidth;
    desc.outputHeight = outputHeight;
    desc.colorTextureFormat = MTLPixelFormatRGBA16Float;
    desc.outputTextureFormat = MTLPixelFormatRGBA16Float;
    desc.depthTextureFormat = MTLPixelFormatR32Float;
    desc.motionTextureFormat = MTLPixelFormatRG16Float;
    desc.normalTextureFormat = MTLPixelFormatRGBA16Float;
    desc.diffuseAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
    desc.specularAlbedoTextureFormat = MTLPixelFormatRGBA16Float;
    desc.roughnessTextureFormat = MTLPixelFormatR16Float;
    desc.autoExposureEnabled = YES;

    metalFXDenoisedScaler_ = [desc newTemporalDenoisedScalerWithDevice:device_];
    if (metalFXDenoisedScaler_ != nil) {
      metalFXDenoisedScaler_.motionVectorScaleX = static_cast<float>(width_);
      metalFXDenoisedScaler_.motionVectorScaleY = static_cast<float>(height_);
      metalFXDenoisedScaler_.depthReversed = NO;
    } else {
      NSLog(@"[MetalFX] denoised scaler creation failed (input=%ux%u output=%ux%u)", width_, height_, outputWidth, outputHeight);
    }
  }

  void teardownMetalFXResources() {
    metalFXDenoisedScaler_ = nil;
    metalFXInputTexture_ = nil;
    metalFXOutputTexture_ = nil;
    metalFXDepthTexture_ = nil;
    metalFXMotionTexture_ = nil;
    metalFXNormalTexture_ = nil;
    metalFXDiffuseAlbedoTexture_ = nil;
    metalFXSpecularAlbedoTexture_ = nil;
    metalFXRoughnessTexture_ = nil;
    metalFXOutputBuffer_ = nil;
    metalFXTonemappedBuffer_ = nil;
    metalFXToonBuffer_ = nil;
    metalFXOutputWidth_ = 0;
    metalFXOutputHeight_ = 0;
  }

  void buildSceneBuffers() {
    std::vector<simd_float4> positions;
    std::vector<simd_float4> normals;
    std::vector<simd_float4> vertexColors;
    std::vector<simd_float2> texcoords;
    std::vector<PackedTriangleIndices> accelIndices;
    std::vector<TriangleShaderData> shaderTriangles;
    std::vector<MaterialShaderData> materials;
    std::vector<TextureShaderData> textures;
    std::vector<simd_float4> texturePixels;
    std::unordered_map<std::string, uint32_t> textureLookup;
    std::unordered_map<std::string, uint32_t> objectIdLookup;
    std::vector<PunctualLightShaderData> lights;

    positions.reserve(4096);
    normals.reserve(4096);
    vertexColors.reserve(4096);
    texcoords.reserve(4096);
    accelIndices.reserve(4096);
    shaderTriangles.reserve(4096);
    materials.reserve(scene_.meshes.size() + scene_.curveNetworks.size());

    uint32_t nextObjectId = 1u;
    for (const rt::RTMesh& mesh : scene_.meshes) {
      if (mesh.vertices.empty() || mesh.indices.empty()) continue;

      const uint32_t baseVertex = static_cast<uint32_t>(positions.size());
      const uint32_t materialIndex = static_cast<uint32_t>(materials.size());
      const std::string objectKey = contourObjectKey(mesh.name);
      auto objectIt = objectIdLookup.find(objectKey);
      uint32_t meshObjectId = 0u;
      if (objectIt != objectIdLookup.end()) {
        meshObjectId = objectIt->second;
      } else {
        meshObjectId = nextObjectId++;
        objectIdLookup.emplace(objectKey, meshObjectId);
      }
      const bool hasVertexNormals = mesh.normals.size() == mesh.vertices.size();
      MaterialShaderData material;
      material.baseColorFactor = simd_make_float4(mesh.baseColorFactor.r, mesh.baseColorFactor.g, mesh.baseColorFactor.b,
                                                  mesh.baseColorFactor.a);
      material.baseColorTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
      material.metallicRoughnessNormal =
          simd_make_float4(mesh.metallicFactor, mesh.roughnessFactor, mesh.normalTextureScale, 0.0f);
      material.metallicRoughnessTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
      material.emissiveFactor = simd_make_float4(mesh.emissiveFactor.r, mesh.emissiveFactor.g, mesh.emissiveFactor.b, 1.0f);
      material.emissiveTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
      material.normalTextureData = simd_make_uint4(0u, 0u, 0u, 0u);
      float opacityPacked = mesh.opacity;
      float transmissionPacked = mesh.transmissionFactor;
      if (opacityPacked < 1e-5f && transmissionPacked < 1e-5f) {
        opacityPacked = 1.0f;
      }
      material.transmissionIor =
          simd_make_float4(transmissionPacked, mesh.indexOfRefraction, mesh.unlit ? 1.0f : 0.0f, opacityPacked);

      if (mesh.hasBaseColorTexture && !mesh.baseColorTexture.pixels.empty()) {
        auto existing = textureLookup.find(mesh.baseColorTexture.cacheKey);
        uint32_t textureIndex = 0;
        if (existing != textureLookup.end()) {
          textureIndex = existing->second;
        } else {
          textureIndex = static_cast<uint32_t>(textures.size());
          const uint32_t textureOffset = static_cast<uint32_t>(texturePixels.size());
          TextureShaderData texture;
          texture.data =
              simd_make_uint4(textureOffset, mesh.baseColorTexture.width, mesh.baseColorTexture.height, 0u);
          textures.push_back(texture);
          textureLookup.emplace(mesh.baseColorTexture.cacheKey, textureIndex);

          for (const glm::vec4& pixel : mesh.baseColorTexture.pixels) {
            texturePixels.push_back(simd_make_float4(pixel.r, pixel.g, pixel.b, pixel.a));
          }
        }
        material.baseColorTextureData = simd_make_uint4(textureIndex, 1u, 0u, 0u);
      }

      if (mesh.hasEmissiveTexture && !mesh.emissiveTexture.pixels.empty()) {
        auto existing = textureLookup.find(mesh.emissiveTexture.cacheKey);
        uint32_t textureIndex = 0;
        if (existing != textureLookup.end()) {
          textureIndex = existing->second;
        } else {
          textureIndex = static_cast<uint32_t>(textures.size());
          const uint32_t textureOffset = static_cast<uint32_t>(texturePixels.size());
          TextureShaderData texture;
          texture.data = simd_make_uint4(textureOffset, mesh.emissiveTexture.width, mesh.emissiveTexture.height, 0u);
          textures.push_back(texture);
          textureLookup.emplace(mesh.emissiveTexture.cacheKey, textureIndex);

          for (const glm::vec4& pixel : mesh.emissiveTexture.pixels) {
            texturePixels.push_back(simd_make_float4(pixel.r, pixel.g, pixel.b, pixel.a));
          }
        }
        material.emissiveTextureData = simd_make_uint4(textureIndex, 1u, 0u, 0u);
      }

      if (mesh.hasMetallicRoughnessTexture && !mesh.metallicRoughnessTexture.pixels.empty()) {
        auto existing = textureLookup.find(mesh.metallicRoughnessTexture.cacheKey);
        uint32_t textureIndex = 0;
        if (existing != textureLookup.end()) {
          textureIndex = existing->second;
        } else {
          textureIndex = static_cast<uint32_t>(textures.size());
          const uint32_t textureOffset = static_cast<uint32_t>(texturePixels.size());
          TextureShaderData texture;
          texture.data = simd_make_uint4(textureOffset, mesh.metallicRoughnessTexture.width,
                                         mesh.metallicRoughnessTexture.height, 0u);
          textures.push_back(texture);
          textureLookup.emplace(mesh.metallicRoughnessTexture.cacheKey, textureIndex);

          for (const glm::vec4& pixel : mesh.metallicRoughnessTexture.pixels) {
            texturePixels.push_back(simd_make_float4(pixel.r, pixel.g, pixel.b, pixel.a));
          }
        }
        material.metallicRoughnessTextureData = simd_make_uint4(textureIndex, 1u, 0u, 0u);
      }

      if (mesh.hasNormalTexture && !mesh.normalTexture.pixels.empty()) {
        auto existing = textureLookup.find(mesh.normalTexture.cacheKey);
        uint32_t textureIndex = 0;
        if (existing != textureLookup.end()) {
          textureIndex = existing->second;
        } else {
          textureIndex = static_cast<uint32_t>(textures.size());
          const uint32_t textureOffset = static_cast<uint32_t>(texturePixels.size());
          TextureShaderData texture;
          texture.data = simd_make_uint4(textureOffset, mesh.normalTexture.width, mesh.normalTexture.height, 0u);
          textures.push_back(texture);
          textureLookup.emplace(mesh.normalTexture.cacheKey, textureIndex);

          for (const glm::vec4& pixel : mesh.normalTexture.pixels) {
            texturePixels.push_back(simd_make_float4(pixel.r, pixel.g, pixel.b, pixel.a));
          }
        }
        material.normalTextureData = simd_make_uint4(textureIndex, 1u, 0u, 0u);
      }
      materials.push_back(material);

      std::vector<glm::vec3> worldVertices(mesh.vertices.size());
      std::vector<glm::vec3> worldNormals(mesh.vertices.size(), glm::vec3(0.0f));
      glm::mat3 normalTransform = glm::transpose(glm::inverse(glm::mat3(mesh.transform)));

      for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        glm::vec4 worldPos = mesh.transform * glm::vec4(mesh.vertices[i], 1.0f);
        worldVertices[i] = glm::vec3(worldPos);
        if (hasVertexNormals) {
          worldNormals[i] = glm::normalize(normalTransform * mesh.normals[i]);
        }
      }

      const bool hasVertexColors = mesh.vertexColors.size() == mesh.vertices.size();
      for (size_t i = 0; i < worldVertices.size(); ++i) {
        positions.push_back(makeFloat4(worldVertices[i], 1.0f));
        normals.push_back(makeFloat4(worldNormals[i], 0.0f));
        if (hasVertexColors) {
          vertexColors.push_back(simd_make_float4(mesh.vertexColors[i].r, mesh.vertexColors[i].g, mesh.vertexColors[i].b, 1.0f));
        } else {
          vertexColors.push_back(simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f));
        }
        glm::vec2 uv = i < mesh.texcoords.size() ? mesh.texcoords[i] : glm::vec2(0.0f);
        texcoords.push_back(simd_make_float2(uv.x, uv.y));
      }

      for (const glm::uvec3& tri : mesh.indices) {
        accelIndices.push_back({baseVertex + tri.x, baseVertex + tri.y, baseVertex + tri.z});
        TriangleShaderData triangle{};
        triangle.indicesMaterial = simd_make_uint4(baseVertex + tri.x, baseVertex + tri.y, baseVertex + tri.z, materialIndex);
        triangle.objectFlags = simd_make_uint4(meshObjectId, hasVertexNormals ? 1u : 0u, 0u, 0u);
        shaderTriangles.push_back(triangle);
      }
    }

    std::vector<CurvePrimitiveShaderData> curvePrimitives;
    curveControlPoints_.clear();
    curveRadii_.clear();

    for (const rt::RTCurveNetwork& curveNet : scene_.curveNetworks) {
      const std::string objectKey = curveNet.name;
      auto objectIt = objectIdLookup.find(objectKey);
      uint32_t curveObjectId = 0u;
      if (objectIt != objectIdLookup.end()) {
        curveObjectId = objectIt->second;
      } else {
        curveObjectId = nextObjectId++;
        objectIdLookup.emplace(objectKey, curveObjectId);
      }

      uint32_t curveMaterialIndex = static_cast<uint32_t>(materials.size());
      MaterialShaderData curveMaterial{};
      curveMaterial.baseColorFactor = simd_make_float4(curveNet.baseColor.r, curveNet.baseColor.g,
                                                       curveNet.baseColor.b, curveNet.baseColor.a);
      curveMaterial.metallicRoughnessNormal = simd_make_float4(curveNet.metallic, curveNet.roughness, 1.0f, 0.0f);
      curveMaterial.transmissionIor = simd_make_float4(0.0f, 1.5f, curveNet.unlit ? 1.0f : 0.0f, 1.0f);
      materials.push_back(curveMaterial);

      float bbPadding = 1.0f;  // DIAGNOSTIC: MASSIVE inflate to test IAS

      for (size_t pi = 0; pi < curveNet.primitives.size(); ++pi) {
        const rt::RTCurvePrimitive& prim = curveNet.primitives[pi];
        // Native Metal curve geometry only supports segments (cylinders), not spheres
        if (prim.type != rt::RTCurvePrimitiveType::Cylinder) continue;

        CurvePrimitiveShaderData shaderPrim;
        shaderPrim.p0_radius = simd_make_float4(prim.p0.x, prim.p0.y, prim.p0.z, prim.radius);
        shaderPrim.p1_type = simd_make_float4(prim.p1.x, prim.p1.y, prim.p1.z, 1.0f);
        shaderPrim.materialObjectId = simd_make_uint4(curveMaterialIndex, curveObjectId, 0u, 0u);
        curvePrimitives.push_back(shaderPrim);

        // Control points for the native Metal curve BLAS
        curveControlPoints_.push_back(simd_make_float3(prim.p0.x, prim.p0.y, prim.p0.z));
        curveControlPoints_.push_back(simd_make_float3(prim.p1.x, prim.p1.y, prim.p1.z));
        curveRadii_.push_back(prim.radius);
        curveRadii_.push_back(prim.radius);

      }
    }

    if (positions.empty() && curvePrimitives.empty()) {
      throw std::runtime_error("The scene did not contain any renderable geometry.");
    }
    if (positions.empty()) {
      for (int j = 0; j < 3; ++j) {
        positions.push_back(simd_make_float4(1e6f, 1e6f, 1e6f, 1.0f));
        normals.push_back(simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f));
        vertexColors.push_back(simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        texcoords.push_back(simd_make_float2(0.0f, 0.0f));
      }
      accelIndices.push_back({0, 1, 2});
      MaterialShaderData dummyMat{};
      materials.push_back(dummyMat);
      TriangleShaderData dummyTri{};
      dummyTri.indicesMaterial = simd_make_uint4(0, 1, 2, static_cast<uint32_t>(materials.size() - 1));
      shaderTriangles.push_back(dummyTri);
    }

    for (const rt::RTPunctualLight& light : scene_.lights) {
      PunctualLightShaderData shaderLight;
      shaderLight.positionRange = simd_make_float4(light.position.x, light.position.y, light.position.z, light.range);
      shaderLight.directionType =
          simd_make_float4(light.direction.x, light.direction.y, light.direction.z,
                           static_cast<float>(static_cast<uint32_t>(light.type)));
      shaderLight.colorIntensity = simd_make_float4(light.color.x, light.color.y, light.color.z, light.intensity);
      shaderLight.spotAngles = simd_make_float4(light.innerConeAngle, light.outerConeAngle, 0.0f, 0.0f);
      lights.push_back(shaderLight);
    }

    positionBuffer_ = [device_ newBufferWithBytes:positions.data()
                                           length:positions.size() * sizeof(simd_float4)
                                          options:MTLResourceStorageModeShared];
    normalVertexBuffer_ = [device_ newBufferWithBytes:normals.data()
                                               length:normals.size() * sizeof(simd_float4)
                                              options:MTLResourceStorageModeShared];
    vertexColorBuffer_ = [device_ newBufferWithBytes:vertexColors.data()
                                              length:vertexColors.size() * sizeof(simd_float4)
                                             options:MTLResourceStorageModeShared];
    texcoordBuffer_ = [device_ newBufferWithBytes:texcoords.data()
                                           length:texcoords.size() * sizeof(simd_float2)
                                          options:MTLResourceStorageModeShared];
    accelIndexBuffer_ = [device_ newBufferWithBytes:accelIndices.data()
                                             length:accelIndices.size() * sizeof(PackedTriangleIndices)
                                            options:MTLResourceStorageModeShared];
    triangleBuffer_ = [device_ newBufferWithBytes:shaderTriangles.data()
                                           length:shaderTriangles.size() * sizeof(TriangleShaderData)
                                          options:MTLResourceStorageModeShared];
    materialBuffer_ = [device_ newBufferWithBytes:materials.data()
                                           length:materials.size() * sizeof(MaterialShaderData)
                                          options:MTLResourceStorageModeShared];
    if (textures.empty()) {
      TextureShaderData defaultTexture;
      defaultTexture.data = simd_make_uint4(0u, 0u, 0u, 0u);
      textures.push_back(defaultTexture);
    }
    if (texturePixels.empty()) {
      texturePixels.push_back(simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f));
    }
    textureMetadataBuffer_ = [device_ newBufferWithBytes:textures.data()
                                                  length:textures.size() * sizeof(TextureShaderData)
                                                 options:MTLResourceStorageModeShared];
    texturePixelBuffer_ = [device_ newBufferWithBytes:texturePixels.data()
                                               length:texturePixels.size() * sizeof(simd_float4)
                                              options:MTLResourceStorageModeShared];
    if (lights.empty()) {
      PunctualLightShaderData defaultLight{};
      lights.push_back(defaultLight);
    }
    lightBuffer_ = [device_ newBufferWithBytes:lights.data()
                                        length:lights.size() * sizeof(PunctualLightShaderData)
                                       options:MTLResourceStorageModeShared];

    if (!curvePrimitives.empty()) {
      curvePrimitiveBuffer_ = [device_ newBufferWithBytes:curvePrimitives.data()
                                                   length:curvePrimitives.size() * sizeof(CurvePrimitiveShaderData)
                                                  options:MTLResourceStorageModeShared];
      curveControlPointBuffer_ = [device_ newBufferWithBytes:curveControlPoints_.data()
                                                      length:curveControlPoints_.size() * sizeof(simd_float3)
                                                     options:MTLResourceStorageModeShared];
      curveRadiusBuffer_ = [device_ newBufferWithBytes:curveRadii_.data()
                                                length:curveRadii_.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
      curveSegmentCount_ = static_cast<uint32_t>(curvePrimitives.size());
    } else {
      CurvePrimitiveShaderData dummy{};
      curvePrimitiveBuffer_ = [device_ newBufferWithBytes:&dummy
                                                   length:sizeof(CurvePrimitiveShaderData)
                                                  options:MTLResourceStorageModeShared];
      curveControlPointBuffer_ = nil;
      curveRadiusBuffer_ = nil;
      curveSegmentCount_ = 0;
    }

    buildAccelerationStructure(static_cast<uint32_t>(accelIndices.size()));
  }

  bool sceneContainsTransmission() const {
    for (const rt::RTMesh& mesh : scene_.meshes) {
      if (mesh.transmissionFactor > 1e-4f) return true;
    }
    return false;
  }

  id<MTLAccelerationStructure> buildAndCompactBLAS(MTLPrimitiveAccelerationStructureDescriptor* descriptor) {
    MTLAccelerationStructureSizes sizes = [device_ accelerationStructureSizesWithDescriptor:descriptor];
    id<MTLAccelerationStructure> uncompacted = [device_ newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch = [device_ newBufferWithLength:sizes.buildScratchBufferSize options:MTLResourceStorageModePrivate];
    id<MTLBuffer> compactedSizeBuf = [device_ newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> cmdBuf = [commandQueue_ commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> enc = [cmdBuf accelerationStructureCommandEncoder];
    [enc buildAccelerationStructure:uncompacted descriptor:descriptor scratchBuffer:scratch scratchBufferOffset:0];
    [enc writeCompactedAccelerationStructureSize:uncompacted toBuffer:compactedSizeBuf offset:0];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    uint32_t compactedSize = 0;
    std::memcpy(&compactedSize, compactedSizeBuf.contents, sizeof(uint32_t));

    id<MTLAccelerationStructure> compacted = [device_ newAccelerationStructureWithSize:compactedSize];

    id<MTLCommandBuffer> compactCmd = [commandQueue_ commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> compactEnc = [compactCmd accelerationStructureCommandEncoder];
    [compactEnc copyAndCompactAccelerationStructure:uncompacted toAccelerationStructure:compacted];
    [compactEnc endEncoding];
    [compactCmd commit];
    [compactCmd waitUntilCompleted];

    return compacted;
  }

  void buildAccelerationStructure(uint32_t triangleCount) {
    @autoreleasepool {
      auto* triGeom = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
      triGeom.vertexBuffer = positionBuffer_;
      triGeom.vertexStride = sizeof(simd_float4);
      triGeom.vertexFormat = MTLAttributeFormatFloat3;
      triGeom.indexBuffer = accelIndexBuffer_;
      triGeom.indexType = MTLIndexTypeUInt32;
      triGeom.triangleCount = triangleCount;
      triGeom.opaque = YES;

      auto* triDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
      triDesc.geometryDescriptors = @[ triGeom ];
      triDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;

      triangleBLAS_ = buildAndCompactBLAS(triDesc);

      bool hasCurves = curveSegmentCount_ > 0 && curveControlPointBuffer_ != nil && curveRadiusBuffer_ != nil;

      if (hasCurves) {
        // Build index buffer: segment i uses control points at [2i, 2i+1].
        // Since index[i+1] = 2*(i+1) != index[i]+1 = 2*i+1, Metal treats every segment
        // as an independent curve with its own end caps.
        std::vector<uint32_t> curveIndices(curveSegmentCount_);
        for (uint32_t i = 0; i < curveSegmentCount_; ++i) curveIndices[i] = i * 2;
        id<MTLBuffer> curveIndexBuf = [device_ newBufferWithBytes:curveIndices.data()
                                                          length:curveIndices.size() * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];

        auto* curveGeom = [MTLAccelerationStructureCurveGeometryDescriptor descriptor];
        curveGeom.controlPointBuffer = curveControlPointBuffer_;
        curveGeom.controlPointCount = curveControlPoints_.size();
        curveGeom.controlPointStride = sizeof(simd_float3);
        curveGeom.controlPointFormat = MTLAttributeFormatFloat3;
        curveGeom.controlPointBufferOffset = 0;
        curveGeom.radiusBuffer = curveRadiusBuffer_;
        curveGeom.radiusBufferOffset = 0;
        curveGeom.indexBuffer = curveIndexBuf;
        curveGeom.indexType = MTLIndexTypeUInt32;
        curveGeom.segmentCount = curveSegmentCount_;
        curveGeom.segmentControlPointCount = 2;
        curveGeom.curveBasis = MTLCurveBasisLinear;
        curveGeom.curveType = MTLCurveTypeRound;
        curveGeom.curveEndCaps = MTLCurveEndCapsDisk;
        curveGeom.intersectionFunctionTableOffset = NSUIntegerMax;

        auto* curveDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        curveDesc.geometryDescriptors = @[ curveGeom ];
        curveDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;

        curveBLAS_ = buildAndCompactBLAS(curveDesc);
      } else {
        curveBLAS_ = nil;
      }

      NSUInteger instanceCount = hasCurves ? 2 : 1;
      std::vector<MTLAccelerationStructureInstanceDescriptor> instances(instanceCount);
      std::memset(instances.data(), 0, instances.size() * sizeof(MTLAccelerationStructureInstanceDescriptor));

      for (NSUInteger i = 0; i < instanceCount; ++i) {
        instances[i].transformationMatrix.columns[0] = {1.0f, 0.0f, 0.0f};
        instances[i].transformationMatrix.columns[1] = {0.0f, 1.0f, 0.0f};
        instances[i].transformationMatrix.columns[2] = {0.0f, 0.0f, 1.0f};
        instances[i].transformationMatrix.columns[3] = {0.0f, 0.0f, 0.0f};
        instances[i].mask = 0xFF;
        instances[i].options = MTLAccelerationStructureInstanceOptionNone;
        instances[i].intersectionFunctionTableOffset = 0;
      }
      instances[0].accelerationStructureIndex = 0;
      if (hasCurves) {
        instances[1].accelerationStructureIndex = 1;
      }

      id<MTLBuffer> instanceBuffer = [device_ newBufferWithBytes:instances.data()
                                                          length:instances.size() * sizeof(MTLAccelerationStructureInstanceDescriptor)
                                                         options:MTLResourceStorageModeShared];

      NSMutableArray* blasArray = [NSMutableArray arrayWithObject:triangleBLAS_];
      if (hasCurves) {
        [blasArray addObject:curveBLAS_];
      }

      auto* instanceDesc = [[MTLInstanceAccelerationStructureDescriptor alloc] init];
      instanceDesc.instancedAccelerationStructures = blasArray;
      instanceDesc.instanceCount = instanceCount;
      instanceDesc.instanceDescriptorBuffer = instanceBuffer;
      instanceDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
      instanceDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;

      MTLAccelerationStructureSizes iasSizes = [device_ accelerationStructureSizesWithDescriptor:instanceDesc];
      accelerationStructure_ = [device_ newAccelerationStructureWithSize:iasSizes.accelerationStructureSize];
      id<MTLBuffer> iasScratch = [device_ newBufferWithLength:iasSizes.buildScratchBufferSize
                                                      options:MTLResourceStorageModePrivate];

      id<MTLCommandBuffer> iasCmdBuf = [commandQueue_ commandBuffer];
      id<MTLAccelerationStructureCommandEncoder> iasEnc = [iasCmdBuf accelerationStructureCommandEncoder];
      [iasEnc buildAccelerationStructure:accelerationStructure_
                              descriptor:instanceDesc
                            scratchBuffer:iasScratch
                      scratchBufferOffset:0];
      [iasEnc endEncoding];
      [iasCmdBuf commit];
      [iasCmdBuf waitUntilCompleted];
    }
  }

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> commandQueue_ = nil;
  id<MTLLibrary> library_ = nil;
  id<MTLComputePipelineState> pathTracePipelineState_ = nil;
  id<MTLComputePipelineState> tonemapPipelineState_ = nil;
  id<MTLComputePipelineState> detailContourPipelineState_ = nil;
  id<MTLComputePipelineState> objectContourPipelineState_ = nil;
  id<MTLComputePipelineState> depthMinMaxPipelineState_ = nil;
  id<MTLComputePipelineState> fxaaPipelineState_ = nil;
  id<MTLComputePipelineState> compositePipelineState_ = nil;

  id<MTLBuffer> positionBuffer_ = nil;
  id<MTLBuffer> normalVertexBuffer_ = nil;
  id<MTLBuffer> vertexColorBuffer_ = nil;
  id<MTLBuffer> texcoordBuffer_ = nil;
  id<MTLBuffer> accelIndexBuffer_ = nil;
  id<MTLBuffer> triangleBuffer_ = nil;
  id<MTLBuffer> materialBuffer_ = nil;
  id<MTLBuffer> textureMetadataBuffer_ = nil;
  id<MTLBuffer> texturePixelBuffer_ = nil;
  id<MTLBuffer> lightBuffer_ = nil;
  id<MTLAccelerationStructure> accelerationStructure_ = nil;
  id<MTLAccelerationStructure> triangleBLAS_ = nil;
  id<MTLAccelerationStructure> curveBLAS_ = nil;
  id<MTLBuffer> curvePrimitiveBuffer_ = nil;
  id<MTLBuffer> curveControlPointBuffer_ = nil;
  id<MTLBuffer> curveRadiusBuffer_ = nil;
  uint32_t curveSegmentCount_ = 0;
  std::vector<simd_float3> curveControlPoints_;
  std::vector<float> curveRadii_;
  id<MTLIntersectionFunctionTable> intersectionFunctionTable_ = nil;

  id<MTLBuffer> cameraBuffer_ = nil;
  id<MTLBuffer> frameBuffer_ = nil;
  id<MTLBuffer> lightingBuffer_ = nil;
  id<MTLBuffer> toonBuffer_ = nil;
  id<MTLBuffer> accumulationBuffer_ = nil;
  id<MTLBuffer> rawColorBuffer_ = nil;
  id<MTLBuffer> tonemappedBuffer_ = nil;
  id<MTLBuffer> detailContourBuffer_ = nil;
  id<MTLBuffer> objectContourBuffer_ = nil;
  id<MTLBuffer> detailContourFxaaBuffer_ = nil;
  id<MTLBuffer> objectContourFxaaBuffer_ = nil;
  id<MTLBuffer> depthMinMaxBuffer_ = nil;
  id<MTLBuffer> outputBuffer_ = nil;
  id<MTLBuffer> depthBuffer_ = nil;
  id<MTLBuffer> linearDepthBuffer_ = nil;
  id<MTLBuffer> normalBuffer_ = nil;
  id<MTLBuffer> objectIdBuffer_ = nil;

  mutable id<MTLCommandBuffer> lastCommandBuffer_ = nil;
  bool lastUseFxaa_ = false;
  bool lastEnableMetalFX_ = false;

  id<MTLComputePipelineState> bufferToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> textureToBufferPipelineState_ = nil;
  id<MTLComputePipelineState> depthToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> roughnessToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> motionToTexturePipelineState_ = nil;

  id<MTLBuffer> diffuseAlbedoBuffer_ = nil;
  id<MTLBuffer> specularAlbedoBuffer_ = nil;
  id<MTLBuffer> roughnessAuxBuffer_ = nil;
  id<MTLBuffer> motionVectorBuffer_ = nil;

  id<MTLTexture> metalFXInputTexture_ = nil;
  id<MTLTexture> metalFXOutputTexture_ = nil;
  id<MTLTexture> metalFXDepthTexture_ = nil;
  id<MTLTexture> metalFXMotionTexture_ = nil;
  id<MTLTexture> metalFXNormalTexture_ = nil;
  id<MTLTexture> metalFXDiffuseAlbedoTexture_ = nil;
  id<MTLTexture> metalFXSpecularAlbedoTexture_ = nil;
  id<MTLTexture> metalFXRoughnessTexture_ = nil;
  id<MTLBuffer> metalFXOutputBuffer_ = nil;
  id<MTLBuffer> metalFXTonemappedBuffer_ = nil;
  id<MTLBuffer> metalFXToonBuffer_ = nil;
  id<MTLFXTemporalDenoisedScaler> metalFXDenoisedScaler_ = nil;
  uint32_t metalFXOutputWidth_ = 0;
  uint32_t metalFXOutputHeight_ = 0;
  simd_float4x4 prevViewProj_;
  bool hasPrevViewProj_ = false;

  rt::RTScene scene_;
  rt::RTCamera camera_;
  mutable rt::RenderBuffer latestBuffer_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t frameIndex_ = 0;
};

class MetalShaderTestHarness final : public rt::IMetalShaderTestHarness {
public:
  explicit MetalShaderTestHarness(const std::string& shaderLibraryPath) {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (device_ == nil) {
        throw std::runtime_error("Metal is unavailable on this machine.");
      }
      if (@available(macOS 11.0, *)) {
        if (![device_ supportsRaytracing]) {
          throw std::runtime_error("This Metal device does not report ray tracing support.");
        }
      }

      commandQueue_ = [device_ newCommandQueue];
      if (commandQueue_ == nil) {
        throw std::runtime_error("Failed to create the Metal command queue.");
      }

      NSError* error = nil;
      std::string resolvedShaderLibraryPath = resolveShaderLibraryPath(shaderLibraryPath);
      NSString* metallibPath = [NSString stringWithUTF8String:resolvedShaderLibraryPath.c_str()];
      library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:metallibPath] error:&error];
      if (library_ == nil) {
        throw std::runtime_error(std::string("Failed to load Metal shader library: ") + buildNSErrorMessage(error));
      }

      tonemapPipelineState_ = createPipeline(@"tonemapKernel");
      detailContourPipelineState_ = createPipeline(@"detailContourKernel");
      objectContourPipelineState_ = createPipeline(@"objectContourKernel");
      depthMinMaxPipelineState_ = createPipeline(@"depthMinMaxKernel");
      fxaaPipelineState_ = createPipeline(@"fxaaKernel");
      compositePipelineState_ = createPipeline(@"compositeKernel");

      toonBuffer_ = [device_ newBufferWithLength:sizeof(ToonShaderUniforms) options:MTLResourceStorageModeShared];
    }
  }

  rt::MetalPostprocessTestOutput runPostprocess(const rt::MetalPostprocessTestInput& input) override {
    if (input.width == 0 || input.height == 0) {
      throw std::runtime_error("Metal shader test input size is invalid.");
    }
    const size_t pixelCount = static_cast<size_t>(input.width) * static_cast<size_t>(input.height);
    if (input.rawColor.size() != pixelCount || input.linearDepth.size() != pixelCount || input.normal.size() != pixelCount ||
        input.objectId.size() != pixelCount) {
      throw std::runtime_error("Metal shader test input buffers have unexpected sizes.");
    }

    std::vector<simd_float4> rawColor(pixelCount);
    std::vector<float> linearDepth = input.linearDepth;
    std::vector<simd_float4> normal(pixelCount);
    for (size_t i = 0; i < pixelCount; ++i) {
      rawColor[i] = simd_make_float4(input.rawColor[i].r, input.rawColor[i].g, input.rawColor[i].b, input.rawColor[i].a);
      glm::vec3 n = input.normal[i];
      if (glm::length(n) > 1e-6f) n = glm::normalize(n);
      normal[i] = simd_make_float4(n.x, n.y, n.z, 0.0f);
    }

    id<MTLBuffer> rawColorBuffer =
        [device_ newBufferWithBytes:rawColor.data() length:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> linearDepthBuffer =
        [device_ newBufferWithBytes:linearDepth.data() length:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> normalBuffer =
        [device_ newBufferWithBytes:normal.data() length:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> objectIdBuffer = [device_ newBufferWithBytes:input.objectId.data()
                                                        length:pixelCount * sizeof(uint32_t)
                                                       options:MTLResourceStorageModeShared];
    id<MTLBuffer> tonemappedBuffer =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> detailContourBuffer =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> objectContourBuffer =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> detailContourFxaaBuffer =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> objectContourFxaaBuffer =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> depthMinMaxBuffer = [device_ newBufferWithLength:2 * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer =
        [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];

    ToonShaderUniforms toon;
    toon.width = input.width;
    toon.height = input.height;
    toon.contourMethod = 2u;
    toon.useFxaa = input.toon.useFxaa ? 1u : 0u;
    toon.detailContourStrength =
        input.toon.enabled && input.toon.enableDetailContour ? std::max(0.0f, input.toon.detailContourStrength) : 0.0f;
    toon.depthThreshold = std::max(0.0f, input.toon.depthThreshold);
    toon.normalThreshold = std::max(0.0f, input.toon.normalThreshold);
    toon.edgeThickness = std::max(1.0f, input.toon.edgeThickness);
    toon.exposure = input.renderMode == rt::RenderMode::Toon ? std::max(0.1f, input.toon.tonemapExposure)
                                                                 : std::max(0.1f, input.lighting.standardExposure);
    toon.gamma = input.renderMode == rt::RenderMode::Toon ? std::max(0.1f, input.toon.tonemapGamma)
                                                              : std::max(0.1f, input.lighting.standardGamma);
    toon.saturation =
        input.renderMode == rt::RenderMode::Toon ? 1.0f : std::max(0.0f, input.lighting.standardSaturation);
    toon.objectContourStrength =
        input.toon.enabled && input.toon.enableObjectContour ? std::max(0.0f, input.toon.objectContourStrength) : 0.0f;
    toon.objectThreshold = std::max(0.0f, input.toon.objectThreshold);
    toon.enableDetailContour = input.toon.enabled && input.toon.enableDetailContour ? 1u : 0u;
    toon.enableObjectContour = input.toon.enabled && input.toon.enableObjectContour ? 1u : 0u;
    toon.enableNormalEdge = input.toon.enableNormalEdge ? 1u : 0u;
    toon.enableDepthEdge = input.toon.enableDepthEdge ? 1u : 0u;
    toon.backgroundColor = makeFloat4(input.lighting.backgroundColor, 1.0f);
    toon.edgeColor = makeFloat4(input.toon.edgeColor, 1.0f);
    std::memcpy(toonBuffer_.contents, &toon, sizeof(ToonShaderUniforms));

    uint32_t init[2];
    float big = 10000.0f;
    std::memcpy(&init[0], &big, sizeof(float));
    init[1] = 0u;
    std::memcpy(depthMinMaxBuffer.contents, init, sizeof(init));

    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];

      id<MTLComputeCommandEncoder> tonemapEncoder = [commandBuffer computeCommandEncoder];
      [tonemapEncoder setComputePipelineState:tonemapPipelineState_];
      [tonemapEncoder setBuffer:rawColorBuffer offset:0 atIndex:0];
      [tonemapEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
      [tonemapEncoder setBuffer:tonemappedBuffer offset:0 atIndex:2];
      dispatchThreads(tonemapEncoder, tonemapPipelineState_, input.width, input.height);
      [tonemapEncoder endEncoding];

      id<MTLComputeCommandEncoder> depthMinMaxEncoder = [commandBuffer computeCommandEncoder];
      [depthMinMaxEncoder setComputePipelineState:depthMinMaxPipelineState_];
      [depthMinMaxEncoder setBuffer:linearDepthBuffer offset:0 atIndex:0];
      [depthMinMaxEncoder setBuffer:depthMinMaxBuffer offset:0 atIndex:1];
      [depthMinMaxEncoder setBuffer:toonBuffer_ offset:0 atIndex:2];
      dispatchThreads(depthMinMaxEncoder, depthMinMaxPipelineState_, input.width, input.height);
      [depthMinMaxEncoder endEncoding];

      id<MTLComputeCommandEncoder> detailEncoder = [commandBuffer computeCommandEncoder];
      [detailEncoder setComputePipelineState:detailContourPipelineState_];
      [detailEncoder setBuffer:linearDepthBuffer offset:0 atIndex:0];
      [detailEncoder setBuffer:normalBuffer offset:0 atIndex:1];
      [detailEncoder setBuffer:objectIdBuffer offset:0 atIndex:2];
      [detailEncoder setBuffer:toonBuffer_ offset:0 atIndex:3];
      [detailEncoder setBuffer:depthMinMaxBuffer offset:0 atIndex:4];
      [detailEncoder setBuffer:detailContourBuffer offset:0 atIndex:5];
      dispatchThreads(detailEncoder, detailContourPipelineState_, input.width, input.height);
      [detailEncoder endEncoding];

      id<MTLComputeCommandEncoder> objectEncoder = [commandBuffer computeCommandEncoder];
      [objectEncoder setComputePipelineState:objectContourPipelineState_];
      [objectEncoder setBuffer:objectIdBuffer offset:0 atIndex:0];
      [objectEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
      [objectEncoder setBuffer:objectContourBuffer offset:0 atIndex:2];
      dispatchThreads(objectEncoder, objectContourPipelineState_, input.width, input.height);
      [objectEncoder endEncoding];

      id<MTLBuffer> detailInput = detailContourBuffer;
      id<MTLBuffer> objectInput = objectContourBuffer;
      if (input.toon.useFxaa) {
        id<MTLComputeCommandEncoder> detailFxaaEncoder = [commandBuffer computeCommandEncoder];
        [detailFxaaEncoder setComputePipelineState:fxaaPipelineState_];
        [detailFxaaEncoder setBuffer:detailContourBuffer offset:0 atIndex:0];
        [detailFxaaEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
        [detailFxaaEncoder setBuffer:detailContourFxaaBuffer offset:0 atIndex:2];
        dispatchThreads(detailFxaaEncoder, fxaaPipelineState_, input.width, input.height);
        [detailFxaaEncoder endEncoding];

        id<MTLComputeCommandEncoder> objectFxaaEncoder = [commandBuffer computeCommandEncoder];
        [objectFxaaEncoder setComputePipelineState:fxaaPipelineState_];
        [objectFxaaEncoder setBuffer:objectContourBuffer offset:0 atIndex:0];
        [objectFxaaEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
        [objectFxaaEncoder setBuffer:objectContourFxaaBuffer offset:0 atIndex:2];
        dispatchThreads(objectFxaaEncoder, fxaaPipelineState_, input.width, input.height);
        [objectFxaaEncoder endEncoding];

        detailInput = detailContourFxaaBuffer;
        objectInput = objectContourFxaaBuffer;
      }

      id<MTLComputeCommandEncoder> compositeEncoder = [commandBuffer computeCommandEncoder];
      [compositeEncoder setComputePipelineState:compositePipelineState_];
      [compositeEncoder setBuffer:tonemappedBuffer offset:0 atIndex:0];
      [compositeEncoder setBuffer:detailInput offset:0 atIndex:1];
      [compositeEncoder setBuffer:objectInput offset:0 atIndex:2];
      [compositeEncoder setBuffer:toonBuffer_ offset:0 atIndex:3];
      [compositeEncoder setBuffer:outputBuffer offset:0 atIndex:4];
      dispatchThreads(compositeEncoder, compositePipelineState_, input.width, input.height);
      [compositeEncoder endEncoding];

      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
    }

    const uint32_t* minmaxBits = static_cast<const uint32_t*>(depthMinMaxBuffer.contents);
    const auto* tonemapped = static_cast<const simd_float4*>(tonemappedBuffer.contents);
    const auto* detailContour = static_cast<const simd_float4*>(detailContourBuffer.contents);
    const auto* objectContour = static_cast<const simd_float4*>(objectContourBuffer.contents);
    const auto* detailContourFxaa = static_cast<const simd_float4*>(detailContourFxaaBuffer.contents);
    const auto* objectContourFxaa = static_cast<const simd_float4*>(objectContourFxaaBuffer.contents);
    const auto* output = static_cast<const simd_float4*>(outputBuffer.contents);

    rt::MetalPostprocessTestOutput result;
    result.width = input.width;
    result.height = input.height;
    std::memcpy(&result.minDepth, &minmaxBits[0], sizeof(float));
    std::memcpy(&result.maxDepth, &minmaxBits[1], sizeof(float));
    result.tonemapped.resize(pixelCount);
    result.detailContour.resize(pixelCount);
    result.objectContour.resize(pixelCount);
    result.detailContourFxaa.resize(pixelCount);
    result.objectContourFxaa.resize(pixelCount);
    result.finalColor.resize(pixelCount);
    for (size_t i = 0; i < pixelCount; ++i) {
      result.tonemapped[i] = glm::vec4(tonemapped[i].x, tonemapped[i].y, tonemapped[i].z, tonemapped[i].w);
      result.detailContour[i] = detailContour[i].x;
      result.objectContour[i] = objectContour[i].x;
      result.detailContourFxaa[i] = input.toon.useFxaa ? detailContourFxaa[i].x : detailContour[i].x;
      result.objectContourFxaa[i] = input.toon.useFxaa ? objectContourFxaa[i].x : objectContour[i].x;
      result.finalColor[i] = glm::vec3(output[i].x, output[i].y, output[i].z);
    }
    return result;
  }

private:
  id<MTLComputePipelineState> createPipeline(NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library_ newFunctionWithName:name];
    id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) {
      throw std::runtime_error(std::string("Failed to create Metal test pipeline ") + name.UTF8String + ": " +
                               buildNSErrorMessage(error));
    }
    return pipeline;
  }

  static void dispatchThreads(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipelineState,
                              uint32_t width, uint32_t height) {
    const NSUInteger threadWidth = pipelineState.threadExecutionWidth;
    const NSUInteger threadHeight = std::max<NSUInteger>(1, pipelineState.maxTotalThreadsPerThreadgroup / threadWidth);
    MTLSize threadsPerGroup = MTLSizeMake(threadWidth, threadHeight, 1);
    MTLSize threadsPerGrid = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
  }

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> commandQueue_ = nil;
  id<MTLLibrary> library_ = nil;
  id<MTLComputePipelineState> tonemapPipelineState_ = nil;
  id<MTLComputePipelineState> detailContourPipelineState_ = nil;
  id<MTLComputePipelineState> objectContourPipelineState_ = nil;
  id<MTLComputePipelineState> depthMinMaxPipelineState_ = nil;
  id<MTLComputePipelineState> fxaaPipelineState_ = nil;
  id<MTLComputePipelineState> compositePipelineState_ = nil;
  id<MTLBuffer> toonBuffer_ = nil;
};

} // namespace

namespace rt {

std::unique_ptr<IRayTracingBackend> createMetalPathTracerBackend(const std::string& shaderLibraryPath) {
  return std::make_unique<MetalPathTracerBackend>(shaderLibraryPath);
}

std::unique_ptr<IMetalShaderTestHarness> createMetalShaderTestHarness(const std::string& shaderLibraryPath) {
  return std::make_unique<MetalShaderTestHarness>(shaderLibraryPath);
}

} // namespace rt
