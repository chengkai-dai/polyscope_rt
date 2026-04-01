#include "rendering/ray_tracing_backend.h"
#include "rendering/metal/metal_device.h"
#include "rendering/metal/metal_test_harness_private.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

#include "glm/glm.hpp"

namespace {

class MetalShaderTestHarness final : public rt::IPostProcessTestHarness {
public:
  explicit MetalShaderTestHarness(const std::string& shaderLibraryPath) {
    @autoreleasepool {
      device_ = metal_rt::createRayTracingDeviceOrThrow();

      commandQueue_ = [device_ newCommandQueue];
      if (commandQueue_ == nil) {
        throw std::runtime_error("Failed to create the Metal command queue.");
      }

      NSError* error = nil;
      std::string resolvedPath = metal_rt::resolveShaderLibraryPath(shaderLibraryPath);
      NSString* metallibPath = [NSString stringWithUTF8String:resolvedPath.c_str()];
      library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:metallibPath] error:&error];
      if (library_ == nil) {
        throw std::runtime_error(std::string("Failed to load Metal shader library: ") +
                                 metal_rt::buildNSErrorMessage(error));
      }

      tonemapPipelineState_       = createPipeline(@"tonemapKernel");
      detailContourPipelineState_ = createPipeline(@"detailContourKernel");
      objectContourPipelineState_ = createPipeline(@"objectContourKernel");
      depthMinMaxPipelineState_   = createPipeline(@"depthMinMaxKernel");
      fxaaPipelineState_          = createPipeline(@"fxaaKernel");
      compositePipelineState_     = createPipeline(@"compositeKernel");

      toonBuffer_ = [device_ newBufferWithLength:sizeof(GPUToonUniforms) options:MTLResourceStorageModeShared];
    }
  }

  rt::PostprocessTestOutput runPostprocess(const rt::PostprocessTestInput& input) override {
    if (input.width == 0 || input.height == 0) {
      throw std::runtime_error("Metal shader test input size is invalid.");
    }
    const size_t pixelCount = static_cast<size_t>(input.width) * static_cast<size_t>(input.height);
    if (input.rawColor.size() != pixelCount || input.linearDepth.size() != pixelCount ||
        input.normal.size() != pixelCount || input.objectId.size() != pixelCount) {
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

    GPUToonUniforms toon = metal_rt::makeToonShaderUniforms(input, input.width, input.height);
    std::memcpy(toonBuffer_.contents, &toon, sizeof(GPUToonUniforms));

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
      metal_rt::dispatchThreads(tonemapEncoder, tonemapPipelineState_, input.width, input.height);
      [tonemapEncoder endEncoding];

      id<MTLComputeCommandEncoder> depthMinMaxEncoder = [commandBuffer computeCommandEncoder];
      [depthMinMaxEncoder setComputePipelineState:depthMinMaxPipelineState_];
      [depthMinMaxEncoder setBuffer:linearDepthBuffer offset:0 atIndex:0];
      [depthMinMaxEncoder setBuffer:depthMinMaxBuffer offset:0 atIndex:1];
      [depthMinMaxEncoder setBuffer:toonBuffer_ offset:0 atIndex:2];
      metal_rt::dispatchThreads(depthMinMaxEncoder, depthMinMaxPipelineState_, input.width, input.height);
      [depthMinMaxEncoder endEncoding];

      id<MTLComputeCommandEncoder> detailEncoder = [commandBuffer computeCommandEncoder];
      [detailEncoder setComputePipelineState:detailContourPipelineState_];
      [detailEncoder setBuffer:linearDepthBuffer offset:0 atIndex:0];
      [detailEncoder setBuffer:normalBuffer offset:0 atIndex:1];
      [detailEncoder setBuffer:objectIdBuffer offset:0 atIndex:2];
      [detailEncoder setBuffer:toonBuffer_ offset:0 atIndex:3];
      [detailEncoder setBuffer:depthMinMaxBuffer offset:0 atIndex:4];
      [detailEncoder setBuffer:detailContourBuffer offset:0 atIndex:5];
      metal_rt::dispatchThreads(detailEncoder, detailContourPipelineState_, input.width, input.height);
      [detailEncoder endEncoding];

      id<MTLComputeCommandEncoder> objectEncoder = [commandBuffer computeCommandEncoder];
      [objectEncoder setComputePipelineState:objectContourPipelineState_];
      [objectEncoder setBuffer:objectIdBuffer offset:0 atIndex:0];
      [objectEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
      [objectEncoder setBuffer:objectContourBuffer offset:0 atIndex:2];
      metal_rt::dispatchThreads(objectEncoder, objectContourPipelineState_, input.width, input.height);
      [objectEncoder endEncoding];

      id<MTLBuffer> detailInput = detailContourBuffer;
      id<MTLBuffer> objectInput = objectContourBuffer;
      if (input.toon.useFxaa) {
        id<MTLComputeCommandEncoder> detailFxaaEncoder = [commandBuffer computeCommandEncoder];
        [detailFxaaEncoder setComputePipelineState:fxaaPipelineState_];
        [detailFxaaEncoder setBuffer:detailContourBuffer offset:0 atIndex:0];
        [detailFxaaEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
        [detailFxaaEncoder setBuffer:detailContourFxaaBuffer offset:0 atIndex:2];
        metal_rt::dispatchThreads(detailFxaaEncoder, fxaaPipelineState_, input.width, input.height);
        [detailFxaaEncoder endEncoding];

        id<MTLComputeCommandEncoder> objectFxaaEncoder = [commandBuffer computeCommandEncoder];
        [objectFxaaEncoder setComputePipelineState:fxaaPipelineState_];
        [objectFxaaEncoder setBuffer:objectContourBuffer offset:0 atIndex:0];
        [objectFxaaEncoder setBuffer:toonBuffer_ offset:0 atIndex:1];
        [objectFxaaEncoder setBuffer:objectContourFxaaBuffer offset:0 atIndex:2];
        metal_rt::dispatchThreads(objectFxaaEncoder, fxaaPipelineState_, input.width, input.height);
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
      metal_rt::dispatchThreads(compositeEncoder, compositePipelineState_, input.width, input.height);
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

    rt::PostprocessTestOutput result;
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
                               metal_rt::buildNSErrorMessage(error));
    }
    return pipeline;
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

BackendAvailability queryMetalTestHarnessAvailability(const std::string& shaderLibraryPath) {
  BackendAvailability availability;
  availability.type = BackendType::Metal;
  availability.name = "Metal";

  metal_rt::MetalDeviceAvailability deviceAvailability = metal_rt::queryRayTracingDeviceAvailability();
  if (!deviceAvailability.available) {
    availability.reason = std::move(deviceAvailability.reason);
    return availability;
  }

  try {
    metal_rt::resolveShaderLibraryPath(shaderLibraryPath);
  } catch (const std::exception& e) {
    availability.reason = e.what();
    return availability;
  }

  availability.available = true;
  return availability;
}

std::unique_ptr<IPostProcessTestHarness> createMetalTestHarnessImpl(const std::string& shaderLibraryPath) {
  return std::make_unique<MetalShaderTestHarness>(shaderLibraryPath);
}

} // namespace rt
