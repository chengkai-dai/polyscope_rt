#include "rendering/metal/metal_backend_internal.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "rendering/environment_model_shared.h"
#include "rendering/metal/metal_backend_private.h"
#include "rendering/metal/metal_device.h"

namespace {

constexpr uint32_t kEnvironmentSampleCellCount = RT_ENVIRONMENT_SAMPLE_WIDTH * RT_ENVIRONMENT_SAMPLE_HEIGHT;

} // namespace

namespace rt::metal_backend_internal {

MetalPathTracerBackend::MetalPathTracerBackend(const std::string& shaderLibraryPath) {
  @autoreleasepool {
    device_ = metal_rt::createRayTracingDeviceOrThrow();

    commandQueue_ = [device_ newCommandQueue];
    if (commandQueue_ == nil) {
      throw std::runtime_error("Failed to create the Metal command queue.");
    }

    NSError* error = nil;
    std::string resolvedPath = metal_rt::resolveShaderLibraryPath(shaderLibraryPath);
    std::fprintf(stderr, "[polyscope_rt] Loading Metal shader library: %s\n", resolvedPath.c_str());
    NSString* metallibPath = [NSString stringWithUTF8String:resolvedPath.c_str()];
    library_ = [device_ newLibraryWithURL:[NSURL fileURLWithPath:metallibPath] error:&error];
    if (library_ == nil) {
      throw std::runtime_error(std::string("Failed to load Metal shader library: ") +
                               metal_rt::buildNSErrorMessage(error));
    }

    id<MTLFunction> pathTraceKernel = [library_ newFunctionWithName:@"pathTraceKernel"];
    id<MTLFunction> sphereIntersectFn = [library_ newFunctionWithName:@"sphereIntersection"];

    MTLLinkedFunctions* linkedFunctions = [[MTLLinkedFunctions alloc] init];
    if (sphereIntersectFn != nil) {
      linkedFunctions.functions = @[ sphereIntersectFn ];
    }

    MTLComputePipelineDescriptor* pipelineDesc = [[MTLComputePipelineDescriptor alloc] init];
    pipelineDesc.computeFunction = pathTraceKernel;
    pipelineDesc.linkedFunctions = linkedFunctions;

    pathTracePipelineState_ = [device_ newComputePipelineStateWithDescriptor:pipelineDesc
                                                                     options:0
                                                                  reflection:nil
                                                                       error:&error];
    if (pathTracePipelineState_ == nil) {
      throw std::runtime_error(std::string("Failed to create the Metal compute pipeline: ") +
                               metal_rt::buildNSErrorMessage(error));
    }

    MTLIntersectionFunctionTableDescriptor* ftDesc = [[MTLIntersectionFunctionTableDescriptor alloc] init];
    ftDesc.functionCount = 1;
    intersectionFunctionTable_ = [pathTracePipelineState_ newIntersectionFunctionTableWithDescriptor:ftDesc];
    if (sphereIntersectFn != nil) {
      id<MTLFunctionHandle> handle = [pathTracePipelineState_ functionHandleWithFunction:sphereIntersectFn];
      if (handle != nil) {
        [intersectionFunctionTable_ setFunction:handle atIndex:0];
      }
    }

    standardPreset_ = metal_rt::createStandardPreset();
    toonPreset_ = metal_rt::createToonPreset();
    standardPreset_->createPipelines(device_, library_);
    toonPreset_->createPipelines(device_, library_);
    denoiser_ = std::make_unique<metal_rt::MetalTemporalDenoiser>(device_, library_);

    cameraBuffer_ = [device_ newBufferWithLength:sizeof(GPUCamera) options:MTLResourceStorageModeShared];
    frameBuffer_ = [device_ newBufferWithLength:sizeof(GPUFrameUniforms) options:MTLResourceStorageModeShared];
    lightingBuffer_ = [device_ newBufferWithLength:sizeof(GPULighting) options:MTLResourceStorageModeShared];
    toonBuffer_ = [device_ newBufferWithLength:sizeof(GPUToonUniforms) options:MTLResourceStorageModeShared];
    environmentSampleBuffer_ =
        [device_ newBufferWithLength:kEnvironmentSampleCellCount * sizeof(GPUEnvironmentSampleCell)
                             options:MTLResourceStorageModeShared];
  }
}

std::string MetalPathTracerBackend::name() const { return "Metal Path Tracer"; }

void MetalPathTracerBackend::setScene(const RTScene& scene) {
  scene_ = scene;
  buildSceneBuffers();
  resetAccumulation();
}

} // namespace rt::metal_backend_internal

namespace rt {

BackendAvailability queryMetalBackendAvailability(const std::string& shaderLibraryPath) {
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

std::unique_ptr<IRayTracingBackend> createMetalBackendImpl(const std::string& shaderLibraryPath) {
  return std::make_unique<metal_backend_internal::MetalPathTracerBackend>(shaderLibraryPath);
}

} // namespace rt
