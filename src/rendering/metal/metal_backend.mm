#include "rendering/ray_tracing_backend.h"
#include "rendering/metal/metal_device.h"
#include "rendering/metal/metal_postprocess.h"
#include "rendering/metal/metal_scene_builder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <simd/simd.h>

#include "glm/glm.hpp"

namespace {

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
      std::string resolvedPath = metal_rt::resolveShaderLibraryPath(shaderLibraryPath);
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

      {
        MTLIntersectionFunctionTableDescriptor* ftDesc = [[MTLIntersectionFunctionTableDescriptor alloc] init];
        ftDesc.functionCount = 1;
        intersectionFunctionTable_ = [pathTracePipelineState_ newIntersectionFunctionTableWithDescriptor:ftDesc];
        if (sphereIntersectFn != nil) {
          id<MTLFunctionHandle> handle = [pathTracePipelineState_ functionHandleWithFunction:sphereIntersectFn];
          if (handle != nil) {
            [intersectionFunctionTable_ setFunction:handle atIndex:0];
          }
        }
      }

      standardPreset_ = metal_rt::createStandardPreset();
      toonPreset_     = metal_rt::createToonPreset();
      standardPreset_->createPipelines(device_, library_);
      toonPreset_->createPipelines(device_, library_);

      auto tryCreatePipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [library_ newFunctionWithName:name];
        if (fn == nil) return nil;
        return [device_ newComputePipelineStateWithFunction:fn error:&error];
      };
      bufferToTexturePipelineState_    = tryCreatePipeline(@"bufferToTextureKernel");
      textureToBufferPipelineState_    = tryCreatePipeline(@"textureToBufferKernel");
      depthToTexturePipelineState_     = tryCreatePipeline(@"depthToTextureKernel");
      roughnessToTexturePipelineState_ = tryCreatePipeline(@"roughnessToTextureKernel");
      motionToTexturePipelineState_    = tryCreatePipeline(@"motionToTextureKernel");

      cameraBuffer_   = [device_ newBufferWithLength:sizeof(GPUCamera)        options:MTLResourceStorageModeShared];
      frameBuffer_    = [device_ newBufferWithLength:sizeof(GPUFrameUniforms) options:MTLResourceStorageModeShared];
      lightingBuffer_ = [device_ newBufferWithLength:sizeof(GPULighting)      options:MTLResourceStorageModeShared];
      toonBuffer_     = [device_ newBufferWithLength:sizeof(GPUToonUniforms)  options:MTLResourceStorageModeShared];
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

    GPUCamera uniforms;
    uniforms.position = metal_rt::makeFloat4(camera.position, 1.0f);
    uniforms.lookDir = metal_rt::makeFloat4(glm::normalize(camera.lookDir));
    uniforms.upDir = metal_rt::makeFloat4(glm::normalize(camera.upDir));
    uniforms.rightDir = metal_rt::makeFloat4(glm::normalize(camera.rightDir));
    uniforms.clipData = simd_make_float4(glm::radians(camera.fovYDegrees), camera.aspect, camera.nearClip, camera.farClip);
    uniforms.viewMatrix = metal_rt::makeFloat4x4(camera.viewMatrix);
    uniforms.projectionMatrix = metal_rt::makeFloat4x4(camera.projectionMatrix);

    std::memcpy(cameraBuffer_.contents, &uniforms, sizeof(GPUCamera));
  }

  void resize(uint32_t width, uint32_t height) override {
    width_ = std::max<uint32_t>(1, width);
    height_ = std::max<uint32_t>(1, height);

    const NSUInteger pixelCount = static_cast<NSUInteger>(width_) * static_cast<NSUInteger>(height_);
    accumulationBuffer_  = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    rawColorBuffer_      = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    outputBuffer_        = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    depthBuffer_         = [device_ newBufferWithLength:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
    linearDepthBuffer_   = [device_ newBufferWithLength:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
    normalBuffer_        = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    objectIdBuffer_      = [device_ newBufferWithLength:pixelCount * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    diffuseAlbedoBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    specularAlbedoBuffer_= [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    roughnessAuxBuffer_  = [device_ newBufferWithLength:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
    motionVectorBuffer_  = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];

    toonPreset_->resize(device_, width_, height_);

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

    if (accumulationBuffer_ != nil) std::memset(accumulationBuffer_.contents, 0, accumulationBuffer_.length);
    if (rawColorBuffer_ != nil)     std::memset(rawColorBuffer_.contents, 0, rawColorBuffer_.length);
    if (outputBuffer_ != nil)       std::memset(outputBuffer_.contents, 0, outputBuffer_.length);
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
    if (diffuseAlbedoBuffer_ != nil)  std::memset(diffuseAlbedoBuffer_.contents, 0, diffuseAlbedoBuffer_.length);
    if (specularAlbedoBuffer_ != nil) std::memset(specularAlbedoBuffer_.contents, 0, specularAlbedoBuffer_.length);
    if (roughnessAuxBuffer_ != nil)   std::memset(roughnessAuxBuffer_.contents, 0, roughnessAuxBuffer_.length);
    if (motionVectorBuffer_ != nil)   std::memset(motionVectorBuffer_.contents, 0, motionVectorBuffer_.length);
    hasPrevViewProj_ = false;
  }

  void renderIteration(const rt::RenderConfig& config) override {
    if (width_ == 0 || height_ == 0) {
      throw std::runtime_error("The render target size is invalid.");
    }
    if (pathTracePipelineState_ == nil || meshCurveAcceleration_ == nil || pointAcceleration_ == nil) {
      throw std::runtime_error("The Metal ray tracing pipeline is not initialized.");
    }

    // --- GPUFrameUniforms ---
    GPUFrameUniforms frame;
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
    frame.planeColorEnabled = metal_rt::makeFloat4(config.groundPlane.color, 1.0f);
    frame.planeParams = simd_make_float4(config.groundPlane.height, config.groundPlane.metallic,
                                         config.groundPlane.roughness, config.groundPlane.reflectance);
    if (config.enableMetalFX) {
      float jx = metal_rt::haltonSequence(frameIndex_ + 1, 2) - 0.5f;
      float jy = metal_rt::haltonSequence(frameIndex_ + 1, 3) - 0.5f;
      frame.jitterOffset = simd_make_float2(jx, jy);
    }
    if (hasPrevViewProj_) {
      frame.prevViewProj = prevViewProj_;
    } else {
      GPUCamera* cam = static_cast<GPUCamera*>(cameraBuffer_.contents);
      frame.prevViewProj = simd_mul(cam->projectionMatrix, cam->viewMatrix);
    }
    std::memcpy(frameBuffer_.contents, &frame, sizeof(GPUFrameUniforms));

    // --- GPULighting ---
    GPULighting lighting;
    lighting.backgroundColor = metal_rt::makeFloat4(config.lighting.backgroundColor, 1.0f);
    lighting.mainLightDirection = metal_rt::makeFloat4(glm::normalize(config.lighting.mainLightDirection), 0.0f);
    lighting.mainLightColorIntensity =
        metal_rt::makeFloat4(config.lighting.mainLightColor, std::max(config.lighting.mainLightIntensity, 0.0f));
    lighting.environmentTintIntensity =
        metal_rt::makeFloat4(config.lighting.environmentTint, config.lighting.environmentIntensity);
    lighting.areaLightCenterEnabled =
        metal_rt::makeFloat4(config.lighting.areaLightCenter, config.lighting.enableAreaLight ? 1.0f : 0.0f);
    lighting.areaLightU = metal_rt::makeFloat4(config.lighting.areaLightU, 0.0f);
    lighting.areaLightV = metal_rt::makeFloat4(config.lighting.areaLightV, 0.0f);
    lighting.areaLightEmission = metal_rt::makeFloat4(config.lighting.areaLightEmission, 0.0f);
    std::memcpy(lightingBuffer_.contents, &lighting, sizeof(GPULighting));

    // --- GPUToonUniforms ---
    GPUToonUniforms toon = metal_rt::makeToonShaderUniforms(config, width_, height_);
    std::memcpy(toonBuffer_.contents, &toon, sizeof(GPUToonUniforms));

    // --- Pass 1: path-trace ---
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuf = [commandQueue_ commandBuffer];
      encodePathTracePass(cmdBuf);
      [cmdBuf commit];
      [cmdBuf waitUntilCompleted];
    }

    // --- Pass 2: post-process ---
    @autoreleasepool {
      id<MTLCommandBuffer> cmdBuf = [commandQueue_ commandBuffer];

      metal_rt::PostProcessBuffers ppBufs;
      ppBufs.rawColor     = rawColorBuffer_;
      ppBufs.linearDepth  = linearDepthBuffer_;
      ppBufs.normal       = normalBuffer_;
      ppBufs.objectId     = objectIdBuffer_;
      ppBufs.toonUniforms = toonBuffer_;
      ppBufs.output       = outputBuffer_;

      metal_rt::IMetalPostProcessPreset* activePreset =
          (config.renderMode == rt::RenderMode::Toon) ? toonPreset_.get() : standardPreset_.get();
      activePreset->encode(cmdBuf, ppBufs, width_, height_, config);

      // MetalFX denoising + upscaling
      if (config.enableMetalFX && config.metalFXOutputWidth > 0 && config.metalFXOutputHeight > 0 &&
          bufferToTexturePipelineState_ != nil && textureToBufferPipelineState_ != nil) {
        ensureMetalFXResources(config.metalFXOutputWidth, config.metalFXOutputHeight);

        if (metalFXDenoisedScaler_ != nil) {
          auto encodeBufferToHalf4Texture = [&](id<MTLBuffer> buf, id<MTLTexture> tex) {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:bufferToTexturePipelineState_];
            [enc setBuffer:buf offset:0 atIndex:0];
            [enc setTexture:tex atIndex:0];
            metal_rt::dispatchThreads(enc, bufferToTexturePipelineState_, width_, height_);
            [enc endEncoding];
          };

          encodeBufferToHalf4Texture(rawColorBuffer_,       metalFXInputTexture_);
          encodeBufferToHalf4Texture(normalBuffer_,         metalFXNormalTexture_);
          encodeBufferToHalf4Texture(diffuseAlbedoBuffer_,  metalFXDiffuseAlbedoTexture_);
          encodeBufferToHalf4Texture(specularAlbedoBuffer_, metalFXSpecularAlbedoTexture_);

          if (depthToTexturePipelineState_ != nil) {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:depthToTexturePipelineState_];
            [enc setBuffer:depthBuffer_ offset:0 atIndex:0];
            [enc setTexture:metalFXDepthTexture_ atIndex:0];
            metal_rt::dispatchThreads(enc, depthToTexturePipelineState_, width_, height_);
            [enc endEncoding];
          }
          if (motionToTexturePipelineState_ != nil) {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:motionToTexturePipelineState_];
            [enc setBuffer:motionVectorBuffer_ offset:0 atIndex:0];
            [enc setTexture:metalFXMotionTexture_ atIndex:0];
            metal_rt::dispatchThreads(enc, motionToTexturePipelineState_, width_, height_);
            [enc endEncoding];
          }
          if (roughnessToTexturePipelineState_ != nil) {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:roughnessToTexturePipelineState_];
            [enc setBuffer:roughnessAuxBuffer_ offset:0 atIndex:0];
            [enc setTexture:metalFXRoughnessTexture_ atIndex:0];
            metal_rt::dispatchThreads(enc, roughnessToTexturePipelineState_, width_, height_);
            [enc endEncoding];
          }

          GPUCamera* cam = static_cast<GPUCamera*>(cameraBuffer_.contents);
          metalFXDenoisedScaler_.colorTexture          = metalFXInputTexture_;
          metalFXDenoisedScaler_.depthTexture          = metalFXDepthTexture_;
          metalFXDenoisedScaler_.motionTexture         = metalFXMotionTexture_;
          metalFXDenoisedScaler_.normalTexture         = metalFXNormalTexture_;
          metalFXDenoisedScaler_.diffuseAlbedoTexture  = metalFXDiffuseAlbedoTexture_;
          metalFXDenoisedScaler_.specularAlbedoTexture = metalFXSpecularAlbedoTexture_;
          metalFXDenoisedScaler_.roughnessTexture      = metalFXRoughnessTexture_;
          metalFXDenoisedScaler_.outputTexture         = metalFXOutputTexture_;
          metalFXDenoisedScaler_.jitterOffsetX         = frame.jitterOffset.x;
          metalFXDenoisedScaler_.jitterOffsetY         = frame.jitterOffset.y;
          metalFXDenoisedScaler_.shouldResetHistory    = !hasPrevViewProj_;
          metalFXDenoisedScaler_.worldToViewMatrix     = cam->viewMatrix;
          metalFXDenoisedScaler_.viewToClipMatrix      = cam->projectionMatrix;
          [metalFXDenoisedScaler_ encodeToCommandBuffer:cmdBuf];

          {
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:textureToBufferPipelineState_];
            [enc setTexture:metalFXOutputTexture_ atIndex:0];
            [enc setBuffer:metalFXOutputBuffer_ offset:0 atIndex:0];
            metal_rt::dispatchThreads(enc, textureToBufferPipelineState_, metalFXOutputWidth_, metalFXOutputHeight_);
            [enc endEncoding];
          }

          {
            GPUToonUniforms mfxToon{};
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
            std::memcpy(metalFXToonBuffer_.contents, &mfxToon, sizeof(GPUToonUniforms));

            metal_rt::PostProcessBuffers mfxBufs;
            mfxBufs.rawColor     = metalFXOutputBuffer_;
            mfxBufs.toonUniforms = metalFXToonBuffer_;
            mfxBufs.output       = metalFXTonemappedBuffer_;
            standardPreset_->encode(cmdBuf, mfxBufs, metalFXOutputWidth_, metalFXOutputHeight_, config);
          }
        }
      } else if (lastEnableMetalFX_) {
        teardownMetalFXResources();
      }

      [cmdBuf commit];
      lastCommandBuffer_ = cmdBuf;
    }

    latestBuffer_.accumulatedSamples += frame.samplesPerIteration;
    frameIndex_ += 1;
    lastUseFxaa_ = config.toon.useFxaa;
    lastEnableMetalFX_ = config.enableMetalFX && metalFXDenoisedScaler_ != nil;
    lastUseToon_ = (config.renderMode == rt::RenderMode::Toon);

    GPUCamera* cam = static_cast<GPUCamera*>(cameraBuffer_.contents);
    prevViewProj_ = simd_mul(cam->projectionMatrix, cam->viewMatrix);
    hasPrevViewProj_ = true;
  }

  void encodePathTracePass(id<MTLCommandBuffer> cmdBuf) {
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pathTracePipelineState_];
    [encoder setBuffer:positionBuffer_         offset:0 atIndex:0];
    [encoder setBuffer:normalVertexBuffer_     offset:0 atIndex:1];
    [encoder setBuffer:texcoordBuffer_         offset:0 atIndex:2];
    [encoder setBuffer:triangleBuffer_         offset:0 atIndex:3];
    [encoder setBuffer:materialBuffer_         offset:0 atIndex:4];
    [encoder setBuffer:textureMetadataBuffer_  offset:0 atIndex:5];
    [encoder setBuffer:texturePixelBuffer_     offset:0 atIndex:6];
    [encoder setBuffer:lightBuffer_            offset:0 atIndex:7];
    [encoder setBuffer:cameraBuffer_           offset:0 atIndex:8];
    [encoder setBuffer:frameBuffer_            offset:0 atIndex:9];
    [encoder setBuffer:lightingBuffer_         offset:0 atIndex:10];
    [encoder setBuffer:accumulationBuffer_     offset:0 atIndex:11];
    [encoder setBuffer:rawColorBuffer_         offset:0 atIndex:12];
    [encoder setBuffer:depthBuffer_            offset:0 atIndex:13];
    [encoder setBuffer:linearDepthBuffer_      offset:0 atIndex:14];
    [encoder setBuffer:normalBuffer_           offset:0 atIndex:15];
    [encoder setBuffer:objectIdBuffer_         offset:0 atIndex:16];
    [encoder setAccelerationStructure:meshCurveAcceleration_ atBufferIndex:17];
    [encoder useResource:meshCurveAcceleration_ usage:MTLResourceUsageRead];
    [encoder setBuffer:diffuseAlbedoBuffer_    offset:0 atIndex:18];
    [encoder setBuffer:specularAlbedoBuffer_   offset:0 atIndex:19];
    [encoder setBuffer:roughnessAuxBuffer_     offset:0 atIndex:20];
    [encoder setBuffer:motionVectorBuffer_     offset:0 atIndex:21];
    [encoder setBuffer:vertexColorBuffer_      offset:0 atIndex:22];
    [encoder setBuffer:curvePrimitiveBuffer_   offset:0 atIndex:23];
    if (intersectionFunctionTable_ != nil) {
      [encoder setIntersectionFunctionTable:intersectionFunctionTable_ atBufferIndex:24];
    }
    [encoder setBuffer:pointPrimitiveBuffer_   offset:0 atIndex:25];
    [encoder setAccelerationStructure:pointAcceleration_ atBufferIndex:26];
    if (triangleBLAS_ != nil && triangleBLAS_ != meshCurveAcceleration_) {
      [encoder useResource:triangleBLAS_ usage:MTLResourceUsageRead];
    }
    if (curveBLAS_ != nil)              [encoder useResource:curveBLAS_              usage:MTLResourceUsageRead];
    if (curvePrimitiveBuffer_ != nil)   [encoder useResource:curvePrimitiveBuffer_   usage:MTLResourceUsageRead];
    if (pointAcceleration_ != nil)      [encoder useResource:pointAcceleration_      usage:MTLResourceUsageRead];
    if (pointBLAS_ != nil)              [encoder useResource:pointBLAS_              usage:MTLResourceUsageRead];
    if (pointPrimitiveBuffer_ != nil)   [encoder useResource:pointPrimitiveBuffer_   usage:MTLResourceUsageRead];
    metal_rt::dispatchThreads(encoder, pathTracePipelineState_, width_, height_);
    [encoder endEncoding];
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
        }
      }
      // Contour data: nearest-neighbor upsampled from render resolution
      // The toon preset's downloadAuxBuffers works at native resolution, but
      // MetalFX output is upscaled. We copy contour data with NN resampling.
      toonPreset_->downloadAuxBuffers(latestBuffer_, lastUseFxaa_);
      // Re-sample contour data from native resolution to output resolution
      if (outW != width_ || outH != height_) {
        rt::RenderBuffer nativeContours;
        nativeContours.width = width_;
        nativeContours.height = height_;
        toonPreset_->downloadAuxBuffers(nativeContours, lastUseFxaa_);
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
            latestBuffer_.detailContourRaw[outIdx] = nativeContours.detailContourRaw[srcIdx];
            latestBuffer_.detailContour[outIdx]    = nativeContours.detailContour[srcIdx];
            latestBuffer_.objectContourRaw[outIdx] = nativeContours.objectContourRaw[srcIdx];
            latestBuffer_.objectContour[outIdx]    = nativeContours.objectContour[srcIdx];
          }
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
      for (size_t i = 0; i < pixelCount; ++i) {
        latestBuffer_.color[i] = glm::vec3(output[i].x, output[i].y, output[i].z);
        latestBuffer_.depth[i] = depth[i];
        latestBuffer_.linearDepth[i] = linearDepth[i];
        latestBuffer_.normal[i] = glm::vec3(normal[i].x, normal[i].y, normal[i].z);
        latestBuffer_.objectId[i] = objectId[i];
      }

      metal_rt::IMetalPostProcessPreset* activePreset =
          lastUseToon_ ? toonPreset_.get() : standardPreset_.get();
      activePreset->downloadAuxBuffers(latestBuffer_, lastUseFxaa_);
    }

    return latestBuffer_;
  }

private:
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

    metalFXInputTexture_           = createPrivateTexture(MTLPixelFormatRGBA16Float, width_, height_);
    metalFXOutputTexture_          = createPrivateTexture(MTLPixelFormatRGBA16Float, outputWidth, outputHeight);
    metalFXDepthTexture_           = createPrivateTexture(MTLPixelFormatR32Float, width_, height_);
    metalFXMotionTexture_          = createPrivateTexture(MTLPixelFormatRG16Float, width_, height_);
    metalFXNormalTexture_          = createPrivateTexture(MTLPixelFormatRGBA16Float, width_, height_);
    metalFXDiffuseAlbedoTexture_   = createPrivateTexture(MTLPixelFormatRGBA16Float, width_, height_);
    metalFXSpecularAlbedoTexture_  = createPrivateTexture(MTLPixelFormatRGBA16Float, width_, height_);
    metalFXRoughnessTexture_       = createPrivateTexture(MTLPixelFormatR16Float, width_, height_);

    const NSUInteger outputPixels = static_cast<NSUInteger>(outputWidth) * static_cast<NSUInteger>(outputHeight);
    metalFXOutputBuffer_     = [device_ newBufferWithLength:outputPixels * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    metalFXTonemappedBuffer_ = [device_ newBufferWithLength:outputPixels * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    metalFXToonBuffer_       = [device_ newBufferWithLength:sizeof(GPUToonUniforms) options:MTLResourceStorageModeShared];

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
    SceneGpuAccumulator acc;
    acc.positions.reserve(4096);
    acc.normals.reserve(4096);
    acc.vertexColors.reserve(4096);
    acc.texcoords.reserve(4096);
    acc.accelIndices.reserve(4096);
    acc.shaderTriangles.reserve(4096);
    acc.materials.reserve(scene_.meshes.size() + scene_.curveNetworks.size() + scene_.vectorFields.size());

    metal_rt::gatherMeshGpuData(acc, scene_);
    metal_rt::gatherVectorFieldGpuData(acc, scene_);
    metal_rt::gatherCurveGpuData(acc, scene_, curveControlPoints_, curveRadii_, pointPrimitives_, pointBboxData_);
    metal_rt::gatherPointBboxData(acc, scene_, pointPrimitives_, pointBboxData_);

    if (acc.positions.empty()) {
      for (int j = 0; j < 3; ++j) {
        acc.positions.push_back(simd_make_float4(1e6f, 1e6f, 1e6f, 1.0f));
        acc.normals.push_back(simd_make_float4(0.0f, 1.0f, 0.0f, 0.0f));
        acc.vertexColors.push_back(simd_make_float4(0.0f, 0.0f, 0.0f, 0.0f));
        acc.texcoords.push_back(simd_make_float2(0.0f, 0.0f));
      }
      acc.accelIndices.push_back({0, 1, 2});
      GPUMaterial dummyMat{};
      acc.materials.push_back(dummyMat);
      GPUTriangle dummyTri{};
      dummyTri.indicesMaterial = simd_make_uint4(0, 1, 2, static_cast<uint32_t>(acc.materials.size() - 1));
      acc.shaderTriangles.push_back(dummyTri);
    }

    metal_rt::gatherLightData(acc, scene_);
    uploadSceneBuffers(acc);
    buildAccelerationStructure(static_cast<uint32_t>(acc.accelIndices.size()));
  }

  void uploadSceneBuffers(SceneGpuAccumulator& acc) {
    positionBuffer_ = [device_ newBufferWithBytes:acc.positions.data()
                                           length:acc.positions.size() * sizeof(simd_float4)
                                          options:MTLResourceStorageModeShared];
    normalVertexBuffer_ = [device_ newBufferWithBytes:acc.normals.data()
                                               length:acc.normals.size() * sizeof(simd_float4)
                                              options:MTLResourceStorageModeShared];
    vertexColorBuffer_ = [device_ newBufferWithBytes:acc.vertexColors.data()
                                              length:acc.vertexColors.size() * sizeof(simd_float4)
                                             options:MTLResourceStorageModeShared];
    texcoordBuffer_ = [device_ newBufferWithBytes:acc.texcoords.data()
                                           length:acc.texcoords.size() * sizeof(simd_float2)
                                          options:MTLResourceStorageModeShared];
    accelIndexBuffer_ = [device_ newBufferWithBytes:acc.accelIndices.data()
                                             length:acc.accelIndices.size() * sizeof(PackedTriangleIndices)
                                            options:MTLResourceStorageModeShared];
    triangleBuffer_ = [device_ newBufferWithBytes:acc.shaderTriangles.data()
                                           length:acc.shaderTriangles.size() * sizeof(GPUTriangle)
                                          options:MTLResourceStorageModeShared];
    materialBuffer_ = [device_ newBufferWithBytes:acc.materials.data()
                                           length:acc.materials.size() * sizeof(GPUMaterial)
                                          options:MTLResourceStorageModeShared];

    if (acc.textures.empty()) {
      GPUTexture defaultTexture;
      defaultTexture.data = simd_make_uint4(0u, 0u, 0u, 0u);
      acc.textures.push_back(defaultTexture);
    }
    if (acc.texturePixels.empty()) {
      acc.texturePixels.push_back(simd_make_float4(1.0f, 1.0f, 1.0f, 1.0f));
    }
    textureMetadataBuffer_ = [device_ newBufferWithBytes:acc.textures.data()
                                                  length:acc.textures.size() * sizeof(GPUTexture)
                                                 options:MTLResourceStorageModeShared];
    texturePixelBuffer_ = [device_ newBufferWithBytes:acc.texturePixels.data()
                                               length:acc.texturePixels.size() * sizeof(simd_float4)
                                              options:MTLResourceStorageModeShared];

    if (acc.lights.empty()) {
      GPUPunctualLight defaultLight{};
      acc.lights.push_back(defaultLight);
    }
    lightBuffer_ = [device_ newBufferWithBytes:acc.lights.data()
                                        length:acc.lights.size() * sizeof(GPUPunctualLight)
                                       options:MTLResourceStorageModeShared];

    if (!acc.curvePrimitives.empty()) {
      curvePrimitiveBuffer_ = [device_ newBufferWithBytes:acc.curvePrimitives.data()
                                                   length:acc.curvePrimitives.size() * sizeof(GPUCurvePrimitive)
                                                  options:MTLResourceStorageModeShared];
      curveControlPointBuffer_ = [device_ newBufferWithBytes:curveControlPoints_.data()
                                                      length:curveControlPoints_.size() * sizeof(simd_float3)
                                                     options:MTLResourceStorageModeShared];
      curveRadiusBuffer_ = [device_ newBufferWithBytes:curveRadii_.data()
                                                length:curveRadii_.size() * sizeof(float)
                                               options:MTLResourceStorageModeShared];
      curveSegmentCount_ = static_cast<uint32_t>(acc.curvePrimitives.size());
    } else {
      GPUCurvePrimitive dummy{};
      curvePrimitiveBuffer_ = [device_ newBufferWithBytes:&dummy
                                                   length:sizeof(GPUCurvePrimitive)
                                                  options:MTLResourceStorageModeShared];
      curveControlPointBuffer_ = nil;
      curveRadiusBuffer_ = nil;
      curveSegmentCount_ = 0;
    }

    if (!pointPrimitives_.empty()) {
      pointPrimitiveBuffer_ = [device_ newBufferWithBytes:pointPrimitives_.data()
                                                   length:pointPrimitives_.size() * sizeof(GPUPointPrimitive)
                                                  options:MTLResourceStorageModeShared];
      pointBboxBuffer_ = [device_ newBufferWithBytes:pointBboxData_.data()
                                              length:pointBboxData_.size() * sizeof(MTLAxisAlignedBoundingBox)
                                             options:MTLResourceStorageModeShared];
    } else {
      GPUPointPrimitive dummy{};
      pointPrimitiveBuffer_ = [device_ newBufferWithBytes:&dummy
                                                   length:sizeof(GPUPointPrimitive)
                                                  options:MTLResourceStorageModeShared];
      pointBboxBuffer_ = nil;
    }

    if (intersectionFunctionTable_ != nil && pointPrimitiveBuffer_ != nil) {
      [intersectionFunctionTable_ setBuffer:pointPrimitiveBuffer_ offset:0 atIndex:25];
    }
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

      bool hasPoints = !pointPrimitives_.empty() && pointBboxBuffer_ != nil;

      auto buildPointIas = [&]() -> id<MTLAccelerationStructure> {
        id<MTLBuffer> bboxBuf = nil;
        NSUInteger bboxCount = 0;

        if (hasPoints) {
          bboxBuf   = pointBboxBuffer_;
          bboxCount = static_cast<NSUInteger>(pointPrimitives_.size());
        } else {
          MTLAxisAlignedBoundingBox dummyBox = { {9.99e9f, 9.99e9f, 9.99e9f},
                                                 {1.001e10f, 1.001e10f, 1.001e10f} };
          bboxBuf   = [device_ newBufferWithBytes:&dummyBox
                                           length:sizeof(dummyBox)
                                          options:MTLResourceStorageModeShared];
          bboxCount = 1;
        }

        auto* bboxGeom = [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        bboxGeom.boundingBoxBuffer = bboxBuf;
        bboxGeom.boundingBoxCount  = bboxCount;
        bboxGeom.boundingBoxStride = sizeof(MTLAxisAlignedBoundingBox);
        bboxGeom.intersectionFunctionTableOffset = 0;

        auto* ptDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        ptDesc.geometryDescriptors = @[ bboxGeom ];
        ptDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;
        pointBLAS_ = buildAndCompactBLAS(ptDesc);

        MTLAccelerationStructureInstanceDescriptor ptInst{};
        ptInst.transformationMatrix.columns[0] = {1.f, 0.f, 0.f};
        ptInst.transformationMatrix.columns[1] = {0.f, 1.f, 0.f};
        ptInst.transformationMatrix.columns[2] = {0.f, 0.f, 1.f};
        ptInst.transformationMatrix.columns[3] = {0.f, 0.f, 0.f};
        ptInst.mask = 0xFF;
        ptInst.options = MTLAccelerationStructureInstanceOptionNone;
        ptInst.intersectionFunctionTableOffset = 0;
        ptInst.accelerationStructureIndex = 0;

        id<MTLBuffer> ptInstBuf = [device_ newBufferWithBytes:&ptInst
                                                        length:sizeof(ptInst)
                                                       options:MTLResourceStorageModeShared];
        auto* ptIasDesc = [[MTLInstanceAccelerationStructureDescriptor alloc] init];
        ptIasDesc.instancedAccelerationStructures = @[ pointBLAS_ ];
        ptIasDesc.instanceCount = 1;
        ptIasDesc.instanceDescriptorBuffer = ptInstBuf;
        ptIasDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
        ptIasDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;

        MTLAccelerationStructureSizes ptSizes = [device_ accelerationStructureSizesWithDescriptor:ptIasDesc];
        id<MTLAccelerationStructure> ptIas = [device_ newAccelerationStructureWithSize:ptSizes.accelerationStructureSize];
        id<MTLBuffer> ptScratch = [device_ newBufferWithLength:ptSizes.buildScratchBufferSize
                                                       options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> ptCmd = [commandQueue_ commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> ptEnc = [ptCmd accelerationStructureCommandEncoder];
        [ptEnc buildAccelerationStructure:ptIas descriptor:ptIasDesc
                             scratchBuffer:ptScratch scratchBufferOffset:0];
        [ptEnc endEncoding];
        [ptCmd commit];
        [ptCmd waitUntilCompleted];
        return ptIas;
      };
      pointAcceleration_ = buildPointIas();

      NSUInteger mcInstanceCount = 1 + (hasCurves ? 1 : 0);
      std::vector<MTLAccelerationStructureInstanceDescriptor> mcInstances(mcInstanceCount);
      std::memset(mcInstances.data(), 0, mcInstances.size() * sizeof(MTLAccelerationStructureInstanceDescriptor));

      for (NSUInteger i = 0; i < mcInstanceCount; ++i) {
        mcInstances[i].transformationMatrix.columns[0] = {1.0f, 0.0f, 0.0f};
        mcInstances[i].transformationMatrix.columns[1] = {0.0f, 1.0f, 0.0f};
        mcInstances[i].transformationMatrix.columns[2] = {0.0f, 0.0f, 1.0f};
        mcInstances[i].transformationMatrix.columns[3] = {0.0f, 0.0f, 0.0f};
        mcInstances[i].mask = 0xFF;
        mcInstances[i].options = MTLAccelerationStructureInstanceOptionNone;
        mcInstances[i].intersectionFunctionTableOffset = 0;
      }
      NSUInteger nextMcInstance = 0;
      mcInstances[nextMcInstance++].accelerationStructureIndex = 0;

      NSMutableArray* mcBlasArray = [NSMutableArray arrayWithObject:triangleBLAS_];
      if (hasCurves) {
        mcInstances[nextMcInstance++].accelerationStructureIndex = static_cast<uint32_t>(mcBlasArray.count);
        [mcBlasArray addObject:curveBLAS_];
      }

      id<MTLBuffer> mcInstanceBuf = [device_ newBufferWithBytes:mcInstances.data()
                                                          length:mcInstances.size() * sizeof(MTLAccelerationStructureInstanceDescriptor)
                                                         options:MTLResourceStorageModeShared];

      auto* mcIasDesc = [[MTLInstanceAccelerationStructureDescriptor alloc] init];
      mcIasDesc.instancedAccelerationStructures = mcBlasArray;
      mcIasDesc.instanceCount = mcInstanceCount;
      mcIasDesc.instanceDescriptorBuffer = mcInstanceBuf;
      mcIasDesc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
      mcIasDesc.usage = MTLAccelerationStructureUsagePreferFastBuild;

      MTLAccelerationStructureSizes mcSizes = [device_ accelerationStructureSizesWithDescriptor:mcIasDesc];
      meshCurveAcceleration_ = [device_ newAccelerationStructureWithSize:mcSizes.accelerationStructureSize];
      id<MTLBuffer> mcScratch = [device_ newBufferWithLength:mcSizes.buildScratchBufferSize
                                                     options:MTLResourceStorageModePrivate];
      id<MTLCommandBuffer> mcCmdBuf = [commandQueue_ commandBuffer];
      id<MTLAccelerationStructureCommandEncoder> mcEnc = [mcCmdBuf accelerationStructureCommandEncoder];
      [mcEnc buildAccelerationStructure:meshCurveAcceleration_
                             descriptor:mcIasDesc
                           scratchBuffer:mcScratch
                     scratchBufferOffset:0];
      [mcEnc endEncoding];
      [mcCmdBuf commit];
      [mcCmdBuf waitUntilCompleted];
    }
  }

  // --- Metal objects ---
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> commandQueue_ = nil;
  id<MTLLibrary> library_ = nil;
  id<MTLComputePipelineState> pathTracePipelineState_ = nil;

  // Scene geometry buffers
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

  // Acceleration structures
  id<MTLAccelerationStructure> meshCurveAcceleration_ = nil;
  id<MTLAccelerationStructure> pointAcceleration_ = nil;
  id<MTLAccelerationStructure> triangleBLAS_ = nil;
  id<MTLAccelerationStructure> curveBLAS_ = nil;

  // Curve data
  id<MTLBuffer> curvePrimitiveBuffer_ = nil;
  id<MTLBuffer> curveControlPointBuffer_ = nil;
  id<MTLBuffer> curveRadiusBuffer_ = nil;
  uint32_t curveSegmentCount_ = 0;
  std::vector<simd_float3> curveControlPoints_;
  std::vector<float> curveRadii_;

  // Point data
  id<MTLAccelerationStructure> pointBLAS_ = nil;
  id<MTLBuffer> pointPrimitiveBuffer_ = nil;
  id<MTLBuffer> pointBboxBuffer_ = nil;
  std::vector<GPUPointPrimitive> pointPrimitives_;
  std::vector<MTLAxisAlignedBoundingBox> pointBboxData_;

  id<MTLIntersectionFunctionTable> intersectionFunctionTable_ = nil;

  // Per-frame uniform buffers
  id<MTLBuffer> cameraBuffer_ = nil;
  id<MTLBuffer> frameBuffer_ = nil;
  id<MTLBuffer> lightingBuffer_ = nil;
  id<MTLBuffer> toonBuffer_ = nil;

  // Render target buffers
  id<MTLBuffer> accumulationBuffer_ = nil;
  id<MTLBuffer> rawColorBuffer_ = nil;
  id<MTLBuffer> outputBuffer_ = nil;
  id<MTLBuffer> depthBuffer_ = nil;
  id<MTLBuffer> linearDepthBuffer_ = nil;
  id<MTLBuffer> normalBuffer_ = nil;
  id<MTLBuffer> objectIdBuffer_ = nil;

  mutable id<MTLCommandBuffer> lastCommandBuffer_ = nil;
  bool lastUseFxaa_ = false;
  bool lastEnableMetalFX_ = false;
  bool lastUseToon_ = false;

  // MetalFX helper pipelines
  id<MTLComputePipelineState> bufferToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> textureToBufferPipelineState_ = nil;
  id<MTLComputePipelineState> depthToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> roughnessToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> motionToTexturePipelineState_ = nil;

  // Denoising auxiliary buffers
  id<MTLBuffer> diffuseAlbedoBuffer_ = nil;
  id<MTLBuffer> specularAlbedoBuffer_ = nil;
  id<MTLBuffer> roughnessAuxBuffer_ = nil;
  id<MTLBuffer> motionVectorBuffer_ = nil;

  // MetalFX resources
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

  // Postprocess presets
  std::unique_ptr<metal_rt::IMetalPostProcessPreset> standardPreset_;
  std::unique_ptr<metal_rt::IMetalPostProcessPreset> toonPreset_;

  // Scene / camera state
  rt::RTScene scene_;
  rt::RTCamera camera_;
  mutable rt::RenderBuffer latestBuffer_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint32_t frameIndex_ = 0;
};

} // namespace

namespace rt {

std::unique_ptr<IRayTracingBackend> createBackend(BackendType type, const std::string& shaderLibraryPath) {
  if (type == BackendType::Metal) return std::make_unique<MetalPathTracerBackend>(shaderLibraryPath);
  throw std::runtime_error("Unsupported backend type");
}

} // namespace rt
