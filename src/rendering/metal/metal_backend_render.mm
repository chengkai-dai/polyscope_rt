#include "rendering/metal/metal_backend_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "rendering/environment_model.h"
#include "rendering/metal/metal_device.h"

namespace rt::metal_backend_internal {

void MetalPathTracerBackend::updateCamera(const RTCamera& camera) {
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

void MetalPathTracerBackend::resize(uint32_t width, uint32_t height) {
  width_ = std::max<uint32_t>(1, width);
  height_ = std::max<uint32_t>(1, height);

  const NSUInteger pixelCount = static_cast<NSUInteger>(width_) * static_cast<NSUInteger>(height_);
  accumulationBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
  rawColorBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
  outputBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
  depthBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
  linearDepthBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
  normalBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
  objectIdBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(uint32_t) options:MTLResourceStorageModeShared];
  diffuseAlbedoBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
  specularAlbedoBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
  roughnessAuxBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(float) options:MTLResourceStorageModeShared];
  motionVectorBuffer_ = [device_ newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];

  toonPreset_->resize(device_, width_, height_);

  latestBuffer_.width = width_;
  latestBuffer_.height = height_;
  latestBuffer_.color.assign(pixelCount, glm::vec3(0.0f));
  latestBuffer_.depth.assign(pixelCount, 1.0f);
  latestBuffer_.linearDepth.assign(pixelCount, -1.0f);
  latestBuffer_.normal.assign(pixelCount, glm::vec3(0.0f, 1.0f, 0.0f));
  latestBuffer_.objectId.assign(pixelCount, 0u);
  latestBuffer_.diffuseAlbedo.assign(pixelCount, glm::vec3(0.0f));
  latestBuffer_.specularAlbedo.assign(pixelCount, glm::vec3(0.0f));
  latestBuffer_.roughness.assign(pixelCount, 0.0f);
  latestBuffer_.detailContour.assign(pixelCount, 0.0f);
  latestBuffer_.detailContourRaw.assign(pixelCount, 0.0f);
  latestBuffer_.objectContour.assign(pixelCount, 0.0f);
  latestBuffer_.objectContourRaw.assign(pixelCount, 0.0f);
  resetAccumulation();
}

void MetalPathTracerBackend::resetAccumulation() {
  frameIndex_ = 0;
  latestBuffer_.accumulatedSamples = 0;

  if (accumulationBuffer_ != nil) std::memset(accumulationBuffer_.contents, 0, accumulationBuffer_.length);
  if (rawColorBuffer_ != nil) std::memset(rawColorBuffer_.contents, 0, rawColorBuffer_.length);
  if (outputBuffer_ != nil) std::memset(outputBuffer_.contents, 0, outputBuffer_.length);
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

void MetalPathTracerBackend::renderIteration(const RenderConfig& config) {
  if (width_ == 0 || height_ == 0) {
    throw std::runtime_error("The render target size is invalid.");
  }
  if (pathTracePipelineState_ == nil || meshCurveAcceleration_ == nil || pointAcceleration_ == nil) {
    throw std::runtime_error("The Metal ray tracing pipeline is not initialized.");
  }

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
  frame.emissiveTriangleCount = emissiveTriangleCount_;
  frame.enableSceneLights = scene_.lights.empty() ? 0u : 1u;
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

  GPULighting lighting;
  lighting.backgroundColor = metal_rt::makeFloat4(config.lighting.backgroundColor, 1.0f);
  lighting.mainLightDirection =
      metal_rt::makeFloat4(glm::normalize(config.lighting.mainLightDirection),
                           std::max(config.lighting.mainLightAngularRadius, 0.0f));
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

  const std::vector<GPUEnvironmentSampleCell> environmentCells = buildEnvironmentSampleCells(config.lighting);
  std::memcpy(environmentSampleBuffer_.contents, environmentCells.data(),
              environmentCells.size() * sizeof(GPUEnvironmentSampleCell));

  GPUToonUniforms toon = metal_rt::makeToonShaderUniforms(config, width_, height_);
  std::memcpy(toonBuffer_.contents, &toon, sizeof(GPUToonUniforms));

  @autoreleasepool {
    id<MTLCommandBuffer> cmdBuf = [commandQueue_ commandBuffer];
    encodePathTracePass(cmdBuf);
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];
  }

  @autoreleasepool {
    id<MTLCommandBuffer> cmdBuf = [commandQueue_ commandBuffer];

    metal_rt::PostProcessBuffers ppBufs;
    ppBufs.rawColor = rawColorBuffer_;
    ppBufs.linearDepth = linearDepthBuffer_;
    ppBufs.normal = normalBuffer_;
    ppBufs.objectId = objectIdBuffer_;
    ppBufs.toonUniforms = toonBuffer_;
    ppBufs.output = outputBuffer_;

    metal_rt::IMetalPostProcessPreset* activePreset =
        (config.renderMode == RenderMode::Toon) ? toonPreset_.get() : standardPreset_.get();
    activePreset->encode(cmdBuf, ppBufs, width_, height_, config);

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

        encodeBufferToHalf4Texture(rawColorBuffer_, metalFXInputTexture_);
        encodeBufferToHalf4Texture(normalBuffer_, metalFXNormalTexture_);
        encodeBufferToHalf4Texture(diffuseAlbedoBuffer_, metalFXDiffuseAlbedoTexture_);
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
          mfxToon.exposure = config.renderMode == RenderMode::Toon
                                 ? std::max(0.1f, config.toon.tonemapExposure)
                                 : std::max(0.1f, config.lighting.standardExposure);
          mfxToon.gamma = config.renderMode == RenderMode::Toon
                              ? std::max(0.1f, config.toon.tonemapGamma)
                              : std::max(0.1f, config.lighting.standardGamma);
          mfxToon.saturation = config.renderMode == RenderMode::Toon
                                   ? 1.0f
                                   : std::max(0.0f, config.lighting.standardSaturation);
          std::memcpy(metalFXToonBuffer_.contents, &mfxToon, sizeof(GPUToonUniforms));

          metal_rt::PostProcessBuffers mfxBufs;
          mfxBufs.rawColor = metalFXOutputBuffer_;
          mfxBufs.toonUniforms = metalFXToonBuffer_;
          mfxBufs.output = metalFXTonemappedBuffer_;
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
  lastUseToon_ = (config.renderMode == RenderMode::Toon);

  GPUCamera* cam = static_cast<GPUCamera*>(cameraBuffer_.contents);
  prevViewProj_ = simd_mul(cam->projectionMatrix, cam->viewMatrix);
  hasPrevViewProj_ = true;
}

void MetalPathTracerBackend::encodePathTracePass(id<MTLCommandBuffer> cmdBuf) {
  id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
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
  [encoder setAccelerationStructure:meshCurveAcceleration_ atBufferIndex:17];
  [encoder useResource:meshCurveAcceleration_ usage:MTLResourceUsageRead];
  [encoder setBuffer:diffuseAlbedoBuffer_ offset:0 atIndex:18];
  [encoder setBuffer:specularAlbedoBuffer_ offset:0 atIndex:19];
  [encoder setBuffer:roughnessAuxBuffer_ offset:0 atIndex:20];
  [encoder setBuffer:motionVectorBuffer_ offset:0 atIndex:21];
  [encoder setBuffer:vertexColorBuffer_ offset:0 atIndex:22];
  [encoder setBuffer:curvePrimitiveBuffer_ offset:0 atIndex:23];
  if (intersectionFunctionTable_ != nil) {
    [encoder setIntersectionFunctionTable:intersectionFunctionTable_ atBufferIndex:24];
  }
  [encoder setBuffer:pointPrimitiveBuffer_ offset:0 atIndex:25];
  [encoder setAccelerationStructure:pointAcceleration_ atBufferIndex:26];
  [encoder setBuffer:isoScalarsBuffer_ offset:0 atIndex:27];
  [encoder setBuffer:emissiveTriangleBuffer_ offset:0 atIndex:28];
  [encoder setBuffer:environmentSampleBuffer_ offset:0 atIndex:29];
  if (triangleBLAS_ != nil && triangleBLAS_ != meshCurveAcceleration_) {
    [encoder useResource:triangleBLAS_ usage:MTLResourceUsageRead];
  }
  if (curveBLAS_ != nil) [encoder useResource:curveBLAS_ usage:MTLResourceUsageRead];
  if (curvePrimitiveBuffer_ != nil) [encoder useResource:curvePrimitiveBuffer_ usage:MTLResourceUsageRead];
  if (pointAcceleration_ != nil) [encoder useResource:pointAcceleration_ usage:MTLResourceUsageRead];
  if (pointBLAS_ != nil) [encoder useResource:pointBLAS_ usage:MTLResourceUsageRead];
  if (pointPrimitiveBuffer_ != nil) [encoder useResource:pointPrimitiveBuffer_ usage:MTLResourceUsageRead];
  metal_rt::dispatchThreads(encoder, pathTracePipelineState_, width_, height_);
  [encoder endEncoding];
}

RenderBuffer MetalPathTracerBackend::downloadRenderBuffer() const {
  if (lastCommandBuffer_ != nil) {
    [lastCommandBuffer_ waitUntilCompleted];
    lastCommandBuffer_ = nil;
  }

  const auto* depth = static_cast<const float*>(depthBuffer_.contents);
  const auto* linearDepth = static_cast<const float*>(linearDepthBuffer_.contents);
  const auto* normal = static_cast<const simd_float4*>(normalBuffer_.contents);
  const auto* objectId = static_cast<const uint32_t*>(objectIdBuffer_.contents);
  const auto* diffuseAlbedo = static_cast<const simd_float4*>(diffuseAlbedoBuffer_.contents);
  const auto* specularAlbedo = static_cast<const simd_float4*>(specularAlbedoBuffer_.contents);
  const auto* roughness = static_cast<const float*>(roughnessAuxBuffer_.contents);

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
    latestBuffer_.diffuseAlbedo.resize(outPixels);
    latestBuffer_.specularAlbedo.resize(outPixels);
    latestBuffer_.roughness.resize(outPixels);

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
        latestBuffer_.diffuseAlbedo[outIdx] =
            glm::vec3(diffuseAlbedo[srcIdx].x, diffuseAlbedo[srcIdx].y, diffuseAlbedo[srcIdx].z);
        latestBuffer_.specularAlbedo[outIdx] =
            glm::vec3(specularAlbedo[srcIdx].x, specularAlbedo[srcIdx].y, specularAlbedo[srcIdx].z);
        latestBuffer_.roughness[outIdx] = roughness[srcIdx];
      }
    }

    toonPreset_->downloadAuxBuffers(latestBuffer_, lastUseFxaa_);
    if (outW != width_ || outH != height_) {
      RenderBuffer nativeContours;
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
          latestBuffer_.detailContour[outIdx] = nativeContours.detailContour[srcIdx];
          latestBuffer_.objectContourRaw[outIdx] = nativeContours.objectContourRaw[srcIdx];
          latestBuffer_.objectContour[outIdx] = nativeContours.objectContour[srcIdx];
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
    latestBuffer_.diffuseAlbedo.resize(pixelCount);
    latestBuffer_.specularAlbedo.resize(pixelCount);
    latestBuffer_.roughness.resize(pixelCount);
    for (size_t i = 0; i < pixelCount; ++i) {
      latestBuffer_.color[i] = glm::vec3(output[i].x, output[i].y, output[i].z);
      latestBuffer_.depth[i] = depth[i];
      latestBuffer_.linearDepth[i] = linearDepth[i];
      latestBuffer_.normal[i] = glm::vec3(normal[i].x, normal[i].y, normal[i].z);
      latestBuffer_.objectId[i] = objectId[i];
      latestBuffer_.diffuseAlbedo[i] = glm::vec3(diffuseAlbedo[i].x, diffuseAlbedo[i].y, diffuseAlbedo[i].z);
      latestBuffer_.specularAlbedo[i] = glm::vec3(specularAlbedo[i].x, specularAlbedo[i].y, specularAlbedo[i].z);
      latestBuffer_.roughness[i] = roughness[i];
    }

    metal_rt::IMetalPostProcessPreset* activePreset =
        lastUseToon_ ? toonPreset_.get() : standardPreset_.get();
    activePreset->downloadAuxBuffers(latestBuffer_, lastUseFxaa_);
  }

  return latestBuffer_;
}

id<MTLTexture> MetalPathTracerBackend::createPrivateTexture(MTLPixelFormat format, uint32_t w, uint32_t h) {
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:w height:h mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = MTLStorageModePrivate;
  return [device_ newTextureWithDescriptor:desc];
}

void MetalPathTracerBackend::ensureMetalFXResources(uint32_t outputWidth, uint32_t outputHeight) {
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
  metalFXTonemappedBuffer_ =
      [device_ newBufferWithLength:outputPixels * sizeof(simd_float4) options:MTLResourceStorageModeShared];
  metalFXToonBuffer_ = [device_ newBufferWithLength:sizeof(GPUToonUniforms) options:MTLResourceStorageModeShared];

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
    NSLog(@"[MetalFX] denoised scaler creation failed (input=%ux%u output=%ux%u)", width_, height_, outputWidth,
          outputHeight);
  }
}

void MetalPathTracerBackend::teardownMetalFXResources() {
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

} // namespace rt::metal_backend_internal
