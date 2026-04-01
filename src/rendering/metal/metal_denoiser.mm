#include "rendering/metal/metal_denoiser.h"

#include <algorithm>
#include <cstring>

#include "rendering/metal/metal_device.h"

namespace metal_rt {

MetalTemporalDenoiser::MetalTemporalDenoiser(id<MTLDevice> device, id<MTLLibrary> library) : device_(device) {
  NSError* error = nil;
  auto tryCreatePipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
    id<MTLFunction> fn = [library newFunctionWithName:name];
    if (fn == nil) return nil;
    return [device_ newComputePipelineStateWithFunction:fn error:&error];
  };

  bufferToTexturePipelineState_ = tryCreatePipeline(@"bufferToTextureKernel");
  textureToBufferPipelineState_ = tryCreatePipeline(@"textureToBufferKernel");
  depthToTexturePipelineState_ = tryCreatePipeline(@"depthToTextureKernel");
  roughnessToTexturePipelineState_ = tryCreatePipeline(@"roughnessToTextureKernel");
  motionToTexturePipelineState_ = tryCreatePipeline(@"motionToTextureKernel");
}

void MetalTemporalDenoiser::resetHistory() {
  hasHistory_ = false;
}

bool MetalTemporalDenoiser::isActive() const {
  return active_ && tonemappedBuffer_ != nil;
}

uint32_t MetalTemporalDenoiser::outputWidth() const {
  return outputWidth_;
}

uint32_t MetalTemporalDenoiser::outputHeight() const {
  return outputHeight_;
}

id<MTLBuffer> MetalTemporalDenoiser::tonemappedBuffer() const {
  return tonemappedBuffer_;
}

id<MTLTexture> MetalTemporalDenoiser::createPrivateTexture(MTLPixelFormat format, uint32_t width, uint32_t height) {
  MTLTextureDescriptor* desc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format width:width height:height mipmapped:NO];
  desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
  desc.storageMode = MTLStorageModePrivate;
  return [device_ newTextureWithDescriptor:desc];
}

void MetalTemporalDenoiser::ensureResources(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth,
                                            uint32_t outputHeight) {
  if (scaler_ != nil && inputWidth_ == inputWidth && inputHeight_ == inputHeight &&
      outputWidth_ == outputWidth && outputHeight_ == outputHeight) {
    return;
  }

  inputWidth_ = inputWidth;
  inputHeight_ = inputHeight;
  outputWidth_ = outputWidth;
  outputHeight_ = outputHeight;

  inputTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, inputWidth, inputHeight);
  outputTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, outputWidth, outputHeight);
  depthTexture_ = createPrivateTexture(MTLPixelFormatR32Float, inputWidth, inputHeight);
  motionTexture_ = createPrivateTexture(MTLPixelFormatRG16Float, inputWidth, inputHeight);
  normalTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, inputWidth, inputHeight);
  diffuseAlbedoTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, inputWidth, inputHeight);
  specularAlbedoTexture_ = createPrivateTexture(MTLPixelFormatRGBA16Float, inputWidth, inputHeight);
  roughnessTexture_ = createPrivateTexture(MTLPixelFormatR16Float, inputWidth, inputHeight);

  const NSUInteger outputPixels = static_cast<NSUInteger>(outputWidth) * static_cast<NSUInteger>(outputHeight);
  outputBuffer_ = [device_ newBufferWithLength:outputPixels * sizeof(simd_float4) options:MTLResourceStorageModeShared];
  tonemappedBuffer_ =
      [device_ newBufferWithLength:outputPixels * sizeof(simd_float4) options:MTLResourceStorageModeShared];
  tonemapUniformBuffer_ = [device_ newBufferWithLength:sizeof(GPUToonUniforms) options:MTLResourceStorageModeShared];

  MTLFXTemporalDenoisedScalerDescriptor* desc = [[MTLFXTemporalDenoisedScalerDescriptor alloc] init];
  desc.inputWidth = inputWidth;
  desc.inputHeight = inputHeight;
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

  scaler_ = [desc newTemporalDenoisedScalerWithDevice:device_];
  if (scaler_ != nil) {
    scaler_.motionVectorScaleX = static_cast<float>(inputWidth);
    scaler_.motionVectorScaleY = static_cast<float>(inputHeight);
    scaler_.depthReversed = NO;
  } else {
    NSLog(@"[MetalFX] denoised scaler creation failed (input=%ux%u output=%ux%u)", inputWidth, inputHeight,
          outputWidth, outputHeight);
  }
}

void MetalTemporalDenoiser::teardown() {
  scaler_ = nil;
  inputTexture_ = nil;
  outputTexture_ = nil;
  depthTexture_ = nil;
  motionTexture_ = nil;
  normalTexture_ = nil;
  diffuseAlbedoTexture_ = nil;
  specularAlbedoTexture_ = nil;
  roughnessTexture_ = nil;
  outputBuffer_ = nil;
  tonemappedBuffer_ = nil;
  tonemapUniformBuffer_ = nil;
  inputWidth_ = 0;
  inputHeight_ = 0;
  outputWidth_ = 0;
  outputHeight_ = 0;
  active_ = false;
  hasHistory_ = false;
}

void MetalTemporalDenoiser::encodeBufferToHalf4Texture(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> buffer,
                                                       id<MTLTexture> texture, uint32_t width, uint32_t height) {
  id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
  [enc setComputePipelineState:bufferToTexturePipelineState_];
  [enc setBuffer:buffer offset:0 atIndex:0];
  [enc setTexture:texture atIndex:0];
  dispatchThreads(enc, bufferToTexturePipelineState_, width, height);
  [enc endEncoding];
}

void MetalTemporalDenoiser::encodeScalarBufferToTexture(id<MTLCommandBuffer> cmdBuf, id<MTLComputePipelineState> pipeline,
                                                        id<MTLBuffer> buffer, id<MTLTexture> texture,
                                                        uint32_t width, uint32_t height) {
  if (pipeline == nil || buffer == nil || texture == nil) return;
  id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
  [enc setComputePipelineState:pipeline];
  [enc setBuffer:buffer offset:0 atIndex:0];
  [enc setTexture:texture atIndex:0];
  dispatchThreads(enc, pipeline, width, height);
  [enc endEncoding];
}

bool MetalTemporalDenoiser::encode(id<MTLCommandBuffer> cmdBuf, const MetalDenoiserInputs& inputs, uint32_t inputWidth,
                                   uint32_t inputHeight, const GPUFrameUniforms& frame, const GPUCamera& camera,
                                   const rt::RenderConfig& config, IMetalPostProcessPreset& standardPreset) {
  const bool enabled = config.enableMetalFX && config.metalFXOutputWidth > 0 && config.metalFXOutputHeight > 0 &&
                       bufferToTexturePipelineState_ != nil && textureToBufferPipelineState_ != nil;
  if (!enabled) {
    if (active_) teardown();
    return false;
  }

  ensureResources(inputWidth, inputHeight, config.metalFXOutputWidth, config.metalFXOutputHeight);
  if (scaler_ == nil) {
    active_ = false;
    hasHistory_ = false;
    return false;
  }

  encodeBufferToHalf4Texture(cmdBuf, inputs.rawColor, inputTexture_, inputWidth, inputHeight);
  encodeBufferToHalf4Texture(cmdBuf, inputs.normal, normalTexture_, inputWidth, inputHeight);
  encodeBufferToHalf4Texture(cmdBuf, inputs.diffuseAlbedo, diffuseAlbedoTexture_, inputWidth, inputHeight);
  encodeBufferToHalf4Texture(cmdBuf, inputs.specularAlbedo, specularAlbedoTexture_, inputWidth, inputHeight);
  encodeScalarBufferToTexture(cmdBuf, depthToTexturePipelineState_, inputs.depth, depthTexture_, inputWidth, inputHeight);
  encodeScalarBufferToTexture(cmdBuf, motionToTexturePipelineState_, inputs.motionVector, motionTexture_, inputWidth,
                              inputHeight);
  encodeScalarBufferToTexture(cmdBuf, roughnessToTexturePipelineState_, inputs.roughness, roughnessTexture_, inputWidth,
                              inputHeight);

  scaler_.colorTexture = inputTexture_;
  scaler_.depthTexture = depthTexture_;
  scaler_.motionTexture = motionTexture_;
  scaler_.normalTexture = normalTexture_;
  scaler_.diffuseAlbedoTexture = diffuseAlbedoTexture_;
  scaler_.specularAlbedoTexture = specularAlbedoTexture_;
  scaler_.roughnessTexture = roughnessTexture_;
  scaler_.outputTexture = outputTexture_;
  scaler_.jitterOffsetX = frame.jitterOffset.x;
  scaler_.jitterOffsetY = frame.jitterOffset.y;
  scaler_.shouldResetHistory = !hasHistory_;
  scaler_.worldToViewMatrix = camera.viewMatrix;
  scaler_.viewToClipMatrix = camera.projectionMatrix;
  [scaler_ encodeToCommandBuffer:cmdBuf];

  {
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:textureToBufferPipelineState_];
    [enc setTexture:outputTexture_ atIndex:0];
    [enc setBuffer:outputBuffer_ offset:0 atIndex:0];
    dispatchThreads(enc, textureToBufferPipelineState_, outputWidth_, outputHeight_);
    [enc endEncoding];
  }

  GPUToonUniforms tonemapUniforms{};
  tonemapUniforms.width = outputWidth_;
  tonemapUniforms.height = outputHeight_;
  tonemapUniforms.exposure = config.renderMode == rt::RenderMode::Toon
                                 ? std::max(0.1f, config.toon.tonemapExposure)
                                 : std::max(0.1f, config.lighting.standardExposure);
  tonemapUniforms.gamma = config.renderMode == rt::RenderMode::Toon
                              ? std::max(0.1f, config.toon.tonemapGamma)
                              : std::max(0.1f, config.lighting.standardGamma);
  tonemapUniforms.saturation = config.renderMode == rt::RenderMode::Toon
                                   ? 1.0f
                                   : std::max(0.0f, config.lighting.standardSaturation);
  std::memcpy(tonemapUniformBuffer_.contents, &tonemapUniforms, sizeof(GPUToonUniforms));

  PostProcessBuffers buffers;
  buffers.rawColor = outputBuffer_;
  buffers.toonUniforms = tonemapUniformBuffer_;
  buffers.output = tonemappedBuffer_;
  standardPreset.encode(cmdBuf, buffers, outputWidth_, outputHeight_, config);

  hasHistory_ = true;
  active_ = true;
  return true;
}

} // namespace metal_rt
