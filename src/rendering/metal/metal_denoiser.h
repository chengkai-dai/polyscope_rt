#pragma once

#include <cstdint>

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include "rendering/gpu_shared_types.h"
#include "rendering/metal/metal_postprocess.h"
#include "rendering/ray_tracing_types.h"

namespace metal_rt {

struct MetalDenoiserInputs {
  id<MTLBuffer> rawColor = nil;
  id<MTLBuffer> depth = nil;
  id<MTLBuffer> normal = nil;
  id<MTLBuffer> diffuseAlbedo = nil;
  id<MTLBuffer> specularAlbedo = nil;
  id<MTLBuffer> roughness = nil;
  id<MTLBuffer> motionVector = nil;
};

class MetalTemporalDenoiser {
public:
  MetalTemporalDenoiser(id<MTLDevice> device, id<MTLLibrary> library);

  void resetHistory();
  bool encode(id<MTLCommandBuffer> cmdBuf, const MetalDenoiserInputs& inputs, uint32_t inputWidth,
              uint32_t inputHeight, const GPUFrameUniforms& frame, const GPUCamera& camera,
              const rt::RenderConfig& config, IMetalPostProcessPreset& standardPreset);

  bool isActive() const;
  uint32_t outputWidth() const;
  uint32_t outputHeight() const;
  id<MTLBuffer> tonemappedBuffer() const;

private:
  id<MTLTexture> createPrivateTexture(MTLPixelFormat format, uint32_t width, uint32_t height);
  void ensureResources(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight);
  void teardown();
  void encodeBufferToHalf4Texture(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> buffer, id<MTLTexture> texture,
                                  uint32_t width, uint32_t height);
  void encodeScalarBufferToTexture(id<MTLCommandBuffer> cmdBuf, id<MTLComputePipelineState> pipeline,
                                   id<MTLBuffer> buffer, id<MTLTexture> texture,
                                   uint32_t width, uint32_t height);

  id<MTLDevice> device_ = nil;

  id<MTLComputePipelineState> bufferToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> textureToBufferPipelineState_ = nil;
  id<MTLComputePipelineState> depthToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> roughnessToTexturePipelineState_ = nil;
  id<MTLComputePipelineState> motionToTexturePipelineState_ = nil;

  id<MTLTexture> inputTexture_ = nil;
  id<MTLTexture> outputTexture_ = nil;
  id<MTLTexture> depthTexture_ = nil;
  id<MTLTexture> motionTexture_ = nil;
  id<MTLTexture> normalTexture_ = nil;
  id<MTLTexture> diffuseAlbedoTexture_ = nil;
  id<MTLTexture> specularAlbedoTexture_ = nil;
  id<MTLTexture> roughnessTexture_ = nil;

  id<MTLBuffer> outputBuffer_ = nil;
  id<MTLBuffer> tonemappedBuffer_ = nil;
  id<MTLBuffer> tonemapUniformBuffer_ = nil;
  id<MTLFXTemporalDenoisedScaler> scaler_ = nil;

  uint32_t inputWidth_ = 0;
  uint32_t inputHeight_ = 0;
  uint32_t outputWidth_ = 0;
  uint32_t outputHeight_ = 0;
  bool hasHistory_ = false;
  bool active_ = false;
};

} // namespace metal_rt
