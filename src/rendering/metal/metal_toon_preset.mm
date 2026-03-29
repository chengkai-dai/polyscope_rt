#include "rendering/metal/metal_postprocess.h"
#include "rendering/metal/metal_device.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace metal_rt {

class MetalToonPreset final : public IMetalPostProcessPreset {
public:
  std::string name() const override { return "Toon"; }

  void createPipelines(id<MTLDevice> device, id<MTLLibrary> library) override {
    NSError* error = nil;

    auto create = [&](NSString* name) -> id<MTLComputePipelineState> {
      id<MTLFunction> fn = [library newFunctionWithName:name];
      id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&error];
      if (ps == nil) {
        throw std::runtime_error(std::string("Failed to create toon pipeline ") +
                                 name.UTF8String + ": " + buildNSErrorMessage(error));
      }
      return ps;
    };

    tonemapPipelineState_       = create(@"tonemapKernel");
    depthMinMaxPipelineState_   = create(@"depthMinMaxKernel");
    detailContourPipelineState_ = create(@"detailContourKernel");
    objectContourPipelineState_ = create(@"objectContourKernel");
    fxaaPipelineState_          = create(@"fxaaKernel");
    compositePipelineState_     = create(@"compositeKernel");
  }

  void resize(id<MTLDevice> device, uint32_t width, uint32_t height) override {
    const NSUInteger pixelCount = static_cast<NSUInteger>(width) * static_cast<NSUInteger>(height);
    tonemappedBuffer_          = [device newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    detailContourBuffer_       = [device newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    objectContourBuffer_       = [device newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    detailContourFxaaBuffer_   = [device newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    objectContourFxaaBuffer_   = [device newBufferWithLength:pixelCount * sizeof(simd_float4) options:MTLResourceStorageModeShared];
    depthMinMaxBuffer_         = [device newBufferWithLength:2 * sizeof(uint32_t) options:MTLResourceStorageModeShared];
  }

  void encode(id<MTLCommandBuffer> cmdBuf,
              const PostProcessBuffers& buffers,
              uint32_t width, uint32_t height,
              const rt::RenderConfig& config) override {
    bool needContours = config.toon.enabled &&
                        (config.toon.enableDetailContour || config.toon.enableObjectContour);

    if (needContours) {
      uint32_t init[2];
      float big = 10000.0f;
      std::memcpy(&init[0], &big, sizeof(float));
      init[1] = 0u;
      std::memcpy(depthMinMaxBuffer_.contents, init, sizeof(init));
    }

    // Tonemap
    {
      id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
      [enc setComputePipelineState:tonemapPipelineState_];
      [enc setBuffer:buffers.rawColor      offset:0 atIndex:0];
      [enc setBuffer:buffers.toonUniforms  offset:0 atIndex:1];
      [enc setBuffer:tonemappedBuffer_     offset:0 atIndex:2];
      dispatchThreads(enc, tonemapPipelineState_, width, height);
      [enc endEncoding];
    }

    if (needContours) {
      // Depth min/max
      {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:depthMinMaxPipelineState_];
        [enc setBuffer:buffers.linearDepth   offset:0 atIndex:0];
        [enc setBuffer:depthMinMaxBuffer_    offset:0 atIndex:1];
        [enc setBuffer:buffers.toonUniforms  offset:0 atIndex:2];
        dispatchThreads(enc, depthMinMaxPipelineState_, width, height);
        [enc endEncoding];
      }

      // Detail contour
      {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:detailContourPipelineState_];
        [enc setBuffer:buffers.linearDepth   offset:0 atIndex:0];
        [enc setBuffer:buffers.normal        offset:0 atIndex:1];
        [enc setBuffer:buffers.objectId      offset:0 atIndex:2];
        [enc setBuffer:buffers.toonUniforms  offset:0 atIndex:3];
        [enc setBuffer:depthMinMaxBuffer_    offset:0 atIndex:4];
        [enc setBuffer:detailContourBuffer_  offset:0 atIndex:5];
        dispatchThreads(enc, detailContourPipelineState_, width, height);
        [enc endEncoding];
      }

      // Object contour
      {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:objectContourPipelineState_];
        [enc setBuffer:buffers.objectId      offset:0 atIndex:0];
        [enc setBuffer:buffers.toonUniforms  offset:0 atIndex:1];
        [enc setBuffer:objectContourBuffer_  offset:0 atIndex:2];
        dispatchThreads(enc, objectContourPipelineState_, width, height);
        [enc endEncoding];
      }

      id<MTLBuffer> detailInput = detailContourBuffer_;
      id<MTLBuffer> objectInput = objectContourBuffer_;

      if (config.toon.useFxaa) {
        {
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:fxaaPipelineState_];
          [enc setBuffer:detailContourBuffer_     offset:0 atIndex:0];
          [enc setBuffer:buffers.toonUniforms     offset:0 atIndex:1];
          [enc setBuffer:detailContourFxaaBuffer_ offset:0 atIndex:2];
          dispatchThreads(enc, fxaaPipelineState_, width, height);
          [enc endEncoding];
        }
        {
          id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
          [enc setComputePipelineState:fxaaPipelineState_];
          [enc setBuffer:objectContourBuffer_     offset:0 atIndex:0];
          [enc setBuffer:buffers.toonUniforms     offset:0 atIndex:1];
          [enc setBuffer:objectContourFxaaBuffer_ offset:0 atIndex:2];
          dispatchThreads(enc, fxaaPipelineState_, width, height);
          [enc endEncoding];
        }
        detailInput = detailContourFxaaBuffer_;
        objectInput = objectContourFxaaBuffer_;
      }

      // Composite
      {
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:compositePipelineState_];
        [enc setBuffer:tonemappedBuffer_     offset:0 atIndex:0];
        [enc setBuffer:detailInput           offset:0 atIndex:1];
        [enc setBuffer:objectInput           offset:0 atIndex:2];
        [enc setBuffer:buffers.toonUniforms  offset:0 atIndex:3];
        [enc setBuffer:buffers.output        offset:0 atIndex:4];
        dispatchThreads(enc, compositePipelineState_, width, height);
        [enc endEncoding];
      }
    } else {
      // No contours: copy tonemapped → output
      id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
      [blit copyFromBuffer:tonemappedBuffer_ sourceOffset:0
                  toBuffer:buffers.output destinationOffset:0
                      size:tonemappedBuffer_.length];
      [blit endEncoding];
    }
  }

  void downloadAuxBuffers(rt::RenderBuffer& buffer, bool usedFxaa) const override {
    const size_t pixelCount = static_cast<size_t>(buffer.width) * static_cast<size_t>(buffer.height);
    buffer.detailContour.resize(pixelCount);
    buffer.detailContourRaw.resize(pixelCount);
    buffer.objectContour.resize(pixelCount);
    buffer.objectContourRaw.resize(pixelCount);

    if (detailContourBuffer_ == nil || objectContourBuffer_ == nil) {
      buffer.detailContour.assign(pixelCount, 0.0f);
      buffer.detailContourRaw.assign(pixelCount, 0.0f);
      buffer.objectContour.assign(pixelCount, 0.0f);
      buffer.objectContourRaw.assign(pixelCount, 0.0f);
      return;
    }

    const auto* detailContourRaw = static_cast<const simd_float4*>(detailContourBuffer_.contents);
    const auto* objectContourRaw = static_cast<const simd_float4*>(objectContourBuffer_.contents);
    const auto* detailContour = static_cast<const simd_float4*>(
        (usedFxaa ? detailContourFxaaBuffer_ : detailContourBuffer_).contents);
    const auto* objectContour = static_cast<const simd_float4*>(
        (usedFxaa ? objectContourFxaaBuffer_ : objectContourBuffer_).contents);

    for (size_t i = 0; i < pixelCount; ++i) {
      buffer.detailContourRaw[i] = detailContourRaw[i].x;
      buffer.detailContour[i]    = detailContour[i].x;
      buffer.objectContourRaw[i] = objectContourRaw[i].x;
      buffer.objectContour[i]    = objectContour[i].x;
    }
  }

private:
  id<MTLComputePipelineState> tonemapPipelineState_       = nil;
  id<MTLComputePipelineState> depthMinMaxPipelineState_   = nil;
  id<MTLComputePipelineState> detailContourPipelineState_ = nil;
  id<MTLComputePipelineState> objectContourPipelineState_ = nil;
  id<MTLComputePipelineState> fxaaPipelineState_          = nil;
  id<MTLComputePipelineState> compositePipelineState_     = nil;

  id<MTLBuffer> tonemappedBuffer_        = nil;
  id<MTLBuffer> detailContourBuffer_     = nil;
  id<MTLBuffer> objectContourBuffer_     = nil;
  id<MTLBuffer> detailContourFxaaBuffer_ = nil;
  id<MTLBuffer> objectContourFxaaBuffer_ = nil;
  id<MTLBuffer> depthMinMaxBuffer_       = nil;
};

std::unique_ptr<IMetalPostProcessPreset> createToonPreset() {
  return std::make_unique<MetalToonPreset>();
}

} // namespace metal_rt
