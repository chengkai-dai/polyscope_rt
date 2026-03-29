#include "rendering/metal/metal_postprocess.h"
#include "rendering/metal/metal_device.h"

#include <stdexcept>
#include <string>

namespace metal_rt {

class MetalStandardPreset final : public IMetalPostProcessPreset {
public:
  std::string name() const override { return "Standard"; }

  void createPipelines(id<MTLDevice> device, id<MTLLibrary> library) override {
    NSError* error = nil;
    id<MTLFunction> tonemapKernel = [library newFunctionWithName:@"tonemapKernel"];
    tonemapPipelineState_ = [device newComputePipelineStateWithFunction:tonemapKernel error:&error];
    if (tonemapPipelineState_ == nil) {
      throw std::runtime_error(std::string("Failed to create tonemap pipeline: ") +
                               buildNSErrorMessage(error));
    }
  }

  void resize(id<MTLDevice> /*device*/, uint32_t /*width*/, uint32_t /*height*/) override {
    // Standard mode has no intermediate buffers to resize.
  }

  void encode(id<MTLCommandBuffer> cmdBuf,
              const PostProcessBuffers& buffers,
              uint32_t width, uint32_t height,
              const rt::RenderConfig& /*config*/) override {
    id<MTLComputeCommandEncoder> tonemapEncoder = [cmdBuf computeCommandEncoder];
    [tonemapEncoder setComputePipelineState:tonemapPipelineState_];
    [tonemapEncoder setBuffer:buffers.rawColor   offset:0 atIndex:0];
    [tonemapEncoder setBuffer:buffers.toonUniforms offset:0 atIndex:1];
    [tonemapEncoder setBuffer:buffers.output      offset:0 atIndex:2];
    dispatchThreads(tonemapEncoder, tonemapPipelineState_, width, height);
    [tonemapEncoder endEncoding];
  }

  void downloadAuxBuffers(rt::RenderBuffer& buffer, bool /*usedFxaa*/) const override {
    const size_t pixelCount = static_cast<size_t>(buffer.width) * static_cast<size_t>(buffer.height);
    buffer.detailContour.assign(pixelCount, 0.0f);
    buffer.detailContourRaw.assign(pixelCount, 0.0f);
    buffer.objectContour.assign(pixelCount, 0.0f);
    buffer.objectContourRaw.assign(pixelCount, 0.0f);
  }

private:
  id<MTLComputePipelineState> tonemapPipelineState_ = nil;
};

std::unique_ptr<IMetalPostProcessPreset> createStandardPreset() {
  return std::make_unique<MetalStandardPreset>();
}

} // namespace metal_rt
