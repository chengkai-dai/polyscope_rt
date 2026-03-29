#pragma once

#import <Metal/Metal.h>
#include <memory>
#include <string>
#include "rendering/ray_tracing_types.h"
#include "rendering/gpu_shared_types.h"

namespace metal_rt {

// Buffers the postprocess preset reads from (filled by the path-trace pass)
// and the final output buffer it writes to.
struct PostProcessBuffers {
  id<MTLBuffer> rawColor;
  id<MTLBuffer> linearDepth;
  id<MTLBuffer> normal;
  id<MTLBuffer> objectId;
  id<MTLBuffer> toonUniforms;
  id<MTLBuffer> output;
};

// Abstract postprocess preset.  The Metal backend owns one preset per
// RenderMode and delegates postprocess encoding to the active one.
class IMetalPostProcessPreset {
public:
  virtual ~IMetalPostProcessPreset() = default;
  virtual std::string name() const = 0;

  virtual void createPipelines(id<MTLDevice> device, id<MTLLibrary> library) = 0;
  virtual void resize(id<MTLDevice> device, uint32_t width, uint32_t height) = 0;

  // Encode all postprocess compute passes.
  // `buffers.output` receives the final composited image.
  virtual void encode(id<MTLCommandBuffer> cmdBuf,
                      const PostProcessBuffers& buffers,
                      uint32_t width, uint32_t height,
                      const rt::RenderConfig& config) = 0;

  // Copy any preset-specific auxiliary data (contour maps, etc.)
  // into the download buffer.
  virtual void downloadAuxBuffers(rt::RenderBuffer& buffer, bool usedFxaa) const = 0;
};

std::unique_ptr<IMetalPostProcessPreset> createStandardPreset();
std::unique_ptr<IMetalPostProcessPreset> createToonPreset();

} // namespace metal_rt
