#include <metal_stdlib>
using namespace metal;
#include "gpu_shared_types.h"
#include "shader_common.h"

[[kernel]] void tonemapKernel(device const float4* input [[buffer(0)]],
                              constant GPUToonUniforms& toon [[buffer(1)]],
                              device float4* output [[buffer(2)]],
                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= toon.width || gid.y >= toon.height) return;
  uint index = pixelIndex(gid.x, gid.y, toon.width);
  float4 color = input[index];
  float3 mapped = toneMapUncharted(color.xyz * toon.exposure, toon.gamma);
  mapped = max(mapped, float3(0.0f));
  mapped = applySaturation(mapped, toon.saturation);
  output[index] = float4(mapped, color.w);
}
