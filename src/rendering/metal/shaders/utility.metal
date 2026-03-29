#include <metal_stdlib>
using namespace metal;
#include "gpu_shared_types.h"

[[kernel]] void bufferToTextureKernel(device const float4* input [[buffer(0)]],
                                      texture2d<half, access::write> output [[texture(0)]],
                                      uint2 gid [[thread_position_in_grid]]) {
  uint w = output.get_width();
  uint h = output.get_height();
  if (gid.x >= w || gid.y >= h) return;
  output.write(half4(input[gid.y * w + gid.x]), gid);
}

[[kernel]] void textureToBufferKernel(texture2d<half, access::read> input [[texture(0)]],
                                      device float4* output [[buffer(0)]],
                                      uint2 gid [[thread_position_in_grid]]) {
  uint w = input.get_width();
  uint h = input.get_height();
  if (gid.x >= w || gid.y >= h) return;
  output[gid.y * w + gid.x] = float4(input.read(gid));
}

[[kernel]] void depthToTextureKernel(device const float* input [[buffer(0)]],
                                     texture2d<float, access::write> output [[texture(0)]],
                                     uint2 gid [[thread_position_in_grid]]) {
  uint w = output.get_width();
  uint h = output.get_height();
  if (gid.x >= w || gid.y >= h) return;
  output.write(float4(input[gid.y * w + gid.x], 0, 0, 0), gid);
}

[[kernel]] void roughnessToTextureKernel(device const float* input [[buffer(0)]],
                                          texture2d<half, access::write> output [[texture(0)]],
                                          uint2 gid [[thread_position_in_grid]]) {
  uint w = output.get_width();
  uint h = output.get_height();
  if (gid.x >= w || gid.y >= h) return;
  output.write(half4(half(input[gid.y * w + gid.x]), 0, 0, 0), gid);
}

[[kernel]] void motionToTextureKernel(device const float4* input [[buffer(0)]],
                                       texture2d<half, access::write> output [[texture(0)]],
                                       uint2 gid [[thread_position_in_grid]]) {
  uint w = output.get_width();
  uint h = output.get_height();
  if (gid.x >= w || gid.y >= h) return;
  float4 v = input[gid.y * w + gid.x];
  output.write(half4(v.x, v.y, 0, 0), gid);
}
