#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;
#include "gpu_shared_types.h"

struct BoundingBoxIntersectionResult {
  bool  accept   [[accept_intersection]];
  float distance [[distance]];
};

// Custom bounding-box intersection function for analytic sphere ray-tracing.
// [[visible]] is required so the function is accessible by name from the host
// and can be inserted into an MTLIntersectionFunctionTable.
[[intersection(bounding_box, instancing)]]
BoundingBoxIntersectionResult sphereIntersection(float3 rayOrigin         [[origin]],
                                                  float3 rayDirection      [[direction]],
                                                  float  rayMinDistance    [[min_distance]],
                                                  float  rayMaxDistance    [[max_distance]],
                                                  uint   primitiveIndex    [[primitive_id]],
                                                  device const GPUPointPrimitive* points [[buffer(25)]]) {
  BoundingBoxIntersectionResult result;
  result.accept = false;

  GPUPointPrimitive pt = points[primitiveIndex];
  float3 center = pt.center_radius.xyz;
  float  radius = pt.center_radius.w;

  float3 oc = rayOrigin - center;
  float  b  = dot(oc, rayDirection);
  float  c  = dot(oc, oc) - radius * radius;
  float  disc = b * b - c;
  if (disc < 0.0f) return result;

  float sqrtDisc = sqrt(disc);
  float t = -b - sqrtDisc;
  if (t < rayMinDistance || t > rayMaxDistance) {
    t = -b + sqrtDisc;
    if (t < rayMinDistance || t > rayMaxDistance) return result;
  }

  result.accept   = true;
  result.distance = t;
  return result;
}
