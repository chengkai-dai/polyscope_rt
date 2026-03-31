#include "rendering/metal/metal_scene_builder.h"

namespace metal_rt {

std::vector<MTLAxisAlignedBoundingBox> makeMetalBoundingBoxes(const std::vector<rt::PackedBoundingBox>& boxes) {
  std::vector<MTLAxisAlignedBoundingBox> metalBoxes;
  metalBoxes.reserve(boxes.size());
  for (const rt::PackedBoundingBox& box : boxes) {
    MTLAxisAlignedBoundingBox metalBox;
    metalBox.min = {box.min.x, box.min.y, box.min.z};
    metalBox.max = {box.max.x, box.max.y, box.max.z};
    metalBoxes.push_back(metalBox);
  }
  return metalBoxes;
}

} // namespace metal_rt
