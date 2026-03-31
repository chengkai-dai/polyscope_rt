#pragma once

#import <Metal/Metal.h>

#include <vector>

#include "rendering/scene_packer.h"

namespace metal_rt {

std::vector<MTLAxisAlignedBoundingBox> makeMetalBoundingBoxes(const std::vector<rt::PackedBoundingBox>& boxes);

} // namespace metal_rt
