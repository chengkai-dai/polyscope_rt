#pragma once

#include <vector>

#include "rendering/ray_tracing_types.h"

namespace rt {

std::vector<glm::vec3> applyToonShading(const RenderBuffer& buffer, const ToonSettings& settings);

} // namespace rt
