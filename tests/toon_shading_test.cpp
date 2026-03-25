#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include "rendering/toon_shading.h"

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
  try {
    rt::RenderBuffer buffer;
    buffer.width = 3;
    buffer.height = 3;
    buffer.color.assign(9, glm::vec3(0.72f, 0.42f, 0.18f));
    buffer.depth.assign(9, 0.4f);
    buffer.linearDepth.assign(9, 2.0f);
    buffer.normal.assign(9, glm::vec3(0.0f, 0.0f, 1.0f));
    buffer.objectId.assign(9, 1u);

    buffer.objectId[0] = 0u;
    buffer.objectId[1] = 0u;
    buffer.objectId[2] = 0u;
    buffer.color[4] = glm::vec3(0.78f, 0.50f, 0.24f);
    buffer.normal[5] = glm::normalize(glm::vec3(1.0f, 0.0f, 0.2f));
    buffer.objectId[7] = 2u;
    buffer.linearDepth[1] = 4.5f;

    rt::ToonSettings settings;
    settings.enabled = true;
    settings.backgroundColor = glm::vec3(1.0f);
    settings.edgeThickness = 1.0f;
    settings.depthThreshold = 1.0f;
    settings.normalThreshold = 0.5f;
    settings.enableDetailContour = true;
    settings.enableObjectContour = true;
    settings.detailContourStrength = 1.0f;
    settings.objectContourStrength = 1.0f;
    settings.edgeColor = glm::vec3(0.0f);

    std::vector<glm::vec3> shaded = rt::applyToonShading(buffer, settings);
    require(shaded.size() == buffer.color.size(), "unexpected toon output size");

    const glm::vec3 center = shaded[4];
    const glm::vec3 corner = shaded[0];
    require(std::isfinite(corner.r) && std::isfinite(corner.g) && std::isfinite(corner.b),
            "background-side pixel should stay finite");
    require(glm::length(corner - buffer.color[0]) > 1e-4f, "background-side pixel should not keep the original surface color");
    require(glm::length(center - buffer.color[4]) > 1e-4f, "contour compositing should affect interior edge pixels");
    require(center.r <= buffer.color[4].r + 1e-4f, "line compositing should not brighten the pixel unexpectedly");

    std::cout << "toon_shading_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "toon_shading_test failed: " << e.what() << std::endl;
    return 1;
  }
}
