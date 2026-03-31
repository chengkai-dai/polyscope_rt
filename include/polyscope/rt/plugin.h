#pragma once

#include <limits>
#include <string>
#include <vector>

#include "glm/glm.hpp"
#include "polyscope/options.h"
#include "polyscope/polyscope.h"
#include "polyscope/rt/material_library.h"
#include "polyscope/view.h"

namespace polyscope {
namespace rt {

using namespace ::polyscope;

namespace options {
using namespace ::polyscope::options;
extern int maxBounces;
extern int samplesPerFrame;
extern float exposure;
extern float gamma;
extern float saturation;
} // namespace options

namespace state {
using namespace ::polyscope::state;
} // namespace state

namespace view {
using namespace ::polyscope::view;
} // namespace view

// Lifecycle
void init(std::string backend = "");
void show(size_t forFrames = std::numeric_limits<size_t>::max());
void frameTick();
void shutdown(bool allowMidFrameShutdown = false);
void removeEverything();
void enable();
void disable();
bool isEnabled();

// Runtime assets
void setShaderLibraryPath(std::string path);
std::string getShaderLibraryPath();

// Lighting
// Main light is an analytic directional light (sun-like). It affects shading
// but does not create visible scene geometry.
void setMainLight(glm::vec3 direction, glm::vec3 color, float intensity);
void setEnvironment(glm::vec3 tint, float intensity);
void setAmbientFloor(float value);
void setBackgroundColor(glm::vec3 color);

// Punctual lights (handle-based)
// Point / directional / spot lights are analytic lights. They illuminate the
// scene efficiently but are not directly visible as bulbs or panels. If you
// want a visible emitter in the render, add emissive geometry or an area light.
size_t addPointLight(glm::vec3 pos, glm::vec3 color, float intensity, float range = 0);
size_t addDirectionalLight(glm::vec3 dir, glm::vec3 color, float intensity);
size_t addSpotLight(glm::vec3 pos, glm::vec3 dir, glm::vec3 color, float intensity,
                    float innerCone, float outerCone);
void removeLight(size_t handle);
void removeAllLights();

// Area light. A finite rectangular analytic light; the current path tracer
// treats the rectangle as double-sided for both direct lighting and visible
// reflections. Use emissive geometry instead when the light source should be an
// explicit mesh in the scene.
void setAreaLight(glm::vec3 center, glm::vec3 u, glm::vec3 v, glm::vec3 emission);
void disableAreaLight();

// Per-mesh material override
void setMaterial(const std::string& meshName, float metallic, float roughness);
void setBaseColor(const std::string& meshName, glm::vec4 color);
// Turns an existing mesh into visible emissive geometry.
void setEmissive(const std::string& meshName, glm::vec3 emissive);
void setUnlitColor(const std::string& meshName, glm::vec3 color);
void setTransmission(const std::string& meshName, float transmission, float ior = 1.5f);
void applyMaterial(const std::string& meshName, const MaterialPreset& preset);
void clearMaterialOverride(const std::string& meshName);

// Render control
void resetAccumulation();
void exportPNG(const std::string& path, int targetSPP = 128);

} // namespace rt
} // namespace polyscope
