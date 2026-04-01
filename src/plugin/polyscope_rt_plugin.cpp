#include "polyscope/rt/plugin.h"

#include <algorithm>
#include <functional>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/simple_triangle_mesh.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"

#include "plugin/polyscope_rt_runtime.h"
#include "rendering/ray_tracing_types.h"

namespace polyscope {
namespace rt {

namespace options {
int maxBounces = 6;
int samplesPerFrame = 1;
float exposure = ::rt::makeDefaultAppearanceConfig().lighting.standardExposure;
float gamma = ::rt::makeDefaultAppearanceConfig().lighting.standardGamma;
float saturation = ::rt::makeDefaultAppearanceConfig().lighting.standardSaturation;
} // namespace options

namespace {

PolyscopeRtRuntime* g_runtime = nullptr;
bool g_ownsRuntime = false;
std::function<void()> g_chainedUserCallback;
std::string g_shaderLibraryPath;
size_t g_nextLightHandle = 1;
std::map<size_t, ::rt::RTPunctualLight> g_apiLights;
std::unordered_map<std::string, ::rt::MaterialOverride> g_materialOverrides;

void dispatchPolyscopeUserCallback();

bool hasInstalledUserCallbackHook() {
  auto* target = polyscope::state::userCallback.target<void (*)()>();
  return target != nullptr && *target == &dispatchPolyscopeUserCallback;
}

void installUserCallbackHook() {
  if (hasInstalledUserCallbackHook()) return;
  g_chainedUserCallback = polyscope::state::userCallback;
  polyscope::state::userCallback = &dispatchPolyscopeUserCallback;
}

void removeUserCallbackHook() {
  if (hasInstalledUserCallbackHook()) {
    polyscope::state::userCallback = g_chainedUserCallback;
  }
  g_chainedUserCallback = nullptr;
}

void bootstrapRuntime() {
  if (g_runtime) return;
  g_runtime = new PolyscopeRtRuntime();
  g_ownsRuntime = true;
}

void setRasterMeshColor(const std::string& structureName, const glm::vec3& color) {
  if (::polyscope::hasSurfaceMesh(structureName)) {
    ::polyscope::getSurfaceMesh(structureName)->setSurfaceColor(color);
  } else if (::polyscope::hasSimpleTriangleMesh(structureName)) {
    ::polyscope::getSimpleTriangleMesh(structureName)->setSurfaceColor(color);
  } else if (::polyscope::hasVolumeMesh(structureName)) {
    ::polyscope::getVolumeMesh(structureName)->setColor(color);
  }
}

void setRasterMeshMaterial(const std::string& structureName, const std::string& materialName) {
  if (::polyscope::hasSurfaceMesh(structureName)) {
    ::polyscope::getSurfaceMesh(structureName)->setMaterial(materialName);
  } else if (::polyscope::hasSimpleTriangleMesh(structureName)) {
    ::polyscope::getSimpleTriangleMesh(structureName)->setMaterial(materialName);
  } else if (::polyscope::hasVolumeMesh(structureName)) {
    ::polyscope::getVolumeMesh(structureName)->setMaterial(materialName);
  }
}

void setRasterMeshTransparency(const std::string& structureName, float transparency) {
  if (::polyscope::hasSurfaceMesh(structureName)) {
    ::polyscope::getSurfaceMesh(structureName)->setTransparency(transparency);
  } else if (::polyscope::hasSimpleTriangleMesh(structureName)) {
    ::polyscope::getSimpleTriangleMesh(structureName)->setTransparency(transparency);
  } else if (::polyscope::hasVolumeMesh(structureName)) {
    ::polyscope::getVolumeMesh(structureName)->setTransparency(transparency);
  }
}

void syncOptionsToRuntime() {
  if (!g_runtime) return;

  g_runtime->setMaxBounces(options::maxBounces);
  g_runtime->setPreviewSamplesPerFrame(options::samplesPerFrame);

  ::rt::AppearanceConfig config = g_runtime->getAppearance();
  config.lighting.standardExposure = options::exposure;
  config.lighting.standardGamma = options::gamma;
  config.lighting.standardSaturation = options::saturation;
  g_runtime->setAppearance(config);

  // Sync lights
  std::vector<::rt::RTPunctualLight> lights;
  lights.reserve(g_apiLights.size());
  for (const auto& [handle, light] : g_apiLights) {
    lights.push_back(light);
  }
  g_runtime->setApiLights(lights);
  g_runtime->setMaterialOverrides(g_materialOverrides);
}

void syncOptionsFromRuntime() {
  if (!g_runtime) return;
  ::rt::AppearanceConfig config = g_runtime->getAppearance();
  options::exposure = config.lighting.standardExposure;
  options::gamma = config.lighting.standardGamma;
  options::saturation = config.lighting.standardSaturation;
}

void dispatchPolyscopeUserCallback() {
  if (g_chainedUserCallback) {
    g_chainedUserCallback();
  }
  if (!g_runtime) return;
  syncOptionsToRuntime();
  g_runtime->tick();
  syncOptionsFromRuntime();
}

} // namespace

void init(std::string backend) {
  if (!::polyscope::isInitialized()) {
    ::polyscope::options::groundPlaneMode = ::polyscope::GroundPlaneMode::None;
    ::polyscope::init(std::move(backend));
  }
  bootstrapRuntime();
  installUserCallbackHook();
  if (g_runtime) g_runtime->setEnabled(true);
}

void show(size_t forFrames) {
  if (!::polyscope::isInitialized()) {
    init();
  } else {
    bootstrapRuntime();
    installUserCallbackHook();
  }
  enable();
  ::polyscope::show(forFrames);
}

void frameTick() {
  if (!::polyscope::isInitialized()) {
    init();
  } else {
    bootstrapRuntime();
    installUserCallbackHook();
  }
  enable();
  ::polyscope::frameTick();
}

void shutdown(bool allowMidFrameShutdown) {
  if (g_runtime) g_runtime->setEnabled(false);
  removeUserCallbackHook();
  g_apiLights.clear();  g_materialOverrides.clear();
  g_nextLightHandle = 1;
  if (g_ownsRuntime) {
    delete g_runtime;
  }
  g_runtime = nullptr;
  g_ownsRuntime = false;
  ::polyscope::shutdown(allowMidFrameShutdown);
}

void removeEverything() {
  ::polyscope::removeEverything();
  if (g_runtime) {
    g_runtime->resetAccumulation();
  }
}

void enable() {
  bootstrapRuntime();
  installUserCallbackHook();
  if (g_runtime) g_runtime->setEnabled(true);
}

void disable() {
  if (g_runtime) g_runtime->setEnabled(false);
}

bool isEnabled() {
  return g_runtime ? g_runtime->isEnabled() : false;
}

void setShaderLibraryPath(std::string path) {
  g_shaderLibraryPath = std::move(path);
}

std::string getShaderLibraryPath() {
  return g_shaderLibraryPath;
}

void setMainLight(glm::vec3 direction, glm::vec3 color, float intensity) {
  if (!g_runtime) return;
  ::rt::AppearanceConfig config = g_runtime->getAppearance();
  config.lighting.mainLightDirection = glm::normalize(direction);
  config.lighting.mainLightColor = color;
  config.lighting.mainLightIntensity = intensity;
  g_runtime->setAppearance(config);
}

void setMainLightAngularRadius(float degrees) {
  if (!g_runtime) return;
  ::rt::AppearanceConfig config = g_runtime->getAppearance();
  config.lighting.mainLightAngularRadius = glm::radians(std::max(degrees, 0.0f));
  g_runtime->setAppearance(config);
}

void setEnvironment(glm::vec3 tint, float intensity) {
  if (!g_runtime) return;
  ::rt::AppearanceConfig config = g_runtime->getAppearance();
  config.lighting.environmentTint = tint;
  config.lighting.environmentIntensity = intensity;
  g_runtime->setAppearance(config);
}

void setAmbientFloor(float value) {
  if (!g_runtime) return;
  ::rt::AppearanceConfig config = g_runtime->getAppearance();
  config.lighting.ambientFloor = value;
  g_runtime->setAppearance(config);
}

void setBackgroundColor(glm::vec3 color) {
  if (!g_runtime) return;
  ::rt::AppearanceConfig config = g_runtime->getAppearance();
  config.lighting.backgroundColor = color;
  g_runtime->setAppearance(config);
}

size_t addPointLight(glm::vec3 pos, glm::vec3 color, float intensity, float range) {
  ::rt::RTPunctualLight light;
  light.type = ::rt::RTPunctualLightType::Point;
  light.position = pos;
  light.color = color;
  light.intensity = intensity;
  light.range = range;
  size_t handle = g_nextLightHandle++;
  g_apiLights[handle] = light;
  return handle;
}

size_t addDirectionalLight(glm::vec3 dir, glm::vec3 color, float intensity) {
  ::rt::RTPunctualLight light;
  light.type = ::rt::RTPunctualLightType::Directional;
  light.direction = glm::normalize(dir);
  light.color = color;
  light.intensity = intensity;
  size_t handle = g_nextLightHandle++;
  g_apiLights[handle] = light;
  return handle;
}

size_t addSpotLight(glm::vec3 pos, glm::vec3 dir, glm::vec3 color, float intensity,
                    float innerCone, float outerCone) {
  ::rt::RTPunctualLight light;
  light.type = ::rt::RTPunctualLightType::Spot;
  light.position = pos;
  light.direction = glm::normalize(dir);
  light.color = color;
  light.intensity = intensity;
  light.innerConeAngle = innerCone;
  light.outerConeAngle = outerCone;
  size_t handle = g_nextLightHandle++;
  g_apiLights[handle] = light;
  return handle;
}

void removeLight(size_t handle) {
  g_apiLights.erase(handle);
}

void removeAllLights() {
  g_apiLights.clear();
}

void setAreaLight(glm::vec3 center, glm::vec3 u, glm::vec3 v, glm::vec3 emission) {
  if (!g_runtime) return;
  ::rt::AppearanceConfig config = g_runtime->getAppearance();
  config.lighting.enableAreaLight = true;
  config.lighting.areaLightCenter = center;
  config.lighting.areaLightU = u;
  config.lighting.areaLightV = v;
  config.lighting.areaLightEmission = emission;
  g_runtime->setAppearance(config);
}

void disableAreaLight() {
  if (!g_runtime) return;
  ::rt::AppearanceConfig config = g_runtime->getAppearance();
  config.lighting.enableAreaLight = false;
  g_runtime->setAppearance(config);
}

void setMaterial(const std::string& meshName, float metallic, float roughness) {
  auto& ov = g_materialOverrides[meshName];
  ov.metallic = metallic;
  ov.roughness = roughness;
}

void setBaseColor(const std::string& meshName, glm::vec4 color) {
  g_materialOverrides[meshName].baseColor = color;
  setRasterMeshColor(meshName, glm::vec3(color));
}

void setEmissive(const std::string& meshName, glm::vec3 emissive) {
  g_materialOverrides[meshName].emissive = emissive;
  setRasterMeshColor(meshName, glm::clamp(emissive, glm::vec3(0.0f), glm::vec3(1.0f)));
  setRasterMeshMaterial(meshName, "flat");
}

void setUnlitColor(const std::string& meshName, glm::vec3 color) {
  auto& ov = g_materialOverrides[meshName];
  ov.baseColor = glm::vec4(color, 1.0f);
  ov.emissive = glm::vec3(0.0f);
  ov.metallic = 0.0f;
  ov.roughness = 1.0f;
  ov.unlit = true;
  setRasterMeshColor(meshName, color);
  setRasterMeshMaterial(meshName, "flat");
  setRasterMeshTransparency(meshName, 1.0f);
}

void setTransmission(const std::string& meshName, float transmission, float ior) {
  auto& ov = g_materialOverrides[meshName];
  ov.transmission = transmission;
  ov.ior = ior;
}

void applyMaterial(const std::string& meshName, const MaterialPreset& preset) {
  auto& ov = g_materialOverrides[meshName];
  ov.baseColor = preset.baseColor;
  ov.metallic = preset.metallic;
  ov.roughness = preset.roughness;
  ov.emissive = preset.emissive;
  ov.transmission = preset.transmission;
  ov.ior = preset.ior;
  ov.opacity = preset.opacity;
  ov.unlit = preset.unlit;
}

void clearMaterialOverride(const std::string& meshName) {
  g_materialOverrides.erase(meshName);
}

void resetAccumulation() {
  if (g_runtime) g_runtime->resetAccumulation();
}

void exportPNG(const std::string& path, int targetSPP) {
  if (g_runtime) g_runtime->exportPNG(path, targetSPP);
}

} // namespace rt
} // namespace polyscope
