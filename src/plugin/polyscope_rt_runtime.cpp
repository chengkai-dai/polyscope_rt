#include "plugin/polyscope_rt_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <ctime>
#include <tuple>
#include <vector>

#include "polyscope/rt/plugin.h"
#include "polyscope/floating_quantity_structure.h"
#include "polyscope/options.h"
#include "polyscope/polyscope.h"
#include "polyscope/view.h"

#include "stb_image_write.h"

namespace {

constexpr const char* kRenderImageName = "Metal Path Tracing";

rt::AppearanceConfig makeDefaultAppearance() {
  rt::AppearanceConfig config;
  config.mode = rt::RenderMode::Standard;
  config.lighting.mainLightIntensity = 0.0f;
  config.lighting.environmentIntensity = 0.3f;
  config.toon.enabled = false;
  config.toon.edgeThickness = 1.0f;
  config.toon.depthThreshold = 0.015f;
  config.toon.normalThreshold = 0.12f;
  config.toon.enableDetailContour = true;
  config.toon.enableObjectContour = true;
  config.toon.enableNormalEdge = true;
  config.toon.enableDepthEdge = true;
  config.toon.detailContourStrength = 1.0f;
  config.toon.objectContourStrength = 1.0f;
  config.toon.edgeColor = glm::vec3(0.3f);
  config.toon.useFxaa = true;
  config.toon.tonemapExposure = 3.0f;
  config.toon.tonemapGamma = 2.2f;
  config.toon.backgroundColor = config.lighting.backgroundColor;
  config.toon.bandCount = config.lighting.toonBandCount;
  return config;
}

bool nearlyEqual(float a, float b, float eps = 1e-4f) { return std::abs(a - b) <= eps; }

bool nearlyEqualVec3(const glm::vec3& a, const glm::vec3& b, float eps = 1e-4f) {
  return glm::length(a - b) <= eps;
}

template <typename T>
void hashBytes(uint64_t& seed, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  for (size_t i = 0; i < sizeof(T); ++i) {
    seed ^= static_cast<uint64_t>(bytes[i]) + 0x9e3779b97f4a7c15ULL + (seed << 6u) + (seed >> 2u);
  }
}

uint64_t computeRenderStateHash(const rt::RenderConfig& config) {
  uint64_t seed = 1469598103934665603ULL;
  hashBytes(seed, config.renderMode);
  hashBytes(seed, config.maxBounces);
  hashBytes(seed, config.lighting.backgroundColor);
  hashBytes(seed, config.lighting.mainLightDirection);
  hashBytes(seed, config.lighting.mainLightColor);
  hashBytes(seed, config.lighting.mainLightIntensity);
  hashBytes(seed, config.lighting.ambientFloor);
  hashBytes(seed, config.lighting.environmentTint);
  hashBytes(seed, config.lighting.environmentIntensity);
  hashBytes(seed, config.lighting.enableAreaLight);
  hashBytes(seed, config.lighting.areaLightCenter);
  hashBytes(seed, config.lighting.areaLightU);
  hashBytes(seed, config.lighting.areaLightV);
  hashBytes(seed, config.lighting.areaLightEmission);
  hashBytes(seed, config.lighting.standardExposure);
  hashBytes(seed, config.lighting.standardGamma);
  hashBytes(seed, config.lighting.standardSaturation);
  hashBytes(seed, config.lighting.toonBandCount);
  hashBytes(seed, config.toon.enabled);
  hashBytes(seed, config.toon.bandCount);
  hashBytes(seed, config.toon.edgeThickness);
  hashBytes(seed, config.toon.depthThreshold);
  hashBytes(seed, config.toon.normalThreshold);
  hashBytes(seed, config.toon.enableDetailContour);
  hashBytes(seed, config.toon.enableObjectContour);
  hashBytes(seed, config.toon.enableNormalEdge);
  hashBytes(seed, config.toon.enableDepthEdge);
  hashBytes(seed, config.toon.detailContourStrength);
  hashBytes(seed, config.toon.objectContourStrength);
  hashBytes(seed, config.toon.edgeColor);
  hashBytes(seed, config.toon.useFxaa);
  hashBytes(seed, config.toon.tonemapExposure);
  hashBytes(seed, config.toon.tonemapGamma);
  hashBytes(seed, config.toon.backgroundColor);
  hashBytes(seed, config.enableMetalFX);
  hashBytes(seed, config.groundPlane.height);
  hashBytes(seed, config.groundPlane.color);
  hashBytes(seed, config.groundPlane.metallic);
  hashBytes(seed, config.groundPlane.roughness);
  hashBytes(seed, config.groundPlane.reflectance);
  return seed;
}


} // namespace

PolyscopeRtRuntime::PolyscopeRtRuntime() {
  polyscope::options::alwaysRedraw = false;
  appearance_ = makeDefaultAppearance();
}

void PolyscopeRtRuntime::tick() {
  buildUI();

  if (!rayTracingEnabled_) {
    if (wasEnabledLastFrame_) {
      removeRenderImage();
      if (backend_) backend_->resetAccumulation();
    }
    wasEnabledLastFrame_ = false;
    previewModeLastFrame_ = false;
    framesSinceDisplayUpdate_ = 0;
    sceneDirty_ = true;
    return;
  }

  try {
    if (!ensureBackend()) return;

    if (shouldRefreshScene()) {
      cachedSnapshot_ = capturePolyscopeSceneSnapshot(materialOverrides_, apiLights_);
      hasCachedSnapshot_ = true;
      sceneDirty_ = false;
      framesSinceSceneSync_ = 0;
    } else {
      framesSinceSceneSync_++;
    }

    if (!hasCachedSnapshot_) {
      setError("Ray tracing scene has not been initialized.");
      return;
    }

    const PolyscopeSceneSnapshot& snapshot = cachedSnapshot_;
    if (snapshot.scene.meshes.empty()) {
      setError("No enabled triangle meshes are registered in Polyscope.");
      removeRenderImage();
      wasEnabledLastFrame_ = false;
      return;
    }

    const rt::RTScene& renderScene = snapshot.scene;

    rt::RTCamera camera = captureCamera();

    if (offlineRenderRequested_ && !offlineRenderActive_) {
      beginOfflineRender(snapshot, camera);
      offlineRenderRequested_ = false;
    }

    if (offlineRenderActive_) {
      processOfflineRender();
      polyscope::requestRedraw();
      return;
    }

    bool cameraMoved = hadPreviousCamera_ && cameraChanged(camera, previousCamera_);
    bool accumulationDirty = previewModeLastFrame_ != cameraMoved;

    uint32_t targetWidth = camera.width;
    uint32_t targetHeight = camera.height;

    if (renderScene.hash != lastSceneHash_) {
      backend_->setScene(renderScene);
      lastSceneHash_ = renderScene.hash;
      accumulationDirty = true;
    }

    if (accumulationDirty || targetWidth != currentWidth_ || targetHeight != currentHeight_ || !wasEnabledLastFrame_) {
      backend_->resize(targetWidth, targetHeight);
      backend_->resetAccumulation();
      currentWidth_ = targetWidth;
      currentHeight_ = targetHeight;
      framesSinceDisplayUpdate_ = 0;
    }

    backend_->updateCamera(camera);

    rt::RenderConfig config = buildRenderConfig(static_cast<uint32_t>(std::max(1, previewSamplesPerFrame_)));
    if (cameraMoved) {
      config.enableMetalFX = false;
      config.maxBounces = std::min(config.maxBounces, 4u);
    }
    if (config.enableMetalFX) {
      config.metalFXOutputWidth = camera.width;
      config.metalFXOutputHeight = camera.height;
      config.accumulate = false;
    }
    uint64_t renderStateHash = computeRenderStateHash(config);
    bool renderStateChanged = renderStateHash != lastRenderStateHash_;
    if (renderStateChanged) {
      backend_->resetAccumulation();
      lastRenderStateHash_ = renderStateHash;
      framesSinceDisplayUpdate_ = 0;
    }

    backend_->renderIteration(config);
    int displayInterval = cameraMoved ? 1 : std::max(1, staticDisplayInterval_);
    bool shouldUpdateDisplay =
        config.enableMetalFX || !wasEnabledLastFrame_ || accumulationDirty || renderStateChanged || framesSinceDisplayUpdate_ >= displayInterval - 1;

    if (shouldUpdateDisplay) {
      lastRenderBuffer_ = backend_->downloadRenderBuffer();
      framesSinceDisplayUpdate_ = 0;

      ensureRenderImage(lastRenderBuffer_.width, lastRenderBuffer_.height);

      if (renderImage_ != nullptr) {
        const auto& color = lastRenderBuffer_.color;
        std::vector<glm::vec4> displayColorAlpha(color.size());
        for (size_t i = 0; i < color.size(); ++i) {
          displayColorAlpha[i] = glm::vec4(color[i], 1.0f);
        }
        renderImage_->colors.data = std::move(displayColorAlpha);
        renderImage_->colors.markHostBufferUpdated();
        renderImage_->setEnabled(true);
      }
    } else {
      framesSinceDisplayUpdate_++;
    }

    backendError_.clear();
    wasEnabledLastFrame_ = true;
    previewModeLastFrame_ = cameraMoved;
    previousCamera_ = camera;
    hadPreviousCamera_ = true;
    polyscope::requestRedraw();
  } catch (const std::exception& e) {
    setError(e.what());
    rayTracingEnabled_ = false;
    removeRenderImage();
    wasEnabledLastFrame_ = false;
  }
}

bool PolyscopeRtRuntime::ensureBackend() {
  if (backend_) return true;

  try {
    backend_ = rt::createMetalPathTracerBackend(polyscope::rt::getShaderLibraryPath());
    backendError_.clear();
    return true;
  } catch (const std::exception& e) {
    setError(e.what());
    return false;
  }
}

rt::RTCamera PolyscopeRtRuntime::captureCamera() const {
  rt::RTCamera camera;
  auto params = polyscope::view::getCameraParametersForCurrentView();
  auto [nearClip, farClip] = polyscope::view::getClipPlanes();
  auto [bufferWidth, bufferHeight] = polyscope::view::getBufferSize();

  camera.position = params.getPosition();
  camera.lookDir = glm::normalize(params.getLookDir());
  camera.upDir = glm::normalize(params.getUpDir());
  camera.rightDir = glm::normalize(params.getRightDir());
  camera.fovYDegrees = params.getFoVVerticalDegrees();
  camera.aspect = params.getAspectRatioWidthOverHeight();
  camera.nearClip = nearClip;
  camera.farClip = farClip;
  camera.viewMatrix = params.getViewMat();
  camera.projectionMatrix = polyscope::view::getCameraPerspectiveMatrix();
  camera.width = static_cast<uint32_t>(std::max(1, bufferWidth));
  camera.height = static_cast<uint32_t>(std::max(1, bufferHeight));

  return camera;
}

bool PolyscopeRtRuntime::cameraChanged(const rt::RTCamera& a, const rt::RTCamera& b) const {
  if (!nearlyEqualVec3(a.position, b.position)) return true;
  if (!nearlyEqualVec3(a.lookDir, b.lookDir)) return true;
  if (!nearlyEqualVec3(a.upDir, b.upDir)) return true;
  if (!nearlyEqualVec3(a.rightDir, b.rightDir)) return true;
  if (!nearlyEqual(a.fovYDegrees, b.fovYDegrees)) return true;
  if (!nearlyEqual(a.aspect, b.aspect)) return true;
  if (a.width != b.width || a.height != b.height) return true;
  return false;
}

void PolyscopeRtRuntime::removeRenderImage() {
  polyscope::removeFloatingQuantity(kRenderImageName, false);
  polyscope::removeFloatingQuantityStructureIfEmpty();
  renderImage_ = nullptr;
  renderImageWidth_ = 0;
  renderImageHeight_ = 0;
}

void PolyscopeRtRuntime::ensureRenderImage(uint32_t width, uint32_t height) {
  if (renderImage_ != nullptr && renderImageWidth_ == width && renderImageHeight_ == height) {
    return;
  }

  removeRenderImage();

  std::vector<glm::vec4> color(width * height, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
  renderImage_ = polyscope::addColorAlphaImageQuantity(kRenderImageName, width, height, color,
                                                       polyscope::ImageOrigin::UpperLeft);
  renderImage_->setShowInImGuiWindow(false);
  renderImage_->setShowFullscreen(true);
  renderImage_->setEnabled(true);
  renderImageWidth_ = width;
  renderImageHeight_ = height;
}

void PolyscopeRtRuntime::setError(std::string message) {
  if (backendError_ != message) {
    std::cerr << "[metal-ray-tracing] " << message << std::endl;
  }
  backendError_ = std::move(message);
}

bool PolyscopeRtRuntime::shouldRefreshScene() {
  if (!hasCachedSnapshot_ || sceneDirty_) return true;
  if (autoSyncScene_ && framesSinceSceneSync_ >= sceneSyncIntervalFrames_) return true;
  return false;
}

void PolyscopeRtRuntime::beginOfflineRender(const PolyscopeSceneSnapshot& snapshot, const rt::RTCamera& camera) {
  offlineSnapshot_ = snapshot;
  offlineCamera_ = camera;
  offlineRenderActive_ = true;
  offlineOutputPath_ = makeOfflineRenderFilename();
  offlineStartTime_ = std::chrono::steady_clock::now();

  backend_->setScene(offlineSnapshot_.scene);
  backend_->resize(offlineCamera_.width, offlineCamera_.height);
  backend_->updateCamera(offlineCamera_);
  backend_->resetAccumulation();
  currentWidth_ = offlineCamera_.width;
  currentHeight_ = offlineCamera_.height;
  framesSinceDisplayUpdate_ = 0;
  lastRenderBuffer_ = {};
}

void PolyscopeRtRuntime::processOfflineRender() {
  backend_->updateCamera(offlineCamera_);

  rt::RenderConfig config = buildRenderConfig(static_cast<uint32_t>(std::max(1, offlineSamplesPerFrame_)));
  config.accumulate = true;

  backend_->renderIteration(config);
  lastRenderBuffer_ = backend_->downloadRenderBuffer();

  ensureRenderImage(lastRenderBuffer_.width, lastRenderBuffer_.height);

  if (renderImage_ != nullptr) {
    const auto& color = lastRenderBuffer_.color;
    std::vector<glm::vec4> displayColorAlpha(color.size());
    for (size_t i = 0; i < color.size(); ++i) {
      displayColorAlpha[i] = glm::vec4(color[i], 1.0f);
    }
    renderImage_->colors.data = std::move(displayColorAlpha);
    renderImage_->colors.markHostBufferUpdated();
    renderImage_->setEnabled(true);
  }

  if (static_cast<int>(lastRenderBuffer_.accumulatedSamples) >= offlineTargetSamples_) {
    saveRenderBufferToFile(lastRenderBuffer_, offlineOutputPath_);
    offlineRenderActive_ = false;
    sceneDirty_ = true;
  }
}

std::string PolyscopeRtRuntime::makeOfflineRenderFilename() const {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm localTm{};
#if defined(_WIN32)
  localtime_s(&localTm, &t);
#else
  localtime_r(&t, &localTm);
#endif

  const char* ext = (offlineFormat_ == ExportFormat::JPG) ? ".jpg" : ".png";
  std::ostringstream filename;
  filename << "render_" << std::put_time(&localTm, "%Y%m%d_%H%M%S") << ext;
  return (std::filesystem::current_path() / filename.str()).string();
}

void PolyscopeRtRuntime::saveRenderBufferToFile(const rt::RenderBuffer& buffer, const std::string& filename) const {
  const size_t npix = buffer.color.size();
  std::vector<unsigned char> pixels(npix * 4, 255);

  auto toByte = [](float value) {
    float mapped = std::clamp(value, 0.0f, 1.0f);
    return static_cast<unsigned char>(std::lround(mapped * 255.0f));
  };

  // Metal buffer is already in standard top-left origin order (pixel[0] = top-left).
  // No flip needed.
  for (size_t i = 0; i < npix; ++i) {
    pixels[4 * i + 0] = toByte(buffer.color[i].r);
    pixels[4 * i + 1] = toByte(buffer.color[i].g);
    pixels[4 * i + 2] = toByte(buffer.color[i].b);
    constexpr uint32_t kGroundPlaneId = 0xFFFFFFFEu;
    unsigned char alpha = 255u;
    if (!offlineWithBackground_) {
      alpha = (buffer.objectId[i] == 0u || buffer.objectId[i] == kGroundPlaneId) ? 0u : 255u;
    }
    pixels[4 * i + 3] = alpha;
  }

  stbi_flip_vertically_on_write(0);

  const int w = static_cast<int>(buffer.width);
  const int h = static_cast<int>(buffer.height);

  if (offlineFormat_ == ExportFormat::JPG) {
    // JPG doesn't support alpha; write 3-channel
    std::vector<unsigned char> rgb(npix * 3);
    for (size_t i = 0; i < npix; ++i) {
      rgb[3 * i + 0] = pixels[4 * i + 0];
      rgb[3 * i + 1] = pixels[4 * i + 1];
      rgb[3 * i + 2] = pixels[4 * i + 2];
    }
    if (!stbi_write_jpg(filename.c_str(), w, h, 3, rgb.data(), 95))
      throw std::runtime_error("Failed to save rendered JPG: " + filename);
  } else {
    stbi_write_png_compression_level = 0;
    if (!stbi_write_png(filename.c_str(), w, h, 4, pixels.data(), w * 4))
      throw std::runtime_error("Failed to save rendered PNG: " + filename);
  }
}

rt::RenderConfig PolyscopeRtRuntime::buildRenderConfig(uint32_t samplesPerIteration) const {
  rt::RenderConfig config;
  config.renderMode = appearance_.mode;
  config.samplesPerIteration = samplesPerIteration;
  config.maxBounces = static_cast<uint32_t>(std::max(1, maxBounces_));
  config.accumulate = progressiveAccumulation_;
  config.enableMetalFX = appearance_.enableMetalFX;
  config.lighting = appearance_.lighting;
  config.toon = appearance_.toon;
  config.toon.enabled = appearance_.mode == rt::RenderMode::Toon;
  config.groundPlane = appearance_.groundPlane;

  // Sync ground plane height with polyscope's bounding box.
  int iP = 1;
  float sign = 1.0f;
  switch (polyscope::view::upDir) {
    case polyscope::UpDir::NegXUp: sign = -1.0f; [[fallthrough]];
    case polyscope::UpDir::XUp:    iP = 0; break;
    case polyscope::UpDir::NegYUp: sign = -1.0f; [[fallthrough]];
    case polyscope::UpDir::YUp:    iP = 1; break;
    case polyscope::UpDir::NegZUp: sign = -1.0f; [[fallthrough]];
    case polyscope::UpDir::ZUp:    iP = 2; break;
  }

  const auto& [bboxMin, bboxMax] = polyscope::state::boundingBox;
  float bboxBottom = (sign > 0.0f) ? bboxMin[iP] : bboxMax[iP];
  float heightEPS  = polyscope::state::lengthScale * 1e-4f;

  switch (polyscope::options::groundPlaneHeightMode) {
    case polyscope::GroundPlaneHeightMode::Automatic:
      config.groundPlane.height =
          bboxBottom - sign * (polyscope::options::groundPlaneHeightFactor.asAbsolute() + heightEPS);
      break;
    case polyscope::GroundPlaneHeightMode::Manual:
      config.groundPlane.height = polyscope::options::groundPlaneHeight;
      break;
  }

  if (!config.toon.enabled) {
    config.lighting.toonBandCount = 0;
    config.toon.enableDetailContour = false;
    config.toon.enableObjectContour = false;
    config.toon.enableNormalEdge = false;
    config.toon.enableDepthEdge = false;
    config.toon.detailContourStrength = 0.0f;
    config.toon.objectContourStrength = 0.0f;
  }

  return config;
}

void PolyscopeRtRuntime::setEnabled(bool enabled) {
  if (rayTracingEnabled_ == enabled) return;
  if (enabled) {
    savedGroundPlaneMode_ = polyscope::options::groundPlaneMode;
    hasSavedGroundPlaneMode_ = true;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
  } else if (hasSavedGroundPlaneMode_) {
    polyscope::options::groundPlaneMode = savedGroundPlaneMode_;
    hasSavedGroundPlaneMode_ = false;
  }
  rayTracingEnabled_ = enabled;
  sceneDirty_ = true;
  framesSinceSceneSync_ = 0;
  if (!enabled) {
    hasCachedSnapshot_ = false;
  }
  polyscope::requestRedraw();
}
bool PolyscopeRtRuntime::isEnabled() const { return rayTracingEnabled_; }
void PolyscopeRtRuntime::setMaxBounces(int bounces) { maxBounces_ = std::clamp(bounces, 1, 12); }
void PolyscopeRtRuntime::setPreviewSamplesPerFrame(int spp) { previewSamplesPerFrame_ = std::clamp(spp, 1, 4); }
void PolyscopeRtRuntime::setAppearance(const rt::AppearanceConfig& config) {
  appearance_ = config;
  syncAppearanceState();
}

rt::AppearanceConfig PolyscopeRtRuntime::getAppearance() const { return appearance_; }

void PolyscopeRtRuntime::setApiLights(const std::vector<rt::RTPunctualLight>& lights) {
  apiLights_ = lights;
  sceneDirty_ = true;
}

void PolyscopeRtRuntime::setMaterialOverrides(const std::unordered_map<std::string, rt::MaterialOverride>& overrides) {
  materialOverrides_ = overrides;
  sceneDirty_ = true;
}

const std::unordered_map<std::string, rt::MaterialOverride>& PolyscopeRtRuntime::getMaterialOverrides() const {
  return materialOverrides_;
}

const std::vector<rt::RTPunctualLight>& PolyscopeRtRuntime::getApiLights() const {
  return apiLights_;
}

void PolyscopeRtRuntime::resetAccumulation() {
  if (backend_) backend_->resetAccumulation();
  lastRenderStateHash_ = 0;
  framesSinceDisplayUpdate_ = 0;
}

void PolyscopeRtRuntime::exportPNG(const std::string& path, int targetSPP) {
  offlineTargetSamples_ = targetSPP;
  offlineOutputPath_ = path;
  offlineRenderRequested_ = true;
}
