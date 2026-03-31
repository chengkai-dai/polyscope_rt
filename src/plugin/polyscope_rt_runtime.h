#pragma once

#include <array>
#include <memory>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

#include "polyscope/color_image_quantity.h"
#include "polyscope/camera_parameters.h"
#include "polyscope/structure.h"

#include "rendering/ray_tracing_backend.h"
#include "scene/polyscope_scene_snapshot.h"

class PolyscopeRtRuntime {
public:
  PolyscopeRtRuntime();
  void tick();

  void setEnabled(bool enabled);
  bool isEnabled() const;
  void setMaxBounces(int bounces);
  void setPreviewSamplesPerFrame(int spp);
  void setAppearance(const rt::AppearanceConfig& config);
  rt::AppearanceConfig getAppearance() const;
  void setApiLights(const std::vector<rt::RTPunctualLight>& lights);
  void setMaterialOverrides(const std::unordered_map<std::string, rt::MaterialOverride>& overrides);
  const std::unordered_map<std::string, rt::MaterialOverride>& getMaterialOverrides() const;
  const std::vector<rt::RTPunctualLight>& getApiLights() const;
  void resetAccumulation();
  void exportPNG(const std::string& path, int targetSPP);

private:
  bool ensureBackend();
  void buildUI();
  void buildPathTracerSection();
  void buildAppearanceSection();
  void buildExperimentalSection();
  void buildScreenshotSection();
  void buildCameraSection();
  void syncAppearanceState();
  rt::RTCamera captureCamera() const;
  bool cameraChanged(const rt::RTCamera& a, const rt::RTCamera& b) const;
  void removeRenderImage();
  void ensureRenderImage(uint32_t width, uint32_t height);
  void setError(std::string message);
  bool shouldRefreshScene();
  void beginOfflineRender(const PolyscopeSceneSnapshot& snapshot, const rt::RTCamera& camera);
  void processOfflineRender();
  std::string makeOfflineRenderFilename() const;
  void saveRenderBufferToFile(const rt::RenderBuffer& buffer, const std::string& filename) const;
  rt::RenderConfig buildRenderConfig(uint32_t samplesPerIteration) const;

  bool rayTracingEnabled_ = false;
  int previewSamplesPerFrame_ = 1;
  int maxBounces_ = 6;
  rt::BackendType backendType_ = rt::BackendType::Metal;
  bool autoSyncScene_ = true;
  int sceneSyncIntervalFrames_ = 30;
  int staticDisplayInterval_ = 2;
  int offlineTargetSamples_ = 128;
  int offlineSamplesPerFrame_ = 4;
  rt::AppearanceConfig appearance_;

  std::unique_ptr<rt::IRayTracingBackend> backend_;
  std::string backendError_;

  bool wasEnabledLastFrame_ = false;
  bool hadPreviousCamera_ = false;
  bool previewModeLastFrame_ = false;
  rt::RTCamera previousCamera_;
  uint64_t lastSceneHash_ = 0;
  uint64_t lastRenderStateHash_ = 0;
  uint32_t currentWidth_ = 0;
  uint32_t currentHeight_ = 0;
  uint32_t renderImageWidth_ = 0;
  uint32_t renderImageHeight_ = 0;
  int framesSinceSceneSync_ = 0;
  int framesSinceDisplayUpdate_ = 0;
  bool sceneDirty_ = true;
  bool hasCachedSnapshot_ = false;
  PolyscopeSceneSnapshot cachedSnapshot_;
  enum class ExportFormat { PNG, JPG };
  ExportFormat offlineFormat_       = ExportFormat::PNG;
  bool offlineWithBackground_       = true;
  bool offlineRenderRequested_ = false;
  bool offlineRenderActive_ = false;  bool hasSavedGroundPlaneMode_ = false;
  PolyscopeSceneSnapshot offlineSnapshot_;
  rt::RTCamera offlineCamera_;
  std::string offlineOutputPath_;
  std::string pendingOutputPath_;  // set by exportPNG(); consumed by beginOfflineRender()
  std::chrono::steady_clock::time_point offlineStartTime_{};

  polyscope::ColorImageQuantity* renderImage_ = nullptr;
  rt::RenderBuffer lastRenderBuffer_;
  std::vector<rt::RTPunctualLight> apiLights_;
  std::unordered_map<std::string, rt::MaterialOverride> materialOverrides_;
  polyscope::GroundPlaneMode savedGroundPlaneMode_ = polyscope::GroundPlaneMode::TileReflection;

  bool cameraFieldActive_ = false;
  bool cameraFlyTo_    = false;
  glm::vec3 camEye_    = {0.f, 0.f,  3.f};
  glm::vec3 camCenter_ = {0.f, 0.f,  0.f};
  glm::vec3 camUp_     = {0.f, 1.f,  0.f};
  float     camFov_    = 45.f;
};
