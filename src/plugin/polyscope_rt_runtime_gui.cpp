#include "plugin/polyscope_rt_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include "imgui.h"
#include "polyscope/internal.h"
#include "polyscope/options.h"
#include "polyscope/view.h"
#include "polyscope/camera_parameters.h"

namespace {

constexpr const char* kRenderModeLabels[] = {"Standard", "Toon"};

glm::vec3 defaultMainLightDirection() {
  return glm::normalize(glm::vec3(-0.5f, -0.5f, 0.70710677f));
}

std::string formatDuration(double seconds) {
  if (!std::isfinite(seconds) || seconds < 0.0) return "--";
  int total = static_cast<int>(std::round(seconds));
  int hours   = total / 3600;
  int minutes = (total % 3600) / 60;
  int secs    = total % 60;
  std::ostringstream out;
  if (hours > 0)        out << hours   << "h " << minutes << "m " << secs << "s";
  else if (minutes > 0) out << minutes << "m " << secs    << "s";
  else                  out << secs    << "s";
  return out.str();
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Top-level
// ─────────────────────────────────────────────────────────────────────────────

void PolyscopeRtRuntime::buildUI() {
  // Position "Polyscope RT" exactly over ##Command UI so it covers the empty wrapper.
  float margin = polyscope::internal::imguiStackMargin;
  float paneW  = polyscope::internal::rightWindowsWidth;
  float x      = polyscope::view::windowWidth - paneW - margin;
  ImGui::SetNextWindowPos(ImVec2(x, margin), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(paneW, 0.f), ImGuiCond_FirstUseEver);

  if (!ImGui::Begin("Polyscope RT")) {
    ImGui::End();
    return;
  }

  ImGui::PushID("polyscope-rt");

  ImGui::Checkbox("Ray Tracing", &rayTracingEnabled_);
  if (!backendError_.empty()) {
    ImGui::TextWrapped("%s", backendError_.c_str());
  } else if (rayTracingEnabled_ && lastRenderBuffer_.width > 0) {
    ImGui::TextDisabled("Accum: %u   %ux%u",
                        lastRenderBuffer_.accumulatedSamples,
                        lastRenderBuffer_.width, lastRenderBuffer_.height);
  }

  if (rayTracingEnabled_) {
    buildCameraSection();
    buildPathTracerSection();
    buildAppearanceSection();
    buildScreenshotSection();
    buildExperimentalSection();
  }

  ImGui::PopID();
  ImGui::End();
}

// ─────────────────────────────────────────────────────────────────────────────
// Path Tracer
// ─────────────────────────────────────────────────────────────────────────────

void PolyscopeRtRuntime::buildPathTracerSection() {
  if (!ImGui::TreeNode("Path Tracer")) return;

  if (ImGui::TreeNode("Sampling")) {
    ImGui::SliderInt("Preview SPP",   &previewSamplesPerFrame_, 1, 4);
    ImGui::SliderInt("Bounces",       &maxBounces_,             1, 12);
    ImGui::SliderInt("Display Every", &staticDisplayInterval_,  1, 6);
    ImGui::Separator();
    ImGui::Checkbox("Auto Sync", &autoSyncScene_);
    ImGui::SameLine();
    if (ImGui::Button("Sync Now")) sceneDirty_ = true;
    ImGui::TreePop();
  }

  if (ImGui::TreeNode("Denoise")) {
    ImGui::Checkbox("MetalFX", &appearance_.enableMetalFX);
    ImGui::TreePop();
  }

  ImGui::TreePop();
}

// ─────────────────────────────────────────────────────────────────────────────
// Appearance
// ─────────────────────────────────────────────────────────────────────────────

void PolyscopeRtRuntime::buildAppearanceSection() {
  if (!ImGui::TreeNode("Appearance")) return;

  if (ImGui::TreeNode("Background")) {
    ImGui::ColorEdit3("Color", &appearance_.lighting.backgroundColor[0]);
    ImGui::TreePop();
  }

  if (ImGui::TreeNode("Light")) {
    ImGui::SliderFloat3("Direction", &appearance_.lighting.mainLightDirection[0], -1.0f, 1.0f, "%.2f");
    if (glm::length(appearance_.lighting.mainLightDirection) < 1e-4f)
      appearance_.lighting.mainLightDirection = defaultMainLightDirection();
    appearance_.lighting.mainLightDirection = glm::normalize(appearance_.lighting.mainLightDirection);
    ImGui::SliderFloat("Intensity",  &appearance_.lighting.mainLightIntensity,   0.0f, 12.0f, "%.2f");
    ImGui::SliderFloat("Env",        &appearance_.lighting.environmentIntensity, 0.0f,  2.0f, "%.2f");
    ImGui::TreePop();
  }

  if (ImGui::TreeNode("Ground Plane")) {
    ImGui::ColorEdit3("Color",       &appearance_.groundPlane.color[0]);
    ImGui::SliderFloat("Metallic",   &appearance_.groundPlane.metallic,     0.0f,  1.0f);
    ImGui::SliderFloat("Roughness",  &appearance_.groundPlane.roughness,    0.01f, 1.0f);
    ImGui::SliderFloat("Reflectance",&appearance_.groundPlane.reflectance,  0.0f,  0.08f, "%.3f");
    ImGui::TreePop();
  }

  if (ImGui::TreeNode("Tone")) {
    ImGui::SliderFloat("Ambient",  &appearance_.lighting.ambientFloor,       0.0f, 0.4f,  "%.2f");
    ImGui::SliderFloat("Exposure", &appearance_.lighting.standardExposure,   0.1f, 8.0f,  "%.2f");
    ImGui::SliderFloat("Gamma",    &appearance_.lighting.standardGamma,      0.8f, 3.0f,  "%.2f");
    ImGui::SliderFloat("Sat",      &appearance_.lighting.standardSaturation, 0.0f, 2.0f,  "%.2f");
    ImGui::TreePop();
  }

  syncAppearanceState();
  ImGui::TreePop();
}

// ─────────────────────────────────────────────────────────────────────────────
// Experimental
// ─────────────────────────────────────────────────────────────────────────────

void PolyscopeRtRuntime::buildExperimentalSection() {
  if (!ImGui::TreeNode("Experimental")) return;

  if (ImGui::TreeNode("Render Preset")) {
    int renderMode = static_cast<int>(appearance_.mode);
    if (ImGui::Combo("##shading", &renderMode, kRenderModeLabels, IM_ARRAYSIZE(kRenderModeLabels))) {
      appearance_.mode         = static_cast<rt::RenderMode>(renderMode);
      appearance_.toon.enabled = appearance_.mode == rt::RenderMode::Toon;
    }

    if (appearance_.mode == rt::RenderMode::Toon) {
      ImGui::SliderFloat("Toon Exposure",   &appearance_.toon.tonemapExposure,       0.1f,  6.0f,  "%.2f");
      ImGui::SliderFloat("Toon Gamma",      &appearance_.toon.tonemapGamma,          0.8f,  3.0f,  "%.2f");
      ImGui::SliderInt  ("Bands",           &appearance_.lighting.toonBandCount,     0,     10);
      ImGui::Separator();
      ImGui::Checkbox("Object Contour",     &appearance_.toon.enableObjectContour);
      ImGui::Checkbox("Detail Contour",     &appearance_.toon.enableDetailContour);
      ImGui::Checkbox("Normal Edge",        &appearance_.toon.enableNormalEdge);
      ImGui::Checkbox("Depth Edge",         &appearance_.toon.enableDepthEdge);
      ImGui::SliderFloat("Edge",            &appearance_.toon.edgeThickness,         1.0f,  3.0f,  "%.1f");
      ImGui::SliderFloat("Depth Thresh",    &appearance_.toon.depthThreshold,        0.01f, 10.0f, "%.2f");
      ImGui::SliderFloat("Normal Thresh",   &appearance_.toon.normalThreshold,       0.01f,  2.0f, "%.2f");
      ImGui::SliderFloat("Obj Mix",         &appearance_.toon.objectContourStrength, 0.0f,   1.0f, "%.2f");
      ImGui::SliderFloat("Dtl Mix",         &appearance_.toon.detailContourStrength, 0.0f,   1.0f, "%.2f");
      ImGui::Checkbox("FXAA",               &appearance_.toon.useFxaa);
      ImGui::ColorEdit3("Line",             &appearance_.toon.edgeColor[0]);
    }

    ImGui::TreePop();
  }

  ImGui::TreePop();
}

// ─────────────────────────────────────────────────────────────────────────────
// Screenshot
// ─────────────────────────────────────────────────────────────────────────────

void PolyscopeRtRuntime::buildScreenshotSection() {
  if (!ImGui::TreeNode("Screenshot")) return;

  ImGui::SliderInt("SPP",   &offlineTargetSamples_,   8, 2048, "%d");
  ImGui::SliderInt("Batch", &offlineSamplesPerFrame_,  1, 32);

  // Format selector
  constexpr const char* kFormatLabels[] = {"PNG", "JPG"};
  int fmt = static_cast<int>(offlineFormat_);
  if (ImGui::Combo("Format", &fmt, kFormatLabels, IM_ARRAYSIZE(kFormatLabels)))
    offlineFormat_ = static_cast<ExportFormat>(fmt);

  // With Background only makes sense for PNG (JPG is always opaque)
  if (offlineFormat_ == ExportFormat::PNG)
    ImGui::Checkbox("With Background", &offlineWithBackground_);

  if (!offlineRenderActive_) {
    if (ImGui::Button("Export")) offlineRenderRequested_ = true;
  } else {
    ImGui::BeginDisabled();
    ImGui::Button("Rendering...");
    ImGui::EndDisabled();

    float progress = std::clamp(
        static_cast<float>(lastRenderBuffer_.accumulatedSamples) / std::max(1, offlineTargetSamples_),
        0.0f, 1.0f);
    ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f));

    auto   now             = std::chrono::steady_clock::now();
    double elapsed         = std::chrono::duration<double>(now - offlineStartTime_).count();
    double samplesPerSec   = elapsed > 0.0 ? static_cast<double>(lastRenderBuffer_.accumulatedSamples) / elapsed : 0.0;
    double remaining       = std::max(0, offlineTargetSamples_ - static_cast<int>(lastRenderBuffer_.accumulatedSamples));
    double eta             = samplesPerSec > 1e-6 ? remaining / samplesPerSec : std::numeric_limits<double>::infinity();

    ImGui::Text("%u / %d spp  |  %.1f spp/s  |  ETA %s",
                lastRenderBuffer_.accumulatedSamples, offlineTargetSamples_,
                samplesPerSec, formatDuration(eta).c_str());
  }

  if (!offlineRenderActive_ && !offlineOutputPath_.empty())
    ImGui::TextWrapped("%s", offlineOutputPath_.c_str());

  ImGui::TreePop();
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera
// ─────────────────────────────────────────────────────────────────────────────

void PolyscopeRtRuntime::buildCameraSection() {
  if (!ImGui::TreeNode("Camera")) return;

  polyscope::CameraParameters cur = polyscope::view::getCameraParametersForCurrentView();
  glm::mat4 E        = cur.getE();
  glm::vec3 liveEye  = cur.getPosition();
  glm::vec3 liveLook = -glm::vec3(E[0][2], E[1][2], E[2][2]);
  glm::vec3 liveUp   =  glm::vec3(E[0][1], E[1][1], E[2][1]);
  float     liveFov  = cur.getFoVVerticalDegrees();

  // When no field is being edited, track live camera
  if (!cameraFieldActive_) {
    camEye_    = liveEye;
    camCenter_ = liveEye + liveLook;
    camUp_     = glm::normalize(liveUp);
    camFov_    = liveFov;
  }

  bool changed = false;
  changed |= ImGui::InputFloat3("Eye",    &camEye_[0],    "%.3f");
  bool a0 = ImGui::IsItemActive();
  changed |= ImGui::InputFloat3("Center", &camCenter_[0], "%.3f");
  bool a1 = ImGui::IsItemActive();
  changed |= ImGui::InputFloat3("Up",     &camUp_[0],     "%.3f");
  bool a2 = ImGui::IsItemActive();
  changed |= ImGui::SliderFloat("FoV",    &camFov_, 5.f, 120.f, "%.1f deg");
  bool a3 = ImGui::IsItemActive();

  cameraFieldActive_ = a0 || a1 || a2 || a3;

  ImGui::Spacing();
  {
    float avail   = ImGui::GetContentRegionAvail().x;
    float spacing = ImGui::GetStyle().ItemSpacing.x;
    float flyW    = ImGui::CalcTextSize("Fly").x
                    + ImGui::GetFrameHeight()
                    + ImGui::GetStyle().ItemInnerSpacing.x;
    float btnW    = (avail - flyW - 2.f * spacing) / 3.f;

    ImGui::Checkbox("Fly", &cameraFlyTo_);
    ImGui::SameLine();
    if (ImGui::Button("Save", ImVec2(btnW, 0))) {
      std::ofstream f(std::filesystem::current_path() / "camera_preset.txt");
      if (f) {
        f << std::fixed << std::setprecision(6);
        f << "eye "    << camEye_.x    << " " << camEye_.y    << " " << camEye_.z    << "\n";
        f << "center " << camCenter_.x << " " << camCenter_.y << " " << camCenter_.z << "\n";
        f << "up "     << camUp_.x     << " " << camUp_.y     << " " << camUp_.z     << "\n";
        f << "fov "    << camFov_ << "\n";
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Load", ImVec2(btnW, 0))) {
      std::ifstream f(std::filesystem::current_path() / "camera_preset.txt");
      if (f) {
        std::string tag;
        while (f >> tag) {
          if      (tag == "eye")    f >> camEye_.x    >> camEye_.y    >> camEye_.z;
          else if (tag == "center") f >> camCenter_.x >> camCenter_.y >> camCenter_.z;
          else if (tag == "up")     f >> camUp_.x     >> camUp_.y     >> camUp_.z;
          else if (tag == "fov")    f >> camFov_;
        }
        cameraFieldActive_ = true;
        glm::vec3 up = glm::length(camUp_) > 1e-5f ? glm::normalize(camUp_) : glm::vec3(0, 1, 0);
        polyscope::view::lookAt(camEye_, camCenter_, up, cameraFlyTo_);
        polyscope::view::fov = camFov_;
      }
    }
  }

  if (changed) {
    glm::vec3 up = glm::length(camUp_) > 1e-5f ? glm::normalize(camUp_) : glm::vec3(0, 1, 0);
    polyscope::view::lookAt(camEye_, camCenter_, up, cameraFlyTo_);
    polyscope::view::fov = camFov_;
  }

  ImGui::TreePop();
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal sync
// ─────────────────────────────────────────────────────────────────────────────

void PolyscopeRtRuntime::syncAppearanceState() {
  appearance_.toon.enabled         = appearance_.mode == rt::RenderMode::Toon;
  appearance_.toon.backgroundColor = appearance_.lighting.backgroundColor;
  appearance_.toon.bandCount       = appearance_.lighting.toonBandCount;
}
