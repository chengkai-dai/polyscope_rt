#pragma once

#include "glm/glm.hpp"

namespace polyscope {
namespace rt {

struct MaterialPreset {
  glm::vec4 baseColor{0.8f, 0.8f, 0.8f, 1.0f};
  float metallic = 0.0f;
  float roughness = 1.0f;
  glm::vec3 emissive{0.0f};
  float transmission = 0.0f;
  float ior = 1.5f;
  float opacity = 1.0f;
  bool unlit = false;
};

inline const MaterialPreset& builtinDefaultMaterial() {
  static const MaterialPreset k;
  return k;
}

inline MaterialPreset PerfectDiffuse() {
  return MaterialPreset{};
}

// ── Metals ──────────────────────────────────────────────

inline MaterialPreset Gold() {
  return {glm::vec4(1.0f, 0.766f, 0.336f, 1.0f), 1.0f, 0.3f};
}
inline MaterialPreset PolishedGold() {
  return {glm::vec4(1.0f, 0.766f, 0.336f, 1.0f), 1.0f, 0.05f};
}
inline MaterialPreset Silver() {
  return {glm::vec4(0.972f, 0.960f, 0.915f, 1.0f), 1.0f, 0.3f};
}
inline MaterialPreset PolishedSilver() {
  return {glm::vec4(0.972f, 0.960f, 0.915f, 1.0f), 1.0f, 0.05f};
}
inline MaterialPreset Copper() {
  return {glm::vec4(0.955f, 0.638f, 0.538f, 1.0f), 1.0f, 0.3f};
}
inline MaterialPreset Aluminum() {
  return {glm::vec4(0.913f, 0.922f, 0.924f, 1.0f), 1.0f, 0.4f};
}
inline MaterialPreset Iron() {
  return {glm::vec4(0.560f, 0.570f, 0.580f, 1.0f), 1.0f, 0.5f};
}
inline MaterialPreset Chrome() {
  return {glm::vec4(0.550f, 0.556f, 0.554f, 1.0f), 1.0f, 0.05f};
}
inline MaterialPreset Titanium() {
  return {glm::vec4(0.542f, 0.497f, 0.449f, 1.0f), 1.0f, 0.35f};
}
inline MaterialPreset BrushedMetal() {
  return {glm::vec4(0.75f, 0.75f, 0.75f, 1.0f), 1.0f, 0.6f};
}

// ── Dielectrics ─────────────────────────────────────────

inline MaterialPreset Plastic() {
  return {glm::vec4(0.8f, 0.8f, 0.8f, 1.0f), 0.0f, 0.4f};
}
inline MaterialPreset GlossyPlastic() {
  return {glm::vec4(0.8f, 0.1f, 0.1f, 1.0f), 0.0f, 0.1f};
}
inline MaterialPreset Rubber() {
  return {glm::vec4(0.15f, 0.15f, 0.15f, 1.0f), 0.0f, 0.9f};
}
inline MaterialPreset Ceramic() {
  return {glm::vec4(0.95f, 0.93f, 0.88f, 1.0f), 0.0f, 0.25f};
}
inline MaterialPreset Porcelain() {
  return {glm::vec4(0.98f, 0.97f, 0.95f, 1.0f), 0.0f, 0.15f};
}
inline MaterialPreset Clay() {
  MaterialPreset p;
  p.baseColor = glm::vec4(0.76f, 0.50f, 0.36f, 1.0f);
  p.roughness = 0.8f;
  return p;
}

// ── Transparent / Refractive ────────────────────────────

inline MaterialPreset Glass() {
  return {glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), 0.0f, 0.0f, glm::vec3(0.0f), 1.0f, 1.5f};
}
inline MaterialPreset FrostedGlass() {
  return {glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), 0.0f, 0.4f, glm::vec3(0.0f), 1.0f, 1.5f};
}
inline MaterialPreset Diamond() {
  return {glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), 0.0f, 0.0f, glm::vec3(0.0f), 1.0f, 2.42f};
}
inline MaterialPreset Water() {
  return {glm::vec4(0.8f, 0.9f, 1.0f, 1.0f), 0.0f, 0.0f, glm::vec3(0.0f), 1.0f, 1.33f};
}
inline MaterialPreset Ice() {
  return {glm::vec4(0.85f, 0.92f, 0.98f, 1.0f), 0.0f, 0.15f, glm::vec3(0.0f), 0.9f, 1.31f};
}
inline MaterialPreset TintedGlass(glm::vec3 tint) {
  return {glm::vec4(tint, 1.0f), 0.0f, 0.0f, glm::vec3(0.0f), 1.0f, 1.5f};
}
inline MaterialPreset ClearGlass(float opacity = 0.0f) {
  MaterialPreset p;
  p.baseColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
  p.roughness = 0.0f;
  p.transmission = 1.0f;
  p.ior = 1.5f;
  p.opacity = opacity;
  return p;
}
inline MaterialPreset ClearGlass(glm::vec3 tint, float opacity = 0.0f) {
  MaterialPreset p;
  p.baseColor = glm::vec4(tint, 1.0f);
  p.roughness = 0.0f;
  p.transmission = 1.0f;
  p.ior = 1.5f;
  p.opacity = opacity;
  return p;
}

// ── Natural / Organic ───────────────────────────────────

inline MaterialPreset Wood() {
  return {glm::vec4(0.55f, 0.35f, 0.17f, 1.0f), 0.0f, 0.7f};
}
inline MaterialPreset Marble() {
  return {glm::vec4(0.93f, 0.90f, 0.87f, 1.0f), 0.0f, 0.2f};
}
inline MaterialPreset Skin() {
  return {glm::vec4(0.87f, 0.68f, 0.57f, 1.0f), 0.0f, 0.6f};
}
inline MaterialPreset Concrete() {
  return {glm::vec4(0.65f, 0.65f, 0.65f, 1.0f), 0.0f, 0.85f};
}
inline MaterialPreset Sand() {
  return {glm::vec4(0.86f, 0.78f, 0.60f, 1.0f), 0.0f, 0.95f};
}

// ── Special ─────────────────────────────────────────────

inline MaterialPreset Mirror() {
  return {glm::vec4(0.95f, 0.95f, 0.95f, 1.0f), 1.0f, 0.0f};
}
inline MaterialPreset Emissive(glm::vec3 color, float strength = 5.0f) {
  return {glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0.0f, 1.0f, color * strength};
}
inline MaterialPreset Unlit(glm::vec3 color) {
  return {glm::vec4(color, 1.0f), 0.0f, 1.0f, glm::vec3(0.0f), 0.0f, 1.5f, 1.0f, true};
}
inline MaterialPreset Transparent(float alpha = 0.3f) {
  MaterialPreset p;
  p.baseColor = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
  p.roughness = 0.4f;
  p.opacity = alpha;
  return p;
}
inline MaterialPreset Transparent(glm::vec3 color, float alpha = 0.3f) {
  MaterialPreset p;
  p.baseColor = glm::vec4(color, 1.0f);
  p.roughness = 0.4f;
  p.opacity = alpha;
  return p;
}

// ── Colored variants (convenience) ──────────────────────

inline MaterialPreset Plastic(glm::vec3 color) {
  return {glm::vec4(color, 1.0f), 0.0f, 0.4f};
}
inline MaterialPreset GlossyPlastic(glm::vec3 color) {
  return {glm::vec4(color, 1.0f), 0.0f, 0.1f};
}
inline MaterialPreset Metal(glm::vec3 color, float roughness = 0.3f) {
  return {glm::vec4(color, 1.0f), 1.0f, roughness};
}

} // namespace rt
} // namespace polyscope
