#pragma once

#include <string_view>

#include "polyscope/rt/material_library.h"
#include "rendering/ray_tracing_types.h"

namespace polyscope {
namespace rt {

inline void applyPhysicalParamsFromPreset(::rt::RTMesh& mesh, const MaterialPreset& preset) {
  mesh.metallicFactor = preset.metallic;
  mesh.roughnessFactor = preset.roughness;
  mesh.emissiveFactor = preset.emissive;
  mesh.transmissionFactor = preset.transmission;
  mesh.indexOfRefraction = preset.ior;
  // Only override opacity when the preset explicitly requests non-opaque (e.g. Transparent()).
  // For fully-opaque presets (opacity==1) we preserve the value already set from
  // Polyscope's per-structure transparency slider so the RT renderer respects it.
  if (preset.opacity < 1.0f) mesh.opacity = preset.opacity;
  mesh.unlit = preset.unlit;
}

inline void applyPhysicalParamsFromPreset(::rt::RTCurveNetwork& curve, const MaterialPreset& preset) {
  curve.metallic = preset.metallic;
  curve.roughness = preset.roughness;
  curve.unlit = preset.unlit;
}

inline MaterialPreset presetFromPolyscopeMaterial(std::string_view materialName) {
  if (materialName == "clay") return Clay();
  if (materialName == "flat") return PerfectDiffuse();
  if (materialName == "candy") {
    auto p = Plastic();
    p.roughness = 0.08f;
    return p;
  }
  if (materialName == "wax") {
    auto p = Plastic();
    p.roughness = 0.35f;
    return p;
  }
  if (materialName == "mud") return Rubber();
  if (materialName == "ceramic") return Ceramic();
  if (materialName == "jade") {
    auto p = Plastic();
    p.roughness = 0.12f;
    return p;
  }
  if (materialName == "normal") {
    auto p = Plastic();
    p.roughness = 0.6f;
    return p;
  }
  return PerfectDiffuse();
}

template <typename T>
inline void applyPolyscopeMaterialPreset(T& target, std::string_view materialName) {
  applyPhysicalParamsFromPreset(target, presetFromPolyscopeMaterial(materialName));
}

} // namespace rt
} // namespace polyscope
