#pragma once

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

} // namespace rt
} // namespace polyscope
