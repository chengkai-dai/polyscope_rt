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
  mesh.opacity = preset.opacity;
  mesh.unlit = preset.unlit;
}

inline void applyPhysicalParamsFromPreset(::rt::RTCurveNetwork& curve, const MaterialPreset& preset) {
  curve.metallic = preset.metallic;
  curve.roughness = preset.roughness;
  curve.unlit = preset.unlit;
}

} // namespace rt
} // namespace polyscope
