#pragma once

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "glm/glm.hpp"

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/simple_triangle_mesh.h"
#include "polyscope/structure.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/surface_vector_quantity.h"
#include "polyscope/vector_quantity.h"
#include "polyscope/volume_mesh.h"

#include "scene/polyscope_scene_snapshot.h"
#include "utility/rt_mesh_material_helpers.h"

namespace snapshot_detail {

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
constexpr float kMinProceduralRadius = 1e-4f;

glm::mat4 makeTranslationTransform(const glm::vec3& offset);
void hashBytes(uint64_t& hash, const void* data, size_t size);
void hashString(uint64_t& hash, std::string_view value);

template <typename T>
void hashVector(uint64_t& hash, const std::vector<T>& values) {
  if (values.empty()) return;
  hashBytes(hash, values.data(), values.size() * sizeof(T));
}

void faceColorsToVertex(const std::vector<glm::vec3>& faceColors,
                        const std::vector<uint32_t>& triVertInds,
                        const std::vector<uint32_t>& triFaceInds,
                        size_t nVerts, glm::vec3 fallback,
                        std::vector<glm::vec3>& outColors);
std::vector<float> faceScalarsToVertex(const std::vector<float>& faceScalars,
                                       const std::vector<uint32_t>& triVertInds,
                                       const std::vector<uint32_t>& triFaceInds,
                                       size_t nVerts, float fallback);

inline rt::RTQuantitySource makeQuantitySource(std::string_view ownerStructureName,
                                               std::string_view quantityName,
                                               rt::RTQuantitySourceKind kind) {
  rt::RTQuantitySource source;
  source.ownerStructureName = ownerStructureName;
  source.quantityName = quantityName;
  source.kind = kind;
  return source;
}

template <typename QuantitiesMap, typename QuantityPtr>
std::string findQuantityName(const QuantitiesMap& quantities, const QuantityPtr* target) {
  if (target == nullptr) return {};
  for (const auto& [name, qPtr] : quantities) {
    if (qPtr.get() == target) return name;
  }
  return {};
}

rt::RTMesh makeMeshFromSimpleTriangleMesh(polyscope::SimpleTriangleMesh& mesh);
rt::RTMesh makeMeshFromSurfaceMesh(polyscope::SurfaceMesh& mesh);
rt::RTMesh makeMeshFromVolumeMesh(polyscope::VolumeMesh& mesh);
rt::RTPointCloud makeRTPointCloud(polyscope::PointCloud& cloud);
rt::RTCurveNetwork makeCurveNetwork(polyscope::CurveNetwork& network);

template <typename T>
void applyMaterialPreset(T& target, std::string_view materialName) {
  polyscope::rt::applyPolyscopeMaterialPreset(target, materialName);
}

void applyMaterialOverride(rt::RTMesh& mesh, const std::unordered_map<std::string, rt::MaterialOverride>& overrides);

void addMeshAndHash(PolyscopeSceneSnapshot& snapshot, rt::RTMesh&& mesh, polyscope::Structure& structure);
void addCurveNetworkAndHash(PolyscopeSceneSnapshot& snapshot, rt::RTCurveNetwork&& curveNet, polyscope::Structure& structure);
void addPointCloudAndHash(PolyscopeSceneSnapshot& snapshot, rt::RTPointCloud&& pc, polyscope::Structure& structure);
void addLightsAndHash(PolyscopeSceneSnapshot& snapshot, const std::vector<rt::RTPunctualLight>& apiLights);

template <typename QT>
rt::RTVectorField vectorQuantityToRT(polyscope::VectorQuantity<QT>& qty) {
  rt::RTVectorField out;
  out.color  = qty.getVectorColor();
  out.radius = static_cast<float>(std::max(1e-5, qty.getVectorRadius()));

  const double scale = qty.getVectorLengthScale();
  const double range = qty.getVectorLengthRange();
  const float mult  = (range > 0.0) ? static_cast<float>(scale / range) : 1.0f;

  qty.vectors.ensureHostBufferPopulated();
  qty.vectorRoots.ensureHostBufferPopulated();

  const size_t n = qty.vectors.data.size();
  out.roots.reserve(n);
  out.directions.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    out.roots.push_back(qty.vectorRoots.data[i]);
    out.directions.push_back(qty.vectors.data[i] * mult);
  }
  return out;
}

template <typename QT>
rt::RTVectorField tangentVectorQuantityToRT(polyscope::TangentVectorQuantity<QT>& qty) {
  rt::RTVectorField out;
  out.color  = qty.getVectorColor();
  out.radius = static_cast<float>(std::max(1e-5, qty.getVectorRadius()));

  const double scale = qty.getVectorLengthScale();
  const double range = qty.getVectorLengthRange();
  const float mult  = (range > 0.0) ? static_cast<float>(scale / range) : 1.0f;

  qty.tangentVectors.ensureHostBufferPopulated();
  qty.tangentBasisX.ensureHostBufferPopulated();
  qty.tangentBasisY.ensureHostBufferPopulated();
  qty.vectorRoots.ensureHostBufferPopulated();

  const size_t n = qty.tangentVectors.data.size();
  out.roots.reserve(n);
  out.directions.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const glm::vec2& tv = qty.tangentVectors.data[i];
    const glm::vec3 dir3 = tv.x * qty.tangentBasisX.data[i] + tv.y * qty.tangentBasisY.data[i];
    out.roots.push_back(qty.vectorRoots.data[i]);
    out.directions.push_back(dir3 * mult);
  }
  return out;
}

std::vector<rt::RTVectorField> makeRTVectorFields(polyscope::SurfaceMesh& mesh);
void addVectorFieldsAndHash(PolyscopeSceneSnapshot& snapshot,
                            std::vector<rt::RTVectorField>&& fields,
                            std::string_view parentName);

} // namespace snapshot_detail
