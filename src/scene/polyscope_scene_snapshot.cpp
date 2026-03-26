#include "scene/polyscope_scene_snapshot.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/point_cloud_scalar_quantity.h"
#include "polyscope/render/color_maps.h"
#include "polyscope/render/engine.h"
#include "polyscope/simple_triangle_mesh.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/surface_color_quantity.h"
#include "polyscope/surface_scalar_quantity.h"
#include "polyscope/volume_mesh.h"

#include "scene/direct_rt_curve_registry.h"
#include "scene/ray_tracing_geometry_primitives.h"
#include "utility/rt_mesh_material_helpers.h"

namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
constexpr float kMinProceduralRadius = 1e-4f;

glm::mat4 makeTranslationTransform(const glm::vec3& offset) {
  glm::mat4 transform(1.0f);
  transform[3] = glm::vec4(offset, 1.0f);
  return transform;
}

void hashBytes(uint64_t& hash, const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= kFnvPrime;
  }
}

void hashString(uint64_t& hash, std::string_view value) {
  hashBytes(hash, value.data(), value.size());
}

template <typename T>
void hashVector(uint64_t& hash, const std::vector<T>& values) {
  if (values.empty()) return;
  hashBytes(hash, values.data(), values.size() * sizeof(T));
}

rt::RTMesh makeMeshFromSimpleTriangleMesh(polyscope::SimpleTriangleMesh& mesh) {
  rt::RTMesh out;
  out.name = mesh.getName();
  out.transform = mesh.getTransform();
  out.baseColorFactor = glm::vec4(mesh.getSurfaceColor(), 1.0f);
  out.vertices = mesh.vertices.data;
  out.indices = mesh.faces.data;
  return out;
}

rt::RTMesh makeMeshFromSurfaceMesh(polyscope::SurfaceMesh& mesh) {
  rt::RTMesh out;
  out.name = mesh.getName();
  out.transform = mesh.getTransform();
  out.baseColorFactor = glm::vec4(mesh.getSurfaceColor(), 1.0f);
  out.vertices = mesh.vertexPositions.data;

  const auto& triIndices = mesh.triangleVertexInds.data;
  if (triIndices.size() % 3 != 0) {
    throw std::runtime_error("Surface mesh triangulation index buffer is malformed.");
  }

  out.indices.reserve(triIndices.size() / 3);
  for (size_t i = 0; i < triIndices.size(); i += 3) {
    out.indices.emplace_back(triIndices[i + 0], triIndices[i + 1], triIndices[i + 2]);
  }

  if (auto* vcq = dynamic_cast<polyscope::SurfaceVertexColorQuantity*>(mesh.dominantQuantity)) {
    out.vertexColors = vcq->colors.data;
    out.baseColorFactor = glm::vec4(1.0f);
  } else if (auto* vsq = dynamic_cast<polyscope::SurfaceVertexScalarQuantity*>(mesh.dominantQuantity)) {
    const auto& scalars = vsq->values.data;
    const auto [vizMin, vizMax] = vsq->getMapRange();
    const double range = (vizMax > vizMin) ? (vizMax - vizMin) : 1.0;
    const polyscope::render::ValueColorMap& cmap =
        polyscope::render::engine->getColorMap(vsq->getColorMap());
    out.vertexColors.reserve(scalars.size());
    for (float s : scalars) {
      double t = (static_cast<double>(s) - vizMin) / range;
      out.vertexColors.push_back(cmap.getValue(t));
    }
    out.baseColorFactor = glm::vec4(1.0f);
  }

  return out;
}

template <typename T>
void applyMaterialPreset(T& target, std::string_view materialName) {
  using polyscope::rt::applyPhysicalParamsFromPreset;
  using polyscope::rt::Ceramic;
  using polyscope::rt::Clay;
  using polyscope::rt::Plastic;
  using polyscope::rt::PerfectDiffuse;
  using polyscope::rt::Rubber;

  if (materialName == "clay") { applyPhysicalParamsFromPreset(target, Clay()); return; }
  if (materialName == "flat") { applyPhysicalParamsFromPreset(target, PerfectDiffuse()); return; }
  if (materialName == "candy") { auto p = Plastic(); p.roughness = 0.08f; applyPhysicalParamsFromPreset(target, p); return; }
  if (materialName == "wax") { auto p = Plastic(); p.roughness = 0.35f; applyPhysicalParamsFromPreset(target, p); return; }
  if (materialName == "mud") { applyPhysicalParamsFromPreset(target, Rubber()); return; }
  if (materialName == "ceramic") { applyPhysicalParamsFromPreset(target, Ceramic()); return; }
  if (materialName == "jade") { auto p = Plastic(); p.roughness = 0.12f; applyPhysicalParamsFromPreset(target, p); return; }
  if (materialName == "normal") { auto p = Plastic(); p.roughness = 0.6f; applyPhysicalParamsFromPreset(target, p); return; }
  applyPhysicalParamsFromPreset(target, PerfectDiffuse());
}

rt::RTPointCloud makeRTPointCloud(polyscope::PointCloud& cloud) {
  rt::RTPointCloud out;
  out.name   = cloud.getName();
  out.radius = std::max(kMinProceduralRadius, static_cast<float>(cloud.getPointRadius()));
  out.baseColor = glm::vec4(cloud.getPointColor(), 1.0f);

  glm::mat4 transform = cloud.getTransform();
  out.centers.reserve(cloud.points.data.size());
  for (const glm::vec3& p : cloud.points.data) {
    out.centers.push_back(glm::vec3(transform * glm::vec4(p, 1.0f)));
  }

  // Reflect Polyscope material preset in roughness/metallic.
  rt::RTMesh tmp;
  tmp.metallicFactor  = 0.0f;
  tmp.roughnessFactor = 0.8f;
  applyMaterialPreset(tmp, cloud.getMaterial());
  out.metallic  = tmp.metallicFactor;
  out.roughness = tmp.roughnessFactor;
  out.unlit     = tmp.unlit;

  // If an enabled scalar quantity exists, compute per-point colors from its colormap.
  for (auto& [qName, qPtr] : cloud.quantities) {
    if (!qPtr->isEnabled()) continue;
    auto* scalarQ = dynamic_cast<polyscope::PointCloudScalarQuantity*>(qPtr.get());
    if (!scalarQ) continue;

    const std::string& cmapName = scalarQ->getColorMap();
    const polyscope::render::ValueColorMap& cmap =
        polyscope::render::engine->getColorMap(cmapName);
    auto [lo, hi] = scalarQ->getMapRange();
    const std::vector<float>& vals = scalarQ->values.data;

    out.colors.reserve(out.centers.size());
    double range = (hi != lo) ? (hi - lo) : 1.0;
    for (size_t i = 0; i < out.centers.size(); ++i) {
      float v = (i < vals.size()) ? vals[i] : 0.0f;
      double t = (v - lo) / range;
      out.colors.push_back(cmap.getValue(t));
    }
    break; // Use the first enabled scalar quantity.
  }

  return out;
}

rt::RTCurveNetwork makeCurveNetwork(polyscope::CurveNetwork& network) {
  rt::RTCurveNetwork out;
  out.name = network.getName();
  out.baseColor = glm::vec4(network.getColor(), 1.0f);
  applyMaterialPreset(out, network.getMaterial());

  float radius = std::max(kMinProceduralRadius, network.getRadius());
  glm::mat4 transform = network.getTransform();

  const auto& nodePositions = network.nodePositions.data;
  const auto& edgeTails = network.edgeTailInds.data;
  const auto& edgeTips = network.edgeTipInds.data;
  size_t edgeCount = std::min(edgeTails.size(), edgeTips.size());

  out.primitives.reserve(nodePositions.size() + edgeCount);

  for (const glm::vec3& node : nodePositions) {
    glm::vec3 worldPos = glm::vec3(transform * glm::vec4(node, 1.0f));
    rt::RTCurvePrimitive prim;
    prim.type = rt::RTCurvePrimitiveType::Sphere;
    prim.p0 = worldPos;
    prim.radius = radius;
    out.primitives.push_back(prim);
  }

  for (size_t i = 0; i < edgeCount; ++i) {
    uint32_t tail = edgeTails[i];
    uint32_t tip = edgeTips[i];
    if (tail >= nodePositions.size() || tip >= nodePositions.size()) continue;
    glm::vec3 worldTail = glm::vec3(transform * glm::vec4(nodePositions[tail], 1.0f));
    glm::vec3 worldTip = glm::vec3(transform * glm::vec4(nodePositions[tip], 1.0f));
    if (glm::length(worldTip - worldTail) < 1e-6f) continue;
    rt::RTCurvePrimitive prim;
    prim.type = rt::RTCurvePrimitiveType::Cylinder;
    prim.p0 = worldTail;
    prim.p1 = worldTip;
    prim.radius = radius;
    out.primitives.push_back(prim);
  }

  return out;
}

rt::RTMesh makeMeshFromVolumeMesh(polyscope::VolumeMesh& mesh) {
  rt::RTMesh out;
  out.name = mesh.getName();
  out.transform = mesh.getTransform();
  out.baseColorFactor = glm::vec4(mesh.getColor(), 1.0f);
  out.vertices = mesh.vertexPositions.data;
  applyMaterialPreset(out, mesh.getMaterial());

  const auto& triIndices = mesh.triangleVertexInds.data;
  const auto& triFaceIndices = mesh.triangleFaceInds.data;
  if (triIndices.size() % 3 != 0 || triFaceIndices.size() != triIndices.size()) {
    throw std::runtime_error("Volume mesh triangulation buffers are malformed.");
  }

  out.indices.reserve(triIndices.size() / 3);
  for (size_t i = 0; i < triIndices.size(); i += 3) {
    uint32_t faceIndex = triFaceIndices[i];
    if (faceIndex < mesh.faceIsInterior.size() && mesh.faceIsInterior[faceIndex]) continue;
    out.indices.emplace_back(triIndices[i + 0], triIndices[i + 1], triIndices[i + 2]);
  }
  return out;
}

void addMeshAndHash(PolyscopeSceneSnapshot& snapshot, rt::RTMesh&& mesh, polyscope::Structure& structure) {
  snapshot.supportedStructureCount++;
  if (snapshot.hostStructure == nullptr) {
    snapshot.hostStructure = &structure;
    snapshot.hostTypeName = structure.typeName();
    snapshot.hostName = structure.getName();
  }

  hashString(snapshot.scene.hash, structure.typeName());
  hashString(snapshot.scene.hash, structure.getName());
  hashVector(snapshot.scene.hash, mesh.vertices);
  hashVector(snapshot.scene.hash, mesh.normals);
  hashVector(snapshot.scene.hash, mesh.indices);
  hashVector(snapshot.scene.hash, mesh.vertexColors);
  hashBytes(snapshot.scene.hash, &mesh.transform[0][0], sizeof(float) * 16);
  hashBytes(snapshot.scene.hash, &mesh.baseColorFactor[0], sizeof(float) * 4);
  hashBytes(snapshot.scene.hash, &mesh.metallicFactor, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.roughnessFactor, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.emissiveFactor[0], sizeof(float) * 3);
  hashBytes(snapshot.scene.hash, &mesh.opacity, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.wireframe, sizeof(bool));
  hashBytes(snapshot.scene.hash, &mesh.edgeColor[0], sizeof(float) * 3);
  hashBytes(snapshot.scene.hash, &mesh.edgeWidth, sizeof(float));

  snapshot.scene.meshes.push_back(std::move(mesh));
}

void addCurveNetworkAndHash(PolyscopeSceneSnapshot& snapshot, rt::RTCurveNetwork&& curveNet, polyscope::Structure& structure) {
  snapshot.supportedStructureCount++;
  if (snapshot.hostStructure == nullptr) {
    snapshot.hostStructure = &structure;
    snapshot.hostTypeName = structure.typeName();
    snapshot.hostName = structure.getName();
  }

  hashString(snapshot.scene.hash, structure.typeName());
  hashString(snapshot.scene.hash, structure.getName());
  hashBytes(snapshot.scene.hash, &curveNet.baseColor[0], sizeof(float) * 4);
  hashBytes(snapshot.scene.hash, &curveNet.metallic, sizeof(float));
  hashBytes(snapshot.scene.hash, &curveNet.roughness, sizeof(float));
  for (const rt::RTCurvePrimitive& prim : curveNet.primitives) {
    hashBytes(snapshot.scene.hash, &prim.type, sizeof(prim.type));
    hashBytes(snapshot.scene.hash, &prim.p0[0], sizeof(float) * 3);
    hashBytes(snapshot.scene.hash, &prim.p1[0], sizeof(float) * 3);
    hashBytes(snapshot.scene.hash, &prim.radius, sizeof(float));
  }

  snapshot.scene.curveNetworks.push_back(std::move(curveNet));
}

void addPointCloudAndHash(PolyscopeSceneSnapshot& snapshot, rt::RTPointCloud&& pc, polyscope::Structure& structure) {
  snapshot.supportedStructureCount++;
  if (snapshot.hostStructure == nullptr) {
    snapshot.hostStructure = &structure;
    snapshot.hostTypeName  = structure.typeName();
    snapshot.hostName      = structure.getName();
  }

  hashString(snapshot.scene.hash, structure.typeName());
  hashString(snapshot.scene.hash, structure.getName());
  hashBytes(snapshot.scene.hash, &pc.baseColor[0], sizeof(float) * 4);
  hashBytes(snapshot.scene.hash, &pc.radius, sizeof(float));
  hashBytes(snapshot.scene.hash, &pc.metallic, sizeof(float));
  hashBytes(snapshot.scene.hash, &pc.roughness, sizeof(float));
  hashVector(snapshot.scene.hash, pc.centers);
  hashVector(snapshot.scene.hash, pc.colors);

  snapshot.scene.pointClouds.push_back(std::move(pc));
}

void addLightsAndHash(PolyscopeSceneSnapshot& snapshot, const std::vector<rt::RTPunctualLight>& apiLights) {
  snapshot.scene.lights = apiLights;
  for (const rt::RTPunctualLight& light : snapshot.scene.lights) {
    hashBytes(snapshot.scene.hash, &light.type, sizeof(light.type));
    hashBytes(snapshot.scene.hash, &light.color[0], sizeof(float) * 3);
    hashBytes(snapshot.scene.hash, &light.intensity, sizeof(float));
    hashBytes(snapshot.scene.hash, &light.position[0], sizeof(float) * 3);
    hashBytes(snapshot.scene.hash, &light.range, sizeof(float));
    hashBytes(snapshot.scene.hash, &light.direction[0], sizeof(float) * 3);
    hashBytes(snapshot.scene.hash, &light.innerConeAngle, sizeof(float));
    hashBytes(snapshot.scene.hash, &light.outerConeAngle, sizeof(float));
  }
}

void applyMaterialOverride(rt::RTMesh& mesh, const std::unordered_map<std::string, rt::MaterialOverride>& overrides) {
  auto it = overrides.find(mesh.name);
  if (it == overrides.end()) return;
  const rt::MaterialOverride& ov = it->second;
  if (ov.metallic) mesh.metallicFactor = *ov.metallic;
  if (ov.roughness) mesh.roughnessFactor = *ov.roughness;
  if (ov.baseColor) mesh.baseColorFactor = *ov.baseColor;
  if (ov.emissive) mesh.emissiveFactor = *ov.emissive;
  if (ov.transmission) mesh.transmissionFactor = *ov.transmission;
  if (ov.ior) mesh.indexOfRefraction = *ov.ior;
  if (ov.opacity) mesh.opacity = *ov.opacity;
  if (ov.unlit) mesh.unlit = *ov.unlit;
}

} // namespace

PolyscopeSceneSnapshot capturePolyscopeSceneSnapshot() {
  static const std::unordered_map<std::string, rt::MaterialOverride> emptyOverrides;
  static const std::vector<rt::RTPunctualLight> emptyLights;
  return capturePolyscopeSceneSnapshot(emptyOverrides, emptyLights);
}

PolyscopeSceneSnapshot capturePolyscopeSceneSnapshot(const std::unordered_map<std::string, rt::MaterialOverride>& materialOverrides,
                                   const std::vector<rt::RTPunctualLight>& apiLights) {
  PolyscopeSceneSnapshot snapshot;
  snapshot.scene.hash = kFnvOffset;

  for (auto& [typeName, structures] : polyscope::state::structures) {
    (void)typeName;
    for (auto& [name, structurePtr] : structures) {
      (void)name;
      polyscope::Structure* structure = structurePtr.get();
      if (structure == nullptr || !structure->isEnabled()) continue;

      if (auto* simpleMesh = dynamic_cast<polyscope::SimpleTriangleMesh*>(structure)) {
        rt::RTMesh mesh = makeMeshFromSimpleTriangleMesh(*simpleMesh);
        applyMaterialPreset(mesh, simpleMesh->getMaterial());
        applyMaterialOverride(mesh, materialOverrides);
        if (mesh.vertices.empty() || mesh.indices.empty()) continue;
        addMeshAndHash(snapshot, std::move(mesh), *structure);
        continue;
      }

      if (auto* surfaceMesh = dynamic_cast<polyscope::SurfaceMesh*>(structure)) {
        if (surfaceMesh->triangleVertexInds.data.empty()) continue;
        rt::RTMesh mesh = makeMeshFromSurfaceMesh(*surfaceMesh);
        applyMaterialPreset(mesh, surfaceMesh->getMaterial());
        applyMaterialOverride(mesh, materialOverrides);
        if (mesh.vertices.empty() || mesh.indices.empty()) continue;
        if (surfaceMesh->getEdgeWidth() > 0.0) {
          mesh.wireframe  = true;
          mesh.edgeColor  = surfaceMesh->getEdgeColor();
          mesh.edgeWidth  = static_cast<float>(surfaceMesh->getEdgeWidth());
        }
        addMeshAndHash(snapshot, std::move(mesh), *structure);
        continue;
      }

      if (auto* pointCloud = dynamic_cast<polyscope::PointCloud*>(structure)) {
        if (pointCloud->points.data.empty()) continue;
        rt::RTPointCloud pc = makeRTPointCloud(*pointCloud);
        addPointCloudAndHash(snapshot, std::move(pc), *pointCloud);
        continue;
      }

      if (auto* curveNetwork = dynamic_cast<polyscope::CurveNetwork*>(structure)) {
        rt::RTCurveNetwork curveNet = makeCurveNetwork(*curveNetwork);
        if (curveNet.primitives.empty()) continue;
        addCurveNetworkAndHash(snapshot, std::move(curveNet), *structure);
        continue;
      }

      if (auto* volumeMesh = dynamic_cast<polyscope::VolumeMesh*>(structure)) {
        if (volumeMesh->triangleVertexInds.data.empty()) continue;
        rt::RTMesh mesh = makeMeshFromVolumeMesh(*volumeMesh);
        applyMaterialOverride(mesh, materialOverrides);
        if (mesh.vertices.empty() || mesh.indices.empty()) continue;
        addMeshAndHash(snapshot, std::move(mesh), *structure);
      }
    }
  }

  // Include curve networks registered directly via polyscope::rt::registerCurveNetwork().
  for (const rt::RTCurveNetwork& directNet : getDirectRtCurveNetworks()) {
    snapshot.supportedStructureCount++;
    hashString(snapshot.scene.hash, directNet.name);
    hashBytes(snapshot.scene.hash, &directNet.baseColor[0], sizeof(float) * 4);
    hashBytes(snapshot.scene.hash, &directNet.metallic, sizeof(float));
    hashBytes(snapshot.scene.hash, &directNet.roughness, sizeof(float));
    for (const rt::RTCurvePrimitive& prim : directNet.primitives) {
      hashBytes(snapshot.scene.hash, &prim.type, sizeof(prim.type));
      hashBytes(snapshot.scene.hash, &prim.p0[0], sizeof(float) * 3);
      hashBytes(snapshot.scene.hash, &prim.p1[0], sizeof(float) * 3);
      hashBytes(snapshot.scene.hash, &prim.radius, sizeof(float));
    }
    snapshot.scene.curveNetworks.push_back(directNet);
  }

  addLightsAndHash(snapshot, apiLights);

  return snapshot;
}
