#include "scene/polyscope_scene_snapshot.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "polyscope/curve_network.h"
#include "polyscope/curve_network_color_quantity.h"
#include "polyscope/curve_network_scalar_quantity.h"
#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"
#include "polyscope/point_cloud_color_quantity.h"
#include "polyscope/point_cloud_scalar_quantity.h"
#include "polyscope/render/color_maps.h"
#include "polyscope/render/engine.h"
#include "polyscope/simple_triangle_mesh.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/surface_color_quantity.h"
#include "polyscope/surface_scalar_quantity.h"
#include "polyscope/surface_vector_quantity.h"
#include "polyscope/vector_quantity.h"
#include "polyscope/volume_mesh.h"

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

// Average per-face colors to per-vertex by summing all adjacent face contributions.
// Vertices not touched by any triangle fall back to `fallback`.
void faceColorsToVertex(const std::vector<glm::vec3>& faceColors,
                        const std::vector<uint32_t>&  triVertInds,
                        const std::vector<uint32_t>&  triFaceInds,
                        size_t nVerts, glm::vec3 fallback,
                        std::vector<glm::vec3>& outColors) {
  std::vector<glm::vec3> sum(nVerts, glm::vec3(0.f));
  std::vector<int>        cnt(nVerts, 0);
  for (size_t i = 0; i < triVertInds.size(); ++i) {
    uint32_t v = triVertInds[i], f = triFaceInds[i];
    if (v < nVerts && f < faceColors.size()) { sum[v] += faceColors[f]; ++cnt[v]; }
  }
  outColors.reserve(nVerts);
  for (size_t i = 0; i < nVerts; ++i)
    outColors.push_back(cnt[i] > 0 ? sum[i] / float(cnt[i]) : fallback);
}

// Average per-face scalars to per-vertex; returns the per-vertex float array.
std::vector<float> faceScalarsToVertex(const std::vector<float>& faceScalars,
                                       const std::vector<uint32_t>& triVertInds,
                                       const std::vector<uint32_t>& triFaceInds,
                                       size_t nVerts, float fallback) {
  std::vector<double> sum(nVerts, 0.0);
  std::vector<int>    cnt(nVerts, 0);
  for (size_t i = 0; i < triVertInds.size(); ++i) {
    uint32_t v = triVertInds[i], f = triFaceInds[i];
    if (v < nVerts && f < faceScalars.size()) { sum[v] += faceScalars[f]; ++cnt[v]; }
  }
  std::vector<float> out;
  out.reserve(nVerts);
  for (size_t i = 0; i < nVerts; ++i)
    out.push_back(cnt[i] > 0 ? float(sum[i] / cnt[i]) : fallback);
  return out;
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

  // Propagate Polyscope's smooth vertex normals so the path tracer interpolates
  // them properly instead of falling back to flat (geometric) normals.
  mesh.vertexNormals.ensureHostBufferPopulated();
  if (mesh.vertexNormals.data.size() == mesh.vertexPositions.data.size()) {
    out.normals = mesh.vertexNormals.data;
  }

  const size_t nVerts = mesh.vertexPositions.data.size();

  if (auto* vcq = dynamic_cast<polyscope::SurfaceVertexColorQuantity*>(mesh.dominantQuantity)) {
    vcq->colors.ensureHostBufferPopulated();
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
    // Propagate isoline settings so the RT shader can draw stripes.
    if (vsq->getIsolinesEnabled()) {
      out.isoScalars          = scalars;
      out.isoSpacing          = static_cast<float>(vsq->getIsolinePeriod());
      out.isoDarkness         = static_cast<float>(vsq->getIsolineDarkness());
      out.isoContourThickness = static_cast<float>(vsq->getIsolineContourThickness());
      out.isoStyle            = (vsq->getIsolineStyle() == polyscope::IsolineStyle::Contour) ? 2 : 1;
    }
  } else if (auto* fcq = dynamic_cast<polyscope::SurfaceFaceColorQuantity*>(mesh.dominantQuantity)) {
    // Face color: average each face's color onto its vertices.
    fcq->colors.ensureHostBufferPopulated();
    mesh.triangleFaceInds.ensureHostBufferPopulated();
    faceColorsToVertex(fcq->colors.data, triIndices, mesh.triangleFaceInds.data,
                       nVerts, mesh.getSurfaceColor(), out.vertexColors);
    out.baseColorFactor = glm::vec4(1.0f);
  } else if (auto* fsq = dynamic_cast<polyscope::SurfaceFaceScalarQuantity*>(mesh.dominantQuantity)) {
    // Face scalar: average adjacent face scalars to each vertex, then colormap.
    mesh.triangleFaceInds.ensureHostBufferPopulated();
    const auto& faceScalars = fsq->values.data;
    const auto [vizMin, vizMax] = fsq->getMapRange();
    const double range = (vizMax > vizMin) ? (vizMax - vizMin) : 1.0;
    const polyscope::render::ValueColorMap& cmap =
        polyscope::render::engine->getColorMap(fsq->getColorMap());

    const std::vector<float> vertScalars = faceScalarsToVertex(
        faceScalars, triIndices, mesh.triangleFaceInds.data, nVerts, float(vizMin));

    out.vertexColors.reserve(nVerts);
    for (float s : vertScalars) {
      double t = (static_cast<double>(s) - vizMin) / range;
      out.vertexColors.push_back(cmap.getValue(t));
    }
    out.baseColorFactor = glm::vec4(1.0f);

    if (fsq->getIsolinesEnabled()) {
      out.isoScalars          = vertScalars;
      out.isoSpacing          = static_cast<float>(fsq->getIsolinePeriod());
      out.isoDarkness         = static_cast<float>(fsq->getIsolineDarkness());
      out.isoContourThickness = static_cast<float>(fsq->getIsolineContourThickness());
      out.isoStyle            = (fsq->getIsolineStyle() == polyscope::IsolineStyle::Contour) ? 2 : 1;
    }
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

  // Priority 1: per-point color quantity (direct RGB, no colormap needed).
  for (auto& [qName, qPtr] : cloud.quantities) {
    if (!qPtr->isEnabled()) continue;
    auto* colorQ = dynamic_cast<polyscope::PointCloudColorQuantity*>(qPtr.get());
    if (!colorQ) continue;
    colorQ->colors.ensureHostBufferPopulated();
    const auto& rawColors = colorQ->colors.data;
    out.colors.reserve(rawColors.size());
    for (const auto& c : rawColors) out.colors.push_back(c);
    break;
  }

  // Priority 2: per-point scalar quantity mapped through a colormap.
  if (out.colors.empty()) {
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

  network.nodePositions.ensureHostBufferPopulated();
  network.edgeTailInds.ensureHostBufferPopulated();
  network.edgeTipInds.ensureHostBufferPopulated();

  const auto& nodePositions = network.nodePositions.data;
  const auto& edgeTails = network.edgeTailInds.data;
  const auto& edgeTips = network.edgeTipInds.data;
  size_t nNodes = nodePositions.size();
  size_t edgeCount = std::min(edgeTails.size(), edgeTips.size());

  out.primitives.reserve(nNodes + edgeCount);

  for (const glm::vec3& node : nodePositions) {
    glm::vec3 worldPos = glm::vec3(transform * glm::vec4(node, 1.0f));
    rt::RTCurvePrimitive prim;
    prim.type = rt::RTCurvePrimitiveType::Sphere;
    prim.p0 = worldPos;
    prim.radius = radius;
    out.primitives.push_back(prim);
  }

  // Track which edge index each cylinder corresponds to (some may be skipped for degenerate edges).
  std::vector<size_t> cylinderEdgeIndex;
  cylinderEdgeIndex.reserve(edgeCount);
  for (size_t i = 0; i < edgeCount; ++i) {
    uint32_t tail = edgeTails[i];
    uint32_t tip = edgeTips[i];
    if (tail >= nNodes || tip >= nNodes) continue;
    glm::vec3 worldTail = glm::vec3(transform * glm::vec4(nodePositions[tail], 1.0f));
    glm::vec3 worldTip = glm::vec3(transform * glm::vec4(nodePositions[tip], 1.0f));
    if (glm::length(worldTip - worldTail) < 1e-6f) continue;
    rt::RTCurvePrimitive prim;
    prim.type = rt::RTCurvePrimitiveType::Cylinder;
    prim.p0 = worldTail;
    prim.p1 = worldTip;
    prim.radius = radius;
    out.primitives.push_back(prim);
    cylinderEdgeIndex.push_back(i);
  }

  // ------------------------------------------------------------------
  // Extract per-primitive colors from the dominant quantity (if any).
  // primitiveColors layout: [0..nNodes-1] = node spheres, [nNodes..] = cylinders.
  // ------------------------------------------------------------------
  const glm::vec3 fallback = glm::vec3(network.getColor());

  auto colormapScalars = [&](const std::vector<float>& scalars,
                              const polyscope::render::ValueColorMap& cmap,
                              double vizMin, double vizMax) -> std::vector<glm::vec3> {
    const double range = (vizMax > vizMin) ? (vizMax - vizMin) : 1.0;
    std::vector<glm::vec3> colors;
    colors.reserve(scalars.size());
    for (float s : scalars) {
      double t = (static_cast<double>(s) - vizMin) / range;
      colors.push_back(cmap.getValue(t));
    }
    return colors;
  };

  if (auto* q = dynamic_cast<polyscope::CurveNetworkNodeColorQuantity*>(network.dominantQuantity)) {
    // Node color: directly interpolate onto edge midpoints.
    q->colors.ensureHostBufferPopulated();
    const auto& nc = q->colors.data;
    out.primitiveColors.reserve(nNodes + cylinderEdgeIndex.size());
    for (size_t i = 0; i < nNodes; ++i)
      out.primitiveColors.push_back(i < nc.size() ? nc[i] : fallback);
    for (size_t ci = 0; ci < cylinderEdgeIndex.size(); ++ci) {
      size_t ei = cylinderEdgeIndex[ci];
      uint32_t t = edgeTails[ei], p = edgeTips[ei];
      glm::vec3 ct = (t < nc.size()) ? nc[t] : fallback;
      glm::vec3 cp = (p < nc.size()) ? nc[p] : fallback;
      out.primitiveColors.push_back((ct + cp) * 0.5f);
    }

  } else if (auto* q = dynamic_cast<polyscope::CurveNetworkEdgeColorQuantity*>(network.dominantQuantity)) {
    // Edge color: node spheres use pre-averaged node colors; cylinders use per-edge color.
    q->colors.ensureHostBufferPopulated();
    q->nodeAverageColors.ensureHostBufferPopulated();
    const auto& ec = q->colors.data;
    const auto& nav = q->nodeAverageColors.data;
    out.primitiveColors.reserve(nNodes + cylinderEdgeIndex.size());
    for (size_t i = 0; i < nNodes; ++i)
      out.primitiveColors.push_back(i < nav.size() ? nav[i] : fallback);
    for (size_t ci = 0; ci < cylinderEdgeIndex.size(); ++ci) {
      size_t ei = cylinderEdgeIndex[ci];
      out.primitiveColors.push_back(ei < ec.size() ? ec[ei] : fallback);
    }

  } else if (auto* q = dynamic_cast<polyscope::CurveNetworkNodeScalarQuantity*>(network.dominantQuantity)) {
    q->values.ensureHostBufferPopulated();
    const auto [vizMin, vizMax] = q->getMapRange();
    const auto& cmap = polyscope::render::engine->getColorMap(q->getColorMap());
    std::vector<glm::vec3> nc = colormapScalars(q->values.data, cmap, vizMin, vizMax);
    out.primitiveColors.reserve(nNodes + cylinderEdgeIndex.size());
    for (size_t i = 0; i < nNodes; ++i)
      out.primitiveColors.push_back(i < nc.size() ? nc[i] : fallback);
    for (size_t ci = 0; ci < cylinderEdgeIndex.size(); ++ci) {
      size_t ei = cylinderEdgeIndex[ci];
      uint32_t t = edgeTails[ei], p = edgeTips[ei];
      glm::vec3 ct = (t < nc.size()) ? nc[t] : fallback;
      glm::vec3 cp = (p < nc.size()) ? nc[p] : fallback;
      out.primitiveColors.push_back((ct + cp) * 0.5f);
    }

  } else if (auto* q = dynamic_cast<polyscope::CurveNetworkEdgeScalarQuantity*>(network.dominantQuantity)) {
    q->values.ensureHostBufferPopulated();
    q->nodeAverageValues.ensureHostBufferPopulated();
    const auto [vizMin, vizMax] = q->getMapRange();
    const auto& cmap = polyscope::render::engine->getColorMap(q->getColorMap());
    const auto& ev = q->values.data;
    // Node spheres use pre-averaged node values.
    std::vector<glm::vec3> nodeColors = colormapScalars(q->nodeAverageValues.data, cmap, vizMin, vizMax);
    out.primitiveColors.reserve(nNodes + cylinderEdgeIndex.size());
    for (size_t i = 0; i < nNodes; ++i)
      out.primitiveColors.push_back(i < nodeColors.size() ? nodeColors[i] : fallback);
    for (size_t ci = 0; ci < cylinderEdgeIndex.size(); ++ci) {
      size_t ei = cylinderEdgeIndex[ci];
      glm::vec3 col = (ei < ev.size()) ? cmap.getValue((ev[ei] - vizMin) / ((vizMax > vizMin) ? (vizMax - vizMin) : 1.0)) : fallback;
      out.primitiveColors.push_back(col);
    }
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
  hashVector(snapshot.scene.hash, mesh.isoScalars);
  hashBytes(snapshot.scene.hash, &mesh.isoSpacing, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.isoDarkness, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.isoContourThickness, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.isoStyle, sizeof(int));

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
  hashVector(snapshot.scene.hash, curveNet.primitiveColors);

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

// ---------------------------------------------------------------------------
// Vector-field helpers
// ---------------------------------------------------------------------------

// Convert one enabled VectorQuantity<T> to an RTVectorField.
// dir = raw_vector * (lengthScale / lengthRange) — matches Polyscope's u_lengthMult.
template <typename QT>
rt::RTVectorField vectorQuantityToRT(polyscope::VectorQuantity<QT>& qty) {
  rt::RTVectorField out;
  out.color  = qty.getVectorColor();
  out.radius = static_cast<float>(std::max(1e-5, qty.getVectorRadius()));

  const double scale = qty.getVectorLengthScale();
  const double range = qty.getVectorLengthRange();
  const float  mult  = (range > 0.0) ? static_cast<float>(scale / range) : 1.0f;

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

// Convert one enabled TangentVectorQuantity<T> to an RTVectorField.
// 3D direction = tangent.x * basisX + tangent.y * basisY, then scaled.
template <typename QT>
rt::RTVectorField tangentVectorQuantityToRT(polyscope::TangentVectorQuantity<QT>& qty) {
  rt::RTVectorField out;
  out.color  = qty.getVectorColor();
  out.radius = static_cast<float>(std::max(1e-5, qty.getVectorRadius()));

  const double scale = qty.getVectorLengthScale();
  const double range = qty.getVectorLengthRange();
  const float  mult  = (range > 0.0) ? static_cast<float>(scale / range) : 1.0f;

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

// Extract all enabled surface vector quantities from a SurfaceMesh.
std::vector<rt::RTVectorField> makeRTVectorFields(polyscope::SurfaceMesh& mesh) {
  std::vector<rt::RTVectorField> fields;
  for (auto& [qName, qPtr] : mesh.quantities) {
    if (!qPtr->isEnabled()) continue;

    if (auto* q = dynamic_cast<polyscope::SurfaceVertexVectorQuantity*>(qPtr.get())) {
      auto field = vectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
    if (auto* q = dynamic_cast<polyscope::SurfaceFaceVectorQuantity*>(qPtr.get())) {
      auto field = vectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
    if (auto* q = dynamic_cast<polyscope::SurfaceFaceTangentVectorQuantity*>(qPtr.get())) {
      auto field = tangentVectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
    if (auto* q = dynamic_cast<polyscope::SurfaceVertexTangentVectorQuantity*>(qPtr.get())) {
      auto field = tangentVectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
    if (auto* q = dynamic_cast<polyscope::SurfaceOneFormTangentVectorQuantity*>(qPtr.get())) {
      auto field = tangentVectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
  }
  return fields;
}

void addVectorFieldsAndHash(PolyscopeSceneSnapshot& snapshot,
                            std::vector<rt::RTVectorField>&& fields,
                            std::string_view parentName) {
  for (rt::RTVectorField& vf : fields) {
    snapshot.supportedStructureCount++;
    hashString(snapshot.scene.hash, parentName);
    hashString(snapshot.scene.hash, vf.name);
    hashBytes(snapshot.scene.hash, &vf.color[0], sizeof(float) * 3);
    hashBytes(snapshot.scene.hash, &vf.radius, sizeof(float));
    hashVector(snapshot.scene.hash, vf.roots);
    hashVector(snapshot.scene.hash, vf.directions);
    snapshot.scene.vectorFields.push_back(std::move(vf));
  }
}

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

        // Also extract any enabled vector-field quantities attached to this mesh.
        auto vfields = makeRTVectorFields(*surfaceMesh);
        addVectorFieldsAndHash(snapshot, std::move(vfields), surfaceMesh->getName());
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

  addLightsAndHash(snapshot, apiLights);

  return snapshot;
}
