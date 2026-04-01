#include "scene/polyscope_scene_snapshot_internal.h"

#include <cstring>
#include <stdexcept>

#include "polyscope/curve_network_color_quantity.h"
#include "polyscope/curve_network_scalar_quantity.h"
#include "polyscope/point_cloud_color_quantity.h"
#include "polyscope/point_cloud_scalar_quantity.h"
#include "polyscope/render/color_maps.h"
#include "polyscope/render/engine.h"
#include "polyscope/surface_color_quantity.h"
#include "polyscope/surface_scalar_quantity.h"

#include "utility/rt_mesh_material_helpers.h"

namespace snapshot_detail {

rt::RTMesh makeMeshFromSimpleTriangleMesh(polyscope::SimpleTriangleMesh& mesh) {
  rt::RTMesh out;
  out.name = mesh.getName();
  out.transform = mesh.getTransform();
  out.baseColorFactor = glm::vec4(mesh.getSurfaceColor(), 1.0f);
  out.opacity = mesh.getTransparency();
  out.doubleSided = mesh.getBackFacePolicy() != polyscope::BackFacePolicy::Cull;
  out.vertices = mesh.vertices.data;
  out.indices = mesh.faces.data;
  return out;
}

rt::RTMesh makeMeshFromSurfaceMesh(polyscope::SurfaceMesh& mesh) {
  rt::RTMesh out;
  out.name = mesh.getName();
  out.transform = mesh.getTransform();
  out.baseColorFactor = glm::vec4(mesh.getSurfaceColor(), 1.0f);
  out.opacity = mesh.getTransparency();
  out.doubleSided = mesh.getBackFacePolicy() != polyscope::BackFacePolicy::Cull;
  out.vertices = mesh.vertexPositions.data;

  const auto& triIndices = mesh.triangleVertexInds.data;
  if (triIndices.size() % 3 != 0) {
    throw std::runtime_error("Surface mesh triangulation index buffer is malformed.");
  }

  out.indices.reserve(triIndices.size() / 3);
  for (size_t i = 0; i < triIndices.size(); i += 3) {
    out.indices.emplace_back(triIndices[i + 0], triIndices[i + 1], triIndices[i + 2]);
  }

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
    const auto& cmap = polyscope::render::engine->getColorMap(vsq->getColorMap());
    out.vertexColors.reserve(scalars.size());
    for (float s : scalars) {
      const double t = (static_cast<double>(s) - vizMin) / range;
      out.vertexColors.push_back(cmap.getValue(t));
    }
    out.baseColorFactor = glm::vec4(1.0f);
    if (vsq->getIsolinesEnabled()) {
      out.isoScalars = scalars;
      out.isoSpacing = static_cast<float>(vsq->getIsolinePeriod());
      out.isoDarkness = static_cast<float>(vsq->getIsolineDarkness());
      out.isoContourThickness = static_cast<float>(vsq->getIsolineContourThickness());
      out.isoStyle = (vsq->getIsolineStyle() == polyscope::IsolineStyle::Contour) ? 2 : 1;
    }
  } else if (auto* fcq = dynamic_cast<polyscope::SurfaceFaceColorQuantity*>(mesh.dominantQuantity)) {
    fcq->colors.ensureHostBufferPopulated();
    mesh.triangleFaceInds.ensureHostBufferPopulated();
    faceColorsToVertex(fcq->colors.data, triIndices, mesh.triangleFaceInds.data,
                       nVerts, mesh.getSurfaceColor(), out.vertexColors);
    out.baseColorFactor = glm::vec4(1.0f);
  } else if (auto* fsq = dynamic_cast<polyscope::SurfaceFaceScalarQuantity*>(mesh.dominantQuantity)) {
    mesh.triangleFaceInds.ensureHostBufferPopulated();
    const auto& faceScalars = fsq->values.data;
    const auto [vizMin, vizMax] = fsq->getMapRange();
    const double range = (vizMax > vizMin) ? (vizMax - vizMin) : 1.0;
    const auto& cmap = polyscope::render::engine->getColorMap(fsq->getColorMap());

    const std::vector<float> vertScalars = faceScalarsToVertex(
        faceScalars, triIndices, mesh.triangleFaceInds.data, nVerts, float(vizMin));

    out.vertexColors.reserve(nVerts);
    for (float s : vertScalars) {
      const double t = (static_cast<double>(s) - vizMin) / range;
      out.vertexColors.push_back(cmap.getValue(t));
    }
    out.baseColorFactor = glm::vec4(1.0f);

    if (fsq->getIsolinesEnabled()) {
      out.isoScalars = vertScalars;
      out.isoSpacing = static_cast<float>(fsq->getIsolinePeriod());
      out.isoDarkness = static_cast<float>(fsq->getIsolineDarkness());
      out.isoContourThickness = static_cast<float>(fsq->getIsolineContourThickness());
      out.isoStyle = (fsq->getIsolineStyle() == polyscope::IsolineStyle::Contour) ? 2 : 1;
    }
  }

  return out;
}

rt::RTPointCloud makeRTPointCloud(polyscope::PointCloud& cloud) {
  rt::RTPointCloud out;
  out.name = cloud.getName();
  out.radius = std::max(kMinProceduralRadius, static_cast<float>(cloud.getPointRadius()));
  out.baseColor = glm::vec4(cloud.getPointColor(), 1.0f);

  const glm::mat4 transform = cloud.getTransform();
  out.centers.reserve(cloud.points.data.size());
  for (const glm::vec3& p : cloud.points.data) {
    out.centers.push_back(glm::vec3(transform * glm::vec4(p, 1.0f)));
  }

  rt::RTMesh tmp;
  tmp.metallicFactor = 0.0f;
  tmp.roughnessFactor = 0.8f;
  applyMaterialPreset(tmp, cloud.getMaterial());
  out.metallic = tmp.metallicFactor;
  out.roughness = tmp.roughnessFactor;
  out.unlit = tmp.unlit;

  for (auto& [qName, qPtr] : cloud.quantities) {
    (void)qName;
    if (!qPtr->isEnabled()) continue;
    auto* colorQ = dynamic_cast<polyscope::PointCloudColorQuantity*>(qPtr.get());
    if (!colorQ) continue;
    colorQ->colors.ensureHostBufferPopulated();
    const auto& rawColors = colorQ->colors.data;
    out.colors.reserve(rawColors.size());
    for (const auto& c : rawColors) out.colors.push_back(c);
    break;
  }

  if (out.colors.empty()) {
    for (auto& [qName, qPtr] : cloud.quantities) {
      (void)qName;
      if (!qPtr->isEnabled()) continue;
      auto* scalarQ = dynamic_cast<polyscope::PointCloudScalarQuantity*>(qPtr.get());
      if (!scalarQ) continue;

      const auto& cmap = polyscope::render::engine->getColorMap(scalarQ->getColorMap());
      auto [lo, hi] = scalarQ->getMapRange();
      const std::vector<float>& vals = scalarQ->values.data;

      out.colors.reserve(out.centers.size());
      const double range = (hi != lo) ? (hi - lo) : 1.0;
      for (size_t i = 0; i < out.centers.size(); ++i) {
        const float v = (i < vals.size()) ? vals[i] : 0.0f;
        const double t = (v - lo) / range;
        out.colors.push_back(cmap.getValue(t));
      }
      break;
    }
  }

  return out;
}

rt::RTCurveNetwork makeCurveNetwork(polyscope::CurveNetwork& network) {
  rt::RTCurveNetwork out;
  out.name = network.getName();
  out.baseColor = glm::vec4(network.getColor(), 1.0f);
  applyMaterialPreset(out, network.getMaterial());

  const float radius = std::max(kMinProceduralRadius, network.getRadius());
  const glm::mat4 transform = network.getTransform();

  network.nodePositions.ensureHostBufferPopulated();
  network.edgeTailInds.ensureHostBufferPopulated();
  network.edgeTipInds.ensureHostBufferPopulated();

  const auto& nodePositions = network.nodePositions.data;
  const auto& edgeTails = network.edgeTailInds.data;
  const auto& edgeTips = network.edgeTipInds.data;
  const size_t nNodes = nodePositions.size();
  const size_t edgeCount = std::min(edgeTails.size(), edgeTips.size());

  out.primitives.reserve(edgeCount + nNodes);

  std::vector<int> nodeDegree(nNodes, 0);
  for (size_t i = 0; i < edgeCount; ++i) {
    const uint32_t tail = edgeTails[i];
    const uint32_t tip  = edgeTips[i];
    if (tail >= nNodes || tip >= nNodes) continue;
    const glm::vec3 wt = glm::vec3(transform * glm::vec4(nodePositions[tail], 1.0f));
    const glm::vec3 wp = glm::vec3(transform * glm::vec4(nodePositions[tip],  1.0f));
    if (glm::length(wp - wt) < 1e-6f) continue;
    nodeDegree[tail]++;
    nodeDegree[tip]++;
  }

  std::vector<size_t> sphereNodeIndices;
  for (size_t ni = 0; ni < nNodes; ++ni) {
    if (nodeDegree[ni] == 2) continue;
    const glm::vec3 worldPos = glm::vec3(transform * glm::vec4(nodePositions[ni], 1.0f));
    rt::RTCurvePrimitive prim;
    prim.type = rt::RTCurvePrimitiveType::Sphere;
    prim.p0 = worldPos;
    prim.radius = radius;
    out.primitives.push_back(prim);
    sphereNodeIndices.push_back(ni);
  }

  std::vector<size_t> cylinderEdgeIndex;
  cylinderEdgeIndex.reserve(edgeCount);
  for (size_t i = 0; i < edgeCount; ++i) {
    const uint32_t tail = edgeTails[i];
    const uint32_t tip = edgeTips[i];
    if (tail >= nNodes || tip >= nNodes) continue;
    const glm::vec3 worldTail = glm::vec3(transform * glm::vec4(nodePositions[tail], 1.0f));
    const glm::vec3 worldTip = glm::vec3(transform * glm::vec4(nodePositions[tip], 1.0f));
    if (glm::length(worldTip - worldTail) < 1e-6f) continue;
    rt::RTCurvePrimitive prim;
    prim.type = rt::RTCurvePrimitiveType::Cylinder;
    prim.p0 = worldTail;
    prim.p1 = worldTip;
    prim.radius = radius;
    out.primitives.push_back(prim);
    cylinderEdgeIndex.push_back(i);
  }

  out.nodePositions.reserve(nNodes);
  for (const glm::vec3& node : nodePositions) {
    out.nodePositions.push_back(glm::vec3(transform * glm::vec4(node, 1.0f)));
  }
  out.edgeTailInds.reserve(cylinderEdgeIndex.size());
  out.edgeTipInds.reserve(cylinderEdgeIndex.size());
  for (size_t ci : cylinderEdgeIndex) {
    out.edgeTailInds.push_back(edgeTails[ci]);
    out.edgeTipInds.push_back(edgeTips[ci]);
  }

  const glm::vec3 fallback = glm::vec3(network.getColor());
  const size_t nSpheres = sphereNodeIndices.size();

  auto colormapScalars = [&](const std::vector<float>& scalars,
                             const polyscope::render::ValueColorMap& cmap,
                             double vizMin, double vizMax) -> std::vector<glm::vec3> {
    const double range = (vizMax > vizMin) ? (vizMax - vizMin) : 1.0;
    std::vector<glm::vec3> colors;
    colors.reserve(scalars.size());
    for (float s : scalars) {
      const double t = (static_cast<double>(s) - vizMin) / range;
      colors.push_back(cmap.getValue(t));
    }
    return colors;
  };

  if (auto* q = dynamic_cast<polyscope::CurveNetworkNodeColorQuantity*>(network.dominantQuantity)) {
    q->colors.ensureHostBufferPopulated();
    const auto& nc = q->colors.data;
    out.primitiveColors.reserve(nSpheres + cylinderEdgeIndex.size());
    out.primitiveColors1.reserve(nSpheres + cylinderEdgeIndex.size());
    for (size_t ni : sphereNodeIndices) {
      const glm::vec3 c = ni < nc.size() ? nc[ni] : fallback;
      out.primitiveColors.push_back(c);
      out.primitiveColors1.push_back(c);
    }
    for (size_t ci = 0; ci < cylinderEdgeIndex.size(); ++ci) {
      const size_t ei = cylinderEdgeIndex[ci];
      const uint32_t t = edgeTails[ei];
      const uint32_t p = edgeTips[ei];
      out.primitiveColors.push_back(t < nc.size() ? nc[t] : fallback);
      out.primitiveColors1.push_back(p < nc.size() ? nc[p] : fallback);
    }
  } else if (auto* q = dynamic_cast<polyscope::CurveNetworkEdgeColorQuantity*>(network.dominantQuantity)) {
    q->colors.ensureHostBufferPopulated();
    const auto& ec = q->colors.data;
    std::vector<glm::vec3> nodeColorSum(nNodes, glm::vec3(0.0f));
    std::vector<int> nodeColorCnt(nNodes, 0);
    for (size_t i = 0; i < edgeCount; ++i) {
      const uint32_t t = edgeTails[i];
      const uint32_t p = edgeTips[i];
      if (t >= nNodes || p >= nNodes || i >= ec.size()) continue;
      nodeColorSum[t] += ec[i];
      nodeColorCnt[t]++;
      nodeColorSum[p] += ec[i];
      nodeColorCnt[p]++;
    }
    out.primitiveColors.reserve(nSpheres + cylinderEdgeIndex.size());
    out.primitiveColors1.reserve(nSpheres + cylinderEdgeIndex.size());
    for (size_t ni : sphereNodeIndices) {
      const glm::vec3 col = (nodeColorCnt[ni] > 0) ? nodeColorSum[ni] / float(nodeColorCnt[ni]) : fallback;
      out.primitiveColors.push_back(col);
      out.primitiveColors1.push_back(col);
    }
    for (size_t ci = 0; ci < cylinderEdgeIndex.size(); ++ci) {
      const size_t ei = cylinderEdgeIndex[ci];
      const glm::vec3 col = ei < ec.size() ? ec[ei] : fallback;
      out.primitiveColors.push_back(col);
      out.primitiveColors1.push_back(col);
    }
  } else if (auto* q = dynamic_cast<polyscope::CurveNetworkNodeScalarQuantity*>(network.dominantQuantity)) {
    q->values.ensureHostBufferPopulated();
    const auto [vizMin, vizMax] = q->getMapRange();
    const auto& cmap = polyscope::render::engine->getColorMap(q->getColorMap());
    const std::vector<glm::vec3> nc = colormapScalars(q->values.data, cmap, vizMin, vizMax);
    out.primitiveColors.reserve(nSpheres + cylinderEdgeIndex.size());
    out.primitiveColors1.reserve(nSpheres + cylinderEdgeIndex.size());
    for (size_t ni : sphereNodeIndices) {
      const glm::vec3 c = ni < nc.size() ? nc[ni] : fallback;
      out.primitiveColors.push_back(c);
      out.primitiveColors1.push_back(c);
    }
    for (size_t ci = 0; ci < cylinderEdgeIndex.size(); ++ci) {
      const size_t ei = cylinderEdgeIndex[ci];
      const uint32_t t = edgeTails[ei];
      const uint32_t p = edgeTips[ei];
      out.primitiveColors.push_back(t < nc.size() ? nc[t] : fallback);
      out.primitiveColors1.push_back(p < nc.size() ? nc[p] : fallback);
    }
  } else if (auto* q = dynamic_cast<polyscope::CurveNetworkEdgeScalarQuantity*>(network.dominantQuantity)) {
    q->values.ensureHostBufferPopulated();
    const auto [vizMin, vizMax] = q->getMapRange();
    const auto& cmap = polyscope::render::engine->getColorMap(q->getColorMap());
    const auto& ev = q->values.data;
    const double range = (vizMax > vizMin) ? (vizMax - vizMin) : 1.0;
    std::vector<float> nodeScalarSum(nNodes, 0.0f);
    std::vector<int> nodeScalarCnt(nNodes, 0);
    for (size_t i = 0; i < edgeCount; ++i) {
      const uint32_t t = edgeTails[i];
      const uint32_t p = edgeTips[i];
      if (t >= nNodes || p >= nNodes || i >= ev.size()) continue;
      nodeScalarSum[t] += ev[i];
      nodeScalarCnt[t]++;
      nodeScalarSum[p] += ev[i];
      nodeScalarCnt[p]++;
    }
    out.primitiveColors.reserve(nSpheres + cylinderEdgeIndex.size());
    out.primitiveColors1.reserve(nSpheres + cylinderEdgeIndex.size());
    for (size_t ni : sphereNodeIndices) {
      glm::vec3 col = fallback;
      if (nodeScalarCnt[ni] > 0) {
        const double avg = nodeScalarSum[ni] / float(nodeScalarCnt[ni]);
        col = cmap.getValue((avg - vizMin) / range);
      }
      out.primitiveColors.push_back(col);
      out.primitiveColors1.push_back(col);
    }
    for (size_t ci = 0; ci < cylinderEdgeIndex.size(); ++ci) {
      const size_t ei = cylinderEdgeIndex[ci];
      const glm::vec3 col = (ei < ev.size()) ? cmap.getValue((ev[ei] - vizMin) / range) : fallback;
      out.primitiveColors.push_back(col);
      out.primitiveColors1.push_back(col);
    }
  }

  return out;
}

rt::RTMesh makeMeshFromVolumeMesh(polyscope::VolumeMesh& mesh) {
  rt::RTMesh out;
  out.name = mesh.getName();
  out.transform = mesh.getTransform();
  out.baseColorFactor = glm::vec4(mesh.getColor(), 1.0f);
  out.opacity = mesh.getTransparency();
  out.vertices = mesh.vertexPositions.data;
  applyMaterialPreset(out, mesh.getMaterial());

  const auto& triIndices = mesh.triangleVertexInds.data;
  const auto& triFaceIndices = mesh.triangleFaceInds.data;
  if (triIndices.size() % 3 != 0 || triFaceIndices.size() != triIndices.size()) {
    throw std::runtime_error("Volume mesh triangulation buffers are malformed.");
  }

  out.indices.reserve(triIndices.size() / 3);
  for (size_t i = 0; i < triIndices.size(); i += 3) {
    const uint32_t faceIndex = triFaceIndices[i];
    if (faceIndex < mesh.faceIsInterior.size() && mesh.faceIsInterior[faceIndex]) continue;
    out.indices.emplace_back(triIndices[i + 0], triIndices[i + 1], triIndices[i + 2]);
  }
  return out;
}

} // namespace snapshot_detail
