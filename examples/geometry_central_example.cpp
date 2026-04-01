#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <limits>
#include <vector>

#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/marching_triangles.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "glm/glm.hpp"
#include "polyscope/rt/curve_network.h"
#include "polyscope/rt/material_library.h"
#include "polyscope/rt/point_cloud.h"
#include "polyscope/rt/polyscope.h"
#include "polyscope/rt/surface_mesh.h"

namespace gc = geometrycentral;
namespace gcs = geometrycentral::surface;
namespace ps = polyscope::rt;

namespace {

struct NormalizeTransform {
  gc::Vector3 center = gc::Vector3::zero();
  double scale = 1.0;
};

NormalizeTransform computeNormalizeTransform(gcs::SurfaceMesh& mesh, gcs::VertexPositionGeometry& geometry) {
  gc::Vector3 minP{std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity(),
                   std::numeric_limits<double>::infinity()};
  gc::Vector3 maxP{-std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity(),
                   -std::numeric_limits<double>::infinity()};

  for (gcs::Vertex v : mesh.vertices()) {
    const gc::Vector3 p = geometry.inputVertexPositions[v];
    minP.x = std::min(minP.x, p.x);
    minP.y = std::min(minP.y, p.y);
    minP.z = std::min(minP.z, p.z);
    maxP.x = std::max(maxP.x, p.x);
    maxP.y = std::max(maxP.y, p.y);
    maxP.z = std::max(maxP.z, p.z);
  }

  NormalizeTransform xf;
  xf.center = 0.5 * (minP + maxP);
  const gc::Vector3 halfExtent = 0.5 * (maxP - minP);
  const double maxHalfExtent = std::max({halfExtent.x, halfExtent.y, halfExtent.z, 1e-6});
  xf.scale = 0.55 / maxHalfExtent;
  return xf;
}

glm::vec3 transformPoint(const gc::Vector3& p, const NormalizeTransform& xf,
                         const glm::vec3& offset = glm::vec3(0.0f)) {
  const gc::Vector3 local = (p - xf.center) * xf.scale;
  return offset + glm::vec3(static_cast<float>(local.x), static_cast<float>(local.y),
                            static_cast<float>(local.z));
}

glm::vec3 transformVector(const gc::Vector3& v, const NormalizeTransform& xf) {
  const gc::Vector3 scaled = v * xf.scale;
  return glm::vec3(static_cast<float>(scaled.x), static_cast<float>(scaled.y),
                   static_cast<float>(scaled.z));
}

std::vector<glm::vec3> offsetPositions(const std::vector<glm::vec3>& basePositions,
                                       const glm::vec3& offset) {
  std::vector<glm::vec3> positions;
  positions.reserve(basePositions.size());
  for (const glm::vec3& p : basePositions) positions.push_back(p + offset);
  return positions;
}

gcs::Vertex chooseShowcaseSource(gcs::SurfaceMesh& mesh, const std::vector<glm::vec3>& normalizedPositions) {
  const glm::vec3 target{-0.18f, 0.22f, 0.30f};
  float bestDistance2 = std::numeric_limits<float>::max();
  gcs::Vertex best = mesh.vertex(0);

  for (gcs::Vertex v : mesh.vertices()) {
    const glm::vec3 delta = normalizedPositions[v.getIndex()] - target;
    const float dist2 = glm::dot(delta, delta);
    if (dist2 < bestDistance2) {
      bestDistance2 = dist2;
      best = v;
    }
  }

  return best;
}

void addSupportMesh(const std::string& name, const std::vector<glm::vec3>& positions,
                    const std::vector<std::vector<size_t>>& faces, const ps::MaterialPreset& preset,
                    const glm::vec4& baseColor) {
  ps::registerSurfaceMesh(name, positions, faces);
  ps::applyMaterial(name, preset);
  ps::setBaseColor(name, baseColor);
}

void addGeodesicColormap(const std::string& meshName, const std::vector<float>& distanceValues,
                         double maxDistance) {
  ps::getSurfaceMesh(meshName)
      ->addVertexScalarQuantity("geodesic distance", distanceValues)
      ->setColorMap("viridis")
      ->setMapRange({0.0f, static_cast<float>(maxDistance)})
      ->setEnabled(true);
}

void addSourceMarker(const std::string& name, const glm::vec3& position, const glm::vec3& color) {
  auto* marker = ps::registerPointCloud(name, std::vector<glm::vec3>{position});
  marker->setPointColor(color);
  marker->setPointRadius(0.03f, false);
}

} // namespace

int main(int argc, char** argv) {
  const std::filesystem::path meshPath =
      argc > 1 ? std::filesystem::path(argv[1])
               : std::filesystem::path(POLYSCOPE_RT_EXAMPLE_DEFAULT_MESH);

  auto [mesh, geometry] = gcs::readSurfaceMesh(meshPath.string());
  std::cout << "Loaded: " << mesh->nVertices() << " vertices, " << mesh->nFaces() << " faces\n";

  geometry->requireVertexNormals();
  geometry->requireFaceNormals();
  geometry->requireFaceAreas();

  const NormalizeTransform xf = computeNormalizeTransform(*mesh, *geometry);
  const auto faceList = mesh->getFaceVertexList();

  std::vector<glm::vec3> normalizedPositions(mesh->nVertices());
  std::vector<glm::vec3> normalizedVertexNormals(mesh->nVertices());
  for (gcs::Vertex v : mesh->vertices()) {
    normalizedPositions[v.getIndex()] = transformPoint(geometry->inputVertexPositions[v], xf);
    normalizedVertexNormals[v.getIndex()] =
        glm::normalize(transformVector(geometry->vertexNormals[v], NormalizeTransform{}));
  }

  const gcs::Vertex sourceVertex = chooseShowcaseSource(*mesh, normalizedPositions);
  const gcs::VertexData<double> geodesicDistance = gcs::heatMethodDistance(*geometry, sourceVertex);

  double maxDistance = 0.0;
  std::vector<float> distanceValues(mesh->nVertices(), 0.0f);
  for (gcs::Vertex v : mesh->vertices()) {
    const double distance = geodesicDistance[v];
    maxDistance = std::max(maxDistance, distance);
    distanceValues[v.getIndex()] = static_cast<float>(distance);
  }

  const int pointStride = std::max<int>(1, static_cast<int>(mesh->nVertices() / 320));
  std::vector<glm::vec3> sampledPositions;
  std::vector<float> sampledDistances;
  sampledPositions.reserve(mesh->nVertices() / pointStride + 1);
  sampledDistances.reserve(mesh->nVertices() / pointStride + 1);
  for (gcs::Vertex v : mesh->vertices()) {
    if (static_cast<int>(v.getIndex()) % pointStride != 0) continue;
    sampledPositions.push_back(normalizedPositions[v.getIndex()] + normalizedVertexNormals[v.getIndex()] * 0.018f);
    sampledDistances.push_back(distanceValues[v.getIndex()]);
  }

  constexpr int kContourLevels = 16;
  constexpr int kContourStride = 2;
  std::vector<glm::vec3> contourNodes;
  std::vector<std::array<size_t, 2>> contourEdges;
  for (int level = 1; level <= kContourLevels; ++level) {
    const double isoValue = maxDistance * static_cast<double>(level) / static_cast<double>(kContourLevels + 1);
    for (const auto& polyline : gcs::marchingTriangles(*geometry, geodesicDistance, isoValue)) {
      if (polyline.size() < 2) continue;

      std::vector<glm::vec3> points;
      for (size_t i = 0; i < polyline.size(); i += static_cast<size_t>(kContourStride)) {
        points.push_back(transformPoint(polyline[i].interpolate(geometry->inputVertexPositions), xf));
      }
      if (points.size() < 2) continue;

      if (polyline.front() == polyline.back() && glm::length(points.front() - points.back()) > 1e-6f) {
        points.push_back(points.front());
      }

      const size_t baseIndex = contourNodes.size();
      contourNodes.insert(contourNodes.end(), points.begin(), points.end());
      for (size_t i = 0; i + 1 < points.size(); ++i) {
        contourEdges.push_back({baseIndex + i, baseIndex + i + 1});
      }
    }
  }

  const int flowStride = std::max<int>(1, static_cast<int>(mesh->nFaces() / 220));
  std::vector<glm::vec3> faceGradientVectors(mesh->nFaces(), glm::vec3(0.0f));
  for (gcs::Face f : mesh->faces()) {
    if (static_cast<int>(f.getIndex()) % flowStride != 0) continue;

    const gc::Vector3 normal = geometry->faceNormals[f];
    const double area = geometry->faceAreas[f];
    if (area < 1e-12) continue;

    gc::Vector3 gradient = gc::Vector3::zero();
    for (gcs::Halfedge he : f.adjacentHalfedges()) {
      const gc::Vector3 oppositeEdge =
          geometry->inputVertexPositions[he.next().next().vertex()] -
          geometry->inputVertexPositions[he.next().vertex()];
      gradient += geodesicDistance[he.vertex()] * gc::cross(normal, oppositeEdge);
    }
    gradient /= (2.0 * area);
    faceGradientVectors[f.getIndex()] = transformVector(gradient, xf);
  }

  const glm::vec3 scalarOffset{-0.95f, 0.0f, -0.72f};
  const glm::vec3 contourOffset{0.95f, 0.0f, -0.72f};
  const glm::vec3 pointOffset{-0.95f, 0.0f, 0.82f};
  const glm::vec3 flowOffset{0.95f, 0.0f, 0.82f};
  const glm::vec3 sourceLocal = normalizedPositions[sourceVertex.getIndex()] +
                                normalizedVertexNormals[sourceVertex.getIndex()] * 0.03f;

  ps::options::programName = "Polyscope RT - Geometry Central Showcase";
  ps::init();

  ps::setBackgroundColor({0.96f, 0.97f, 0.99f});
  ps::setMainLight({-0.40f, -1.0f, 0.28f}, {1.0f, 0.98f, 0.95f}, 2.0f);
  ps::setMainLightAngularRadius(1.0f);
  ps::setEnvironment({1.0f, 1.0f, 1.0f}, 0.40f);
  ps::setAmbientFloor(0.08f);

  addSupportMesh("gc scalar surface", offsetPositions(normalizedPositions, scalarOffset), faceList,
                 ps::Ceramic(), {1.0f, 1.0f, 1.0f, 1.0f});
  addGeodesicColormap("gc scalar surface", distanceValues, maxDistance);
  addSourceMarker("gc scalar source", sourceLocal + scalarOffset, {1.0f, 0.92f, 0.35f});

  addSupportMesh("gc contour surface", offsetPositions(normalizedPositions, contourOffset), faceList,
                 ps::Ceramic(), {1.0f, 1.0f, 1.0f, 1.0f});
  addGeodesicColormap("gc contour surface", distanceValues, maxDistance);
  auto contourNodesOffset = offsetPositions(contourNodes, contourOffset);
  auto* contourNetwork = ps::registerCurveNetwork("gc geodesic contours", contourNodesOffset, contourEdges);
  contourNetwork->setColor({0.97f, 0.98f, 1.0f});
  contourNetwork->setRadius(0.008f, false);
  addSourceMarker("gc contour source", sourceLocal + contourOffset, {1.0f, 0.92f, 0.35f});

  addSupportMesh("gc sample surface", offsetPositions(normalizedPositions, pointOffset), faceList,
                 ps::Ceramic(), {1.0f, 1.0f, 1.0f, 1.0f});
  addGeodesicColormap("gc sample surface", distanceValues, maxDistance);
  auto sampledPositionsOffset = offsetPositions(sampledPositions, pointOffset);
  auto* pointCloud = ps::registerPointCloud("gc surface samples", sampledPositionsOffset);
  pointCloud->setPointRadius(0.022f, false);
  pointCloud->addScalarQuantity("geodesic distance", sampledDistances)
      ->setColorMap("viridis")
      ->setMapRange({0.0f, static_cast<float>(maxDistance)})
      ->setEnabled(true);
  addSourceMarker("gc sample source", sourceLocal + pointOffset, {1.0f, 0.92f, 0.35f});

  addSupportMesh("gc flow surface", offsetPositions(normalizedPositions, flowOffset), faceList,
                 ps::Ceramic(), {1.0f, 1.0f, 1.0f, 1.0f});
  addGeodesicColormap("gc flow surface", distanceValues, maxDistance);
  auto* flowQuantity =
      ps::getSurfaceMesh("gc flow surface")->addFaceVectorQuantity("geodesic flow", faceGradientVectors);
  flowQuantity->setVectorColor({0.12f, 0.82f, 1.0f})
      ->setVectorRadius(0.006f, false)
      ->setVectorLengthScale(0.16f, false)
      ->setEnabled(true);
  addSourceMarker("gc flow source", sourceLocal + flowOffset, {1.0f, 0.92f, 0.35f});

  ps::view::lookAt({0.0f, 1.35f, -4.8f}, {0.0f, -0.02f, 0.05f}, {0.0f, 1.0f, 0.0f});
  ps::view::fov = 48.0f;

  ps::show();
  return 0;
}
