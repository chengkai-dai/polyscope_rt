#include <filesystem>
#include <iostream>
#include <vector>

#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/marching_triangles.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "polyscope/render/color_maps.h"
#include "polyscope/render/engine.h"
#include "polyscope/rt/curve_network.h"
#include "polyscope/rt/point_cloud.h"
#include "polyscope/rt/polyscope.h"
#include "polyscope/rt/surface_mesh.h"

namespace gc = geometrycentral;
namespace gcs = geometrycentral::surface;
namespace ps = polyscope::rt; // ← fallback to polyscope:: for non-RT
// namespace ps  = polyscope;

static void registerGeodesicColorMap() {
  static const std::vector<glm::vec3> kEntries = {
      {1.0000f, 1.0000f, 1.0000f}, {1.0000f, 1.0000f, 0.9365f},
      {1.0000f, 1.0000f, 0.8730f}, {1.0000f, 1.0000f, 0.8095f},
      {1.0000f, 1.0000f, 0.7460f}, {1.0000f, 1.0000f, 0.6825f},
      {1.0000f, 1.0000f, 0.6190f}, {1.0000f, 1.0000f, 0.5556f},
      {1.0000f, 1.0000f, 0.4921f}, {1.0000f, 1.0000f, 0.4286f},
      {1.0000f, 1.0000f, 0.3651f}, {1.0000f, 1.0000f, 0.3016f},
      {1.0000f, 1.0000f, 0.2381f}, {1.0000f, 1.0000f, 0.1746f},
      {1.0000f, 1.0000f, 0.1111f}, {1.0000f, 1.0000f, 0.0476f},
      {1.0000f, 0.9894f, 0.0000f}, {1.0000f, 0.9471f, 0.0000f},
      {1.0000f, 0.9048f, 0.0000f}, {1.0000f, 0.8624f, 0.0000f},
      {1.0000f, 0.8201f, 0.0000f}, {1.0000f, 0.7778f, 0.0000f},
      {1.0000f, 0.7354f, 0.0000f}, {1.0000f, 0.6931f, 0.0000f},
      {1.0000f, 0.6508f, 0.0000f}, {1.0000f, 0.6085f, 0.0000f},
      {1.0000f, 0.5661f, 0.0000f}, {1.0000f, 0.5238f, 0.0000f},
      {1.0000f, 0.4815f, 0.0000f}, {1.0000f, 0.4392f, 0.0000f},
      {1.0000f, 0.3968f, 0.0000f}, {1.0000f, 0.3545f, 0.0000f},
      {1.0000f, 0.3122f, 0.0000f}, {1.0000f, 0.2698f, 0.0000f},
      {1.0000f, 0.2275f, 0.0000f}, {1.0000f, 0.1852f, 0.0000f},
      {1.0000f, 0.1429f, 0.0000f}, {1.0000f, 0.1005f, 0.0000f},
      {1.0000f, 0.0582f, 0.0000f}, {1.0000f, 0.0159f, 0.0000f},
      {0.9735f, 0.0000f, 0.0000f}, {0.9312f, 0.0000f, 0.0000f},
      {0.8889f, 0.0000f, 0.0000f}, {0.8466f, 0.0000f, 0.0000f},
      {0.8042f, 0.0000f, 0.0000f}, {0.7619f, 0.0000f, 0.0000f},
      {0.7196f, 0.0000f, 0.0000f}, {0.6772f, 0.0000f, 0.0000f},
      {0.6349f, 0.0000f, 0.0000f}, {0.5926f, 0.0000f, 0.0000f},
      {0.5503f, 0.0000f, 0.0000f}, {0.5079f, 0.0000f, 0.0000f},
      {0.4656f, 0.0000f, 0.0000f}, {0.4233f, 0.0000f, 0.0000f},
      {0.3810f, 0.0000f, 0.0000f}, {0.3386f, 0.0000f, 0.0000f},
      {0.2963f, 0.0000f, 0.0000f}, {0.2540f, 0.0000f, 0.0000f},
      {0.2116f, 0.0000f, 0.0000f}, {0.1693f, 0.0000f, 0.0000f},
      {0.1270f, 0.0000f, 0.0000f}, {0.0847f, 0.0000f, 0.0000f},
      {0.0423f, 0.0000f, 0.0000f}, {0.0000f, 0.0000f, 0.0000f},
  };

  auto cmap = std::make_unique<polyscope::render::ValueColorMap>();
  cmap->name = "geodesic_hot_r";
  cmap->values = kEntries;
  polyscope::render::engine->colorMaps.emplace_back(std::move(cmap));
}

int main(int argc, char **argv) {
  // ── Load mesh ─────────────────────────────────────────────────────────────
  std::filesystem::path meshPath =
      argc > 1 ? argv[1]
               : std::filesystem::path(POLYSCOPE_RT_EXAMPLE_DEFAULT_MESH);

  auto [mesh, geometry] = gcs::readSurfaceMesh(meshPath.string());
  std::cout << "Loaded: " << mesh->nVertices() << " vertices, "
            << mesh->nFaces() << " faces\n";

  gcs::VertexData<double> distToSource =
      gcs::heatMethodDistance(*geometry, mesh->vertex(0));

  double maxDist = 0.0;
  std::vector<float> distValues(mesh->nVertices());
  for (gcs::Vertex v : mesh->vertices()) {
    double d = distToSource[v];
    maxDist = std::max(maxDist, d);
    distValues[v.getIndex()] = float(d);
  }

  constexpr int kIsoLevels = 24;
  constexpr int kPolyStride = 2;

  std::vector<gc::Vector3> isoNodes;
  std::vector<std::array<size_t, 2>> isoEdges;

  for (int k = 1; k <= kIsoLevels; ++k) {
    double iso = maxDist * k / double(kIsoLevels + 1);
    for (const auto &polyline :
         gcs::marchingTriangles(*geometry, distToSource, iso)) {
      if (polyline.size() < 2)
        continue;

      std::vector<gc::Vector3> pts;
      for (size_t i = 0; i < polyline.size(); i += size_t(kPolyStride))
        pts.push_back(polyline[i].interpolate(geometry->inputVertexPositions));
      if (pts.size() < 2)
        continue;

      // Close loops
      if (polyline.front() == polyline.back() &&
          gc::norm(pts.front() - pts.back()) > 1e-6)
        pts.push_back(pts.front());

      size_t base = isoNodes.size();
      for (const auto &p : pts)
        isoNodes.push_back(p);
      for (size_t i = 0; i + 1 < pts.size(); ++i)
        isoEdges.push_back({base + i, base + i + 1});
    }
  }

  ps::init();
  registerGeodesicColorMap();

  // mesh
  auto *psMesh = ps::registerSurfaceMesh(
      "bunny", geometry->inputVertexPositions, mesh->getFaceVertexList());
  // Contour-style isolines on the geodesic scalar — 20 thin lines across the
  // full range, mirroring Polyscope's CONTOUR_VALUECOLOR.  Switch to
  // IsolineStyle::Stripe for flat periodic bands instead.
  psMesh->addVertexScalarQuantity("geodesic distance", distValues)
      ->setColorMap("geodesic_hot_r")
      ->setMapRange({0.0f, static_cast<float>(maxDist)})
      ->setIsolinesEnabled(true)
      ->setIsolineStyle(ps::IsolineStyle::Contour)
      ->setIsolinePeriod(maxDist / 20.0, /*isRelative=*/false)
      ->setIsolineDarkness(0.8)
      ->setIsolineContourThickness(0.4)
      ->setEnabled(true);

  // curve
  auto *psCurve =
      ps::registerCurveNetwork("geodesic isocontours", isoNodes, isoEdges)
          ->setColor({1.0f, 1.0f, 1.0f})
          ->setRadius(0.0002f, /*isRelative=*/false)
          ->setEnabled(false);

  constexpr int kStep = 20;
  std::vector<glm::vec3> sampledPositions;
  std::vector<float> sampledDist;
  for (gcs::Vertex v : mesh->vertices()) {
    if (static_cast<int>(v.getIndex()) % kStep != 0)
      continue;
    auto p = geometry->inputVertexPositions[v];
    sampledPositions.push_back({float(p.x), float(p.y), float(p.z)});
    sampledDist.push_back(distValues[v.getIndex()]);
  }

  auto *psPoints = ps::registerPointCloud("surface samples", sampledPositions);
  psPoints->addScalarQuantity("geodesic distance", sampledDist)
      ->setColorMap("geodesic_hot_r")
      ->setMapRange({0.0f, static_cast<float>(maxDist)})
      ->setEnabled(true);
  psPoints->setPointRadius(0.0015f, /*isRelative=*/false);

  // vector field
  geometry->requireFaceNormals();
  geometry->requireFaceAreas();

  constexpr int kVFStep = 50;
  std::vector<glm::vec3> faceGradVecs(mesh->nFaces(), glm::vec3(0.0f));

  for (gcs::Face f : mesh->faces()) {
    if (static_cast<int>(f.getIndex()) % kVFStep != 0)
      continue;

    const gc::Vector3 N = geometry->faceNormals[f];
    const double A = geometry->faceAreas[f];
    if (A < 1e-12)
      continue;

    gc::Vector3 grad = gc::Vector3::zero();
    for (gcs::Halfedge he : f.adjacentHalfedges()) {
      // Edge opposite to he.vertex() runs from he.next().vertex() to
      // he.next().next().vertex()
      gc::Vector3 eOpp =
          geometry->inputVertexPositions[he.next().next().vertex()] -
          geometry->inputVertexPositions[he.next().vertex()];
      grad += distToSource[he.vertex()] * gc::cross(N, eOpp);
    }
    grad /= (2.0 * A);
    faceGradVecs[f.getIndex()] = {float(grad.x), float(grad.y), float(grad.z)};
  }

  auto *vfQty = psMesh->addFaceVectorQuantity("geodesic flow", faceGradVecs);
  vfQty->setVectorColor({0.2f, 0.8f, 1.0f})
      ->setVectorRadius(0.0004, /*isRelative=*/false)
      ->setVectorLengthScale(0.015, /*isRelative=*/false)
      ->setEnabled(false);

  ps::show();
  return 0;
}
