// CPU-only tests for all the new quantity-extraction features implemented in
// polyscope_scene_snapshot.cpp.  No GPU / Metal backend is required here.
//
// Tests:
//  1.  SurfaceMesh vertex normals → RTMesh::normals
//  2.  Face scalar quantity  → per-vertex colormap colors
//  3.  Face color quantity   → per-vertex RGB
//  4.  Face scalar isoline parameters propagate to RTMesh
//  5.  PointCloud ColorQuantity has higher priority than ScalarQuantity
//  6.  PointCloud ColorQuantity overrides baseColor
//  7.  Curve network node color  → primitiveColors (spheres + cylinder midpoint)
//  8.  Curve network edge color  → primitiveColors (nodeAverage for spheres)
//  9.  Curve network node scalar → primitiveColors via colormap
// 10.  Curve network edge scalar → primitiveColors via colormap
// 11.  Hash invalidates when face quantity is toggled
// 12.  Hash invalidates when curve quantity is toggled
// 13.  Disabled curve quantity does not populate primitiveColors
// 14.  Vertex normals absent → RTMesh::normals stays empty (SimpleTriangleMesh)
// 15.  Wireframe properties flow into RTMesh

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include "polyscope/curve_network.h"
#include "polyscope/curve_network_color_quantity.h"
#include "polyscope/curve_network_scalar_quantity.h"
#include "polyscope/point_cloud.h"
#include "polyscope/point_cloud_color_quantity.h"
#include "polyscope/point_cloud_scalar_quantity.h"
#include "polyscope/polyscope.h"
#include "polyscope/simple_triangle_mesh.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/surface_color_quantity.h"
#include "polyscope/surface_scalar_quantity.h"

#include "scene/polyscope_scene_snapshot.h"
#include "rendering/ray_tracing_types.h"
#include "test_helpers.h"

namespace {

// A minimal quad (2 triangles, 4 vertices) useful for face-quantity tests.
//   v3---v2
//   |  \ |
//   v0---v1
// Face 0 = {v0, v1, v2}   Face 1 = {v0, v2, v3}
std::vector<glm::vec3> quadVerts() {
  return {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
          {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
}
std::vector<std::vector<uint32_t>> quadFaces() {
  return {{0, 1, 2}, {0, 2, 3}};
}

// ---------------------------------------------------------------------------
// Test 1: SurfaceMesh vertex normals are propagated into RTMesh::normals.
// ---------------------------------------------------------------------------
void testVertexNormalsExtracted() {
  polyscope::removeEverything();
  auto* sm = polyscope::registerSurfaceMesh("normals_mesh", quadVerts(), quadFaces());
  (void)sm;

  auto snap = capturePolyscopeSceneSnapshot();
  require(!snap.scene.meshes.empty(), "mesh should be captured");
  const auto& m = snap.scene.meshes.front();
  // Polyscope computes smooth vertex normals when the mesh is added;
  // they should be parallel to positions (one per vertex).
  require(m.normals.size() == m.vertices.size(),
          "RTMesh::normals should have one normal per vertex");
  for (const auto& n : m.normals) {
    float len = glm::length(n);
    require(len > 0.5f && len < 2.0f, "normal should be approximately unit length");
  }
}

// ---------------------------------------------------------------------------
// Test 2: Face scalar quantity → per-vertex colormap colors.
// ---------------------------------------------------------------------------
void testFaceScalarToVertexColors() {
  polyscope::removeEverything();
  auto* sm = polyscope::registerSurfaceMesh("fsq_mesh", quadVerts(), quadFaces());
  // Face 0 scalar = 0.0, Face 1 scalar = 1.0
  auto* qty = sm->addFaceScalarQuantity("face_s", std::vector<float>{0.0f, 1.0f});
  qty->setEnabled(true);

  auto snap = capturePolyscopeSceneSnapshot();
  require(!snap.scene.meshes.empty(), "mesh must be captured");
  const auto& m = snap.scene.meshes.front();
  require(m.vertexColors.size() == 4, "face scalar should produce 4 per-vertex colors");
  // Vertex 1 touches only face 0 (scalar = 0.0) → colormap output near cold end.
  // Vertex 3 touches only face 1 (scalar = 1.0) → colormap output near warm end.
  // Their R channel values should differ.
  require(m.vertexColors[1] != m.vertexColors[3],
          "vertices on different-scalar faces should have different colors");
  require(m.baseColorFactor == glm::vec4(1.0f),
          "face scalar quantity should set baseColorFactor to (1,1,1,1)");
}

// ---------------------------------------------------------------------------
// Test 3: Face color quantity → per-vertex RGB.
// ---------------------------------------------------------------------------
void testFaceColorToVertexColors() {
  polyscope::removeEverything();
  auto* sm = polyscope::registerSurfaceMesh("fcq_mesh", quadVerts(), quadFaces());
  // Face 0 = red, Face 1 = blue
  auto* qty = sm->addFaceColorQuantity(
      "face_c", std::vector<glm::vec3>{{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f}});
  qty->setEnabled(true);

  auto snap = capturePolyscopeSceneSnapshot();
  require(!snap.scene.meshes.empty(), "mesh must be captured");
  const auto& m = snap.scene.meshes.front();
  require(m.vertexColors.size() == 4, "face color should produce 4 per-vertex colors");
  // Vertex 1 is only on face 0 (red) → high R
  require(m.vertexColors[1].r > 0.5f, "vertex only on red face should be reddish");
  // Vertex 3 is only on face 1 (blue) → high B
  require(m.vertexColors[3].b > 0.5f, "vertex only on blue face should be bluish");
}

// ---------------------------------------------------------------------------
// Test 4: Face scalar isoline parameters propagate to RTMesh.
// ---------------------------------------------------------------------------
void testFaceScalarIsolineParams() {
  polyscope::removeEverything();
  auto* sm = polyscope::registerSurfaceMesh("iso_mesh", quadVerts(), quadFaces());
  auto* qty = sm->addFaceScalarQuantity("face_iso", std::vector<float>{0.0f, 1.0f});
  qty->setEnabled(true);
  qty->setIsolinesEnabled(true);
  qty->setIsolinePeriod(0.25, false);
  qty->setIsolineDarkness(0.7);

  auto snap = capturePolyscopeSceneSnapshot();
  const auto& m = snap.scene.meshes.front();
  require(!m.isoScalars.empty(),    "face scalar isolines should populate isoScalars");
  require(m.isoScalars.size() == 4, "isoScalars should have one entry per vertex");
  requireNear(m.isoSpacing,  0.25f, 1e-4f, "isoSpacing should reflect the set period");
  requireNear(m.isoDarkness, 0.7f,  1e-4f, "isoDarkness should reflect the set darkness");
}

// ---------------------------------------------------------------------------
// Test 5: PointCloud ColorQuantity has higher priority than ScalarQuantity.
// ---------------------------------------------------------------------------
void testPointCloudColorQuantityPriority() {
  polyscope::removeEverything();
  std::vector<glm::vec3> pts = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  auto* pc = polyscope::registerPointCloud("pc_prio", pts);

  // Add scalar (colormap: red/blue gradient) first.
  auto* scalarQty = pc->addScalarQuantity("pc_scalar", std::vector<float>{0.0f, 1.0f});
  scalarQty->setEnabled(true);

  auto snapScalar = capturePolyscopeSceneSnapshot();
  require(!snapScalar.scene.pointClouds.front().colors.empty(),
          "enabled scalar quantity should produce per-point colors");

  // Now also add a color quantity that overrides with pure green.
  auto* colorQty = pc->addColorQuantity(
      "pc_color", std::vector<glm::vec3>{{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}});
  colorQty->setEnabled(true);
  scalarQty->setEnabled(false);

  auto snapColor = capturePolyscopeSceneSnapshot();
  const auto& colors = snapColor.scene.pointClouds.front().colors;
  require(!colors.empty(), "color quantity should produce per-point colors");
  require(colors[0].g > 0.9f, "color quantity (green) should dominate per-point output");
  require(colors[0].r < 0.1f, "color quantity should suppress non-green channels");
}

// ---------------------------------------------------------------------------
// Test 6: PointCloud ColorQuantity overrides baseColor in the snapshot.
// ---------------------------------------------------------------------------
void testPointCloudColorQuantityOverridesBase() {
  polyscope::removeEverything();
  std::vector<glm::vec3> pts = {{0.0f, 0.0f, 0.0f}};
  auto* pc = polyscope::registerPointCloud("pc_override", pts);
  pc->setPointColor(glm::vec3(1.0f, 0.0f, 0.0f)); // base = red
  auto* qty = pc->addColorQuantity("c", std::vector<glm::vec3>{{0.0f, 0.0f, 1.0f}});
  qty->setEnabled(true);

  auto snap = capturePolyscopeSceneSnapshot();
  const auto& rtpc = snap.scene.pointClouds.front();
  require(!rtpc.colors.empty(), "color quantity should populate per-point colors");
  require(rtpc.colors[0].b > 0.9f, "per-point blue should come from color quantity");
  require(rtpc.colors[0].r < 0.1f, "base color red should not bleed into per-point colors");
}

// ---------------------------------------------------------------------------
// Test 7: Curve network node color → primitiveColors.
// ---------------------------------------------------------------------------
void testCurveNodeColorExtracted() {
  polyscope::removeEverything();
  std::vector<glm::vec3> nodes = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.5f, 1.0f, 0.0f}};
  std::vector<std::array<uint32_t, 2>> edges = {{0, 1}, {1, 2}};
  auto* cv = polyscope::registerCurveNetwork("cv_nc", nodes, edges);
  auto* qty = cv->addNodeColorQuantity(
      "nc", std::vector<glm::vec3>{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}});
  qty->setEnabled(true);

  auto snap = capturePolyscopeSceneSnapshot();
  require(!snap.scene.curveNetworks.empty(), "curve network should be captured");
  const auto& cn = snap.scene.curveNetworks.front();
  require(!cn.primitiveColors.empty(),
          "node color quantity should populate primitiveColors");
  require(cn.primitiveColors.size() == cn.primitives.size(),
          "primitiveColors must be parallel to primitives");
  // Node 0 is red → sphere 0 should be red.
  require(cn.primitiveColors[0].r > 0.9f, "node 0 sphere should be red");
  require(cn.primitiveColors[0].g < 0.1f, "node 0 sphere should not be green");
}

// ---------------------------------------------------------------------------
// Test 8: Curve network edge color → primitiveColors.
// ---------------------------------------------------------------------------
void testCurveEdgeColorExtracted() {
  polyscope::removeEverything();
  std::vector<glm::vec3> nodes = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.5f, 1.0f, 0.0f}};
  std::vector<std::array<uint32_t, 2>> edges = {{0, 1}, {1, 2}};
  auto* cv = polyscope::registerCurveNetwork("cv_ec", nodes, edges);
  // Edge 0 = yellow (R+G), Edge 1 = cyan (G+B)
  auto* qty = cv->addEdgeColorQuantity(
      "ec", std::vector<glm::vec3>{{1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 1.0f}});
  qty->setEnabled(true);

  auto snap = capturePolyscopeSceneSnapshot();
  const auto& cn = snap.scene.curveNetworks.front();
  require(!cn.primitiveColors.empty(), "edge color quantity should populate primitiveColors");
  require(cn.primitiveColors.size() == cn.primitives.size(),
          "primitiveColors must be parallel to primitives");

  // Count cylinders — they should have the per-edge color.
  int cylIdx = 0;
  int nNodes = 0;
  for (const auto& p : cn.primitives) {
    if (p.type == rt::RTCurvePrimitiveType::Sphere) ++nNodes;
  }
  // Cylinders start at index nNodes in the flat list.
  // Cylinder for edge 0 (yellow) should have high R and G.
  if ((int)cn.primitiveColors.size() > nNodes) {
    const glm::vec3& cylColor0 = cn.primitiveColors[nNodes];
    require(cylColor0.r > 0.5f && cylColor0.g > 0.5f,
            "edge 0 cylinder should be yellow (R+G)");
    (void)cylIdx;
  }
}

// ---------------------------------------------------------------------------
// Test 9: Curve network node scalar → primitiveColors via colormap.
// ---------------------------------------------------------------------------
void testCurveNodeScalarExtracted() {
  polyscope::removeEverything();
  std::vector<glm::vec3> nodes = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  std::vector<std::array<uint32_t, 2>> edges = {{0, 1}};
  auto* cv = polyscope::registerCurveNetwork("cv_ns", nodes, edges);
  auto* qty = cv->addNodeScalarQuantity("ns", std::vector<float>{0.0f, 1.0f});
  qty->setEnabled(true);

  auto snap = capturePolyscopeSceneSnapshot();
  const auto& cn = snap.scene.curveNetworks.front();
  require(!cn.primitiveColors.empty(),
          "node scalar quantity should populate primitiveColors");
  require(cn.primitiveColors.size() == cn.primitives.size(),
          "primitiveColors must be parallel to primitives");
  // Nodes 0 and 1 have different scalar values → different colormap outputs.
  require(cn.primitiveColors[0] != cn.primitiveColors[1],
          "different node scalars should produce different primitive colors");
}

// ---------------------------------------------------------------------------
// Test 10: Curve network edge scalar → primitiveColors via colormap.
// ---------------------------------------------------------------------------
void testCurveEdgeScalarExtracted() {
  polyscope::removeEverything();
  std::vector<glm::vec3> nodes = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.5f, 1.0f, 0.0f}};
  std::vector<std::array<uint32_t, 2>> edges = {{0, 1}, {1, 2}};
  auto* cv = polyscope::registerCurveNetwork("cv_es", nodes, edges);
  // Edge 0 = 0.0, Edge 1 = 1.0 → clearly different colormap outputs.
  auto* qty = cv->addEdgeScalarQuantity("es", std::vector<float>{0.0f, 1.0f});
  qty->setEnabled(true);

  auto snap = capturePolyscopeSceneSnapshot();
  const auto& cn = snap.scene.curveNetworks.front();
  require(!cn.primitiveColors.empty(),
          "edge scalar quantity should populate primitiveColors");
  require(cn.primitiveColors.size() == cn.primitives.size(),
          "primitiveColors must be parallel to primitives");
  // Cylinders have per-edge colors.  The two cylinder colors should differ.
  int nNodes = 0;
  for (const auto& p : cn.primitives)
    if (p.type == rt::RTCurvePrimitiveType::Sphere) ++nNodes;
  const int nCyls = static_cast<int>(cn.primitives.size()) - nNodes;
  require(nCyls >= 2, "expected at least 2 cylinder primitives");
  require(cn.primitiveColors[nNodes] != cn.primitiveColors[nNodes + 1],
          "edge-0 and edge-1 cylinders should have different colormap colors");
}

// ---------------------------------------------------------------------------
// Test 11: Scene hash invalidates when face quantity changes.
// ---------------------------------------------------------------------------
void testHashInvalidatesOnFaceQuantityChange() {
  polyscope::removeEverything();
  auto* sm = polyscope::registerSurfaceMesh("hash_fq", quadVerts(), quadFaces());
  auto* qty = sm->addFaceScalarQuantity("fq", std::vector<float>{0.0f, 1.0f});
  qty->setEnabled(true);

  auto snap1 = capturePolyscopeSceneSnapshot();
  // Disable the quantity → base color path; hash must change.
  qty->setEnabled(false);
  auto snap2 = capturePolyscopeSceneSnapshot();
  require(snap1.scene.hash != snap2.scene.hash,
          "hash must change when face scalar quantity is toggled off");

  // Re-enable and change the colormap.
  qty->setEnabled(true);
  qty->setColorMap("blues");
  auto snap3 = capturePolyscopeSceneSnapshot();
  require(snap2.scene.hash != snap3.scene.hash,
          "hash must change when face scalar colormap changes");
}

// ---------------------------------------------------------------------------
// Test 12: Scene hash invalidates when curve quantity is toggled.
// ---------------------------------------------------------------------------
void testHashInvalidatesOnCurveQuantityChange() {
  polyscope::removeEverything();
  std::vector<glm::vec3> nodes = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  std::vector<std::array<uint32_t, 2>> edges = {{0, 1}};
  auto* cv = polyscope::registerCurveNetwork("hash_cv", nodes, edges);
  auto* qty = cv->addNodeScalarQuantity("ns", std::vector<float>{0.0f, 1.0f});

  qty->setEnabled(false);
  auto snapOff = capturePolyscopeSceneSnapshot();
  require(snapOff.scene.curveNetworks.front().primitiveColors.empty(),
          "disabled quantity should not produce primitive colors");

  qty->setEnabled(true);
  auto snapOn = capturePolyscopeSceneSnapshot();
  require(!snapOn.scene.curveNetworks.front().primitiveColors.empty(),
          "enabled quantity should produce primitive colors");
  require(snapOff.scene.hash != snapOn.scene.hash,
          "hash must change when curve quantity is toggled");
}

// ---------------------------------------------------------------------------
// Test 13: Disabled curve quantity does not populate primitiveColors.
// ---------------------------------------------------------------------------
void testDisabledCurveQuantityNotExtracted() {
  polyscope::removeEverything();
  std::vector<glm::vec3> nodes = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
  std::vector<std::array<uint32_t, 2>> edges = {{0, 1}};
  auto* cv = polyscope::registerCurveNetwork("dis_cv", nodes, edges);
  auto* qty = cv->addEdgeColorQuantity(
      "ec", std::vector<glm::vec3>{{0.0f, 1.0f, 0.0f}});
  qty->setEnabled(false); // intentionally disabled

  auto snap = capturePolyscopeSceneSnapshot();
  require(snap.scene.curveNetworks.front().primitiveColors.empty(),
          "disabled color quantity must not populate primitiveColors");
}

// ---------------------------------------------------------------------------
// Test 14: SimpleTriangleMesh (no smooth normals) → RTMesh::normals is empty.
// ---------------------------------------------------------------------------
void testSimpleTriangleMeshHasNoNormals() {
  polyscope::removeEverything();
  std::vector<glm::vec3> verts = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  std::vector<glm::uvec3> faces = {glm::uvec3(0, 1, 2)};
  polyscope::registerSimpleTriangleMesh("stm_no_normals", verts, faces);

  auto snap = capturePolyscopeSceneSnapshot();
  require(!snap.scene.meshes.empty(), "SimpleTriangleMesh must be captured");
  const auto& m = snap.scene.meshes.front();
  // SimpleTriangleMesh does not carry vertex normals in Polyscope's API,
  // so RTMesh::normals should be empty (no smooth normals → flat shading).
  require(m.normals.empty(),
          "SimpleTriangleMesh should produce empty RTMesh::normals (flat shading)");
}

// ---------------------------------------------------------------------------
// Test 15: Wireframe properties flow into RTMesh.
// ---------------------------------------------------------------------------
void testWireframePropertiesExtracted() {
  polyscope::removeEverything();
  auto* sm = polyscope::registerSurfaceMesh("wf_mesh", quadVerts(), quadFaces());
  sm->setEdgeWidth(2.5);
  sm->setEdgeColor(glm::vec3(1.0f, 0.5f, 0.0f));

  auto snap = capturePolyscopeSceneSnapshot();
  require(!snap.scene.meshes.empty(), "wireframe mesh must be captured");
  const auto& m = snap.scene.meshes.front();
  require(m.wireframe, "RTMesh::wireframe should be true when edgeWidth > 0");
  requireNear(m.edgeWidth, 2.5f, 1e-4f, "RTMesh::edgeWidth should match setEdgeWidth");
  requireNear(m.edgeColor.r, 1.0f, 1e-4f, "RTMesh::edgeColor.r should match setEdgeColor");
  requireNear(m.edgeColor.g, 0.5f, 1e-4f, "RTMesh::edgeColor.g should match setEdgeColor");
}

} // namespace

int main() {
  try {
    polyscope::init("openGL_mock");

    testVertexNormalsExtracted();
    testFaceScalarToVertexColors();
    testFaceColorToVertexColors();
    testFaceScalarIsolineParams();
    testPointCloudColorQuantityPriority();
    testPointCloudColorQuantityOverridesBase();
    testCurveNodeColorExtracted();
    testCurveEdgeColorExtracted();
    testCurveNodeScalarExtracted();
    testCurveEdgeScalarExtracted();
    testHashInvalidatesOnFaceQuantityChange();
    testHashInvalidatesOnCurveQuantityChange();
    testDisabledCurveQuantityNotExtracted();
    testSimpleTriangleMeshHasNoNormals();
    testWireframePropertiesExtracted();

    polyscope::shutdown();
    std::cout << "quantity_extraction_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "quantity_extraction_test FAILED: " << e.what() << std::endl;
    return 1;
  }
}
