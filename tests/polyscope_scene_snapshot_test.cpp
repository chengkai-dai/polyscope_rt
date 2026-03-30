#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>

#include "glm/gtc/matrix_transform.hpp"

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/simple_triangle_mesh.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"

#include "scene/polyscope_scene_snapshot.h"
#include "rendering/ray_tracing_types.h"
#include "test_helpers.h"

namespace {

} // namespace

int main() {
  try {
    polyscope::init("openGL_mock");
    polyscope::removeEverything();

    std::vector<glm::vec3> vertices{
        {-1.0f, 0.0f, -1.0f},
        {1.0f, 0.0f, -1.0f},
        {0.0f, 0.0f, 1.0f},
    };
    std::vector<glm::uvec3> faces{glm::uvec3(0, 1, 2)};

    auto* mesh = polyscope::registerSimpleTriangleMesh("snapshot_mesh", vertices, faces);
    mesh->setSurfaceColor(glm::vec3(0.2f, 0.4f, 0.8f));
    mesh->setTransform(glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 2.0f, 3.0f)));

    PolyscopeSceneSnapshot snapshot = capturePolyscopeSceneSnapshot();
    require(snapshot.supportedStructureCount == 1, "expected exactly one supported mesh");
    require(snapshot.scene.meshes.size() == 1, "expected exactly one extracted RT mesh");
    require(snapshot.hostStructure != nullptr, "expected a host structure");
    require(snapshot.hostName == "snapshot_mesh", "unexpected host structure name");
    require(snapshot.hostTypeName == polyscope::SimpleTriangleMesh::structureTypeName, "unexpected host structure type");

    const auto& rtMesh = snapshot.scene.meshes.front();
    require(rtMesh.vertices.size() == 3, "unexpected vertex count");
    require(rtMesh.indices.size() == 1, "unexpected face count");
    require(std::abs(rtMesh.baseColorFactor.x - 0.2f) < 1e-5f, "unexpected mesh albedo");
    require(std::abs(rtMesh.transform[3][0] - 1.0f) < 1e-5f, "unexpected transform x");
    require(std::abs(rtMesh.transform[3][1] - 2.0f) < 1e-5f, "unexpected transform y");
    require(std::abs(rtMesh.transform[3][2] - 3.0f) < 1e-5f, "unexpected transform z");

    mesh->setEnabled(false);
    PolyscopeSceneSnapshot disabledSnapshot = capturePolyscopeSceneSnapshot();
    require(disabledSnapshot.scene.meshes.empty(), "disabled meshes should not be exported");
    require(disabledSnapshot.supportedStructureCount == 0, "disabled meshes should not count as supported");

    polyscope::removeEverything();
    auto* pointCloud = polyscope::registerPointCloud("snapshot_points",
                                                     std::vector<glm::vec3>{{-0.5f, 0.0f, 0.0f}, {0.5f, 0.0f, 0.0f}});
    pointCloud->setPointColor(glm::vec3(0.9f, 0.3f, 0.1f));
    pointCloud->setPointRadius(0.15, false);

    auto* curveNetwork = polyscope::registerCurveNetwork(
        "snapshot_curve", std::vector<glm::vec3>{{0.0f, 0.0f, 0.0f}, {0.0f, 0.6f, 0.0f}, {0.5f, 1.0f, 0.0f}},
        std::vector<std::array<uint32_t, 2>>{{0, 1}, {1, 2}});
    curveNetwork->setColor(glm::vec3(0.1f, 0.7f, 0.4f));
    curveNetwork->setRadius(0.08f, false);

    std::vector<glm::vec3> volumeVertices{
        {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
        {-0.5f, -0.5f, 0.5f},  {0.5f, -0.5f, 0.5f},  {0.5f, 0.5f, 0.5f},  {-0.5f, 0.5f, 0.5f},
    };
    std::vector<std::array<uint32_t, 8>> volumeCells{{0, 1, 2, 3, 4, 5, 6, 7}};
    auto* volumeMesh = polyscope::registerVolumeMesh("snapshot_volume", volumeVertices, volumeCells);
    volumeMesh->setColor(glm::vec3(0.2f, 0.5f, 0.9f));

    PolyscopeSceneSnapshot mixedSnapshot = capturePolyscopeSceneSnapshot();
    require(mixedSnapshot.supportedStructureCount == 3, "expected point, curve, and volume structures to be exported");
    require(mixedSnapshot.scene.meshes.size() == 1, "expected one RT mesh (volume only; point cloud goes to pointClouds)");
    require(mixedSnapshot.scene.curveNetworks.size() == 1, "expected one curve network");
    require(mixedSnapshot.scene.pointClouds.size() == 1, "expected one RTPointCloud");

    const auto& rtpc = mixedSnapshot.scene.pointClouds.front();
    require(rtpc.name == "snapshot_points", "RTPointCloud name mismatch");
    require(rtpc.centers.size() == 2, "RTPointCloud should have 2 centers");
    requireNear(rtpc.radius, 0.15f, 1e-4f, "RTPointCloud radius mismatch");

    bool foundVolumeMesh = false;
    for (const auto& rtMesh2 : mixedSnapshot.scene.meshes) {
      if (rtMesh2.name == "snapshot_volume") {
        foundVolumeMesh = true;
        require(!rtMesh2.indices.empty(), "volume mesh should export its boundary triangles");
      }
    }
    require(foundVolumeMesh, "volume mesh export missing");

    const auto& curveNet = mixedSnapshot.scene.curveNetworks.front();
    require(curveNet.name == "snapshot_curve", "curve network name mismatch");
    require(!curveNet.primitives.empty(), "curve network should generate analytical primitives");
    require(std::abs(curveNet.baseColor.x - 0.1f) < 1e-5f, "unexpected curve network color");

    bool hasSpheres = false;
    bool hasCylinders = false;
    for (const auto& prim : curveNet.primitives) {
      if (prim.type == rt::RTCurvePrimitiveType::Sphere) hasSpheres = true;
      if (prim.type == rt::RTCurvePrimitiveType::Cylinder) hasCylinders = true;
      require(prim.radius > 0.0f, "curve primitive should have positive radius");
    }
    // Catmull-Rom curves are C1-continuous through every degree-2 node,
    // so junction spheres are NOT generated for interior nodes.
    // However, degree-1 endpoint nodes (and degree-≥3 branch nodes) still
    // get sphere primitives to cap the open tube ends.
    require(hasSpheres, "snapshot should generate sphere primitives for endpoint nodes (degree≠2)");
    require(hasCylinders, "curve network should contain cylinder primitives for edges");

    // -----------------------------------------------------------------------
    // Vector field extraction tests
    // -----------------------------------------------------------------------
    polyscope::removeEverything();

    // Register a tiny triangle mesh and add a vertex vector quantity.
    std::vector<glm::vec3> triVerts{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    std::vector<std::vector<uint32_t>> triFaces{{0, 1, 2}};
    auto* smesh = polyscope::registerSurfaceMesh("vf_mesh", triVerts, triFaces);

    std::vector<glm::vec3> vecs{{0.0f, 0.1f, 0.0f}, {0.0f, 0.2f, 0.0f}, {0.0f, 0.15f, 0.0f}};
    auto* vfQty = smesh->addVertexVectorQuantity("vf", vecs);
    vfQty->setEnabled(true);

    PolyscopeSceneSnapshot vfSnapshot = capturePolyscopeSceneSnapshot();
    require(!vfSnapshot.scene.vectorFields.empty(),
            "enabled vertex vector quantity should produce an RTVectorField");

    const rt::RTVectorField& vf = vfSnapshot.scene.vectorFields.front();
    require(vf.name == "vf", "RTVectorField name should match quantity name");
    require(vf.roots.size() == 3, "RTVectorField should have 3 roots (one per vertex)");
    require(vf.directions.size() == 3, "RTVectorField should have 3 directions");
    require(vf.radius > 0.0f, "RTVectorField radius should be positive");

    // Disabling the quantity should remove it from the snapshot.
    vfQty->setEnabled(false);
    PolyscopeSceneSnapshot noVfSnapshot = capturePolyscopeSceneSnapshot();
    require(noVfSnapshot.scene.vectorFields.empty(),
            "disabled vector quantity should NOT produce an RTVectorField");

    // Scene hash must change between enabled and disabled states.
    require(vfSnapshot.scene.hash != noVfSnapshot.scene.hash,
            "scene hash must differ when vector field is toggled on/off");

    // Re-enable and change vector color — hash must change again.
    vfQty->setEnabled(true);
    vfQty->setVectorColor(glm::vec3(1.0f, 0.0f, 0.0f));
    PolyscopeSceneSnapshot coloredVfSnapshot = capturePolyscopeSceneSnapshot();
    require(coloredVfSnapshot.scene.hash != vfSnapshot.scene.hash,
            "scene hash must change when vector field color changes");

    polyscope::shutdown();
    std::cout << "polyscope_scene_snapshot_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "polyscope_scene_snapshot_test failed: " << e.what() << std::endl;
    return 1;
  }
}
