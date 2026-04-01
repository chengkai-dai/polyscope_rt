#include "scene/polyscope_scene_snapshot.h"

#include "polyscope/polyscope.h"

#include "scene/polyscope_scene_snapshot_internal.h"

PolyscopeSceneSnapshot capturePolyscopeSceneSnapshot() {
  static const std::unordered_map<std::string, rt::MaterialOverride> emptyOverrides;
  static const std::vector<rt::RTPunctualLight> emptyLights;
  return capturePolyscopeSceneSnapshot(emptyOverrides, emptyLights);
}

PolyscopeSceneSnapshot
capturePolyscopeSceneSnapshot(const std::unordered_map<std::string, rt::MaterialOverride>& materialOverrides,
                              const std::vector<rt::RTPunctualLight>& apiLights) {
  PolyscopeSceneSnapshot snapshot;
  snapshot.scene.hash = snapshot_detail::kFnvOffset;

  for (auto& [typeName, structures] : polyscope::state::structures) {
    (void)typeName;
    for (auto& [name, structurePtr] : structures) {
      (void)name;
      polyscope::Structure* structure = structurePtr.get();
      if (structure == nullptr || !structure->isEnabled()) continue;

      if (auto* simpleMesh = dynamic_cast<polyscope::SimpleTriangleMesh*>(structure)) {
        rt::RTMesh mesh = snapshot_detail::makeMeshFromSimpleTriangleMesh(*simpleMesh);
        snapshot_detail::applyMaterialPreset(mesh, simpleMesh->getMaterial());
        snapshot_detail::applyMaterialOverride(mesh, materialOverrides);
        if (mesh.vertices.empty() || mesh.indices.empty()) continue;
        snapshot_detail::addMeshAndHash(snapshot, std::move(mesh), *structure);
        continue;
      }

      if (auto* surfaceMesh = dynamic_cast<polyscope::SurfaceMesh*>(structure)) {
        if (surfaceMesh->triangleVertexInds.data.empty()) continue;
        rt::RTMesh mesh = snapshot_detail::makeMeshFromSurfaceMesh(*surfaceMesh);
        snapshot_detail::applyMaterialPreset(mesh, surfaceMesh->getMaterial());
        snapshot_detail::applyMaterialOverride(mesh, materialOverrides);
        if (mesh.vertices.empty() || mesh.indices.empty()) continue;
        if (surfaceMesh->getEdgeWidth() > 0.0) {
          mesh.wireframe = true;
          mesh.edgeColor = surfaceMesh->getEdgeColor();
          mesh.edgeWidth = static_cast<float>(surfaceMesh->getEdgeWidth());
        }
        snapshot_detail::addMeshAndHash(snapshot, std::move(mesh), *structure);

        auto vfields = snapshot_detail::makeRTVectorFields(*surfaceMesh);
        snapshot_detail::addVectorFieldsAndHash(snapshot, std::move(vfields), surfaceMesh->getName());
        continue;
      }

      if (auto* pointCloud = dynamic_cast<polyscope::PointCloud*>(structure)) {
        if (pointCloud->points.data.empty()) continue;
        rt::RTPointCloud pc = snapshot_detail::makeRTPointCloud(*pointCloud);
        snapshot_detail::addPointCloudAndHash(snapshot, std::move(pc), *pointCloud);
        continue;
      }

      if (auto* curveNetwork = dynamic_cast<polyscope::CurveNetwork*>(structure)) {
        rt::RTCurveNetwork curveNet = snapshot_detail::makeCurveNetwork(*curveNetwork);
        if (curveNet.primitives.empty()) continue;
        snapshot_detail::addCurveNetworkAndHash(snapshot, std::move(curveNet), *structure);
        continue;
      }

      if (auto* volumeMesh = dynamic_cast<polyscope::VolumeMesh*>(structure)) {
        if (volumeMesh->triangleVertexInds.data.empty()) continue;
        rt::RTMesh mesh = snapshot_detail::makeMeshFromVolumeMesh(*volumeMesh);
        snapshot_detail::applyMaterialOverride(mesh, materialOverrides);
        if (mesh.vertices.empty() || mesh.indices.empty()) continue;
        snapshot_detail::addMeshAndHash(snapshot, std::move(mesh), *structure);
      }
    }
  }

  snapshot_detail::addLightsAndHash(snapshot, apiLights);
  return snapshot;
}
