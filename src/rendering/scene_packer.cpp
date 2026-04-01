#include "rendering/scene_packer.h"

#include "rendering/scene_packer_internal.h"

namespace rt {

PackedSceneData packScene(const RTScene& scene) {
  PackedSceneData packed;
  packed.acc.positions.reserve(4096);
  packed.acc.normals.reserve(4096);
  packed.acc.vertexColors.reserve(4096);
  packed.acc.texcoords.reserve(4096);
  packed.acc.accelIndices.reserve(4096);
  packed.acc.shaderTriangles.reserve(4096);
  packed.acc.materials.reserve(scene.meshes.size() + scene.curveNetworks.size() + scene.vectorFields.size());

  scene_packer_detail::gatherMeshGpuData(packed.acc, scene);
  scene_packer_detail::gatherVectorFieldGpuData(packed.acc, scene);
  scene_packer_detail::gatherCurveGpuData(packed.acc, scene, packed.curveControlPoints, packed.curveRadii,
                                          packed.pointPrimitives, packed.pointBoundingBoxes);
  scene_packer_detail::gatherPointBboxData(packed.acc, scene, packed.pointPrimitives, packed.pointBoundingBoxes);
  scene_packer_detail::gatherLightData(packed.acc, scene);
  return packed;
}

} // namespace rt
