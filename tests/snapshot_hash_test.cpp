#include <exception>
#include <iostream>
#include <vector>

#include "glm/glm.hpp"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "scene/polyscope_scene_snapshot.h"
#include "test_helpers.h"

namespace {

polyscope::SurfaceMesh* registerTinyMesh() {
  std::vector<glm::vec3> verts{
      {-1.0f, 0.0f, -1.0f},
      {1.0f, 0.0f, -1.0f},
      {0.0f, 0.0f, 1.0f},
  };
  std::vector<std::vector<uint32_t>> faces{{0, 1, 2}};
  return polyscope::registerSurfaceMesh("hash_mesh", verts, faces);
}

void testMaterialOverrideFieldsAffectHash() {
  polyscope::removeEverything();
  auto* mesh = registerTinyMesh();
  mesh->setEnabled(true);

  const PolyscopeSceneSnapshot base = capturePolyscopeSceneSnapshot();

  std::unordered_map<std::string, rt::MaterialOverride> overrides;
  overrides["hash_mesh"].transmission = 0.75f;
  overrides["hash_mesh"].ior = 1.6f;
  overrides["hash_mesh"].opacity = 0.5f;
  overrides["hash_mesh"].unlit = true;
  const PolyscopeSceneSnapshot overridden = capturePolyscopeSceneSnapshot(overrides, {});

  require(base.scene.hash != overridden.scene.hash,
          "material overrides that affect shading should change the scene hash");
}

void testBackFacePolicyAffectsHash() {
  polyscope::removeEverything();
  auto* mesh = registerTinyMesh();
  mesh->setBackFacePolicy(polyscope::BackFacePolicy::Cull);
  const PolyscopeSceneSnapshot culled = capturePolyscopeSceneSnapshot();

  mesh->setBackFacePolicy(polyscope::BackFacePolicy::Different);
  const PolyscopeSceneSnapshot doubleSided = capturePolyscopeSceneSnapshot();

  require(culled.scene.hash != doubleSided.scene.hash,
          "back-face policy should flow through doubleSided and affect the scene hash");
}

void testApiLightsAffectHash() {
  polyscope::removeEverything();
  registerTinyMesh();

  std::vector<rt::RTPunctualLight> lightsA(1);
  lightsA[0].type = rt::RTPunctualLightType::Point;
  lightsA[0].position = {0.0f, 1.0f, 0.0f};
  lightsA[0].intensity = 2.0f;

  std::vector<rt::RTPunctualLight> lightsB = lightsA;
  lightsB[0].intensity = 3.0f;

  const PolyscopeSceneSnapshot snapA = capturePolyscopeSceneSnapshot({}, lightsA);
  const PolyscopeSceneSnapshot snapB = capturePolyscopeSceneSnapshot({}, lightsB);
  require(snapA.scene.hash != snapB.scene.hash,
          "API light changes should affect the scene hash");
}

} // namespace

int main() {
  try {
    polyscope::init("openGL_mock");
    testMaterialOverrideFieldsAffectHash();
    testBackFacePolicyAffectsHash();
    testApiLightsAffectHash();
    polyscope::shutdown();
    std::cout << "snapshot_hash_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "snapshot_hash_test failed: " << e.what() << std::endl;
    return 1;
  }
}
