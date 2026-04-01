#include <cmath>
#include <exception>
#include <iostream>
#include <string>

#include "glm/gtc/matrix_transform.hpp"

#include "rendering/scene_packer.h"
#include "test_helpers.h"

namespace {

rt::RTTexture makeSolidTexture(const std::string& cacheKey, const glm::vec4& color) {
  rt::RTTexture texture;
  texture.width = 1;
  texture.height = 1;
  texture.cacheKey = cacheKey;
  texture.pixels = {color};
  return texture;
}

rt::RTMesh makeTriangleMesh(const std::string& name) {
  rt::RTMesh mesh;
  mesh.name = name;
  mesh.vertices = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  mesh.normals = {{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}};
  mesh.texcoords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
  mesh.indices = {glm::uvec3(0u, 1u, 2u)};
  return mesh;
}

glm::vec3 unpackVec3(const simd_float4& value) {
  return {value.x, value.y, value.z};
}

void testMaterialFlagsAndTexturePacking() {
  rt::RTScene scene;
  rt::RTMesh mesh = makeTriangleMesh("flagged_mesh");
  mesh.baseColorFactor = {0.2f, 0.4f, 0.6f, 0.8f};
  mesh.metallicFactor = 0.25f;
  mesh.roughnessFactor = 0.75f;
  mesh.emissiveFactor = {0.3f, 0.2f, 0.1f};
  mesh.transmissionFactor = 0.65f;
  mesh.indexOfRefraction = 1.7f;
  mesh.opacity = 0.35f;
  mesh.unlit = true;
  mesh.doubleSided = true;
  mesh.normalTextureScale = 0.8f;
  mesh.hasBaseColorTexture = true;
  mesh.baseColorTexture = makeSolidTexture("base_tex", {0.8f, 0.2f, 0.1f, 1.0f});
  mesh.hasEmissiveTexture = true;
  mesh.emissiveTexture = makeSolidTexture("emissive_tex", {0.4f, 0.6f, 0.9f, 1.0f});
  mesh.hasMetallicRoughnessTexture = true;
  mesh.metallicRoughnessTexture = makeSolidTexture("mr_tex", {0.0f, 0.7f, 0.2f, 1.0f});
  mesh.hasNormalTexture = true;
  mesh.normalTexture = makeSolidTexture("normal_tex", {0.5f, 0.5f, 1.0f, 1.0f});
  scene.meshes.push_back(mesh);

  const rt::PackedSceneData packed = rt::packScene(scene);
  require(packed.acc.materials.size() == 1u, "expected exactly one packed material");
  require(packed.acc.textures.size() == 4u, "expected four distinct packed textures");
  require(packed.acc.texturePixels.size() == 4u, "expected one pixel per packed texture");
  require(packed.acc.shaderTriangles.size() == 1u, "expected one packed triangle");

  const GPUMaterial& material = packed.acc.materials.front();
  requireNear(material.baseColorFactor.x, 0.2f, 1e-6f, "base color red should be preserved");
  requireNear(material.baseColorFactor.y, 0.4f, 1e-6f, "base color green should be preserved");
  requireNear(material.baseColorFactor.z, 0.6f, 1e-6f, "base color blue should be preserved");
  requireNear(material.baseColorFactor.w, 0.8f, 1e-6f, "base color alpha should be preserved");
  requireNear(material.metallicRoughnessNormal.x, 0.25f, 1e-6f, "metallic factor should be preserved");
  requireNear(material.metallicRoughnessNormal.y, 0.75f, 1e-6f, "roughness factor should be preserved");
  requireNear(material.metallicRoughnessNormal.z, 0.8f, 1e-6f, "normal texture scale should be preserved");
  requireNear(material.transmissionIor.x, 0.65f, 1e-6f, "transmission should be preserved");
  requireNear(material.transmissionIor.y, 1.7f, 1e-6f, "IOR should be preserved");
  requireNear(material.transmissionIor.w, 0.35f, 1e-6f, "opacity should be preserved");
  require(material.materialFlags.x == 1u, "unlit flag should be packed");
  require(material.materialFlags.y == 1u, "double-sided flag should be packed");
  require(material.baseColorTextureData.y == 1u, "base color texture should be marked present");
  require(material.emissiveTextureData.y == 1u, "emissive texture should be marked present");
  require(material.metallicRoughnessTextureData.y == 1u, "metallic-roughness texture should be marked present");
  require(material.normalTextureData.y == 1u, "normal texture should be marked present");
}

void testTextureDeduplicationByCacheKey() {
  rt::RTScene scene;
  rt::RTTexture sharedTexture = makeSolidTexture("shared_tex", {0.2f, 0.3f, 0.4f, 1.0f});

  rt::RTMesh meshA = makeTriangleMesh("mesh_a");
  meshA.hasBaseColorTexture = true;
  meshA.baseColorTexture = sharedTexture;
  scene.meshes.push_back(meshA);

  rt::RTMesh meshB = makeTriangleMesh("mesh_b");
  meshB.transform = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f));
  meshB.hasBaseColorTexture = true;
  meshB.baseColorTexture = sharedTexture;
  scene.meshes.push_back(meshB);

  const rt::PackedSceneData packed = rt::packScene(scene);
  require(packed.acc.materials.size() == 2u, "expected two packed materials");
  require(packed.acc.textures.size() == 1u, "textures with the same cache key should be deduplicated");
  require(packed.acc.materials[0].baseColorTextureData.x == packed.acc.materials[1].baseColorTextureData.x,
          "deduplicated textures should share the same packed texture index");
}

void testTriangleWindingAlignsWithSuppliedNormals() {
  rt::RTScene scene;
  rt::RTMesh mesh = makeTriangleMesh("flipped_winding");
  mesh.normals = {{0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, -1.0f}};
  scene.meshes.push_back(mesh);

  const rt::PackedSceneData packed = rt::packScene(scene);
  require(packed.acc.accelIndices.size() == 1u, "expected one packed acceleration triangle");
  const PackedTriangleIndices accelTri = packed.acc.accelIndices.front();
  require(accelTri.i0 == 0u && accelTri.i1 == 2u && accelTri.i2 == 1u,
          "packed triangle winding should flip to match the supplied vertex normals");

  const GPUTriangle& tri = packed.acc.shaderTriangles.front();
  require(tri.indicesMaterial.x == 0u && tri.indicesMaterial.y == 2u && tri.indicesMaterial.z == 1u,
          "shader triangle indices should match the winding-corrected triangle");
}

void testTransformAppliesToPositionsAndNormals() {
  rt::RTScene scene;
  rt::RTMesh mesh = makeTriangleMesh("transformed_mesh");
  const glm::vec3 diagonalNormal = glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f));
  mesh.normals = {diagonalNormal, diagonalNormal, diagonalNormal};
  mesh.transform = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, -1.0f, 2.0f)) *
                   glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 1.0f, 0.5f));
  scene.meshes.push_back(mesh);

  const rt::PackedSceneData packed = rt::packScene(scene);
  require(packed.acc.positions.size() >= 1u && packed.acc.normals.size() >= 1u,
          "transformed mesh should pack positions and normals");

  const glm::vec3 packedPosition = unpackVec3(packed.acc.positions[0]);
  requireNear(packedPosition.x, 3.0f, 1e-6f, "mesh transform should translate X");
  requireNear(packedPosition.y, -1.0f, 1e-6f, "mesh transform should translate Y");
  requireNear(packedPosition.z, 2.0f, 1e-6f, "mesh transform should translate Z");

  const glm::vec3 expectedNormal = glm::normalize(glm::vec3(0.5f, 0.0f, 2.0f));
  const glm::vec3 packedNormal = unpackVec3(packed.acc.normals[0]);
  require(glm::length(packedNormal - expectedNormal) < 1e-5f,
          "packed normals should use inverse-transpose transformed world normals");
}

void testEmissiveTriangleWeightsScaleByAreaAndPower() {
  rt::RTScene scene;

  rt::RTMesh small = makeTriangleMesh("small_emissive");
  small.emissiveFactor = {1.0f, 1.0f, 1.0f};
  scene.meshes.push_back(small);

  rt::RTMesh large = makeTriangleMesh("large_emissive");
  large.vertices = {{0.0f, 0.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {0.0f, 2.0f, 0.0f}};
  large.normals = {{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 1.0f}};
  large.texcoords = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
  large.indices = {glm::uvec3(0u, 1u, 2u)};
  large.emissiveFactor = {2.0f, 2.0f, 2.0f};
  large.transform = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0.0f, 0.0f));
  scene.meshes.push_back(large);

  const rt::PackedSceneData packed = rt::packScene(scene);
  require(packed.acc.emissiveTriangles.size() == 2u, "expected two emissive triangles");

  const GPUEmissiveTriangle& smallTri = packed.acc.emissiveTriangles[0];
  const GPUEmissiveTriangle& largeTri = packed.acc.emissiveTriangles[1];
  requireNear(smallTri.params.x, 0.5f, 1e-6f, "small emissive triangle area should be packed");
  requireNear(largeTri.params.x, 2.0f, 1e-6f, "large emissive triangle area should be packed");
  requireNear(largeTri.params.y / smallTri.params.y, 8.0f, 1e-5f,
              "emissive selection weight should scale with triangle area times emissive power");
}

void testContourPrimitiveNamesShareObjectId() {
  rt::RTScene scene;
  rt::RTMesh primitive0 = makeTriangleMesh("contour_mesh/primitive_0");
  rt::RTMesh primitive1 = makeTriangleMesh("contour_mesh/primitive_1");
  primitive1.transform = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f));
  scene.meshes.push_back(primitive0);
  scene.meshes.push_back(primitive1);

  const rt::PackedSceneData packed = rt::packScene(scene);
  require(packed.acc.shaderTriangles.size() == 2u, "expected one packed triangle per primitive mesh");
  require(packed.acc.shaderTriangles[0].objectFlags.x == packed.acc.shaderTriangles[1].objectFlags.x,
          "contour primitive meshes should collapse to the same object ID");
}

void testObjectIdsStayStableAcrossEquivalentPackingRuns() {
  rt::RTScene scene;
  rt::RTMesh meshA = makeTriangleMesh("mesh_a");
  rt::RTMesh meshB = makeTriangleMesh("mesh_b");
  meshB.transform = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f));
  scene.meshes.push_back(meshA);
  scene.meshes.push_back(meshB);

  const rt::PackedSceneData packedA = rt::packScene(scene);
  const rt::PackedSceneData packedB = rt::packScene(scene);
  require(packedA.acc.shaderTriangles.size() == packedB.acc.shaderTriangles.size(),
          "equivalent scenes should produce the same triangle count across packing runs");
  for (size_t i = 0; i < packedA.acc.shaderTriangles.size(); ++i) {
    require(packedA.acc.shaderTriangles[i].objectFlags.x == packedB.acc.shaderTriangles[i].objectFlags.x,
            "object IDs should stay stable across equivalent packing runs");
  }
}

void testVectorFieldArrowAddsClosedCapGeometry() {
  rt::RTScene scene;
  rt::RTVectorField vf;
  vf.name = "vf_caps";
  vf.roots = {{0.0f, 0.0f, 0.0f}};
  vf.directions = {{0.0f, 1.0f, 0.0f}};
  vf.radius = 0.1f;
  scene.vectorFields.push_back(vf);

  const rt::PackedSceneData packed = rt::packScene(scene);
  require(packed.acc.materials.size() == 1u, "vector field should pack one material");
  require(packed.acc.shaderTriangles.size() == 40u,
          "vector field arrow should pack side geometry plus closed root/base caps");

  bool foundRootCenter = false;
  bool foundConeBaseCenter = false;
  for (size_t i = 0; i < packed.acc.positions.size(); ++i) {
    const glm::vec3 position = unpackVec3(packed.acc.positions[i]);
    const glm::vec3 normal = unpackVec3(packed.acc.normals[i]);
    if (glm::length(position - glm::vec3(0.0f, 0.0f, 0.0f)) < 1e-6f) {
      foundRootCenter = true;
      require(normal.y < -0.99f, "root cap center should face backward along the arrow axis");
    }
    if (glm::length(position - glm::vec3(0.0f, 0.8f, 0.0f)) < 1e-6f) {
      foundConeBaseCenter = true;
      require(normal.y < -0.99f, "cone base cap center should face backward along the arrow axis");
    }
  }

  require(foundRootCenter, "vector field arrow should contain a root cap center vertex");
  require(foundConeBaseCenter, "vector field arrow should contain a cone base cap center vertex");
}

} // namespace

int main() {
  try {
    testMaterialFlagsAndTexturePacking();
    testTextureDeduplicationByCacheKey();
    testTriangleWindingAlignsWithSuppliedNormals();
    testTransformAppliesToPositionsAndNormals();
    testEmissiveTriangleWeightsScaleByAreaAndPower();
    testContourPrimitiveNamesShareObjectId();
    testObjectIdsStayStableAcrossEquivalentPackingRuns();
    testVectorFieldArrowAddsClosedCapGeometry();
    std::cout << "scene_packer_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "scene_packer_test failed: " << e.what() << std::endl;
    return 1;
  }
}
