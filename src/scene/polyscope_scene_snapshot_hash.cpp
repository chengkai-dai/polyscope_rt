#include "scene/polyscope_scene_snapshot_internal.h"

namespace snapshot_detail {

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

void faceColorsToVertex(const std::vector<glm::vec3>& faceColors,
                        const std::vector<uint32_t>& triVertInds,
                        const std::vector<uint32_t>& triFaceInds,
                        size_t nVerts, glm::vec3 fallback,
                        std::vector<glm::vec3>& outColors) {
  std::vector<glm::vec3> sum(nVerts, glm::vec3(0.f));
  std::vector<int> cnt(nVerts, 0);
  for (size_t i = 0; i < triVertInds.size(); ++i) {
    const uint32_t v = triVertInds[i];
    const uint32_t f = triFaceInds[i];
    if (v < nVerts && f < faceColors.size()) { sum[v] += faceColors[f]; ++cnt[v]; }
  }
  outColors.reserve(nVerts);
  for (size_t i = 0; i < nVerts; ++i) {
    outColors.push_back(cnt[i] > 0 ? sum[i] / float(cnt[i]) : fallback);
  }
}

std::vector<float> faceScalarsToVertex(const std::vector<float>& faceScalars,
                                       const std::vector<uint32_t>& triVertInds,
                                       const std::vector<uint32_t>& triFaceInds,
                                       size_t nVerts, float fallback) {
  std::vector<double> sum(nVerts, 0.0);
  std::vector<int> cnt(nVerts, 0);
  for (size_t i = 0; i < triVertInds.size(); ++i) {
    const uint32_t v = triVertInds[i];
    const uint32_t f = triFaceInds[i];
    if (v < nVerts && f < faceScalars.size()) { sum[v] += faceScalars[f]; ++cnt[v]; }
  }
  std::vector<float> out;
  out.reserve(nVerts);
  for (size_t i = 0; i < nVerts; ++i) {
    out.push_back(cnt[i] > 0 ? float(sum[i] / cnt[i]) : fallback);
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
  hashBytes(snapshot.scene.hash, &mesh.normalTextureScale, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.emissiveFactor[0], sizeof(float) * 3);
  hashBytes(snapshot.scene.hash, &mesh.transmissionFactor, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.indexOfRefraction, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.opacity, sizeof(float));
  hashBytes(snapshot.scene.hash, &mesh.doubleSided, sizeof(bool));
  hashBytes(snapshot.scene.hash, &mesh.unlit, sizeof(bool));
  hashBytes(snapshot.scene.hash, &mesh.hasBaseColorTexture, sizeof(bool));
  hashBytes(snapshot.scene.hash, &mesh.hasMetallicRoughnessTexture, sizeof(bool));
  hashBytes(snapshot.scene.hash, &mesh.hasEmissiveTexture, sizeof(bool));
  hashBytes(snapshot.scene.hash, &mesh.hasNormalTexture, sizeof(bool));
  if (mesh.hasBaseColorTexture) hashString(snapshot.scene.hash, mesh.baseColorTexture.cacheKey);
  if (mesh.hasMetallicRoughnessTexture) hashString(snapshot.scene.hash, mesh.metallicRoughnessTexture.cacheKey);
  if (mesh.hasEmissiveTexture) hashString(snapshot.scene.hash, mesh.emissiveTexture.cacheKey);
  if (mesh.hasNormalTexture) hashString(snapshot.scene.hash, mesh.normalTexture.cacheKey);
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
    snapshot.hostTypeName = structure.typeName();
    snapshot.hostName = structure.getName();
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

} // namespace snapshot_detail
