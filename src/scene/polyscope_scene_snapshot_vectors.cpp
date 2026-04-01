#include "scene/polyscope_scene_snapshot_internal.h"

namespace snapshot_detail {

std::vector<rt::RTVectorField> makeRTVectorFields(polyscope::SurfaceMesh& mesh) {
  std::vector<rt::RTVectorField> fields;
  for (auto& [qName, qPtr] : mesh.quantities) {
    if (!qPtr->isEnabled()) continue;

    if (auto* q = dynamic_cast<polyscope::SurfaceVertexVectorQuantity*>(qPtr.get())) {
      auto field = vectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
    if (auto* q = dynamic_cast<polyscope::SurfaceFaceVectorQuantity*>(qPtr.get())) {
      auto field = vectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
    if (auto* q = dynamic_cast<polyscope::SurfaceFaceTangentVectorQuantity*>(qPtr.get())) {
      auto field = tangentVectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
    if (auto* q = dynamic_cast<polyscope::SurfaceVertexTangentVectorQuantity*>(qPtr.get())) {
      auto field = tangentVectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
    if (auto* q = dynamic_cast<polyscope::SurfaceOneFormTangentVectorQuantity*>(qPtr.get())) {
      auto field = tangentVectorQuantityToRT(*q);
      field.name = qName;
      if (!field.roots.empty()) fields.push_back(std::move(field));
      continue;
    }
  }
  return fields;
}

void addVectorFieldsAndHash(PolyscopeSceneSnapshot& snapshot,
                            std::vector<rt::RTVectorField>&& fields,
                            std::string_view parentName) {
  for (rt::RTVectorField& vf : fields) {
    snapshot.supportedStructureCount++;
    hashString(snapshot.scene.hash, parentName);
    hashString(snapshot.scene.hash, vf.name);
    hashBytes(snapshot.scene.hash, &vf.color[0], sizeof(float) * 3);
    hashBytes(snapshot.scene.hash, &vf.radius, sizeof(float));
    hashVector(snapshot.scene.hash, vf.roots);
    hashVector(snapshot.scene.hash, vf.directions);
    snapshot.scene.vectorFields.push_back(std::move(vf));
  }
}

} // namespace snapshot_detail
