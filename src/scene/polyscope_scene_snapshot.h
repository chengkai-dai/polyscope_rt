#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "polyscope/structure.h"

#include "rendering/ray_tracing_types.h"

struct PolyscopeSceneSnapshot {
  rt::RTScene scene;
  polyscope::Structure* hostStructure = nullptr;
  std::string hostTypeName;
  std::string hostName;
  size_t supportedMeshCount = 0;
};

PolyscopeSceneSnapshot capturePolyscopeSceneSnapshot();
PolyscopeSceneSnapshot
capturePolyscopeSceneSnapshot(const std::unordered_map<std::string, rt::MaterialOverride>& materialOverrides,
                              const std::vector<rt::RTPunctualLight>& apiLights);
