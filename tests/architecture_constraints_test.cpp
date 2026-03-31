#include <fstream>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "polyscope/rt/options.h"
#include "rendering/ray_tracing_types.h"
#include "test_helpers.h"

namespace {

const std::filesystem::path kSourceDir = RT_SOURCE_DIR;

std::string readTextFile(const std::string& path) {
  const std::filesystem::path resolvedPath =
      std::filesystem::path(path).is_absolute() ? std::filesystem::path(path) : (kSourceDir / path);
  std::ifstream in(resolvedPath);
  if (!in.good()) throw std::runtime_error("failed to open file: " + resolvedPath.string());
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

void testBackendNeutralScenePackerHeader() {
  const std::string contents = readTextFile("src/rendering/scene_packer.h");
  require(contents.find("Metal/Metal.h") == std::string::npos,
          "scene_packer.h must not include Metal headers");
  require(contents.find("MTL") == std::string::npos,
          "scene_packer.h must not expose Metal-specific types");
}

void testRuntimeUsesBackendFactory() {
  const std::string contents = readTextFile("src/plugin/polyscope_rt_runtime.cpp");
  require(contents.find("createMetalPathTracerBackend(") == std::string::npos,
          "runtime should not call the Metal-only backend factory directly");
  require(contents.find("createBackend(") != std::string::npos,
          "runtime should go through the backend-neutral factory");
}

void testPluginOptionDefaultsMatchAuthoritativeDefaults() {
  const rt::AppearanceConfig defaults = rt::makeDefaultAppearanceConfig();
  requireNear(polyscope::rt::options::exposure, defaults.lighting.standardExposure, 1e-6f,
              "plugin exposure default should match makeDefaultAppearanceConfig()");
  requireNear(polyscope::rt::options::gamma, defaults.lighting.standardGamma, 1e-6f,
              "plugin gamma default should match makeDefaultAppearanceConfig()");
  requireNear(polyscope::rt::options::saturation, defaults.lighting.standardSaturation, 1e-6f,
              "plugin saturation default should match makeDefaultAppearanceConfig()");
}

} // namespace

int main() {
  try {
    testBackendNeutralScenePackerHeader();
    testRuntimeUsesBackendFactory();
    testPluginOptionDefaultsMatchAuthoritativeDefaults();
    std::cout << "architecture_constraints_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "architecture_constraints_test failed: " << e.what() << std::endl;
    return 1;
  }
}
