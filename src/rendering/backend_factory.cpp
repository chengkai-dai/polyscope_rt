#include "rendering/ray_tracing_backend.h"

#include <stdexcept>

#include "rendering/metal/metal_backend_private.h"
#include "rendering/metal/metal_test_harness_private.h"

namespace rt {

std::unique_ptr<IRayTracingBackend> createBackend(BackendType type, const std::string& shaderLibraryPath) {
  switch (type) {
    case BackendType::Metal:
      return createMetalBackendImpl(shaderLibraryPath);
    case BackendType::Vulkan:
      throw std::runtime_error("Vulkan backend is not implemented yet.");
  }
  throw std::runtime_error("Unsupported backend type");
}

std::unique_ptr<IPostProcessTestHarness> createTestHarness(BackendType type, const std::string& shaderLibraryPath) {
  switch (type) {
    case BackendType::Metal:
      return createMetalTestHarnessImpl(shaderLibraryPath);
    case BackendType::Vulkan:
      throw std::runtime_error("Vulkan postprocess harness is not implemented yet.");
  }
  throw std::runtime_error("Unsupported backend type");
}

} // namespace rt
