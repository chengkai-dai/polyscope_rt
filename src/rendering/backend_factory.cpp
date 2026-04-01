#include "rendering/ray_tracing_backend.h"

#include <stdexcept>

#include "rendering/metal/metal_backend_private.h"
#include "rendering/metal/metal_test_harness_private.h"

namespace rt {

namespace {

BackendAvailability unavailableBackend(BackendType type, std::string name, std::string reason) {
  BackendAvailability availability;
  availability.type = type;
  availability.name = std::move(name);
  availability.available = false;
  availability.reason = std::move(reason);
  return availability;
}

void throwIfUnavailable(const BackendAvailability& availability) {
  if (!availability.available) {
    throw std::runtime_error(availability.reason.empty() ? "Requested backend is unavailable." : availability.reason);
  }
}

} // namespace

BackendAvailability queryBackendAvailability(BackendType type, const std::string& shaderLibraryPath) {
  switch (type) {
    case BackendType::Metal:
      return queryMetalBackendAvailability(shaderLibraryPath);
    case BackendType::Vulkan:
      return unavailableBackend(type, "Vulkan", "Vulkan backend is not implemented yet.");
  }
  return unavailableBackend(type, "Unknown", "Unsupported backend type.");
}

BackendAvailability queryTestHarnessAvailability(BackendType type, const std::string& shaderLibraryPath) {
  switch (type) {
    case BackendType::Metal:
      return queryMetalTestHarnessAvailability(shaderLibraryPath);
    case BackendType::Vulkan:
      return unavailableBackend(type, "Vulkan", "Vulkan postprocess harness is not implemented yet.");
  }
  return unavailableBackend(type, "Unknown", "Unsupported backend type.");
}

std::unique_ptr<IRayTracingBackend> createBackend(BackendType type, const std::string& shaderLibraryPath) {
  const BackendAvailability availability = queryBackendAvailability(type, shaderLibraryPath);
  throwIfUnavailable(availability);
  switch (type) {
    case BackendType::Metal:
      return createMetalBackendImpl(shaderLibraryPath);
    case BackendType::Vulkan:
      break;
  }
  throw std::runtime_error("Unsupported backend type.");
}

std::unique_ptr<IPostProcessTestHarness> createTestHarness(BackendType type, const std::string& shaderLibraryPath) {
  const BackendAvailability availability = queryTestHarnessAvailability(type, shaderLibraryPath);
  throwIfUnavailable(availability);
  switch (type) {
    case BackendType::Metal:
      return createMetalTestHarnessImpl(shaderLibraryPath);
    case BackendType::Vulkan:
      break;
  }
  throw std::runtime_error("Unsupported backend type.");
}

} // namespace rt
