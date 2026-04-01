#include <exception>
#include <iostream>
#include <stdexcept>

#include "rendering/ray_tracing_backend.h"
#include "test_helpers.h"

namespace {

void testVulkanAvailabilityQueryIsExplicit() {
  const rt::BackendAvailability availability = rt::queryBackendAvailability(rt::BackendType::Vulkan);
  require(!availability.available, "Vulkan should report unavailable until implemented");
  require(availability.reason.find("not implemented") != std::string::npos,
          "Vulkan availability should explain that the backend is not implemented");

  bool threw = false;
  try {
    (void)rt::createBackend(rt::BackendType::Vulkan);
  } catch (const std::exception& e) {
    threw = true;
    require(std::string(e.what()).find("not implemented") != std::string::npos,
            "Vulkan createBackend() should surface the same not-implemented reason");
  }
  require(threw, "createBackend(Vulkan) should throw while the backend is unimplemented");
}

void testMetalAvailabilityQueryMatchesFactory() {
  const rt::BackendAvailability availability = rt::queryBackendAvailability(rt::BackendType::Metal);
  if (!availability.available) {
    require(isSkippableBackendError(availability.reason),
            "unavailable Metal backend should explain whether Metal or ray tracing support is missing");
    return;
  }

  auto backend = rt::createBackend(rt::BackendType::Metal);
  require(backend != nullptr, "available Metal backend should be creatable");
}

void testHarnessAvailabilityQueryMatchesFactory() {
  const rt::BackendAvailability availability = rt::queryTestHarnessAvailability(rt::BackendType::Metal);
  if (!availability.available) {
    require(isSkippableBackendError(availability.reason),
            "unavailable Metal harness should explain whether Metal or ray tracing support is missing");
    return;
  }

  auto harness = rt::createTestHarness(rt::BackendType::Metal);
  require(harness != nullptr, "available Metal harness should be creatable");
}

} // namespace

int main() {
  try {
    testVulkanAvailabilityQueryIsExplicit();
    testMetalAvailabilityQueryMatchesFactory();
    testHarnessAvailabilityQueryMatchesFactory();
    std::cout << "backend_factory_test passed" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "backend_factory_test failed: " << e.what() << std::endl;
    return 1;
  }
}
