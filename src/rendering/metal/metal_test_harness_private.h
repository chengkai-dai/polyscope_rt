#pragma once

#include <memory>
#include <string>

#include "rendering/ray_tracing_backend.h"

namespace rt {

BackendAvailability queryMetalTestHarnessAvailability(const std::string& shaderLibraryPath = {});
std::unique_ptr<IPostProcessTestHarness> createMetalTestHarnessImpl(const std::string& shaderLibraryPath = {});

} // namespace rt
