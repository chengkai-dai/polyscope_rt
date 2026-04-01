#pragma once

#include <memory>
#include <string>

#include "rendering/ray_tracing_backend.h"

namespace rt {

BackendAvailability queryMetalBackendAvailability(const std::string& shaderLibraryPath = {});
std::unique_ptr<IRayTracingBackend> createMetalBackendImpl(const std::string& shaderLibraryPath = {});

} // namespace rt
