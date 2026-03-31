#pragma once

#include <memory>
#include <string>

#include "rendering/ray_tracing_backend.h"

namespace rt {

std::unique_ptr<IRayTracingBackend> createMetalBackendImpl(const std::string& shaderLibraryPath = {});

} // namespace rt
