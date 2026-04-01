#include "rendering/metal/metal_device.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#import <Foundation/Foundation.h>
#import <mach-o/dyld.h>

#include "glm/glm.hpp"

namespace {

constexpr const char* kShaderLibraryEnvVar = "POLYSCOPE_RT_SHADER_LIBRARY";
constexpr const char* kLegacyShaderLibraryEnvVar = "POLYSCOPE_RT_METALLIB_PATH";

std::filesystem::path executableDirectory() {
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  if (size == 0) return {};

  std::string buffer(size, '\0');
  if (_NSGetExecutablePath(buffer.data(), &size) != 0) return {};

  std::error_code ec;
  std::filesystem::path executablePath = std::filesystem::weakly_canonical(std::filesystem::path(buffer.c_str()), ec);
  if (ec) executablePath = std::filesystem::path(buffer.c_str());
  return executablePath.parent_path();
}

std::vector<std::filesystem::path> shaderLibraryCandidates(const std::string& requestedPath) {
  std::vector<std::filesystem::path> candidates;

  auto appendCandidate = [&candidates](const std::filesystem::path& path) {
    if (path.empty()) return;
    if (std::find(candidates.begin(), candidates.end(), path) == candidates.end()) {
      candidates.push_back(path);
    }
  };

  if (!requestedPath.empty()) {
    appendCandidate(requestedPath);
  }
  if (const char* envPath = std::getenv(kShaderLibraryEnvVar)) {
    appendCandidate(envPath);
  }
  if (const char* envPath = std::getenv(kLegacyShaderLibraryEnvVar)) {
    appendCandidate(envPath);
  }

  const std::filesystem::path executableDir = executableDirectory();
  if (!executableDir.empty()) {
    appendCandidate(executableDir / POLYSCOPE_RT_METALLIB_FILENAME);
    appendCandidate(executableDir / "polyscope_rt" / POLYSCOPE_RT_METALLIB_FILENAME);
    appendCandidate(executableDir / ".." / "lib" / "polyscope_rt" / POLYSCOPE_RT_METALLIB_FILENAME);
    appendCandidate(executableDir / ".." / "share" / "polyscope_rt" / POLYSCOPE_RT_METALLIB_FILENAME);
  }

  appendCandidate(POLYSCOPE_RT_BUILD_METALLIB_PATH);
  return candidates;
}

} // namespace

namespace metal_rt {

MetalDeviceAvailability queryRayTracingDeviceAvailability() {
  @autoreleasepool {
    MetalDeviceAvailability availability;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      availability.reason = "Metal is unavailable on this machine.";
      return availability;
    }
    if (@available(macOS 11.0, *)) {
      if (![device supportsRaytracing]) {
        availability.reason = "This Metal device does not report ray tracing support.";
        return availability;
      }
    }
    availability.available = true;
    return availability;
  }
}

id<MTLDevice> createRayTracingDeviceOrThrow() {
  MetalDeviceAvailability availability = queryRayTracingDeviceAvailability();
  if (!availability.available) {
    throw std::runtime_error(availability.reason);
  }
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) {
    throw std::runtime_error("Metal is unavailable on this machine.");
  }
  return device;
}

std::string resolveShaderLibraryPath(const std::string& requestedPath) {
  std::error_code ec;
  std::vector<std::filesystem::path> candidates = shaderLibraryCandidates(requestedPath);
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate, ec) && !ec) {
      return std::filesystem::absolute(candidate, ec).string();
    }
    ec.clear();
  }

  std::string message = "Failed to locate Metal shader library. Checked:";
  for (const auto& candidate : candidates) {
    message += "\n - " + candidate.string();
  }
  throw std::runtime_error(message);
}

std::string buildNSErrorMessage(NSError* error) {
  if (error == nil) return "unknown error";

  std::string message;
  auto appendPart = [&](NSString* value) {
    if (value == nil || value.length == 0) return;
    if (!message.empty()) message += " | ";
    message += value.UTF8String;
  };

  appendPart(error.localizedDescription);
  appendPart(error.localizedFailureReason);
  appendPart(error.localizedRecoverySuggestion);

  for (id value in error.userInfo.allValues) {
    if ([value isKindOfClass:[NSError class]]) {
      NSError* nested = (NSError*)value;
      appendPart(nested.localizedDescription);
      appendPart(nested.localizedFailureReason);
      continue;
    }
    if ([value isKindOfClass:[NSArray class]]) {
      for (id nestedValue in (NSArray*)value) {
        if (![nestedValue isKindOfClass:[NSError class]]) continue;
        NSError* nested = (NSError*)nestedValue;
        appendPart(nested.localizedDescription);
        appendPart(nested.localizedFailureReason);
      }
    }
  }

  return message.empty() ? "unknown error" : message;
}

std::string contourObjectKey(const std::string& meshName) {
  constexpr const char* primitiveTag = "/primitive_";
  const size_t pos = meshName.rfind(primitiveTag);
  if (pos == std::string::npos) return meshName;
  return meshName.substr(0, pos);
}

float haltonSequence(uint32_t index, uint32_t base) {
  float f = 1.0f, r = 0.0f;
  float invBase = 1.0f / static_cast<float>(base);
  while (index > 0) {
    f *= invBase;
    r += f * static_cast<float>(index % base);
    index /= base;
  }
  return r;
}

simd_float4x4 makeFloat4x4(const glm::mat4& m) {
  simd_float4x4 result;
  result.columns[0] = simd_make_float4(m[0][0], m[0][1], m[0][2], m[0][3]);
  result.columns[1] = simd_make_float4(m[1][0], m[1][1], m[1][2], m[1][3]);
  result.columns[2] = simd_make_float4(m[2][0], m[2][1], m[2][2], m[2][3]);
  result.columns[3] = simd_make_float4(m[3][0], m[3][1], m[3][2], m[3][3]);
  return result;
}

simd_float4 makeFloat4(const glm::vec3& v, float w) {
  return simd_make_float4(v.x, v.y, v.z, w);
}

void dispatchThreads(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipelineState,
                     uint32_t width, uint32_t height) {
  const NSUInteger threadWidth  = pipelineState.threadExecutionWidth;
  const NSUInteger threadHeight = std::max<NSUInteger>(1, pipelineState.maxTotalThreadsPerThreadgroup / threadWidth);
  MTLSize threadsPerGroup = MTLSizeMake(threadWidth, threadHeight, 1);
  MTLSize threadsPerGrid  = MTLSizeMake(width, height, 1);
  [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerGroup];
}

uint32_t registerTextureInAcc(SceneGpuAccumulator& acc, const rt::RTTexture& tex) {
  auto existing = acc.textureLookup.find(tex.cacheKey);
  if (existing != acc.textureLookup.end()) return existing->second;
  const uint32_t idx    = static_cast<uint32_t>(acc.textures.size());
  const uint32_t offset = static_cast<uint32_t>(acc.texturePixels.size());
  GPUTexture td;
  td.data = simd_make_uint4(offset, tex.width, tex.height, 0u);
  acc.textures.push_back(td);
  acc.textureLookup.emplace(tex.cacheKey, idx);
  for (const glm::vec4& pixel : tex.pixels) {
    acc.texturePixels.push_back(simd_make_float4(pixel.r, pixel.g, pixel.b, pixel.a));
  }
  return idx;
}

} // namespace metal_rt
