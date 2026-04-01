#pragma once

#include <cmath>
#include <stdexcept>
#include <string>

#include "rendering/ray_tracing_backend.h"

inline void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

inline void requireNear(float a, float b, float tol, const char* message) {
  if (std::abs(a - b) > tol) throw std::runtime_error(message);
}

inline bool isSkippableBackendError(const std::string& message) {
  return message.find("Metal is unavailable") != std::string::npos ||
         message.find("does not report ray tracing support") != std::string::npos;
}

inline bool isBackendAvailable(rt::BackendType type) {
  return rt::queryBackendAvailability(type).available;
}

inline bool isTestHarnessAvailable(rt::BackendType type) {
  return rt::queryTestHarnessAvailability(type).available;
}
