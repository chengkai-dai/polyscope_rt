#pragma once

#include <cmath>
#include <stdexcept>

inline void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

inline void requireNear(float a, float b, float tol, const char* message) {
  if (std::abs(a - b) > tol) throw std::runtime_error(message);
}
