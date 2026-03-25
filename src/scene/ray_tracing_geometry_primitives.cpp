#include "scene/ray_tracing_geometry_primitives.h"

#include <cmath>
#include <cstdint>
#include <unordered_map>

#include "glm/gtc/constants.hpp"

namespace rt_geometry {
TriangleMeshData makeUvSphere(float radius, uint32_t latSegments, uint32_t lonSegments) {
  TriangleMeshData mesh;
  if (radius <= 0.0f || latSegments < 2u || lonSegments < 3u) return mesh;

  for (uint32_t y = 0; y <= latSegments; ++y) {
    float v = static_cast<float>(y) / static_cast<float>(latSegments);
    float theta = v * glm::pi<float>();
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    for (uint32_t x = 0; x <= lonSegments; ++x) {
      float u = static_cast<float>(x) / static_cast<float>(lonSegments);
      float phi = u * glm::two_pi<float>();
      float sinPhi = std::sin(phi);
      float cosPhi = std::cos(phi);
      mesh.vertices.emplace_back(radius * sinTheta * cosPhi, radius * cosTheta, radius * sinTheta * sinPhi);
    }
  }

  auto idx = [lonSegments](uint32_t y, uint32_t x) { return y * (lonSegments + 1u) + x; };
  for (uint32_t y = 0; y < latSegments; ++y) {
    for (uint32_t x = 0; x < lonSegments; ++x) {
      uint32_t i0 = idx(y, x);
      uint32_t i1 = idx(y, x + 1u);
      uint32_t i2 = idx(y + 1u, x + 1u);
      uint32_t i3 = idx(y + 1u, x);
      if (y != 0u) mesh.faces.emplace_back(i0, i3, i1);
      if (y + 1u != latSegments) mesh.faces.emplace_back(i1, i3, i2);
    }
  }

  return mesh;
}

TriangleMeshData makeCylinder(const glm::vec3& start, const glm::vec3& end, float radius, uint32_t radialSegments) {
  TriangleMeshData mesh;
  if (radius <= 0.0f || radialSegments < 3u) return mesh;

  glm::vec3 axis = end - start;
  float length = glm::length(axis);
  if (length <= 1e-6f) return mesh;
  axis /= length;

  glm::vec3 tangent = std::abs(axis.y) < 0.999f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
  glm::vec3 bitangent = glm::normalize(glm::cross(axis, tangent));
  tangent = glm::normalize(glm::cross(bitangent, axis));

  mesh.vertices.reserve(static_cast<size_t>(radialSegments) * 2u);
  for (uint32_t i = 0; i < radialSegments; ++i) {
    float angle = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(radialSegments);
    glm::vec3 radial = std::cos(angle) * tangent + std::sin(angle) * bitangent;
    mesh.vertices.push_back(start + radius * radial);
    mesh.vertices.push_back(end + radius * radial);
  }

  for (uint32_t i = 0; i < radialSegments; ++i) {
    uint32_t next = (i + 1u) % radialSegments;
    uint32_t base0 = 2u * i;
    uint32_t top0 = base0 + 1u;
    uint32_t base1 = 2u * next;
    uint32_t top1 = base1 + 1u;
    mesh.faces.emplace_back(base0, base1, top0);
    mesh.faces.emplace_back(top0, base1, top1);
  }

  return mesh;
}

void appendMesh(TriangleMeshData& target, const TriangleMeshData& source, const glm::mat4& transform) {
  if (source.vertices.empty() || source.faces.empty()) return;

  uint32_t baseIndex = static_cast<uint32_t>(target.vertices.size());
  target.vertices.reserve(target.vertices.size() + source.vertices.size());
  target.faces.reserve(target.faces.size() + source.faces.size());

  for (const glm::vec3& vertex : source.vertices) {
    target.vertices.push_back(glm::vec3(transform * glm::vec4(vertex, 1.0f)));
  }
  for (const glm::uvec3& face : source.faces) {
    target.faces.emplace_back(baseIndex + face.x, baseIndex + face.y, baseIndex + face.z);
  }
}

void weldVertices(TriangleMeshData& mesh, float epsilon) {
  if (mesh.vertices.empty()) return;
  if (epsilon <= 0.0f) return;

  struct Key {
    int64_t x;
    int64_t y;
    int64_t z;
    bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
  };
  struct KeyHash {
    size_t operator()(const Key& k) const {
      return (static_cast<size_t>(k.x) * 73856093u) ^ (static_cast<size_t>(k.y) * 19349663u) ^
             (static_cast<size_t>(k.z) * 83492791u);
    }
  };

  const float invEps = 1.0f / epsilon;
  std::unordered_map<Key, uint32_t, KeyHash> grid;
  grid.reserve(mesh.vertices.size());

  std::vector<uint32_t> oldToNew(mesh.vertices.size());
  std::vector<glm::vec3> newVerts;
  newVerts.reserve(mesh.vertices.size());

  for (size_t i = 0; i < mesh.vertices.size(); ++i) {
    const glm::vec3& p = mesh.vertices[i];
    Key k{static_cast<int64_t>(std::llround(static_cast<double>(p.x * invEps))),
          static_cast<int64_t>(std::llround(static_cast<double>(p.y * invEps))),
          static_cast<int64_t>(std::llround(static_cast<double>(p.z * invEps)))};
    auto it = grid.find(k);
    if (it != grid.end()) {
      oldToNew[i] = it->second;
    } else {
      uint32_t idx = static_cast<uint32_t>(newVerts.size());
      oldToNew[i] = idx;
      grid.emplace(k, idx);
      newVerts.push_back(p);
    }
  }

  std::vector<glm::uvec3> newFaces;
  newFaces.reserve(mesh.faces.size());
  for (const glm::uvec3& f : mesh.faces) {
    glm::uvec3 nf(oldToNew[f.x], oldToNew[f.y], oldToNew[f.z]);
    if (nf.x == nf.y || nf.y == nf.z || nf.z == nf.x) continue;
    newFaces.push_back(nf);
  }

  mesh.vertices = std::move(newVerts);
  mesh.faces = std::move(newFaces);
}

} // namespace rt_geometry
