#pragma once

#include <cstdint>
#include <vector>

#include "glm/glm.hpp"

namespace rt_geometry {

struct TriangleMeshData {
  std::vector<glm::vec3> vertices;
  std::vector<glm::uvec3> faces;
};

TriangleMeshData makeUvSphere(float radius, uint32_t latSegments, uint32_t lonSegments);
TriangleMeshData makeCylinder(const glm::vec3& start, const glm::vec3& end, float radius, uint32_t radialSegments);
void appendMesh(TriangleMeshData& target, const TriangleMeshData& source, const glm::mat4& transform);

/// Merge vertices within `epsilon` (quantized grid), remap faces, drop degenerate triangles.
void weldVertices(TriangleMeshData& mesh, float epsilon);

} // namespace rt_geometry
