#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "glm/glm.hpp"
#include "polyscope/rt/polyscope.h"
#include "polyscope/rt/surface_mesh.h"

namespace {

std::string trim(const std::string& value) {
  size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }

  size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }

  return value.substr(begin, end - begin);
}

size_t parseObjIndexToken(const std::string& token, size_t vertexCount) {
  const size_t slash = token.find('/');
  const std::string vertexToken = slash == std::string::npos ? token : token.substr(0, slash);
  if (vertexToken.empty()) {
    throw std::runtime_error("OBJ face uses an empty vertex index.");
  }

  const int parsedIndex = std::stoi(vertexToken);
  if (parsedIndex == 0) {
    throw std::runtime_error("OBJ indices are 1-based; found index 0.");
  }

  if (parsedIndex > 0) {
    return static_cast<size_t>(parsedIndex - 1);
  }

  const int relativeIndex = static_cast<int>(vertexCount) + parsedIndex;
  if (relativeIndex < 0) {
    throw std::runtime_error("OBJ face references a vertex before the beginning of the mesh.");
  }
  return static_cast<size_t>(relativeIndex);
}

void readObjMesh(const std::filesystem::path& path, std::vector<glm::vec3>& vertices,
                 std::vector<std::array<size_t, 3>>& faces) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open OBJ file: " + path.string());
  }

  std::string line;
  while (std::getline(input, line)) {
    const std::string trimmed = trim(line);
    if (trimmed.empty() || trimmed[0] == '#') continue;

    std::istringstream stream(trimmed);
    std::string prefix;
    stream >> prefix;

    if (prefix == "v") {
      glm::vec3 position{0.0f};
      if (!(stream >> position.x >> position.y >> position.z)) {
        throw std::runtime_error("Malformed vertex record in OBJ file: " + path.string());
      }
      vertices.push_back(position);
      continue;
    }

    if (prefix == "f") {
      std::vector<size_t> polygon;
      std::string token;
      while (stream >> token) {
        polygon.push_back(parseObjIndexToken(token, vertices.size()));
      }

      if (polygon.size() < 3) {
        throw std::runtime_error("OBJ face has fewer than 3 vertices: " + path.string());
      }

      for (size_t i = 1; i + 1 < polygon.size(); ++i) {
        faces.push_back({polygon[0], polygon[i], polygon[i + 1]});
      }
    }
  }

  if (vertices.empty() || faces.empty()) {
    throw std::runtime_error("OBJ file does not contain a renderable triangle mesh: " + path.string());
  }
}

std::filesystem::path defaultMeshPath() {
  return std::filesystem::path(POLYSCOPE_RT_EXAMPLE_DEFAULT_MESH);
}

} // namespace

int main(int argc, char** argv) {
  const std::filesystem::path meshPath = argc > 1 ? std::filesystem::path(argv[1]) : defaultMeshPath();

  std::vector<glm::vec3> meshVertices;
  std::vector<std::array<size_t, 3>> meshFaces;

  try {
    readObjMesh(meshPath, meshVertices, meshFaces);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  polyscope::rt::options::programName = "Polyscope RT Surface Mesh Example";

  // The migrated user-facing flow is intentionally the same as standard Polyscope:
  // init -> load mesh -> registerSurfaceMesh -> show.
  polyscope::rt::init();
  polyscope::rt::registerSurfaceMesh("input mesh", meshVertices, meshFaces);
  polyscope::rt::show();

  return 0;
}
