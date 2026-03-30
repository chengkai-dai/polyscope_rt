// curve_network_example.cpp
//
// Demonstrates the four new per-primitive curve-network quantity features
// that polyscope_rt now renders via ray tracing:
//
//   1. Node scalar quantity  →  colormap along a helix (temperature)
//   2. Edge scalar quantity  →  local curvature of a Lissajous figure
//   3. Node color quantity   →  RGB rainbow on a figure-8 knot
//   4. Edge color quantity   →  hue-shifted spring coil
//
// Each curve lives at a different world-space location so they are all
// visible at once without overlap.

#include <array>
#include <cmath>
#include <vector>

#include "glm/glm.hpp"
#include "polyscope/rt/curve_network.h"
#include "polyscope/rt/polyscope.h"

namespace ps = polyscope::rt;

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------
static constexpr float kTwoPi = 6.283185307f;

static glm::vec3 hsv2rgb(float h, float s, float v) {
  float c = v * s;
  float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
  float m = v - c;
  glm::vec3 rgb;
  int seg = int(h * 6.0f) % 6;
  switch (seg) {
    case 0: rgb = {c, x, 0}; break;
    case 1: rgb = {x, c, 0}; break;
    case 2: rgb = {0, c, x}; break;
    case 3: rgb = {0, x, c}; break;
    case 4: rgb = {x, 0, c}; break;
    default: rgb = {c, 0, x}; break;
  }
  return rgb + m;
}

// ---------------------------------------------------------------------------
// 1. Helix with NODE SCALAR quantity ("temperature")
//    Placed at origin.
// ---------------------------------------------------------------------------
static void addHelixNodeScalar() {
  constexpr int N = 120;        // nodes
  constexpr float R = 0.18f;   // helix radius
  constexpr float pitch = 0.4f; // vertical rise per turn

  std::vector<glm::vec3> nodes(N);
  std::vector<std::array<size_t, 2>> edges;
  std::vector<float> temperature(N);

  for (int i = 0; i < N; ++i) {
    float t = float(i) / float(N - 1);
    float angle = t * 4.0f * kTwoPi;
    nodes[i] = {R * std::cos(angle), t * pitch - pitch * 0.5f,
                R * std::sin(angle)};
    // "temperature" rises from 0 at bottom to 1 at top, with a small
    // oscillation to give the scalar some interesting structure.
    temperature[i] = t + 0.08f * std::sin(angle * 2.0f);
    if (i + 1 < N) edges.push_back({size_t(i), size_t(i + 1)});
  }

  auto* net =
      ps::registerCurveNetwork("helix – node scalar", nodes, edges);
  net->setRadius(0.006f, /*isRelative=*/false);

  net->addNodeScalarQuantity("temperature", temperature)
      ->setColorMap("coolwarm")
      ->setMapRange({0.0f, 1.0f})
      ->setEnabled(true);
}

// ---------------------------------------------------------------------------
// 2. Lissajous figure with EDGE SCALAR quantity ("curvature")
//    Placed +0.6 in X.
// ---------------------------------------------------------------------------
static void addLissajousEdgeScalar() {
  constexpr int N = 200;
  constexpr float ox = 0.6f; // x offset

  std::vector<glm::vec3> nodes(N);
  std::vector<std::array<size_t, 2>> edges;

  // Lissajous 3:2:1 in 3-D
  for (int i = 0; i < N; ++i) {
    float t = float(i) / float(N) * kTwoPi;
    nodes[i] = {ox + 0.2f * std::cos(3.0f * t + 0.5f),
                0.2f * std::sin(2.0f * t),
                0.2f * std::cos(1.0f * t)};
    if (i + 1 < N) edges.push_back({size_t(i), size_t(i + 1)});
  }
  // Close the loop
  edges.push_back({size_t(N - 1), 0});

  // Discrete curvature per edge: angle between consecutive tangents.
  int E = int(edges.size());
  std::vector<float> curvature(E);
  for (int e = 0; e < E; ++e) {
    size_t a = edges[e][0], b = edges[e][1];
    size_t c = edges[(e + 1) % E][1];
    glm::vec3 t0 = glm::normalize(nodes[b] - nodes[a]);
    glm::vec3 t1 = glm::normalize(nodes[c] - nodes[b]);
    float cosA = glm::clamp(glm::dot(t0, t1), -1.0f, 1.0f);
    curvature[e] = std::acos(cosA); // 0 = straight, π = sharp turn
  }

  auto* net =
      ps::registerCurveNetworkLoop("lissajous – edge scalar", nodes);
  net->setRadius(0.006f, /*isRelative=*/false);

  net->addEdgeScalarQuantity("curvature", curvature)
      ->setColorMap("viridis")
      ->setEnabled(true);
}

// ---------------------------------------------------------------------------
// 3. Figure-8 knot with NODE COLOR quantity (RGB rainbow)
//    Placed -0.6 in X.
// ---------------------------------------------------------------------------
static void addFigure8NodeColor() {
  constexpr int N = 160;
  constexpr float ox = -0.6f;

  std::vector<glm::vec3> nodes(N);
  std::vector<std::array<size_t, 2>> edges;
  std::vector<glm::vec3> nodeColors(N);

  // Figure-8 knot parametrisation
  for (int i = 0; i < N; ++i) {
    float t = float(i) / float(N) * kTwoPi;
    float x = (2.0f + std::cos(2.0f * t)) * std::cos(3.0f * t);
    float y = (2.0f + std::cos(2.0f * t)) * std::sin(3.0f * t);
    float z = std::sin(4.0f * t);
    // Rescale to fit in ~0.4 units
    nodes[i] = {ox + x * 0.07f, y * 0.07f, z * 0.07f};
    nodeColors[i] = hsv2rgb(float(i) / float(N), 1.0f, 1.0f);
    if (i + 1 < N) edges.push_back({size_t(i), size_t(i + 1)});
  }
  edges.push_back({size_t(N - 1), 0}); // close the knot

  auto* net = ps::registerCurveNetworkLoop("figure-8 knot – node color", nodes);
  net->setRadius(0.007f, /*isRelative=*/false);

  net->addNodeColorQuantity("rainbow", nodeColors)->setEnabled(true);
}

// ---------------------------------------------------------------------------
// 4. Spring coil with EDGE COLOR quantity (hue shift per edge)
//    Placed 0 in X, +0.6 in Z.
// ---------------------------------------------------------------------------
static void addSpringEdgeColor() {
  constexpr int N = 150;
  constexpr float oz = 0.6f;
  constexpr float R = 0.16f;
  constexpr float turns = 5.0f;

  std::vector<glm::vec3> nodes(N);
  std::vector<std::array<size_t, 2>> edges;

  for (int i = 0; i < N; ++i) {
    float t = float(i) / float(N - 1);
    float angle = t * turns * kTwoPi;
    nodes[i] = {R * std::cos(angle),
                t * 0.5f - 0.25f,
                oz + R * std::sin(angle)};
    if (i + 1 < N) edges.push_back({size_t(i), size_t(i + 1)});
  }

  int E = int(edges.size());
  std::vector<glm::vec3> edgeColors(E);
  for (int e = 0; e < E; ++e) {
    // Hue cycles through the full spectrum twice along the coil, with
    // saturation dropping towards the ends to create a fade-out effect.
    float t = float(e) / float(E - 1);
    float hue = std::fmod(t * 2.0f, 1.0f);
    float sat = 0.4f + 0.6f * std::sin(t * kTwoPi); // gentle pulse
    edgeColors[e] = hsv2rgb(hue, glm::clamp(sat, 0.0f, 1.0f), 1.0f);
  }

  auto* net = ps::registerCurveNetwork("spring coil – edge color", nodes, edges);
  net->setRadius(0.006f, /*isRelative=*/false);

  net->addEdgeColorQuantity("spectrum", edgeColors)->setEnabled(true);
}

// ---------------------------------------------------------------------------
int main() {
  ps::options::programName = "Polyscope RT – Curve Network Features";

  ps::init();

  addHelixNodeScalar();      // centre-left,  uses node scalar → coolwarm
  addLissajousEdgeScalar();  // right,         uses edge scalar → viridis
  addFigure8NodeColor();     // left,          uses node colors → RGB rainbow
  addSpringEdgeColor();      // centre-back,   uses edge colors → spectrum

  ps::show();
  return 0;
}
