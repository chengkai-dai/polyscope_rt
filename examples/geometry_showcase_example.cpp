// geometry_showcase_example.cpp
//
// A comprehensive showcase of polyscope_rt surface-mesh features.
// All geometry is generated procedurally — no external file required.
//
// Scene layout (top view, camera at front-upper):
//
//   z = -1.4  ─ Row A: Metals         (Gold, Silver, Copper, Chrome, Mirror)
//   z = -0.7  ─ Row B: Dielectrics    (Plastic, GlossyPlastic, Rubber, Marble, Ceramic)
//   z =  0.0  ─ Row C: Transmissive   (Glass, FrostedGlass, Water, TintedGlass, Diamond)
//   z = +0.7  ─ Row D: Transparent    (Transparent 0.15 … 0.9 in five steps)
//
//   Left  (x ≈ -2.0)  ─ Mesh quantity: vertex scalar + isolines  (colormap viridis)
//   Right (x ≈ +2.0)  ─ Mesh quantity: face color (per-face RGB)
//   Far   (z ≈ +1.6)  ─ Mesh quantity: vertex color + wireframe
//   Center top          ─ Emissive sphere (visible light)
//   Front right         ─ Point cloud with per-point scalar colormap
//
// Lighting:
//   • Soft warm directional main light from upper-left
//   • Cold blue point light behind the scene
//   • Warm orange area light above the central axis

#include <array>
#include <cmath>
#include <vector>

#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "polyscope/point_cloud.h"
#include "polyscope/rt/material_library.h"
#include "polyscope/rt/polyscope.h"
#include "polyscope/rt/surface_mesh.h"

namespace ps = polyscope::rt;
using glm::vec3;
using glm::vec4;

static constexpr float kPi    = glm::pi<float>();
static constexpr float kTwoPi = glm::two_pi<float>();

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

struct Mesh {
  std::vector<glm::vec3>            verts;
  std::vector<std::array<size_t,3>> faces;
};

// UV sphere centred at 'centre' with the given radius and subdivision.
static Mesh makeSphere(glm::vec3 centre, float radius,
                       int latDiv = 32, int lonDiv = 48) {
  Mesh m;
  for (int lat = 0; lat <= latDiv; ++lat) {
    float theta = kPi * float(lat) / float(latDiv);   // 0 … π
    for (int lon = 0; lon <= lonDiv; ++lon) {
      float phi = kTwoPi * float(lon) / float(lonDiv); // 0 … 2π
      glm::vec3 p = {
        radius * std::sin(theta) * std::cos(phi),
        radius * std::cos(theta),
        radius * std::sin(theta) * std::sin(phi)
      };
      m.verts.push_back(centre + p);
    }
  }
  auto idx = [&](int la, int lo) -> size_t {
    return size_t(la * (lonDiv + 1) + lo);
  };
  for (int la = 0; la < latDiv; ++la) {
    for (int lo = 0; lo < lonDiv; ++lo) {
      m.faces.push_back({idx(la,lo),   idx(la+1,lo),   idx(la,  lo+1)});
      m.faces.push_back({idx(la,lo+1), idx(la+1,lo),   idx(la+1,lo+1)});
    }
  }
  return m;
}

// Simple flat torus (tube on a ring) centred at 'centre', ring radius R,
// tube radius r.
static Mesh makeTorus(glm::vec3 centre, float R, float r,
                      int ringDiv = 48, int tubeDiv = 24) {
  Mesh m;
  for (int u = 0; u <= ringDiv; ++u) {
    float phi = kTwoPi * float(u) / float(ringDiv);
    for (int v = 0; v <= tubeDiv; ++v) {
      float theta = kTwoPi * float(v) / float(tubeDiv);
      glm::vec3 p = {
        (R + r * std::cos(theta)) * std::cos(phi),
        r * std::sin(theta),
        (R + r * std::cos(theta)) * std::sin(phi)
      };
      m.verts.push_back(centre + p);
    }
  }
  auto idx = [&](int u, int v) -> size_t {
    return size_t(u * (tubeDiv + 1) + v);
  };
  for (int u = 0; u < ringDiv; ++u) {
    for (int v = 0; v < tubeDiv; ++v) {
      m.faces.push_back({idx(u,v),   idx(u+1,v),   idx(u,  v+1)});
      m.faces.push_back({idx(u,v+1), idx(u+1,v),   idx(u+1,v+1)});
    }
  }
  return m;
}

// Thin disc (flat cylinder cap) — used for the area-light quad visualisation.
static Mesh makeDisc(glm::vec3 centre, float radius, int divs = 32) {
  Mesh m;
  m.verts.push_back(centre); // centre vertex 0
  for (int i = 0; i <= divs; ++i) {
    float angle = kTwoPi * float(i) / float(divs);
    m.verts.push_back(centre + glm::vec3{radius * std::cos(angle), 0.0f,
                                         radius * std::sin(angle)});
  }
  for (int i = 0; i < divs; ++i) {
    m.faces.push_back({0, size_t(i + 1), size_t(i + 2)});
  }
  return m;
}

// HSV → RGB for rainbow colours.
static glm::vec3 hsv(float h, float s, float v) {
  float c = v * s;
  float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
  float m = v - c;
  glm::vec3 rgb;
  switch (int(h * 6.0f) % 6) {
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
// Section A: row of metal-material spheres
// ---------------------------------------------------------------------------
static void addMetalRow() {
  constexpr float z    = -1.4f;
  constexpr float r    = 0.28f;
  const float xs[]     = {-1.4f, -0.7f, 0.0f, 0.7f, 1.4f};
  const char* names[]  = {"metal-gold","metal-silver","metal-copper",
                           "metal-chrome","metal-mirror"};
  const ps::MaterialPreset presets[] = {
    ps::Gold(), ps::Silver(), ps::Copper(), ps::Chrome(), ps::Mirror()
  };

  for (int i = 0; i < 5; ++i) {
    auto m = makeSphere({xs[i], 0.0f, z}, r);
    ps::registerSurfaceMesh(names[i], m.verts, m.faces);
    ps::applyMaterial(names[i], presets[i]);
  }
}

// ---------------------------------------------------------------------------
// Section B: row of dielectric-material spheres
// ---------------------------------------------------------------------------
static void addDielectricRow() {
  constexpr float z    = -0.7f;
  constexpr float r    = 0.28f;
  const float xs[]     = {-1.4f, -0.7f, 0.0f, 0.7f, 1.4f};
  const char* names[]  = {"diel-plastic","diel-glossy","diel-rubber",
                           "diel-ceramic","diel-marble"};
  const ps::MaterialPreset presets[] = {
    ps::Plastic(),        ps::GlossyPlastic(),  ps::Rubber(),
    ps::Ceramic(),        ps::Marble()
  };

  for (int i = 0; i < 5; ++i) {
    auto m = makeSphere({xs[i], 0.0f, z}, r);
    ps::registerSurfaceMesh(names[i], m.verts, m.faces);
    ps::applyMaterial(names[i], presets[i]);
  }
}

// ---------------------------------------------------------------------------
// Section C: row of transmissive / refractive spheres
// ---------------------------------------------------------------------------
static void addTransmissiveRow() {
  constexpr float z    = 0.0f;
  constexpr float r    = 0.28f;
  const float xs[]     = {-1.4f, -0.7f, 0.0f, 0.7f, 1.4f};
  const char* names[]  = {"trans-glass","trans-frosted","trans-water",
                           "trans-tinted","trans-diamond"};
  const ps::MaterialPreset presets[] = {
    ps::Glass(),
    ps::FrostedGlass(),
    ps::Water(),
    ps::TintedGlass({0.6f, 0.85f, 1.0f}),  // pale-blue tint
    ps::Diamond()
  };

  for (int i = 0; i < 5; ++i) {
    auto m = makeSphere({xs[i], 0.0f, z}, r);
    ps::registerSurfaceMesh(names[i], m.verts, m.faces);
    ps::applyMaterial(names[i], presets[i]);
  }
}

// ---------------------------------------------------------------------------
// Section D: row of alpha-transparent spheres (showcases stochastic opacity)
// Each sphere wraps a small solid GlossyPlastic ball inside it to make
// the transparency visible.
// ---------------------------------------------------------------------------
static void addTransparentRow() {
  constexpr float z = 0.7f;
  constexpr float r = 0.28f;
  const float xs[]  = {-1.4f, -0.7f, 0.0f, 0.7f, 1.4f};

  const glm::vec3 tintColors[] = {
    {0.9f, 0.9f, 0.9f},
    {0.4f, 0.8f, 1.0f},
    {0.6f, 1.0f, 0.5f},
    {1.0f, 0.6f, 0.3f},
    {0.9f, 0.4f, 0.8f}
  };
  const float alphas[] = {0.20f, 0.35f, 0.50f, 0.65f, 0.80f};
  const glm::vec3 innerColors[] = {
    {1.0f, 0.3f, 0.3f},
    {0.3f, 1.0f, 0.3f},
    {0.3f, 0.4f, 1.0f},
    {1.0f, 0.9f, 0.2f},
    {0.9f, 0.3f, 0.9f}
  };

  for (int i = 0; i < 5; ++i) {
    // Outer transparent shell
    std::string outerName = "alpha-outer-" + std::to_string(i);
    auto outerMesh = makeSphere({xs[i], 0.0f, z}, r);
    ps::registerSurfaceMesh(outerName, outerMesh.verts, outerMesh.faces);
    ps::applyMaterial(outerName, ps::Transparent(tintColors[i], alphas[i]));

    // Inner solid sphere
    std::string innerName = "alpha-inner-" + std::to_string(i);
    auto innerMesh = makeSphere({xs[i], 0.0f, z}, r * 0.5f);
    ps::registerSurfaceMesh(innerName, innerMesh.verts, innerMesh.faces);
    ps::applyMaterial(innerName, ps::GlossyPlastic(innerColors[i]));
  }
}

// ---------------------------------------------------------------------------
// Section E: Mesh with vertex scalar + isolines  (left side)
// ---------------------------------------------------------------------------
static void addVertexScalarMesh() {
  constexpr glm::vec3 centre = {-2.3f, 0.0f, -0.35f};
  auto m = makeTorus(centre, 0.38f, 0.14f);

  int N = int(m.verts.size());
  std::vector<float> scalars(N);
  for (int i = 0; i < N; ++i) {
    // Signed distance from XZ plane (height), normalised to [0,1]
    float y = m.verts[i].y - centre.y;
    scalars[i] = (y / 0.14f) * 0.5f + 0.5f;
  }

  auto* mesh = ps::registerSurfaceMesh("qty-vertex-scalar", m.verts, m.faces);
  auto* qty  = mesh->addVertexScalarQuantity("height", scalars);
  qty->setColorMap("viridis")
     ->setMapRange({0.0f, 1.0f})
     ->setIsolinesEnabled(true)
     ->setIsolinePeriod(0.1, /*relative=*/false)
     ->setIsolineDarkness(0.6)
     ->setEnabled(true);
}

// ---------------------------------------------------------------------------
// Section F: Mesh with per-face colour quantity  (right side)
// ---------------------------------------------------------------------------
static void addFaceColorMesh() {
  constexpr glm::vec3 centre = {2.3f, 0.0f, -0.35f};
  auto m = makeTorus(centre, 0.38f, 0.14f);

  int F = int(m.faces.size());
  std::vector<glm::vec3> faceColors(F);
  for (int i = 0; i < F; ++i) {
    faceColors[i] = hsv(float(i) / float(F), 1.0f, 0.9f);
  }

  auto* mesh = ps::registerSurfaceMesh("qty-face-color", m.verts, m.faces);
  mesh->addFaceColorQuantity("rainbow", faceColors)->setEnabled(true);
}

// ---------------------------------------------------------------------------
// Section G: Mesh with per-vertex colour + wireframe overlay  (far centre)
// ---------------------------------------------------------------------------
static void addVertexColorWireframeMesh() {
  constexpr glm::vec3 centre = {0.0f, 0.0f, 1.6f};
  auto m = makeSphere(centre, 0.38f, 24, 32);

  int N = int(m.verts.size());
  std::vector<glm::vec3> vertColors(N);
  for (int i = 0; i < N; ++i) {
    glm::vec3 dir = glm::normalize(m.verts[i] - centre);
    // Map normal direction to hue (atan2 in XZ plane), brightness from Y.
    float hue    = (std::atan2(dir.z, dir.x) / kTwoPi + 1.0f);
    float bright = dir.y * 0.5f + 0.5f;
    vertColors[i] = hsv(std::fmod(hue, 1.0f), 0.8f, 0.6f + 0.4f * bright);
  }

  auto* mesh = ps::registerSurfaceMesh("qty-vert-color-wire", m.verts, m.faces);
  mesh->addVertexColorQuantity("dir-rainbow", vertColors)->setEnabled(true);
  mesh->setEdgeWidth(0.7);                          // enable wireframe overlay
  mesh->setEdgeColor({0.05f, 0.05f, 0.05f});
}

// ---------------------------------------------------------------------------
// Section H: Mesh with per-face scalar + contour lines  (far left)
// ---------------------------------------------------------------------------
static void addFaceScalarContourMesh() {
  constexpr glm::vec3 centre = {-2.3f, 0.0f, 1.3f};
  auto m = makeSphere(centre, 0.38f, 24, 32);

  // Face scalar: distance of face centroid from the Y axis (ring-like bands).
  int F = int(m.faces.size());
  std::vector<float> faceScalars(F);
  for (int fi = 0; fi < F; ++fi) {
    glm::vec3 c = (m.verts[m.faces[fi][0]] +
                   m.verts[m.faces[fi][1]] +
                   m.verts[m.faces[fi][2]]) / 3.0f - centre;
    faceScalars[fi] = std::sqrt(c.x*c.x + c.z*c.z) / 0.38f;  // 0 at poles
  }

  auto* mesh = ps::registerSurfaceMesh("qty-face-scalar-contour", m.verts, m.faces);
  auto* qty  = mesh->addFaceScalarQuantity("ring-dist", faceScalars);
  qty->setColorMap("coolwarm")
     ->setIsolinesEnabled(true)
     ->setIsolineStyle(polyscope::IsolineStyle::Contour)
     ->setIsolinePeriod(0.12, /*relative=*/false)
     ->setIsolineContourThickness(0.3)
     ->setEnabled(true);
}

// ---------------------------------------------------------------------------
// Section I: Emissive sphere (acts as a visible light source)
// ---------------------------------------------------------------------------
static void addEmissiveSphere() {
  auto m = makeSphere({0.0f, 1.5f, -0.3f}, 0.12f, 16, 24);
  ps::registerSurfaceMesh("emissive-sun", m.verts, m.faces);
  ps::applyMaterial("emissive-sun", ps::Emissive({1.0f, 0.95f, 0.8f}, 8.0f));
}

// ---------------------------------------------------------------------------
// Section J: Rainbow point cloud (right side)
// ---------------------------------------------------------------------------
static void addPointCloud() {
  constexpr int N      = 800;
  constexpr float rMax = 0.45f;
  constexpr glm::vec3 centre = {2.3f, 0.0f, 1.3f};

  std::vector<glm::vec3> pts(N);
  std::vector<float>     scalars(N);

  // Fibonacci sphere distribution for uniform point cloud.
  const float goldenRatio = (1.0f + std::sqrt(5.0f)) * 0.5f;
  for (int i = 0; i < N; ++i) {
    float t     = float(i) / float(N - 1);
    float theta = std::acos(1.0f - 2.0f * t);
    float phi   = kTwoPi * float(i) / goldenRatio;
    float r     = rMax * std::pow(t, 0.33f);  // denser at centre
    pts[i]      = centre + r * glm::vec3{std::sin(theta) * std::cos(phi),
                                          std::cos(theta),
                                          std::sin(theta) * std::sin(phi)};
    scalars[i]  = t; // [0,1] from bottom to top
  }

  auto* pc = ps::registerPointCloud("point-cloud", pts);
  pc->setPointRadius(0.008f, /*relative=*/false);
  pc->addScalarQuantity("height", scalars)
    ->setColorMap("plasma")
    ->setEnabled(true);
}

// ---------------------------------------------------------------------------
// Section K: Mirror-ball that demonstrates reflections
// ---------------------------------------------------------------------------
static void addMirrorBall() {
  auto m = makeSphere({0.0f, 0.0f, -1.4f}, 0.36f, 48, 64);
  ps::registerSurfaceMesh("mirror-ball", m.verts, m.faces);
  ps::applyMaterial("mirror-ball", ps::Mirror());
}

// ---------------------------------------------------------------------------
int main() {
  ps::options::programName = "Polyscope RT – Geometry Feature Showcase";

  ps::init();

  // ── Lighting ────────────────────────────────────────────────────────────
  // Bright white sky so bounce rays reflect a light environment (critical for metals).
  ps::setBackgroundColor({0.85f, 0.90f, 0.95f});

  // Warm main directional light from upper-left front.
  ps::setMainLight({-0.5f, -1.0f, 0.4f},
                   {1.0f, 0.95f, 0.85f},
                   2.0f);

  // Cold-blue fill light in front of Row A/B.
  ps::addPointLight({0.0f, 1.8f, -4.0f},
                    {0.5f, 0.65f, 1.0f},
                    5.0f);

  // Warm orange point light on the right side.
  ps::addPointLight({2.5f, 1.5f, -0.3f},
                    {1.0f, 0.6f, 0.2f},
                    4.0f);

  // Cold fill from the left, aimed at the metal / dielectric rows.
  ps::addPointLight({-2.5f, 1.5f, -1.0f},
                    {0.8f, 0.9f, 1.0f},
                    3.5f);

  // Large area light above spanning all four rows.
  // Centre (z=-0.35), v-half-width 1.4 → covers z ≈ -1.75 … +1.05.
  ps::setAreaLight({0.0f, 2.2f, -0.35f},
                   {1.2f, 0.0f, 0.0f},
                   {0.0f, 0.0f, 1.4f},
                   {1.5f, 1.46f, 1.43f});   // slightly brighter warm-white emission

  ps::setAmbientFloor(0.07f);
  // Environment tint: bright sky-blue, high intensity so metals pick up IBL.
  ps::setEnvironment({0.85f, 0.90f, 1.0f}, 0.60f);

  // ── Geometry ────────────────────────────────────────────────────────────
  addMetalRow();              // Row A: Gold, Silver, Copper, Chrome, Mirror
  addDielectricRow();         // Row B: Plastic, GlossyPlastic, Rubber, Ceramic, Marble
  addTransmissiveRow();       // Row C: Glass, FrostedGlass, Water, TintedGlass, Diamond
  addTransparentRow();        // Row D: semi-transparent shells with solid inner balls

  addVertexScalarMesh();      // Left torus: vertex scalar + isolines (viridis)
  addFaceColorMesh();         // Right torus: per-face rainbow colours

  addVertexColorWireframeMesh();   // Far centre sphere: vertex colour + wireframe
  addFaceScalarContourMesh();      // Far left sphere: face scalar + contour lines

  addMirrorBall();            // Large mirror ball at back-centre
  addEmissiveSphere();        // Glowing sphere above scene

  addPointCloud();            // Rainbow Fibonacci point cloud (right back)

  ps::show();
  return 0;
}
