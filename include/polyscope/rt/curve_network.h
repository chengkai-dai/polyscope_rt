#pragma once

#include <string>
#include <utility>
#include <vector>

#include "glm/glm.hpp"
#include "polyscope/curve_network.h"
#include "polyscope/rt/plugin.h"

namespace polyscope {
namespace rt {

// Bring in the standard polyscope template overloads so that
//   polyscope::rt::registerCurveNetwork(name, nodes, edges)
// with standard polyscope edge types (e.g. std::vector<std::array<size_t,2>>)
// works identically to polyscope::registerCurveNetwork and returns
// polyscope::CurveNetwork*.  This makes polyscope::rt a drop-in replacement.
using ::polyscope::registerCurveNetwork;
using ::polyscope::registerCurveNetworkLine;
using ::polyscope::registerCurveNetworkLoop;
using ::polyscope::registerCurveNetworkSegments;
using ::polyscope::removeCurveNetwork;
using ::polyscope::getCurveNetwork;
using ::polyscope::hasCurveNetwork;

// ---------------------------------------------------------------------------
// RT-enhanced registration
//
// Use this variant when you want to control tube radius, base color, and
// unlit shading directly at registration time, bypassing Polyscope's GUI
// material system.  Edges are pairs of (tail_index, tip_index).
//
// Both the RT renderer AND Polyscope's OpenGL fallback are registered, so
// disabling RT renders the same curves via OpenGL automatically.
// ---------------------------------------------------------------------------
void registerCurveNetwork(const std::string& name,
                          const std::vector<glm::vec3>& nodes,
                          const std::vector<std::pair<uint32_t, uint32_t>>& edges,
                          float radius = 0.005f,
                          glm::vec4 color = {0.8f, 0.8f, 0.8f, 1.0f},
                          bool unlit = false);

// Remove a directly-registered RT curve network by name.
void removeCurveNetworkRT(const std::string& name);

// Remove all directly-registered RT curve networks.
void clearCurveNetworksRT();

} // namespace rt
} // namespace polyscope
