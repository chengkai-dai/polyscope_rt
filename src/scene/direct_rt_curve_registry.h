#pragma once

#include <string>
#include <vector>

#include "rendering/ray_tracing_types.h"

// Direct RT curve registry: stores RTCurveNetwork objects registered directly
// via polyscope::rt::registerCurveNetwork(), bypassing Polyscope's structure system.

void registerDirectRtCurveNetwork(rt::RTCurveNetwork network);
void removeDirectRtCurveNetwork(const std::string& name);
void clearDirectRtCurveNetworks();
const std::vector<rt::RTCurveNetwork>& getDirectRtCurveNetworks();
