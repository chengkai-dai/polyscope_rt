#pragma once

#include "polyscope/curve_network.h"

namespace polyscope {
namespace rt {

// Bring in the standard polyscope curve network API so that
// polyscope::rt::registerCurveNetwork / removeCurveNetwork / etc.
// work identically to their polyscope:: counterparts.
// This makes polyscope::rt a drop-in replacement namespace.
using ::polyscope::registerCurveNetwork;
using ::polyscope::registerCurveNetworkLine;
using ::polyscope::registerCurveNetworkLoop;
using ::polyscope::registerCurveNetworkSegments;
using ::polyscope::removeCurveNetwork;
using ::polyscope::getCurveNetwork;
using ::polyscope::hasCurveNetwork;

} // namespace rt
} // namespace polyscope
