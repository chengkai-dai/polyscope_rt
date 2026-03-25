#include "scene/direct_rt_curve_registry.h"

#include <algorithm>
#include <utility>

namespace {

std::vector<rt::RTCurveNetwork>& registry() {
  static std::vector<rt::RTCurveNetwork> value;
  return value;
}

} // namespace

void registerDirectRtCurveNetwork(rt::RTCurveNetwork network) {
  auto& reg = registry();
  // Replace existing entry with the same name.
  for (auto& existing : reg) {
    if (existing.name == network.name) {
      existing = std::move(network);
      return;
    }
  }
  reg.push_back(std::move(network));
}

void removeDirectRtCurveNetwork(const std::string& name) {
  auto& reg = registry();
  reg.erase(std::remove_if(reg.begin(), reg.end(),
                           [&](const rt::RTCurveNetwork& n) { return n.name == name; }),
            reg.end());
}

void clearDirectRtCurveNetworks() {
  registry().clear();
}

const std::vector<rt::RTCurveNetwork>& getDirectRtCurveNetworks() {
  return registry();
}
