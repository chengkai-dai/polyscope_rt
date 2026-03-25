#include "polyscope/rt/polyscope.h"
#include "polyscope/rt/surface_mesh.h"

int main() {
  polyscope::rt::disable();
  return polyscope::rt::isEnabled() ? 1 : 0;
}
