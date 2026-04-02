# knitgen Implementation Plan

## Positioning

`knitgen` is a standalone project between `autoknit` and `polyscope_rt`.

- `autoknit` owns knitting semantics and machine-feasible structure.
- `knitgen` owns stitch-aware geometry, yarn curves, and renderable knit objects.
- `polyscope_rt` owns rendering only.

The goal is to avoid putting knitting logic in the renderer and avoid putting rendering concerns in `autoknit`.

## Goals

1. Import `autoknit` outputs without changing `polyscope_rt`'s renderer-facing API.
2. Build a stable knitting-domain intermediate representation instead of a one-off converter.
3. Support two quality modes:
   - preview: fast curve/tube visualization
   - final: stitch-aware yarn geometry suitable for realistic rendering
4. Keep the system extensible enough for future Blender, glTF, or offline-renderer export.

## Non-Goals

- Re-implement `autoknit` scheduling or machine planning.
- Make `polyscope_rt` aware of stitch types such as knit/tuck/increase.
- Solve full yarn simulation in the first milestone.
- Depend on DCC-specific concepts such as UV workflows as the canonical representation.

## Recommended Repository Boundary

Create a new sibling project:

```text
workspace/
  autoknit/
  knitgen/
  polyscope_rt/
```

`knitgen` can later be consumed in three ways:

- as a standalone CLI
- as a C++ library
- as a thin viewer example linked against `polyscope_rt`

## Core Architecture

### Layer 1: Import

Adapters that read upstream formats and normalize them into knitgen's internal domain types.

Initial adapter:

- `autoknit .st` traced stitches

Future adapters:

- knitout-derived stitch traces
- Blender-authored stitch layouts
- custom JSON stitch meshes

### Layer 2: Topology / Domain Model

This is the canonical representation layer. It should not depend on Polyscope or renderer code.

Proposed types:

```text
TracedStitchSet
  raw imported stitches with yarn id, stitch type, direction, in/out, anchor point

YarnSequenceSet
  per-yarn ordered stitch sequences reconstructed from traced order

StitchGraph
  explicit stitch adjacency and stitch semantic labels

StitchMesh
  stitch-level surface/connectivity representation for knit-aware geometry generation

YarnCurveSet
  piecewise spline centerlines for each yarn

RenderableKnitObject
  final render payload:
    - curves for preview
    - mesh for final render
    - optional per-yarn / per-stitch material groups
```

### Layer 3: Geometry

This layer turns stitch semantics into shape.

Sub-stages:

1. `stitch layout normalization`
   - validate imported connectivity
   - reconstruct yarn ordering
   - compute local stitch frames

2. `stitch template expansion`
   - assign a canonical geometric template per stitch type
   - initial supported types:
     - knit
     - tuck
     - miss
     - increase
     - decrease

3. `curve generation`
   - connect template segments into per-yarn centerlines
   - fit splines
   - resample for preview or final output

4. `relaxation`
   - MVP: local smoothing with stitch constraints
   - later: stitch-mesh-aware or yarn-aware relaxation

5. `surface generation`
   - preview: curve radii only
   - final: sweep yarn cross-sections to triangle meshes

### Layer 4: Output / Export

Outputs should be renderer-agnostic first.

Initial outputs:

- `CurvePreview`
- `TriangleMesh`
- debug JSON dump

Renderer adapters:

- `polyscope_rt` curve network exporter
- `polyscope_rt` surface mesh exporter

Future outputs:

- OBJ
- glTF
- Alembic or custom cache format

## Why This Matches Current Practice

Modern knit graphics pipelines are usually not:

`path -> render`

They are closer to:

`stitch-aware representation -> yarn curves -> relaxation -> appearance/render`

That means `autoknit` traced paths are a starting point, not the final render object.

## Proposed Module Layout

```text
knitgen/
  CMakeLists.txt
  README.md
  include/knitgen/
    io/
      autoknit_st.h
    topology/
      traced_stitch.h
      yarn_sequence.h
      stitch_graph.h
      stitch_mesh.h
    geometry/
      stitch_frame.h
      stitch_template.h
      yarn_curve.h
      relaxation.h
      tube_sweep.h
    export/
      triangle_mesh.h
      curve_preview.h
      polyscope_rt_export.h
  src/
    io/
    topology/
    geometry/
    export/
  examples/
    autoknit_curve_preview.cpp
    autoknit_surface_preview.cpp
  tests/
    autoknit_import_test.cpp
    yarn_sequence_test.cpp
    stitch_graph_test.cpp
    stitch_template_test.cpp
    tube_sweep_test.cpp
```

## Proposed Data Contracts

### Import Contract from autoknit

For MVP, `knitgen` only assumes:

- stitch type
- stitch direction
- yarn id
- stitch-to-stitch `in/out` relations
- anchor point `at`

This matches the current `autoknit` traced stitch export and keeps the coupling minimal.

### Export Contract to polyscope_rt

`polyscope_rt` should only consume:

- `std::vector<glm::vec3>` nodes + edge list for curve preview
- `std::vector<glm::vec3>` vertices + triangle indices for final mesh
- material groups / colors

No stitch semantics cross the boundary.

## MVP Scope

The first usable version of `knitgen` should do exactly this:

1. Read an `autoknit .st` file.
2. Reconstruct yarn sequences.
3. Build a simple stitch graph.
4. Generate per-yarn centerlines with stitch-aware tangent estimation.
5. Export:
   - a `CurvePreview`
   - a swept tube mesh
6. View both in a small `polyscope_rt` example.

This gets us a clean end-to-end pipeline without committing too early to a heavy physical model.

## Milestone Plan

### Milestone 0: Project Skeleton

- create standalone `knitgen` repo
- set up CMake
- add core domain types
- add sample `autoknit` fixture data

Exit condition:

- project builds
- tests run
- fixtures load

### Milestone 1: autoknit Import + Preview Curves

- parse `.st`
- validate `in/out` consistency
- reconstruct per-yarn sequences
- generate preview centerlines
- export to `polyscope_rt::registerCurveNetwork`

Exit condition:

- traced stitches can be loaded and shown as yarn-level curves

### Milestone 2: Stitch Templates + Tube Sweep

- define canonical templates for supported stitch types
- compute local stitch frames
- generate denser centerlines
- sweep circular or elliptical cross-sections into triangle meshes

Exit condition:

- `autoknit .st` becomes a visibly knitted object, not just debug tubes

### Milestone 3: Relaxation

- add constrained smoothing
- preserve stitch ordering and branch structure
- avoid self-intersection where possible

Exit condition:

- generated knit geometry looks less rigid and less graph-like

### Milestone 4: Appearance

- material groups per yarn
- yarn flattening / twist / roughness controls
- optional normal perturbation for fiber appearance

Exit condition:

- plausible final render quality in `polyscope_rt`

### Milestone 5: Generalized Export

- OBJ / glTF export
- serialized knitgen cache
- optional Blender interchange path

Exit condition:

- knitgen output is reusable outside the viewer

## Geometry Strategy

### Preview Mode

Use a cheap but stable construction:

- one ordered centerline per yarn
- local Catmull-Rom or cubic spline interpolation
- constant or gently varying radius
- no expensive relaxation

This is the default for debugging and iteration.

### Final Mode

Use a stitch-aware build:

- derive a local frame for each stitch from incoming/outgoing neighbors
- instantiate a per-stitch geometric motif
- blend motifs into continuous yarn centerlines
- perform a relaxation pass
- sweep to mesh

This matches the long-term direction of stitch-mesh and yarn-level rendering pipelines.

## Testing Strategy

### Unit Tests

- `.st` parsing
- stitch relation validation
- yarn ordering reconstruction
- template expansion per stitch type
- tube sweep mesh topology

### Golden Tests

- compare node/edge counts for preview output
- compare triangle/vertex counts for final output
- hash stable exported JSON summaries

### Visual Tests

- small set of canonical fixtures:
  - straight strip
  - tube
  - increase/decrease patch
  - multi-yarn color sample

## Open Design Decisions

These do not block the MVP, but should be made explicit:

1. Canonical IR granularity:
   - keep both `StitchGraph` and `StitchMesh`
   - or derive `StitchMesh` on demand

2. Relaxation backend:
   - custom local optimizer first
   - or bring in a geometry library early

3. Output ownership:
   - export plain arrays only
   - or introduce a richer scene object with materials and named groups

4. Appearance scope:
   - geometry-only MVP
   - or include yarn shading controls from the first release

## Recommended First Build Order

Implement in this order:

1. `TracedStitchSet`
2. `autoknit .st` importer
3. `YarnSequenceSet`
4. `CurvePreview`
5. `polyscope_rt` curve example
6. `StitchGraph`
7. local stitch frames
8. template-expanded `YarnCurveSet`
9. `TriangleMesh` sweep
10. `polyscope_rt` mesh example

This sequence keeps progress visible and avoids blocking on the hardest geometry work too early.

## Immediate Next Step

Create the standalone `knitgen` repo and implement Milestone 0 plus Milestone 1 first.

That gives us a clean proof that the boundary is right:

`autoknit .st -> knitgen curves -> polyscope_rt`

After that, we can safely iterate on higher-fidelity stitch geometry without destabilizing either upstream project.
