## Context

The workspace currently contains only a minimal OpenSpec scaffold and no application code. The target system must be reproducible on NixOS, use `uv` and `bun`, and produce a working local demo rather than a research note. The design needs to preserve a clear path from a mid-difficulty practice-footage demo to broader footage support without prematurely optimizing for distributed infrastructure or lab-grade biomechanics fitting.

## Goals / Non-Goals

**Goals:**
- Deliver a runnable Gradio-first demo that analyzes a local dance video and produces annotated outputs.
- Use C-RADIOv4 as the shared visual backbone behind a pluggable pipeline rather than binding the system to a single monolithic model.
- Keep the core metrics deterministic enough for testing by using confidence-aware 2D-first approximations.
- Make the system easy to extend with stronger helper models, including Sapiens and optional 3D fitting stages.

**Non-Goals:**
- Full paper-faithful reproduction of every C-RADIOv4 benchmark in the first iteration.
- Production deployment, multi-GPU orchestration, or remote job management.
- Guaranteed lab-grade COM estimation from monocular footage.

## Decisions

### Python-first inference core with Bun orchestration
The analysis pipeline, metrics, and Gradio UI will live in Python so the project can use the existing vision and audio tooling directly. `bun` will provide a small command surface for local DX, repeatable scripts, and future frontend extensions without forcing a separate web application in v1.

Alternative considered:
- Separate Bun frontend and Python API. Rejected for v1 because it adds coordination overhead before the analysis contract is stable.

### Adapter-based layered pipeline
The pipeline will be divided into ingest, visual analysis, kinematic scoring, audio analysis, and rendering layers. Each layer will exchange typed records and write cached intermediates to disk so reruns after manual review can be partial.

Alternative considered:
- One end-to-end monolithic inference call. Rejected because it makes debugging, testing, and model substitution harder.

### C-RADIOv4 backbone plus helper stages
C-RADIOv4 will provide a shared feature extraction path and be represented as a dedicated adapter. Helper stages for detection, tracking, segmentation, and pose will be isolated behind interfaces so the project can start with heuristic or lightweight implementations and later swap in stronger models like Sapiens without changing the app contract.

Alternative considered:
- Pure C-RADIOv4-only pipeline. Rejected because the immediate goal is a usable analysis demo, not a constrained ablation study.

### 2D-first biomechanics with optional 3D refinement
The default metrics will use 2D joints and simple anthropometric mass weights to estimate COM, velocity, acceleration, and related movement qualities. A later high-confidence refinement path can attach 3D fitting without becoming a v1 dependency.

Alternative considered:
- Full SMPL-X-first design. Rejected for v1 because it is less robust on messy footage and would dominate implementation complexity.

### Confidence-aware graceful degradation
Every stage will emit confidence values and explanatory flags. When a stage is uncertain, the system will surface that uncertainty in both JSON outputs and the visualization instead of silently inventing precise values.

Alternative considered:
- Fail hard on low-confidence frames. Rejected because interactive footage review benefits more from visible uncertainty than from hard stops.

## Risks / Trade-offs

- [Model availability differs across machines] -> Keep adapters optional and provide deterministic fallback analyzers for local development.
- [Dance footage can break identity tracking during crossings] -> Preserve manual correction hooks and emit track instability warnings.
- [Beat estimation may be noisy in crowd-heavy footage] -> Limit v1 to basic beat/on-beat scoring with visible confidence and warnings.
- [Ada-specific optimization cannot be fully validated in this environment] -> Provide runtime profiles and profiling hooks, then verify on target hardware later.

## Migration Plan

Create the scaffold, ship the baseline demo with deterministic adapters, and keep model-backed stages behind configuration flags. This allows the repo to remain runnable without heavyweight weights while preserving a clean path to stronger local inference profiles.

## Open Questions

- Exact external model weights to enable by default once a target Ada runtime is available.
- Whether optional 3D refinement should use SMPL-X directly or an adapter built on a more deployment-friendly fitting stack.
