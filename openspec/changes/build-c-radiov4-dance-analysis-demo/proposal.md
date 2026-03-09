## Why

Battle dance footage is difficult to analyze reliably because dancers cross paths, poses are extreme, and the footage quality is inconsistent. A local, reproducible workbench is needed now so C-RADIOv4 can be evaluated as a practical visual backbone for dancer isolation, motion analysis, and music alignment instead of remaining a paper-only capability.

## What Changes

- Add a NixOS-friendly local project scaffold using `uv` for Python inference code and `bun` for developer orchestration.
- Add an interactive Gradio workbench that ingests local video, exposes a segment selector, and renders overlays for tracks, pose, COM, and beat alignment.
- Add a layered analysis pipeline that uses C-RADIOv4 as the shared feature backbone and supports helper adapters for person detection, tracking, pose, and optional 3D refinement.
- Add per-frame biomechanics outputs centered on a statistically approximated COM, plus basic beat grid and on-beat scoring for dancer segments.
- Add runtime profiles, caching, and failure/confidence reporting suitable for a single Ada Lovelace GPU.

## Capabilities

### New Capabilities
- `interactive-dance-analysis-workbench`: Interactive upload, review, correction, and visualization workflow for one or two dancers in local video.
- `dancer-isolation-pipeline`: Layered video analysis pipeline for segmenting, tracking, and stabilizing one or two dancer identities with confidence-aware outputs.
- `motion-and-beat-scoring`: Frame-level motion metrics, COM proxies, and basic beat alignment scoring with visualized confidence.

### Modified Capabilities
- None.

## Impact

- Adds project infrastructure files for Nix, Python, and Bun.
- Adds a Python application package, Gradio app, pipeline modules, configuration profiles, and tests.
- Introduces model adapter interfaces for C-RADIOv4 and helper models such as Sapiens-compatible pose/segmentation stages.
- Defines JSON output contracts for frame-level tracks, pose, biomechanics, and music metrics.
