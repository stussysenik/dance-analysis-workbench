# Ingredients and Experiments

This document defines the exact building blocks in the current repo and the next experiments to run when moving from the deterministic fallback path to an Ada-targeted model-backed path.

## Exact Ingredients

### Runtime and packaging

- Nix flake shell for the system boundary
- Python 3.12+ app managed by `uv`
- Bun-managed scripts for top-level commands
- Gradio for the interactive workbench
- OpenCV for video ingest and overlay rendering
- NumPy for metrics and audio signal processing
- Pydantic for stable analysis contracts

### Current pipeline ingredients

- Segment loader: reads a user-selected video range and samples frames by runtime profile
- Fallback person proposal: contour-based large-body proposals from thresholded frames
- Track stabilizer: nearest-center assignment with optional user-provided seed centers
- Pose proxy: box-derived joints through the Sapiens adapter seam
- COM calculator: anthropometric weighted 2D joint aggregation
- Motion metrics: velocity, acceleration, cumulative travel, and expansion proxy
- Audio path: extracted mono WAV or explicit WAV override
- Beat estimator: simple energy-envelope autocorrelation with confidence
- Alignment scorer: compares motion peaks to beat timestamps for a normalized on-beat score
- Renderer: annotated MP4 preview plus JSON artifact bundle

### Planned model-backed ingredients

- `C-RADIOv4` as the shared feature backbone adapter
- `Sapiens` or similar articulation model for higher-quality body joints and part reasoning
- Optional 3D refinement path for better COM approximation when visibility is high
- Better tracking stage for dancer crossings and partial occlusions

## Experiment Matrix

### Experiment 1: Fallback pipeline sanity baseline

Goal:
- Verify end-to-end correctness, schema stability, and UI clarity without external weights

Inputs:
- One clean single-dancer clip
- One mid-difficulty two-dancer practice clip

Success criteria:
- App runs locally
- JSON outputs are complete
- Overlay readability is acceptable
- Track instability warnings appear when crossings happen

### Experiment 2: Runtime profile comparison

Goal:
- Compare `fast-preview`, `balanced`, and `ada-high` on the same clip

Measurements:
- Wall-clock runtime
- Frames processed
- Track warning count
- Beat confidence
- Visual stability of COM trail

Success criteria:
- `fast-preview` is clearly faster
- `balanced` is the default quality baseline
- `ada-high` preserves semantics while increasing analysis depth

### Experiment 3: Human-in-the-loop correction value

Goal:
- Measure how much seeded correction improves identity stability in crossings

Procedure:
- Run without hints
- Re-run with seeded centers
- Compare warning counts and visual identity swaps

Success criteria:
- Corrected runs reduce instability warnings and improve continuity

### Experiment 4: Beat alignment face validity

Goal:
- Check whether the score separates on-beat and off-beat excerpts in a practically useful way

Procedure:
- Use short clips with clear rhythmic motion
- Compare scores across intentionally aligned and misaligned segments

Success criteria:
- On-beat excerpts consistently score higher than off-beat excerpts
- Low-confidence audio is flagged instead of overstated

### Experiment 5: Ada bring-up

Goal:
- Replace fallback-heavy execution with model-backed execution on the target Ada machine

Procedure:
- Add CUDA-facing dependencies to the Nix shell
- Point environment variables at local model weights
- Profile a fixed clip under `ada-high`

Measurements:
- GPU memory usage
- End-to-end runtime
- Stage-by-stage latency
- Overlay quality compared with fallback mode

Success criteria:
- Model-backed path runs without changing UI contracts
- Throughput is acceptable for local iterative analysis
- Quality improves on pose fidelity and track stability

## Minimal Logging to Capture

For each experiment, save:

- Input clip identifier
- Runtime profile
- Adapter availability states
- Processing duration
- Output artifact directory
- Track warnings
- Beat confidence and score

## Current Boundary

The repo is already suitable for baseline experiments and UI iteration. The next meaningful jump is not more architecture; it is attaching the Ada machine, model weights, and a profiler without breaking the current contracts.
