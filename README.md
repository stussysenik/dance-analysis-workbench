# Dance Analysis Workbench

![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Bun](https://img.shields.io/badge/bun-1.3.10-black)
![Nix](https://img.shields.io/badge/nix-flake-5277C3)
![Gradio](https://img.shields.io/badge/gradio-6.x-orange)

Local Gradio-first workbench for isolating one or two dancers from battle footage, overlaying motion qualities, and scoring basic beat alignment. The current implementation uses a deterministic fallback pipeline with adapter seams for `C-RADIOv4` and helper human-analysis models.

## Why

Dance battle footage is hard to analyze because dancers overlap, poses get extreme, and camera quality varies. This project turns that problem into a local interactive workflow so you can inspect isolation quality, COM behavior, and basic musical timing without depending on a cloud service or a one-off research notebook.

## How It Works

The system runs in layers:

1. Load a local video segment and sample frames with a selected runtime profile.
2. Propose one or two dancer tracks with the fallback tracker or future model-backed adapters.
3. Estimate proxy pose joints, then compute 2D center-of-mass, velocity, acceleration, travel distance, and expansion per frame.
4. Extract or override audio, estimate a beat grid, and compare motion peaks against beat times for a simple on-beat score.
5. Render an annotated video plus structured JSON so the same run can be reviewed visually and programmatically.

## Stack

- `Nix` for reproducible shell setup
- `uv` for Python environment and execution
- `bun` for top-level scripts
- `Gradio` for the interactive workbench

## Long-Term Package Management

Treat package management as a layered system:

1. `Nix` owns system-level tooling and native dependencies.
2. `uv` owns Python packages and the local virtual environment.
3. `bun` owns JavaScript tooling and developer scripts only.

Rules for keeping the repo stable on NixOS:

- Add compilers, `ffmpeg`, CUDA-adjacent system libraries, and shell tools in [flake.nix](/teamspace/studios/this_studio/flake.nix), not in ad hoc shell setup.
- Add Python runtime and test dependencies in [pyproject.toml](/teamspace/studios/this_studio/pyproject.toml), then refresh `uv.lock` with `uv sync`.
- Add JavaScript-only tooling in [package.json](/teamspace/studios/this_studio/package.json); avoid duplicating Python package concerns there.
- Prefer deterministic entrypoints like `bun run dev`, `bun run test`, and `uv run ...` over undocumented manual commands.
- Keep heavyweight model weights outside the repo and pass them in through environment variables or external paths.

Recommended maintenance workflow:

```bash
nix develop
uv sync --extra dev
bun install
bun run test
```

When dependencies change:

- Update `flake.nix` for system/toolchain changes.
- Update `pyproject.toml` and commit the resulting `uv.lock` for Python changes.
- Update `package.json` and commit the resulting `bun.lock` if Bun dependencies are added later.
- Re-run tests after each layer changes so dependency drift is caught early.

## Quick start

```bash
nix develop
uv sync --extra dev
bun run dev
```

Open the Gradio URL printed in the terminal.

## Runtime profiles

Profiles are defined in [configs/runtime_profiles.json](/teamspace/studios/this_studio/configs/runtime_profiles.json).

- `fast-preview`: low-overhead deterministic fallback pipeline
- `balanced`: default local profile for analysis and overlays
- `ada-high`: higher frame budget and adapter-preferred profile intended for Ada Lovelace GPUs

## Current model path

The baseline pipeline is runnable without heavyweight weights. Model-backed integration points already exist:

- `CRadioV4Adapter`: shared visual backbone seam for local C-RADIOv4 embeddings
- `SapiensPoseAdapter`: helper pose/part-analysis seam for higher-quality articulation estimates

Adapters are optional. If model weights are unavailable, the workbench falls back to contour-based proposals and proxy joints while preserving the same structured outputs.

## Ingredients and Experiments

The exact implementation ingredients and the experiment matrix for baseline, runtime-profile, correction-loop, beat-alignment, and Ada bring-up testing are documented in [docs/ingredients-and-experiments.md](/teamspace/studios/this_studio/docs/ingredients-and-experiments.md).

## Current limitations

- COM is a 2D proxy derived from joints and anthropometric weights, not a lab-grade estimate.
- Beat scoring is basic and confidence-aware. If no usable audio can be extracted, the app reports a warning and skips authoritative scoring.
- The annotated preview focuses on clarity and debug visibility over polished presentation.
