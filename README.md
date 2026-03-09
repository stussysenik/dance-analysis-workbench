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

Open the Gradio URL printed in the terminal. The app prefers `0.0.0.0:7860` and will automatically move to the next open port if `7860` is already in use. You can still force a starting port with `PORT=7861 bun run dev`.

On Lightning, prefer the `Lightning preview URL:` printed at startup over the raw `litng.ai` port subdomain.

## Headless C-RADIOv4 Run

If you want actual model-backed outputs without the Gradio surface, use the headless runner:

```bash
python3 scripts/run_radio_headless.py "path/to/video.mp4" --start 0 --end 0.12 --dancers 2 --profile balanced --device cpu
```

Notes:

- The headless runner uses `CRADIOV4_MODEL_ID=nvidia/C-RADIOv4-SO400M` by default.
- Override the model with `--model-id nvidia/C-RADIOv4-H` if your runtime can support it.
- Outputs are written under `artifacts/runs/<timestamp>/` as `annotated.mp4`, `preview.jpg`, and `analysis.json`.
- In the current Lightning VM, CUDA may not be exposed even when PyTorch has CUDA support installed. If `torch.cuda.is_available()` is `False`, use `--device cpu`.

## Runtime profiles

Profiles are defined in [configs/runtime_profiles.json](/teamspace/studios/this_studio/configs/runtime_profiles.json).

- `fast-preview`: low-overhead deterministic fallback pipeline
- `balanced`: default local profile for analysis and overlays
- `ada-high`: higher frame budget and adapter-preferred profile intended for Ada Lovelace GPUs

## Current model path

The baseline pipeline is runnable without heavyweight weights. Model-backed integration points already exist:

- `CRadioV4Adapter`: shared visual backbone seam for local C-RADIOv4 embeddings, with optional Hugging Face loading through `CRADIOV4_MODEL_ID`
- `SapiensPoseAdapter`: helper pose/part-analysis seam for higher-quality articulation estimates

Adapters are optional. If model weights are unavailable, the workbench falls back to contour-based proposals and proxy joints while preserving the same structured outputs. When `CRADIOV4_MODEL_ID` is set and the dependencies are available, the tracker uses real C-RADIOv4 crop embeddings to improve assignment stability.

## Ingredients and Experiments

The exact implementation ingredients and the experiment matrix for baseline, runtime-profile, correction-loop, beat-alignment, and Ada bring-up testing are documented in [docs/ingredients-and-experiments.md](/teamspace/studios/this_studio/docs/ingredients-and-experiments.md).

## Current limitations

- COM is a 2D proxy derived from joints and anthropometric weights, not a lab-grade estimate.
- Beat scoring is basic and confidence-aware. If no usable audio can be extracted, the app reports a warning and skips authoritative scoring.
- The annotated preview focuses on clarity and debug visibility over polished presentation.
- The current repo does not yet include SAM3-based text or box-prompt segmentation. The shipped headless path is C-RADIOv4-backed tracking plus overlay rendering.
