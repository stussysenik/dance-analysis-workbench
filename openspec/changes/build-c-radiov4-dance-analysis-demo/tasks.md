## 1. Project Scaffold

- [x] 1.1 Add Nix, `uv`, and `bun` project scaffolding with runnable dev commands
- [x] 1.2 Create the Python package layout, configuration files, and output directories for the analysis app

## 2. Analysis Contracts

- [x] 2.1 Implement typed request and result models for tracks, pose, biomechanics, audio, and rendered outputs
- [x] 2.2 Implement configuration-driven runtime profiles and pipeline settings for fallback and model-backed stages

## 3. Baseline Pipeline

- [x] 3.1 Implement video ingest, segment handling, and fallback dancer proposal/tracking analyzers
- [x] 3.2 Implement frame-level COM and motion-quality scoring from tracked joints
- [x] 3.3 Implement basic beat extraction and on-beat scoring for a selected segment

## 4. Interactive Workbench

- [x] 4.1 Build the Gradio workbench for upload, segment selection, overlay toggles, and result summaries
- [x] 4.2 Implement annotated preview rendering and human-in-the-loop rerun inputs for unstable tracks

## 5. Verification

- [x] 5.1 Add unit and integration tests covering contracts, motion scoring, beat scoring, and end-to-end analysis
- [x] 5.2 Document setup, runtime profiles, and the current model adapter path including C-RADIOv4 integration points
