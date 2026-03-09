# Package Management Policy

This project is intended to stay healthy on NixOS over time, even as the model stack evolves. The easiest way to keep that true is to keep dependency ownership strict.

## Ownership Boundaries

### Nix

`Nix` is the outer environment manager.

Use it for:

- Python interpreter selection
- `uv`, `bun`, `ffmpeg`, and shell tooling
- Native libraries and compiler/runtime dependencies
- Future CUDA toolkit integration when the Ada machine is active

Do not use it for:

- Pinning individual Python libraries already managed by `uv`
- Replacing app-level JS tooling already managed by `bun`

### uv

`uv` is the Python dependency and execution layer.

Use it for:

- App dependencies
- Test dependencies
- Locked Python resolution via `uv.lock`
- Running Python entrypoints with `uv run`

Policy:

- Every Python dependency change must update `pyproject.toml`.
- Commit `uv.lock` after dependency updates.
- Do not `pip install` packages manually into the project environment.

### Bun

`bun` is the JS task runner and future frontend dependency layer.

Use it for:

- Consistent project scripts
- Future frontend helpers or small asset pipeline tools
- Workspace-level DX commands

Policy:

- Keep Bun responsibilities narrow until a real JS frontend exists.
- Do not mirror Python package management in Bun.
- If Bun dependencies are introduced, commit the Bun lockfile.

## Operational Defaults

- Start work with `nix develop`.
- Run Python through `uv run`.
- Run project commands through `bun run` when scripts exist.
- Keep model weights and large artifacts out of git.
- Prefer adapter seams over hard-wiring environment-specific model paths.

## Dependency Update Procedure

### System layer

Edit [flake.nix](/teamspace/studios/this_studio/flake.nix), enter `nix develop`, and verify the shell still resolves cleanly.

### Python layer

Edit [pyproject.toml](/teamspace/studios/this_studio/pyproject.toml), run `uv sync --extra dev`, and verify [uv.lock](/teamspace/studios/this_studio/uv.lock) changes are committed.

### JS layer

Edit [package.json](/teamspace/studios/this_studio/package.json), run `bun install`, and commit the generated Bun lockfile once the project starts using Bun packages.

## GPU Readiness Notes

When the Ada Lovelace machine is ready:

- Extend `flake.nix` with the required CUDA-facing packages instead of relying on host-global installs.
- Keep GPU-specific model paths configurable through environment variables.
- Add a dedicated runtime profile instead of modifying the fallback profile in place.
