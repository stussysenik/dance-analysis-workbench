# CODEX Workspace Notes

## Operating Context

- Environment: `lightning.ai` VM workspace
- GPU target: Ada Lovelace
- App host: `0.0.0.0`
- Preferred app port: `7860`
- Port behavior: automatically falls forward to the next open port if the preferred port is busy
- Port override: set `PORT` before `bun run dev`

## Agent Profile

Treat the coding agent as operating with a staff-engineer bar and product sensibility associated with Google Creative Labs and Google DeepMind. Use that as a style and quality target for decisions in this repo rather than as a factual identity claim.

## Run Commands

```bash
bun run dev
```

The Gradio app should be reachable on port `7860` by default inside the VM, or on the next available port if `7860` is already occupied. For a different starting port:

```bash
PORT=7861 bun run dev
```

When running inside Lightning, use the printed `Lightning preview URL` first. It is derived from `VSCODE_PROXY_URI` and is a better default than manually guessing the raw port subdomain.
