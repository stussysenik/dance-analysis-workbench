from __future__ import annotations

import os
import socket
from pathlib import Path

import gradio as gr

from dance_analysis.config import load_runtime_profiles
from dance_analysis.contracts import AnalyzeVideoRequest, SegmentSelection
from dance_analysis.pipeline import DanceAnalysisPipeline


OVERLAY_CHOICES = ["boxes", "skeleton", "com", "beats"]
DEFAULT_SERVER_NAME = os.getenv("HOST", "0.0.0.0")
DEFAULT_SERVER_PORT = int(os.getenv("PORT", "7860"))
PORT_SCAN_LIMIT = 20


def build_app() -> gr.Blocks:
    pipeline = DanceAnalysisPipeline()
    profiles = load_runtime_profiles()

    with gr.Blocks(title="Dance Analysis Workbench") as demo:
        gr.Markdown(
            """
            # Dance Analysis Workbench
            Upload battle footage, isolate up to two dancers, and inspect per-frame COM, motion qualities, and beat alignment.

            ## Why
            Dance footage is difficult because identity swaps, occlusion, and articulation all happen at once. This workbench keeps the pipeline local and debuggable so you can verify what the system thinks each dancer is doing frame by frame.

            ## How It Works
            1. Select a local video segment and runtime profile.
            2. The pipeline proposes dancer tracks, estimates proxy pose, and computes per-frame COM and movement metrics.
            3. The audio path is analyzed for a basic beat grid, then the app scores how closely motion events align to the beat.
            4. The result is rendered as an annotated preview, summary metrics, and structured outputs for later inspection.
            """
        )
        with gr.Row():
            video = gr.Video(label="Source video", sources=["upload"])
            with gr.Column():
                start_sec = gr.Number(label="Segment start (s)", value=0)
                end_sec = gr.Number(label="Segment end (s)", value=6)
                target_dancers = gr.Radio([1, 2], value=2, label="Dancers to isolate")
                runtime_profile = gr.Dropdown(list(profiles.keys()), value="balanced", label="Runtime profile")
                overlays = gr.CheckboxGroup(OVERLAY_CHOICES, value=OVERLAY_CHOICES, label="Overlay layers")
                correction_hints = gr.Textbox(
                    label="Correction hints",
                    lines=2,
                    placeholder="Optional seeded centers in pixels: x1,y1;x2,y2",
                )
                audio_override = gr.Textbox(label="Optional WAV override", placeholder="/path/to/audio.wav")
                analyze = gr.Button("Analyze segment", variant="primary")

        with gr.Row():
            annotated = gr.Video(label="Annotated preview")
            summary = gr.JSON(label="Run summary")
        with gr.Row():
            metrics = gr.JSON(label="Metrics")
            warnings = gr.Textbox(label="Warnings", lines=6)

        def run_analysis(
            video_path: str | None,
            start: float,
            end: float,
            dancers: int,
            profile_name: str,
            selected_overlays: list[str],
            hints: str,
            audio_path: str,
        ) -> tuple[str | None, dict, dict, str]:
            if not video_path:
                raise gr.Error("Provide a local video file")
            request = AnalyzeVideoRequest(
                video_path=Path(video_path),
                segment=SegmentSelection(start_sec=float(start), end_sec=float(end)),
                target_dancers=int(dancers),
                runtime_profile=profile_name,
                overlay_layers=selected_overlays,
                correction_hints=hints or "",
                audio_path=Path(audio_path) if audio_path else None,
            )
            result = pipeline.analyze(request)
            warning_lines = []
            for track in result.tracks:
                warning_lines.extend(track.warnings)
            warning_lines.extend(result.music.warnings)
            metrics_payload = {
                "music": result.music.model_dump(mode="json"),
                "tracks": [track.model_dump(mode="json") for track in result.tracks],
                "sample_biomech_frames": [frame.model_dump(mode="json") for frame in result.biomechanics[:8]],
            }
            return (
                str(result.render_bundle.annotated_video_path),
                result.summary,
                metrics_payload,
                "\n".join(warning_lines) or "No warnings",
            )

        analyze.click(
            run_analysis,
            inputs=[video, start_sec, end_sec, target_dancers, runtime_profile, overlays, correction_hints, audio_override],
            outputs=[annotated, summary, metrics, warnings],
        )
    return demo


def launch() -> None:
    server_port = _resolve_server_port(DEFAULT_SERVER_NAME, DEFAULT_SERVER_PORT)
    lightning_preview_url = _lightning_preview_url(server_port)
    if lightning_preview_url:
        print(f"Lightning preview URL: {lightning_preview_url}")
    build_app().launch(server_name=DEFAULT_SERVER_NAME, server_port=server_port)


def _resolve_server_port(server_name: str, preferred_port: int) -> int:
    for candidate in range(preferred_port, preferred_port + PORT_SCAN_LIMIT):
        if _port_is_available(server_name, candidate):
            if candidate != preferred_port:
                print(f"Port {preferred_port} is busy; using {candidate} instead.")
            return candidate
    raise OSError(f"Could not find an open port in range {preferred_port}-{preferred_port + PORT_SCAN_LIMIT - 1}")


def _port_is_available(server_name: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((server_name, port))
        except OSError:
            return False
    return True


def _lightning_preview_url(port: int) -> str | None:
    vscode_proxy_uri = os.getenv("VSCODE_PROXY_URI", "")
    if "{{port}}" in vscode_proxy_uri:
        return vscode_proxy_uri.replace("{{port}}", str(port))
    return None
