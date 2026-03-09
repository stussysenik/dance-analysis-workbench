from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dance_analysis.contracts import AnalyzeVideoRequest, SegmentSelection
from dance_analysis.pipeline import DanceAnalysisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a headless C-RADIOv4-backed analysis segment.")
    parser.add_argument("video_path")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=4.0)
    parser.add_argument("--dancers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--audio-path")
    parser.add_argument("--overlay", action="append", default=["boxes", "skeleton", "com", "beats"])
    parser.add_argument("--hints", default="")
    parser.add_argument("--model-id", default="nvidia/C-RADIOv4-SO400M")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ["CRADIOV4_MODEL_ID"] = args.model_id
    os.environ["CRADIOV4_DEVICE"] = args.device

    request = AnalyzeVideoRequest(
        video_path=Path(args.video_path),
        segment=SegmentSelection(start_sec=args.start, end_sec=args.end),
        target_dancers=args.dancers,
        runtime_profile=args.profile,
        overlay_layers=args.overlay,
        correction_hints=args.hints,
        audio_path=Path(args.audio_path) if args.audio_path else None,
    )
    result = DanceAnalysisPipeline().analyze(request)
    print(f"Annotated video: {result.render_bundle.annotated_video_path}")
    print(f"Preview frame: {result.render_bundle.preview_frame_path}")
    print(f"Analysis json: {result.render_bundle.json_path}")
    print(f"Summary: {result.summary}")


if __name__ == "__main__":
    main()
