from __future__ import annotations

import argparse
from pathlib import Path

from dance_analysis.contracts import AnalyzeVideoRequest, SegmentSelection
from dance_analysis.pipeline import DanceAnalysisPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local dance analysis.")
    parser.add_argument("video_path")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=6.0)
    parser.add_argument("--dancers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--audio-path")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--hints", default="")
    args = parser.parse_args()

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
    print(result.render_bundle.json_path)


if __name__ == "__main__":
    main()
