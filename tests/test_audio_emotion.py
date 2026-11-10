"""
Standalone test script for VibeLingual Acoustic Emotion & Speech Feature Analysis.

Usage:
    .venv\\Scripts\\python.exe tests/test_audio_emotion.py media/malayalam_sample.wav.m4a [--multimodal]
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.audio_emotion_service import (
    extract_local_acoustic_features,
    analyze_audio_emotion,
    AudioEmotionServiceError,
)


def main():
    parser = argparse.ArgumentParser(
        description="Test VibeLingual Acoustic Emotion & Audio Feature Analysis."
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the audio file to analyze (e.g. media/malayalam_sample.wav.m4a)",
    )
    parser.add_argument(
        "--transcript",
        "-t",
        type=str,
        default=None,
        help="Optional reference transcript for verbal cross-checking",
    )
    parser.add_argument(
        "--multimodal",
        "-m",
        action="store_true",
        help="Execute live Gemini 3.8 Flash multimodal audio perception test",
    )

    args = parser.parse_args()

    if not args.audio_path.is_file():
        print(f"Error: Audio file not found at {args.audio_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("VibeLingual Acoustic Emotion Diagnostic")
    print("=" * 60)
    print(f"Audio File: {args.audio_path}")

    # 1. Local DSP Analysis
    print("\n[1] Running Local DSP Acoustic Feature Extraction...")
    try:
        dsp_features = extract_local_acoustic_features(args.audio_path)
        print("-> Local DSP Extraction Succeeded:")
        print(json.dumps(dsp_features, indent=2))
    except AudioEmotionServiceError as e:
        print(f"-> Local DSP extraction error: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Multimodal Acoustic Emotion (Optional CLI flag)
    if args.multimodal:
        print("\n[2] Running Gemini 3.8 Flash Multimodal Acoustic Emotion Analysis...")
        try:
            analysis = analyze_audio_emotion(args.audio_path, transcript=args.transcript)
            print("-> Multimodal Acoustic Analysis Succeeded:")
            print(json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False))
        except AudioEmotionServiceError as e:
            print(f"-> Multimodal analysis failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n(Note: Pass --multimodal to trigger live Gemini audio perception call)")

    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()

