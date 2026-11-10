"""
End-to-End Voice Translation Pipeline for VibeLingual.

Chains the core VibeLingual components:
    Stage 1: Speech-to-Text (Gnani Prisma v2.5) -> Malayalam transcript
    Stage 2: Acoustic & Contextual Intelligence (Gemini 3.8 Flash) -> Natural English translation + emotional/cultural analysis
    Stage 3: Emotion-Expressive Voice Synthesis (Gnani Timbre v2.5) -> English audio synthesis (WAV)

Usage:
    .venv\\Scripts\\python.exe tests/test_pipeline.py [--input media/malayalam_sample.wav.m4a] [--voice Kaveri]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root directory to path to ensure services can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.stt_service import transcribe_audio, STTServiceError
from services.intelligence_service import analyze_and_translate, IntelligenceServiceError
from services.tts_service import synthesize_speech, TTSServiceError
from services.audio_emotion_service import extract_local_acoustic_features


def run_pipeline(
    input_audio_path: Path,
    output_audio_path: Path,
    output_metadata_path: Path,
    source_language: str = "ml-IN",
    target_language: str = "en",
    tts_language_code: str = "en-IN",
    tts_voice: str = "Kaveri",
    tts_speed: float = 1.0,
) -> bool:
    """Executes the full end-to-end voice translation pipeline.

    Returns:
        bool: True if all stages succeeded, False otherwise.
    """
    print("=" * 70)
    print("VibeLingual End-to-End Pipeline: Beyond Words. Beyond Translation.")
    print("=" * 70)
    print(f"Input Audio          : {input_audio_path}")
    print(f"Output Audio Target  : {output_audio_path}")
    print(f"Metadata Target      : {output_metadata_path}")
    print(f"Language Flow        : {source_language} -> {target_language} (TTS: {tts_language_code}, Voice: {tts_voice})")
    print("=" * 70)

    # Pre-flight check: Verify input exists
    if not input_audio_path.is_file():
        print(f"\n[PIPELINE ABORTED] Input audio file does not exist: {input_audio_path}", file=sys.stderr)
        return False

    # Pre-flight check: Ensure target outputs do not already exist (never overwrite)
    if output_audio_path.exists():
        print(f"\n[PIPELINE ABORTED] Target output audio already exists: {output_audio_path}. Overwrite is disallowed.", file=sys.stderr)
        return False

    if output_metadata_path.exists():
        print(f"\n[PIPELINE ABORTED] Target metadata file already exists: {output_metadata_path}. Overwrite is disallowed.", file=sys.stderr)
        return False

    pipeline_start_time = time.time()
    timing_records = {}

    # --------------------------------------------------------------------------
    # STAGE 1: Speech-to-Text (Gnani Prisma v2.5)
    # --------------------------------------------------------------------------
    print("\n[STAGE 1/3] Converting speech to text with Gnani Prisma v2.5...")
    stt_start = time.time()
    try:
        stt_result = transcribe_audio(
            audio_path=input_audio_path,
            language_code=source_language,
        )
        transcript = stt_result.get("transcript", "").strip()
        timing_records["stt_seconds"] = round(time.time() - stt_start, 2)

        if not transcript:
            print("[STAGE 1 FAILED] STT returned an empty transcript.", file=sys.stderr)
            return False

        print(f"-> STT Succeeded ({timing_records['stt_seconds']}s)")
        print(f"-> Original Malayalam Transcript: \"{transcript}\"")

    except STTServiceError as err:
        print(f"[STAGE 1 FAILED] STT processing error: {err}", file=sys.stderr)
        return False
    except Exception as unexpected:
        print(f"[STAGE 1 FAILED] Unexpected error in STT stage: {unexpected}", file=sys.stderr)
        return False

    # --------------------------------------------------------------------------
    # STAGE 2: Acoustic & Contextual Intelligence (Google Gemini 3.8 Flash)
    # --------------------------------------------------------------------------
    print("\n[STAGE 2/3] Analyzing vocal acoustics, emotion & translating with Gemini 3.8 Flash...")
    gemini_start = time.time()
    try:
        # Pre-extract local DSP acoustic features
        local_dsp = extract_local_acoustic_features(input_audio_path)

        analysis_result = analyze_and_translate(
            text=transcript,
            source_language=source_language,
            target_language=target_language,
            audio_path=input_audio_path,
            audio_emotion_data=local_dsp,
        )
        timing_records["intelligence_seconds"] = round(time.time() - gemini_start, 2)

        natural_translation = analysis_result.get("natural_translation", "").strip()
        literal_translation = analysis_result.get("literal_translation", "").strip()
        emotion = analysis_result.get("emotion", "Calm / Neutral")
        confidence = analysis_result.get("emotion_confidence", "medium")
        intensity = analysis_result.get("emotional_intensity", "moderate")
        intent = analysis_result.get("conversational_intent", "")
        cultural_context = analysis_result.get("cultural_context", "")
        delivery_style = analysis_result.get("suggested_delivery_style", "Natural")
        recommended_speed = analysis_result.get("tts_speed_recommendation", tts_speed)

        if not natural_translation:
            print("[STAGE 2 FAILED] Gemini did not generate a natural translation.", file=sys.stderr)
            return False

        print(f"-> Intelligence Succeeded ({timing_records['intelligence_seconds']}s)")
        print(f"-> Literal Translation   : \"{literal_translation}\"")
        print(f"-> Natural Translation   : \"{natural_translation}\"")
        print(f"-> Detected Emotion      : {emotion} (Confidence: {confidence}, Intensity: {intensity})")
        print(f"-> Conversational Intent : {intent}")
        print(f"-> Delivery Style        : {delivery_style} (Speed factor: {recommended_speed}x)")
        if cultural_context and cultural_context.lower() != "none":
            print(f"-> Cultural Context      : {cultural_context}")

    except IntelligenceServiceError as err:
        print(f"[STAGE 2 FAILED] Gemini analysis error: {err}", file=sys.stderr)
        return False
    except Exception as unexpected:
        print(f"[STAGE 2 FAILED] Unexpected error in Intelligence stage: {unexpected}", file=sys.stderr)
        return False

    # --------------------------------------------------------------------------
    # STAGE 3: Emotion-Expressive Text-to-Speech (Gnani Timbre v2.5)
    # --------------------------------------------------------------------------
    effective_speed = recommended_speed if (tts_speed == 1.0) else tts_speed
    print(f"\n[STAGE 3/3] Synthesizing English speech with Gnani Timbre v2.5 (Speed: {effective_speed}x)...")
    tts_start = time.time()
    try:
        audio_bytes = synthesize_speech(
            text=natural_translation,
            language_code=tts_language_code,
            voice=tts_voice,
            speed=effective_speed,
            emotion=emotion,
        )
        timing_records["tts_seconds"] = round(time.time() - tts_start, 2)

        # Ensure output directory exists
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_audio_path, "wb") as f:
            f.write(audio_bytes)

        file_size_kb = len(audio_bytes) / 1024
        print(f"-> TTS Succeeded ({timing_records['tts_seconds']}s)")
        print(f"-> Synthesized Voice : {tts_voice} ({tts_language_code}) @ {effective_speed}x")
        print(f"-> Audio Size        : {file_size_kb:.1f} KB")
        print(f"-> Saved Audio To    : {output_audio_path}")

    except TTSServiceError as err:
        print(f"[STAGE 3 FAILED] TTS synthesis error: {err}", file=sys.stderr)
        return False
    except Exception as unexpected:
        print(f"[STAGE 3 FAILED] Unexpected error in TTS stage: {unexpected}", file=sys.stderr)
        return False

    # --------------------------------------------------------------------------
    # SAVE METADATA & FINAL REPORT
    # --------------------------------------------------------------------------
    total_duration = round(time.time() - pipeline_start_time, 2)
    timing_records["total_pipeline_seconds"] = total_duration

    pipeline_metadata = {
        "pipeline_status": "SUCCESS",
        "input_file": str(input_audio_path),
        "output_audio_file": str(output_audio_path),
        "source_language": source_language,
        "target_language": target_language,
        "tts_language_code": tts_language_code,
        "tts_voice": tts_voice,
        "effective_tts_speed": effective_speed,
        "timing_metrics": timing_records,
        "stt_data": stt_result,
        "gemini_analysis": analysis_result,
    }

    try:
        output_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_metadata_path, "w", encoding="utf-8") as meta_f:
            json.dump(pipeline_metadata, meta_f, indent=2, ensure_ascii=False)
        print(f"-> Saved Metadata To : {output_metadata_path}")
    except Exception as meta_err:
        print(f"[WARNING] Audio succeeded but metadata write failed: {meta_err}", file=sys.stderr)

    print("\n" + "=" * 70)
    print(f"VIBELINGUAL PIPELINE EXECUTION COMPLETED IN {total_duration}s")
    print("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run VibeLingual End-to-End Voice Translation Pipeline."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("media/malayalam_sample.wav.m4a"),
        help="Input audio file path (default: media/malayalam_sample.wav.m4a)",
    )
    parser.add_argument(
        "--output-audio",
        "-o",
        type=Path,
        default=Path("outputs/vibelingual_pipeline_run.wav"),
        help="Output audio path (default: outputs/vibelingual_pipeline_run.wav)",
    )
    parser.add_argument(
        "--output-json",
        "-j",
        type=Path,
        default=Path("outputs/vibelingual_pipeline_run.json"),
        help="Output metadata JSON path (default: outputs/vibelingual_pipeline_run.json)",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default="Kaveri",
        help="Voice for English TTS synthesis (default: Kaveri, verified for en-IN in Gnani documentation)",
    )

    args = parser.parse_args()

    success = run_pipeline(
        input_audio_path=args.input,
        output_audio_path=args.output_audio,
        output_metadata_path=args.output_json,
        tts_voice=args.voice,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
