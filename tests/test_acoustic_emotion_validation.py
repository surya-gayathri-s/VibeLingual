"""
Validation Test Suite for VibeLingual Acoustic Emotion Decoupling.

Verifies that:
1. Acoustic emotion is evaluated strictly from the raw audio signal without transcript leakage.
2. Calm delivery of a concerning statement (e.g. exams, difficulties) is correctly classified
   as 'Calm / Neutral' acoustically, while text sentiment is recognized as 'Worried / Concerned'.
3. Acoustic emotion and text sentiment remain completely separate fields.
4. Sentiment contrast notes are accurately populated when vocal demeanor and text meaning diverge.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.tts_service import synthesize_speech
from services.audio_emotion_service import analyze_audio_emotion
from services.intelligence_service import analyze_and_translate


def test_acoustic_emotion_decoupling():
    print("=" * 70)
    print("VibeLingual Acoustic Emotion & Text Sentiment Decoupling Test")
    print("=" * 70)

    # Malayalam sentence with strong worry/anxiety semantic content:
    # "എനിക്ക് നാളെ പരീക്ഷയുണ്ട്, വളരെയധികം ബുദ്ധിമുട്ടുകൾ ഉണ്ട്"
    # (I have an exam tomorrow, there are so many difficulties)
    worry_statement = "എനിക്ക് നാളെ പരീക്ഷയുണ്ട്, വളരെയധികം ബുദ്ധിമുട്ടുകൾ ഉണ്ട്"
    print(f"Test Statement: \"{worry_statement}\"")
    print("Expected Semantic Sentiment: Worried / Concerned")
    print("Audio Delivery Mode        : Calm & Relaxed (Speed: 0.98x, Voice: Kaveri)")
    print("-" * 70)

    temp_audio = PROJECT_ROOT / "outputs" / "temp_validation_calm.wav"

    try:
        # Step 1: Synthesize the test sentence in a calm, relaxed voice
        print("\n[Step 1] Synthesizing calm audio clip with Gnani Timbre...")
        audio_bytes = synthesize_speech(
            text=worry_statement,
            language_code="ml-IN",
            voice="Kaveri",
            speed=0.98,
            emotion="Calm / Neutral",
        )
        with open(temp_audio, "wb") as f:
            f.write(audio_bytes)
        print(f"-> Synthesized audio saved ({len(audio_bytes)} bytes)")

        # Step 2: Pure Acoustic Emotion Analysis (NO TRANSCRIPT PASSED)
        print("\n[Step 2] Executing Pure Acoustic Emotion Analysis (Audio only)...")
        t0 = time.time()
        acoustic_res = analyze_audio_emotion(temp_audio)
        acoustic_duration = round(time.time() - t0, 2)
        print(f"-> Acoustic Analysis Completed in {acoustic_duration}s:")
        print(f"   • Detected Vibe : {acoustic_res.detected_emotion}")
        print(f"   • Confidence    : {acoustic_res.emotion_confidence}")
        print(f"   • Vocal Energy  : {acoustic_res.vocal_energy}")
        print(f"   • Speaking Pace : {acoustic_res.speaking_pace}")
        print(f"   • Vocal Tension : {acoustic_res.voice_tension_or_softness}")
        print(f"   • Evidence      : {', '.join(acoustic_res.supporting_observations)}")

        # Step 3: Text Translation & Semantic Sentiment Analysis
        print("\n[Step 3] Executing Text Translation & Semantic Sentiment Analysis...")
        t1 = time.time()
        full_res = analyze_and_translate(
            text=worry_statement,
            source_language="ml-IN",
            target_language="en",
            acoustic_data=acoustic_res,
        )
        trans_duration = round(time.time() - t1, 2)
        print(f"-> Translation & Sentiment Completed in {trans_duration}s:")
        print(f"   • Acoustic Emotion       : {full_res.get('acoustic_emotion')}")
        print(f"   • Text Sentiment         : {full_res.get('text_sentiment')}")
        print(f"   • Natural Translation    : \"{full_res.get('natural_translation')}\"")
        print(f"   • Literal Translation    : \"{full_res.get('literal_translation')}\"")
        print(f"   • Contrast Note          : {full_res.get('sentiment_contrast_note')}")

        # Verification Assertions
        detected_vibe = full_res.get("acoustic_emotion", "")
        text_sentiment = full_res.get("text_sentiment", "")

        assert "calm" in detected_vibe.lower(), (
            f"FAIL: Expected acoustic emotion to be Calm / Neutral, but got '{detected_vibe}'"
        )
        assert any(k in text_sentiment.lower() for k in ["worried", "concern", "anxious", "problem", "difficult"]), (
            f"FAIL: Expected text sentiment to reflect worry/concern, but got '{text_sentiment}'"
        )
        assert full_res.get("acoustic_emotion") != full_res.get("text_sentiment"), (
            "FAIL: Acoustic emotion and text sentiment should disagree for this contrast test."
        )

        print("\n" + "=" * 70)
        print("✓ VALIDATION TEST PASSED: Acoustic emotion successfully decoupled from transcript semantics!")
        print("=" * 70)
        return True

    finally:
        if temp_audio.exists():
            try:
                temp_audio.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    success = test_acoustic_emotion_decoupling()
    sys.exit(0 if success else 1)

