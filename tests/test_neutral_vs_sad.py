"""
Verification Test Suite: Calm/Neutral vs Sad Classification in VibeLingual.

Verifies:
1. Real human neutral recording is classified as 'Calm / Neutral' (NOT Sad).
2. Calm neutral voice saying a sad sentence yields Acoustic: 'Calm / Neutral' & Text: 'Sad'.
3. Calm neutral voice saying a happy sentence yields Acoustic: 'Calm / Neutral' & Text: 'Happy'.
4. Clearly excited vocal delivery yields Acoustic: 'Happy / Excited'.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.audio_emotion_service import analyze_audio_emotion
from services.intelligence_service import analyze_and_translate
from services.tts_service import synthesize_speech


def test_neutral_vs_sad():
    print("=" * 75)
    print("VibeLingual Mandatory Acoustic Emotion Validation Suite")
    print("=" * 75)

    results = {}

    # --------------------------------------------------------------------------
    # Case 1: Real Human Neutral Recording
    # --------------------------------------------------------------------------
    human_audio = PROJECT_ROOT / "outputs" / "latest_browser_recording.wav"
    print("\n[Case 1] Real Human Neutral Recording (Browser Recording):")
    print(f"Path: {human_audio} ({human_audio.stat().st_size} bytes)")
    
    t0 = time.time()
    res1 = analyze_audio_emotion(human_audio)
    t1 = time.time() - t0
    
    print(f"-> Detected Acoustic Emotion: {res1.detected_emotion} (Confidence: {res1.emotion_confidence}, in {round(t1, 2)}s)")
    print(f"   Vocal Tension: {res1.voice_tension_or_softness}, Energy: {res1.vocal_energy}, Pace: {res1.speaking_pace}")
    print(f"   Observations : {res1.supporting_observations}")
    
    is_neutral = any(k in res1.detected_emotion.lower() for k in ["neutral", "calm"])
    not_sad = "sad" not in res1.detected_emotion.lower()
    assert is_neutral and not_sad, f"FAIL Case 1: Expected Neutral/Calm, got '{res1.detected_emotion}'"
    print("✓ Case 1 PASSED: Real human neutral speech classified as Calm / Neutral (NOT Sad)!")
    results["Case 1 (Real Human Neutral)"] = "PASSED"

    # --------------------------------------------------------------------------
    # Case 2: Calm Neutral Voice Saying a Sad Sentence
    # --------------------------------------------------------------------------
    sad_statement = "എന്റെ ജീവിതത്തിൽ വലിയൊരു നഷ്ടം സംഭവിച്ചു, ഞാൻ ആകെ തകർന്നുപോയി"
    print("\n[Case 2] Calm Neutral Voice Saying a Sad Sentence:")
    print(f"Statement: \"{sad_statement}\" (A big loss happened, I am shattered)")
    
    temp_sad_calm = PROJECT_ROOT / "outputs" / "temp_sad_calm.wav"
    try:
        audio_bytes = synthesize_speech(
            text=sad_statement,
            language_code="ml-IN",
            voice="Kaveri",
            speed=0.95,
            emotion="Calm / Neutral",
        )
        with open(temp_sad_calm, "wb") as f:
            f.write(audio_bytes)
        
        t0 = time.time()
        res2_audio = analyze_audio_emotion(temp_sad_calm)
        t_audio = time.time() - t0
        print(f"-> Acoustic Emotion : {res2_audio.detected_emotion} (Confidence: {res2_audio.emotion_confidence}, in {round(t_audio, 2)}s)")
        print(f"   Vocal Tension    : {res2_audio.voice_tension_or_softness}, Energy: {res2_audio.vocal_energy}")

        t0 = time.time()
        res2_trans = analyze_and_translate(
            text=sad_statement,
            source_language="ml-IN",
            target_language="en",
            acoustic_data=res2_audio,
        )
        t_trans = time.time() - t0
        print(f"-> Text Sentiment   : {res2_trans.get('text_sentiment')} (in {round(t_trans, 2)}s)")
        print(f"-> Natural Trans    : \"{res2_trans.get('natural_translation')}\"")
        print(f"-> Contrast Note    : {res2_trans.get('sentiment_contrast_note')}")

        is_calm_audio = any(k in res2_audio.detected_emotion.lower() for k in ["neutral", "calm"])
        is_sad_text = any(k in res2_trans.get("text_sentiment", "").lower() for k in ["sad", "somber", "grief", "loss", "devastat", "sorrow", "hurt"])
        
        assert is_calm_audio, f"FAIL Case 2 Acoustic: Expected Calm/Neutral, got '{res2_audio.detected_emotion}'"
        assert is_sad_text, f"FAIL Case 2 Text: Expected Sadness in text sentiment, got '{res2_trans.get('text_sentiment')}'"
        assert res2_trans.get("sentiment_contrast_note") is not None, "FAIL Case 2: Contrast note should be populated"
        
        print("✓ Case 2 PASSED: Acoustic remained Neutral, Text Sentiment captured Sadness, Contrast documented!")
        results["Case 2 (Calm voice + Sad words)"] = "PASSED"
    finally:
        if temp_sad_calm.exists():
            try:
                temp_sad_calm.unlink()
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # Case 3: Calm Neutral Voice Saying a Happy Sentence
    # --------------------------------------------------------------------------
    happy_statement = "ഇന്ന് എനിക്ക് ജോലിയിൽ സ്ഥാനക്കയറ്റം ലഭിച്ചു, വലിയ സന്തോഷമായി"
    print("\n[Case 3] Calm Neutral Voice Saying a Happy Sentence:")
    print(f"Statement: \"{happy_statement}\" (Today I got promoted at work, great joy)")
    
    temp_happy_calm = PROJECT_ROOT / "outputs" / "temp_happy_calm.wav"
    try:
        audio_bytes = synthesize_speech(
            text=happy_statement,
            language_code="ml-IN",
            voice="Kaveri",
            speed=1.0,
            emotion="Calm / Neutral",
        )
        with open(temp_happy_calm, "wb") as f:
            f.write(audio_bytes)
        
        t0 = time.time()
        res3_audio = analyze_audio_emotion(temp_happy_calm)
        t_audio = time.time() - t0
        print(f"-> Acoustic Emotion : {res3_audio.detected_emotion} (in {round(t_audio, 2)}s)")

        t0 = time.time()
        res3_trans = analyze_and_translate(
            text=happy_statement,
            source_language="ml-IN",
            target_language="en",
            acoustic_data=res3_audio,
        )
        t_trans = time.time() - t0
        print(f"-> Text Sentiment   : {res3_trans.get('text_sentiment')} (in {round(t_trans, 2)}s)")
        print(f"-> Natural Trans    : \"{res3_trans.get('natural_translation')}\"")

        is_calm_audio3 = any(k in res3_audio.detected_emotion.lower() for k in ["neutral", "calm"])
        is_happy_text = any(k in res3_trans.get("text_sentiment", "").lower() for k in ["happy", "joy", "excit", "celebrat", "pleas", "proud"])
        
        assert is_calm_audio3, f"FAIL Case 3 Acoustic: Expected Calm/Neutral, got '{res3_audio.detected_emotion}'"
        assert is_happy_text, f"FAIL Case 3 Text: Expected Happy text sentiment, got '{res3_trans.get('text_sentiment')}'"
        
        print("✓ Case 3 PASSED: Acoustic remained Neutral, Text Sentiment captured Joy/Happiness!")
        results["Case 3 (Calm voice + Happy words)"] = "PASSED"
    finally:
        if temp_happy_calm.exists():
            try:
                temp_happy_calm.unlink()
            except OSError:
                pass

    # --------------------------------------------------------------------------
    # Case 4: Clearly Excited Voice (Human Audio Sample)
    # --------------------------------------------------------------------------
    excited_audio = PROJECT_ROOT / "media" / "malayalam_sample.wav.m4a"
    print("\n[Case 4] Clearly Excited Human Voice Sample:")
    print(f"Path: {excited_audio} ({excited_audio.stat().st_size} bytes)")
    
    t0 = time.time()
    res4 = analyze_audio_emotion(excited_audio)
    t4 = time.time() - t0
    print(f"-> Detected Acoustic Emotion: {res4.detected_emotion} (Confidence: {res4.emotion_confidence}, in {round(t4, 2)}s)")
    print(f"   Energy: {res4.vocal_energy}, Pace: {res4.speaking_pace}, Style: {res4.speaking_style}")
    print(f"   Observations: {res4.supporting_observations}")
    
    is_excited = "happy" in res4.detected_emotion.lower() or "excit" in res4.detected_emotion.lower()
    assert is_excited, f"FAIL Case 4: Expected Happy/Excited, got '{res4.detected_emotion}'"
    print("✓ Case 4 PASSED: Clearly animated human voice recognized as Happy / Excited!")
    results["Case 4 (Clearly Excited Human Audio)"] = "PASSED"

    print("\n" + "=" * 75)
    print("ALL MANDATORY TEST CASES PASSED:")
    for test_name, status in results.items():
        print(f"  • {test_name:38}: {status}")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = test_neutral_vs_sad()
    sys.exit(0 if success else 1)

