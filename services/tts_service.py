"""
Text-to-Speech (TTS) Service for VibeLingual.

Integrates with Gnani Timbre v2.5 via REST API to synthesize natural speech
in Indian languages and Indian English (en-IN), adapting speech pacing to
reflect the speaker's emotional expression.
"""

import os
from typing import Optional
import requests
from dotenv import load_dotenv

# Verified endpoint for Gnani TTS inference
GNANI_TTS_ENDPOINT = "https://api.vachana.ai/api/v1/tts/inference"

# Request timeout: 10s connect, 60s read for synthesis
TTS_TIMEOUT_CONFIG = (10.0, 60.0)

# Emotion-to-Pacing modulation table for Timbre v2.5
EMOTION_SPEED_MAP = {
    "Happy / Excited": 1.12,
    "Surprised": 1.10,
    "Frustrated / Angry": 1.08,
    "Calm / Neutral": 1.00,
    "Worried / Anxious": 1.04,
    "Sad": 0.90,
    "Uncertain / Mixed": 1.00,
}


class TTSServiceError(Exception):
    """Raised when an error occurs during Text-to-Speech synthesis."""
    pass


def is_valid_wav(data: bytes) -> bool:
    """Verifies that the byte stream conforms to standard RIFF/WAVE header specifications."""
    return len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def resolve_speech_speed(
    speed: Optional[float] = None,
    emotion: Optional[str] = None,
) -> float:
    """Resolves the effective speech rate based on explicit speed or detected emotion."""
    if speed is not None and speed != 1.0:
        return speed

    if emotion:
        # Match case-insensitively or via prefix
        for emo_key, rate in EMOTION_SPEED_MAP.items():
            if emo_key.lower() in emotion.lower() or emotion.lower() in emo_key.lower():
                return rate

    return 1.0 if speed is None else speed


def synthesize_speech(
    text: str,
    language_code: str = "en-IN",
    voice: str = "Kaveri",
    speed: Optional[float] = None,
    emotion: Optional[str] = None,
) -> bytes:
    """Synthesizes text into WAV audio using Gnani Timbre v2.5 with emotional rate modulation.

    Args:
        text: Text string to synthesize into speech.
        language_code: Target language code (e.g., 'en-IN', 'ml-IN').
        voice: Voice persona identifier from Gnani Timbre catalog (default 'Kaveri').
        speed: Explicit speech rate multiplier between 0.5 and 2.0. If omitted or 1.0,
               the rate is automatically modulated by the emotion parameter.
        emotion: Detected emotion used to calculate appropriate expressive speaking tempo.

    Returns:
        Raw bytes of the synthesized 48 kHz mono 16-bit PCM WAV audio file.

    Raises:
        TTSServiceError: If credentials, inputs, network, or audio validation fail.
    """
    # 1. Load credentials securely
    load_dotenv()
    api_key = os.getenv("GNANI_API_KEY")

    if not api_key or not api_key.strip():
        raise TTSServiceError(
            "GNANI_API_KEY is not configured in the environment or .env file."
        )

    # 2. Validate input text
    cleaned_text = text.strip()
    if not cleaned_text:
        raise TTSServiceError("Input text for speech synthesis cannot be empty.")

    # 3. Resolve effective speaking speed
    effective_speed = resolve_speech_speed(speed=speed, emotion=emotion)
    if not (0.5 <= effective_speed <= 2.0):
        effective_speed = max(0.5, min(2.0, effective_speed))

    # 4. Prepare request headers and JSON payload
    headers = {
        "X-API-Key-ID": api_key.strip(),
        "Content-Type": "application/json",
    }

    payload = {
        "text": cleaned_text,
        "model": "timbre-v2.5",
        "voice": voice,
        "language": language_code,
        "speed": round(effective_speed, 2),
        "audio_config": {
            "encoding": "linear_pcm",
            "container": "wav",
            "num_channels": 1,
            "sample_rate": 48000,
            "sample_width": 2,
        },
    }

    try:
        response = requests.post(
            GNANI_TTS_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=TTS_TIMEOUT_CONFIG,
        )

        if response.status_code == 200:
            audio_bytes = response.content
            if not is_valid_wav(audio_bytes):
                try:
                    err_json = response.json()
                    detail = f"Server returned JSON payload instead of audio: {err_json}"
                except ValueError:
                    detail = f"Payload header is not a valid RIFF/WAVE structure (first 50 bytes: {audio_bytes[:50]!r})"
                raise TTSServiceError(f"Invalid audio payload received: {detail}")

            return audio_bytes
        else:
            try:
                err_data = response.json()
                detail = err_data.get("detail") or err_data.get("message") or err_data
            except ValueError:
                detail = response.text[:200]
            raise TTSServiceError(
                f"Gnani TTS API failed with HTTP {response.status_code}: {detail}"
            )

    except requests.exceptions.Timeout:
        raise TTSServiceError(
            f"Gnani TTS request timed out after {TTS_TIMEOUT_CONFIG[1]} seconds."
        )
    except requests.exceptions.ConnectionError:
        raise TTSServiceError(
            "Failed to connect to Gnani TTS service at https://api.vachana.ai."
        )
    except requests.exceptions.RequestException as req_err:
        raise TTSServiceError(f"Network error during TTS request: {req_err}")
