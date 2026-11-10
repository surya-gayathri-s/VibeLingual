"""
Speech-to-Text (STT) Service for VibeLingual.

Integrates with Gnani Prisma v2.5 via REST API to provide accurate,
low-latency transcription for Indian languages (including Malayalam).
Includes automatic 48 kHz PCM normalization to prevent Cloudflare gateway
timeouts (HTTP 504), robust MIME type resolution, and diagnostic logging.
"""

import logging
import os
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import requests
from dotenv import load_dotenv

# Setup safe diagnostic logging without logging credentials or audio contents
logger = logging.getLogger("vibelingual.stt")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [VibeLingual.STT] %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# Verified endpoint for Gnani STT v3 (Prisma v2.5 is default model)
GNANI_STT_ENDPOINT = "https://api.vachana.ai/stt/v3"

# Request timeout: 10s connect, 45s read for model inference
STT_TIMEOUT_CONFIG = (10.0, 45.0)

# Target optimal sample rate for Gnani STT inference
OPTIMAL_SAMPLE_RATE = 48000

# MIME mapping based on audio file extension
STT_MIME_MAP = {
    ".wav": "audio/wav",
    ".m4a": "audio/m4a",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
}


class STTErrorCategory:
    """Classified error categories for safe UI presentation and diagnostics."""
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    INVALID_AUDIO_FORMAT = "INVALID_AUDIO_FORMAT"
    SERVER_GATEWAY_ERROR = "SERVER_GATEWAY_ERROR"
    NETWORK_CONNECTION_ERROR = "NETWORK_CONNECTION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class STTServiceError(Exception):
    """Raised when an error occurs during Speech-to-Text processing."""
    def __init__(
        self,
        message: str,
        category: str = STTErrorCategory.UNKNOWN_ERROR,
        status_code: Optional[int] = None,
        safe_user_message: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.safe_user_message = safe_user_message or message
        self.original_error = original_error


def resolve_stt_mime_type(audio_path: Path) -> str:
    """Resolves accurate MIME type for Gnani STT multipart payload."""
    ext = audio_path.suffix.lower()
    return STT_MIME_MAP.get(ext, "audio/wav")


def ensure_48khz_pcm_wav(path_obj: Path) -> Tuple[Path, bool]:
    """Ensures WAV audio is standard 48 kHz 16-bit mono PCM to prevent Gnani STT gateway timeouts.

    Gnani STT's inference engine processes 48 kHz audio in < 0.8s, whereas non-48 kHz
    audio (e.g. 16 kHz browser recordings) triggers server-side upsampling that exceeds
    Cloudflare's strict 5-second upstream timeout, resulting in HTTP 504.

    Returns:
        Tuple of (effective_path, is_temporary)
    """
    if path_obj.suffix.lower() != ".wav":
        return path_obj, False

    try:
        with wave.open(str(path_obj), "rb") as w:
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            framerate = w.getframerate()
            n_frames = w.getnframes()

            # Already at optimal 48kHz mono 16-bit PCM
            if framerate == OPTIMAL_SAMPLE_RATE and channels == 1 and sampwidth == 2:
                return path_obj, False

            raw_bytes = w.readframes(n_frames)

        logger.info(
            f"Converting WAV to optimal Gnani format (original: {framerate}Hz, "
            f"{channels}ch, {sampwidth*8}-bit -> target: {OPTIMAL_SAMPLE_RATE}Hz, 1ch, 16-bit)..."
        )

        if sampwidth == 2:
            samples = np.frombuffer(raw_bytes, dtype=np.int16)
        elif sampwidth == 1:
            samples = ((np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128) * 256).astype(np.int16)
        elif sampwidth == 4:
            samples = (np.frombuffer(raw_bytes, dtype=np.int32) // 65536).astype(np.int16)
        else:
            return path_obj, False

        # If stereo, downmix to mono
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

        # Resample to 48000 Hz if needed
        if framerate != OPTIMAL_SAMPLE_RATE and len(samples) > 1:
            target_len = int(len(samples) * (float(OPTIMAL_SAMPLE_RATE) / framerate))
            samples = np.interp(
                np.linspace(0, len(samples) - 1, target_len),
                np.arange(len(samples)),
                samples
            ).astype(np.int16)

        # Write converted WAV to a temp file
        temp_converted = tempfile.NamedTemporaryFile(delete=False, suffix="_48k.wav")
        temp_converted.close()

        with wave.open(temp_converted.name, "wb") as w_out:
            w_out.setnchannels(1)
            w_out.setsampwidth(2)
            w_out.setframerate(OPTIMAL_SAMPLE_RATE)
            w_out.writeframes(samples.tobytes())

        logger.info(f"WAV converted successfully: {len(samples)} samples @ {OPTIMAL_SAMPLE_RATE}Hz.")
        return Path(temp_converted.name), True

    except Exception as conv_err:
        logger.warning(f"Audio normalization skipped (using original): {conv_err}")
        return path_obj, False


def classify_stt_error(status_code: Optional[int], response_text: str) -> Tuple[str, str]:
    """Classifies an HTTP status code and response into an error category and safe user message."""
    if status_code in [401, 403]:
        return (
            STTErrorCategory.AUTHENTICATION_ERROR,
            "Gnani API authentication failed. Please verify GNANI_API_KEY in .env.",
        )
    if status_code == 429:
        return (
            STTErrorCategory.QUOTA_EXCEEDED,
            "Gnani STT transcription quota or rate limit exceeded. Please wait a moment before trying again.",
        )
    if status_code in [400, 415, 422]:
        return (
            STTErrorCategory.INVALID_AUDIO_FORMAT,
            f"Gnani STT rejected the audio format or encoding (HTTP {status_code}). Ensure a clear recording.",
        )
    if status_code in [502, 503, 504]:
        return (
            STTErrorCategory.SERVER_GATEWAY_ERROR,
            f"Gnani STT server gateway timed out (HTTP {status_code}). Cloudflare upstream timeout occurred.",
        )
    return (
        STTErrorCategory.UNKNOWN_ERROR,
        f"Gnani STT service returned an unexpected status code (HTTP {status_code}).",
    )


def transcribe_audio(
    audio_path: Union[str, Path],
    language_code: str = "ml-IN",
) -> Dict[str, Any]:
    """Transcribes an audio file using Gnani Prisma v2.5 with 48kHz normalization and logging.

    Args:
        audio_path: Path to the input audio file (WAV, M4A, MP3, etc.).
        language_code: Target language locale code (e.g., 'ml-IN', 'hi-IN').

    Returns:
        Dict containing:
            - 'transcript': Extracted text transcription.
            - 'detected_language': Detected language code if available.
            - 'request_id': Gnani request tracking identifier.
            - 'processing_time': Server processing duration in seconds.

    Raises:
        STTServiceError: If credentials, files, or network requests fail.
    """
    # 1. Load credentials securely
    load_dotenv()
    api_key = os.getenv("GNANI_API_KEY")

    if not api_key or not api_key.strip():
        logger.error("GNANI_API_KEY is not configured in the environment or .env file.")
        raise STTServiceError(
            message="GNANI_API_KEY is not configured.",
            category=STTErrorCategory.AUTHENTICATION_ERROR,
            safe_user_message="Gnani API key is missing. Please set GNANI_API_KEY in your .env file.",
        )

    # 2. Validate input audio path and format
    path_obj = Path(audio_path).resolve()
    if not path_obj.is_file():
        logger.error(f"Audio file not found at: {path_obj}")
        raise STTServiceError(
            message=f"Audio file not found at: {path_obj}",
            category=STTErrorCategory.INVALID_AUDIO_FORMAT,
            safe_user_message="Audio recording file could not be found on disk.",
        )

    file_size_bytes = path_obj.stat().st_size
    if file_size_bytes == 0:
        logger.error(f"Audio file is empty (0 bytes): {path_obj.name}")
        raise STTServiceError(
            message="Audio file is empty (0 bytes).",
            category=STTErrorCategory.INVALID_AUDIO_FORMAT,
            safe_user_message="The audio recording is empty (0 bytes). Please record again.",
        )

    # 3. Ensure optimal 48kHz audio format to prevent Cloudflare 504 gateway timeouts
    effective_path, is_temp_norm = ensure_48khz_pcm_wav(path_obj)
    filename = effective_path.name
    mime_type = resolve_stt_mime_type(effective_path)
    file_size_kb = effective_path.stat().st_size / 1024

    logger.info(
        f"Preparing Gnani STT request: file='{filename}' ({file_size_kb:.1f} KB), "
        f"mime='{mime_type}', language='{language_code}', endpoint='{GNANI_STT_ENDPOINT}'"
    )

    # 4. Construct headers and form payload (credentials never exposed)
    headers = {
        "X-API-Key-ID": api_key.strip(),
    }
    data_payload = {
        "language_code": language_code,
    }

    try:
        with open(effective_path, "rb") as audio_file:
            files_payload = {
                "audio_file": (filename, audio_file, mime_type),
            }

            logger.info("Sending STT request to Gnani API...")
            response = requests.post(
                GNANI_STT_ENDPOINT,
                headers=headers,
                files=files_payload,
                data=data_payload,
                timeout=STT_TIMEOUT_CONFIG,
            )

        logger.info(f"Gnani STT responded with HTTP {response.status_code}.")

        # Success path (HTTP 200)
        if response.status_code == 200:
            try:
                result = response.json()
            except ValueError as json_err:
                logger.exception(f"Received HTTP 200 but body was not valid JSON: {response.text[:200]!r}")
                raise STTServiceError(
                    message=f"Received HTTP 200, but response was invalid JSON: {json_err}",
                    category=STTErrorCategory.UNKNOWN_ERROR,
                    status_code=200,
                    safe_user_message="Gnani STT returned an invalid response body.",
                    original_error=json_err,
                )

            transcript = result.get("transcript", "").strip()
            req_id = result.get("request_id", "N/A")
            proc_time = result.get("processing_time", 0.0)

            logger.info(
                f"Gnani STT succeeded! RequestID={req_id}, Time={proc_time}s, "
                f"Transcript='{transcript[:60]}...'"
            )

            return {
                "transcript": transcript,
                "detected_language": result.get("detected_language_code") or language_code,
                "request_id": req_id,
                "processing_time": proc_time,
            }

        # Error path
        safe_snippet = response.text[:300].strip() if response.text else "(empty body)"
        cat, safe_msg = classify_stt_error(response.status_code, safe_snippet)
        logger.error(
            f"Gnani STT request failed with HTTP {response.status_code} [{cat}]. "
            f"Response snippet: {safe_snippet}"
        )
        raise STTServiceError(
            message=f"Gnani STT failed with HTTP {response.status_code}: {safe_snippet[:120]}",
            category=cat,
            status_code=response.status_code,
            safe_user_message=safe_msg,
        )

    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Gnani STT request timed out: {timeout_err}")
        raise STTServiceError(
            message=f"Gnani STT request timed out after {STT_TIMEOUT_CONFIG[1]}s.",
            category=STTErrorCategory.NETWORK_CONNECTION_ERROR,
            safe_user_message=f"Gnani STT server timed out after {int(STT_TIMEOUT_CONFIG[1])}s. Please try again.",
            original_error=timeout_err,
        ) from timeout_err

    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Failed to connect to Gnani STT endpoint: {conn_err}")
        raise STTServiceError(
            message="Failed to connect to Gnani STT service at https://api.vachana.ai.",
            category=STTErrorCategory.NETWORK_CONNECTION_ERROR,
            safe_user_message="Could not connect to Gnani STT server. Please check your internet connection.",
            original_error=conn_err,
        ) from conn_err

    except requests.exceptions.RequestException as req_err:
        logger.exception(f"Network error during STT request: {req_err}")
        raise STTServiceError(
            message=f"Network error during STT request: {req_err}",
            category=STTErrorCategory.NETWORK_CONNECTION_ERROR,
            safe_user_message="Network communication error occurred while sending audio to Gnani STT.",
            original_error=req_err,
        ) from req_err

    finally:
        # Clean up temporary resampled file
        if is_temp_norm and effective_path.exists():
            try:
                effective_path.unlink()
            except OSError:
                pass
