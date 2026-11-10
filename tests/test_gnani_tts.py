"""
Standalone test script for Gnani Timbre v2.5 Text-to-Speech (TTS) API.

Usage:
    .venv\\Scripts\\python.exe tests/test_gnani_tts.py "നമസ്കാരം, സുഖമാണോ?" [--language-code ml-IN] [--voice Kaveri] [--output outputs/tts_sample.wav]
"""

import argparse
import os
import sys
import requests
from dotenv import load_dotenv

# Endpoint for Gnani TTS inference
GNANI_TTS_ENDPOINT = "https://api.vachana.ai/api/v1/tts/inference"

# Request timeout: 10s connect, 60s read for synthesis
TIMEOUT_CONFIG = (10.0, 60.0)


def is_valid_wav(data: bytes) -> bool:
    """Verifies that the byte stream conforms to standard RIFF/WAVE header specifications."""
    return len(data) >= 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def run_tts_test(
    text: str,
    language_code: str = "ml-IN",
    voice: str = "Kaveri",
    speed: float = 1.0,
    output_path: str = "outputs/tts_output.wav",
) -> bool:
    """Executes a test text synthesis against Gnani Timbre v2.5."""
    # 1. Load environment variables securely
    load_dotenv()
    api_key = os.getenv("GNANI_API_KEY")

    if not api_key or not api_key.strip():
        print(
            "Error: GNANI_API_KEY is not set. Please provide it in your .env file or environment.",
            file=sys.stderr,
        )
        return False

    # 2. Validate input text
    cleaned_text = text.strip()
    if not cleaned_text:
        print("Error: Input text cannot be empty.", file=sys.stderr)
        return False

    # 3. Validate speech speed range
    if not (0.5 <= speed <= 2.0):
        print(
            f"Error: Speed must be between 0.5 and 2.0 (got {speed}).",
            file=sys.stderr,
        )
        return False

    print(f"Preparing TTS request for text: '{cleaned_text[:60]}{'...' if len(cleaned_text) > 60 else ''}'")
    print(f"Model: timbre-v2.5 | Language: {language_code} | Voice: {voice} | Speed: {speed}")
    print(f"Service endpoint: {GNANI_TTS_ENDPOINT}")

    # 4. Prepare headers and JSON payload
    # Note: Never log or print the headers or API key
    headers = {
        "X-API-Key-ID": api_key.strip(),
        "Content-Type": "application/json",
    }

    payload = {
        "text": cleaned_text,
        "model": "timbre-v2.5",
        "voice": voice,
        "language": language_code,
        "speed": speed,
        "audio_config": {
            "encoding": "linear_pcm",
            "container": "wav",
            "num_channels": 1,
            "sample_rate": 48000,
            "sample_width": 2,
        },
    }

    try:
        print("Sending request to Gnani TTS API...")
        response = requests.post(
            GNANI_TTS_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=TIMEOUT_CONFIG,
        )

        # 5. Handle HTTP responses
        if response.status_code == 200:
            audio_bytes = response.content
            byte_size = len(audio_bytes)

            # 6. Verify valid WAV content
            if not is_valid_wav(audio_bytes):
                print(
                    f"Error: Received HTTP 200, but response payload does not contain a valid WAV header.",
                    file=sys.stderr,
                )
                try:
                    err_json = response.json()
                    print(f"Response JSON: {err_json}", file=sys.stderr)
                except ValueError:
                    print(f"First 100 bytes: {audio_bytes[:100]}", file=sys.stderr)
                return False

            # Ensure destination directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(audio_bytes)

            file_size_kb = byte_size / 1024
            print("\n" + "=" * 50)
            print("GNANI TTS TEST SUCCESSFUL")
            print("=" * 50)
            print(f"Synthesized text : {cleaned_text}")
            print(f"Voice / Language : {voice} ({language_code})")
            print(f"Audio Size       : {file_size_kb:.1f} KB ({byte_size} bytes)")
            print(f"Output saved to  : {output_path}")
            print("=" * 50)
            return True

        else:
            print(
                f"\nError: API returned HTTP status {response.status_code}",
                file=sys.stderr,
            )
            try:
                err_json = response.json()
                detail = err_json.get("detail") or err_json.get("message") or err_json
                print(f"Error details: {detail}", file=sys.stderr)
            except ValueError:
                print(f"Error message: {response.text[:200]}", file=sys.stderr)
            return False

    except requests.exceptions.Timeout:
        print(
            f"Error: Request timed out after {TIMEOUT_CONFIG[1]}s waiting for Gnani TTS server.",
            file=sys.stderr,
        )
        return False
    except requests.exceptions.ConnectionError:
        print(
            "Error: Connection failed. Could not reach https://api.vachana.ai.",
            file=sys.stderr,
        )
        return False
    except requests.exceptions.RequestException as e:
        print(f"Error: Network or request exception occurred: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Gnani Timbre v2.5 Text-to-Speech (TTS) API."
    )
    parser.add_argument(
        "text",
        type=str,
        help="Text string to synthesize (e.g. 'നമസ്കാരം, സുഖമാണോ?')",
    )
    parser.add_argument(
        "--language-code",
        "-l",
        type=str,
        default="ml-IN",
        help="Language code (default: ml-IN). Examples: ml-IN, hi-IN, ta-IN, te-IN, kn-IN",
    )
    parser.add_argument(
        "--voice",
        "-v",
        type=str,
        default="Kaveri",
        help="Voice name (default: Kaveri). Check Gnani Timbre v2.5 catalog for regional personas.",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=1.0,
        help="Speech speed factor between 0.5 and 2.0 (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/tts_malayalam_output.wav",
        help="Output WAV audio file path (default: outputs/tts_malayalam_output.wav)",
    )

    args = parser.parse_args()
    success = run_tts_test(
        text=args.text,
        language_code=args.language_code,
        voice=args.voice,
        speed=args.speed,
        output_path=args.output,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

