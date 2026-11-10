"""
Standalone test script for Gnani Prisma v2.5 Speech-to-Text (STT) API.

Usage:
    .venv\\Scripts\\python.exe tests/test_gnani_stt.py <path_to_audio_file> [--language-code hi-IN]
"""

import argparse
import os
import sys
import requests
from dotenv import load_dotenv

# Endpoint for Gnani STT v3 (Prisma v2.5 is the default model)
GNANI_STT_ENDPOINT = "https://api.vachana.ai/stt/v3"

# Request timeout: 10s connect, 60s read for audio inference
TIMEOUT_CONFIG = (10.0, 60.0)


def run_stt_test(audio_file_path: str, language_code: str) -> bool:
    """Executes a test transcription against Gnani Prisma v2.5."""
    # 1. Load environment variables securely
    load_dotenv()
    api_key = os.getenv("GNANI_API_KEY")

    if not api_key or not api_key.strip():
        print(
            "Error: GNANI_API_KEY is not set. Please provide it in your .env file or environment.",
            file=sys.stderr,
        )
        return False

    # 2. Validate audio file existence
    if not os.path.isfile(audio_file_path):
        print(
            f"Error: Audio file not found at '{audio_file_path}'.",
            file=sys.stderr,
        )
        return False

    filename = os.path.basename(audio_file_path)
    file_size_kb = os.path.getsize(audio_file_path) / 1024

    print(f"Preparing STT request for file: {filename} ({file_size_kb:.1f} KB)")
    print(f"Target language code: {language_code}")
    print(f"Service endpoint: {GNANI_STT_ENDPOINT}")

    # 3. Prepare headers and multipart payload
    # Note: Never log or print the headers or API key
    headers = {
        "X-API-Key-ID": api_key.strip()
    }

    # Prisma v2.5 is default; no model parameter is included
    data_payload = {
        "language_code": language_code
    }

    try:
        with open(audio_file_path, "rb") as audio_file:
            files_payload = {
                "audio_file": (filename, audio_file, "application/octet-stream")
            }

            print("Sending request to Gnani STT API...")
            response = requests.post(
                GNANI_STT_ENDPOINT,
                headers=headers,
                files=files_payload,
                data=data_payload,
                timeout=TIMEOUT_CONFIG,
            )

        # 4. Handle HTTP responses
        if response.status_code == 200:
            try:
                result = response.json()
            except ValueError:
                print(
                    f"Error: Received HTTP 200, but response was not valid JSON.",
                    file=sys.stderr,
                )
                print(f"Raw response: {response.text[:200]}", file=sys.stderr)
                return False

            print("\n" + "=" * 50)
            print("GNANI STT TEST SUCCESSFUL")
            print("=" * 50)

            transcript = result.get("transcript")
            detected_lang = result.get("detected_language_code")
            request_id = result.get("request_id")
            processing_time = result.get("processing_time")

            print(f"Transcript        : {transcript}")
            if detected_lang:
                print(f"Detected Language : {detected_lang}")
            if request_id:
                print(f"Request ID        : {request_id}")
            if processing_time is not None:
                print(f"Processing Time   : {processing_time}s")
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
            f"Error: Request timed out after {TIMEOUT_CONFIG[1]}s waiting for Gnani STT server.",
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
        description="Test Gnani Prisma v2.5 Speech-to-Text (STT) API."
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file (e.g. sample.wav, sample.mp3)",
    )
    parser.add_argument(
        "--language-code",
        "-l",
        type=str,
        default="hi-IN",
        help="Language code (default: hi-IN). Examples: hi-IN, en-IN, ta-IN, te-IN, kn-IN",
    )

    args = parser.parse_args()
    success = run_stt_test(args.audio_path, args.language_code)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

