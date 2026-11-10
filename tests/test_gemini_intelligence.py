"""
Standalone test script for VibeLingual Gemini Intelligence & Translation Component.

This script sends input text to Google Gemini for contextual understanding,
intent & emotion detection, cultural nuance analysis, and dual translation
(literal vs. natural) with structured JSON output.

Usage:
    .venv\\Scripts\\python.exe tests/test_gemini_intelligence.py "Text to analyze" [--source-language ml-IN] [--target-language en]
"""

import argparse
import json
import os
import sys
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ==============================================================================
# Model Configuration
# ==============================================================================
# gemini-3.8-flash is Google's flagship flash model for fast, accurate multilingual
# understanding (including Indic languages) and native structured JSON generation.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


# ==============================================================================
# Structured Output Schema (Pydantic)
# ==============================================================================
class VibeLingualAnalysis(BaseModel):
    """Schema defining the structured analysis returned by Gemini."""

    source_text: str = Field(
        description="The original input text provided by the user."
    )
    source_language: str = Field(
        description="Human-readable name of the source language (e.g. Malayalam)."
    )
    target_language: str = Field(
        description="Human-readable name of the target language (e.g. English)."
    )
    literal_translation: str = Field(
        description="A close, word-for-word or direct translation preserving source phrasing."
    )
    natural_translation: str = Field(
        description="A fluent, idiomatic, natural translation that sounds native in the target language."
    )
    emotion: str = Field(
        description="Detected emotional tone (e.g., happy, enthusiastic, frustrated, neutral, or uncertain)."
    )
    emotion_confidence: str = Field(
        description="Confidence level of detected emotion: 'low', 'medium', or 'high'."
    )
    conversational_intent: str = Field(
        description="What the speaker is aiming to communicate or achieve in conversation."
    )
    cultural_context: str = Field(
        description="Relevant cultural meaning, context, or social nuance (or 'None' if purely factual)."
    )
    idioms_or_expressions: List[str] = Field(
        default_factory=list,
        description="List of idioms, slang, or colloquial expressions detected, if any."
    )
    explanation: str = Field(
        description="Brief explanation of key linguistic choices or nuances between literal and natural translation."
    )


# ==============================================================================
# System Prompt & Linguistic Rules
# ==============================================================================
SYSTEM_INSTRUCTION = """
You are VibeLingual, an expert multilingual linguist and cultural translator specializing in Indian languages.
Your purpose is to move "Beyond Translation" by analyzing the emotional tone, conversational intent, and cultural nuances of the speaker.

Follow these strict analysis rules:
1. Preserve the speaker's original meaning faithfully.
2. Do not invent cultural context, idioms, or metaphors that are not present.
3. Identify sarcasm or irony only when there is clear textual evidence.
4. Do not assume emotion from text alone with certainty; if ambiguous, mark confidence as 'low' or emotion as 'neutral/uncertain'.
5. Clearly distinguish literal meaning from implied conversational meaning.
6. If context is insufficient, state 'uncertain' or 'unknown' rather than guessing.
7. Preserve the speaker's politeness, informality, and conversational register.
8. Do not add information that the speaker did not communicate.
9. Return the output strictly conforming to the requested JSON schema.
"""


def analyze_text_with_gemini(
    text: str,
    source_language: str = "ml-IN",
    target_language: str = "en",
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> bool:
    """Executes a contextual translation and intelligence analysis using Gemini."""
    # 1. Load environment variables securely from .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not api_key.strip():
        print(
            "Error: GEMINI_API_KEY is not set. Please provide it in your .env file or environment.",
            file=sys.stderr,
        )
        return False

    # 2. Validate input text
    cleaned_text = text.strip()
    if not cleaned_text:
        print("Error: Input text cannot be empty.", file=sys.stderr)
        return False

    print("=" * 60)
    print("VibeLingual Gemini Intelligence Test")
    print("=" * 60)
    print(f"Model           : {model_name}")
    print(f"Source Language : {source_language}")
    print(f"Target Language : {target_language}")
    print(f"Input Text      : {cleaned_text}")
    print("=" * 60)

    # 3. Import Google GenAI SDK
    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import APIError
    except ImportError as e:
        print(
            f"Error: google-genai package is missing or corrupted: {e}",
            file=sys.stderr,
        )
        return False

    # 4. Initialize GenAI client securely (never log or print api_key)
    try:
        client = genai.Client(api_key=api_key.strip())
    except Exception as e:
        print(f"Error: Failed to initialize Gemini Client: {e}", file=sys.stderr)
        return False

    # 5. Build user prompt
    user_prompt = (
        f"Analyze the following {source_language} statement and translate it into {target_language}.\n"
        f"Source Text: \"{cleaned_text}\"\n\n"
        f"Provide the complete cultural, emotional, and linguistic analysis in the specified JSON structure."
    )

    # 6. Configure generation with structured output schema for Gemini 3.8 Flash
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=VibeLingualAnalysis,
    )

    # 7. Execute model call
    try:
        print("Sending request to Gemini API...")
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=config,
        )

        if not response or not response.text:
            print("Error: Gemini returned an empty response.", file=sys.stderr)
            return False

        # 8. Parse and validate JSON
        try:
            parsed_data = json.loads(response.text)
        except json.JSONDecodeError as err:
            print(f"Error: Response is not valid JSON: {err}", file=sys.stderr)
            print(f"Raw response: {response.text[:200]}", file=sys.stderr)
            return False

        # Validate with Pydantic model
        validated_analysis = VibeLingualAnalysis.model_validate(parsed_data)

        print("\n" + "=" * 60)
        print("GEMINI INTELLIGENCE TEST SUCCESSFUL")
        print("=" * 60)
        formatted_json = json.dumps(
            validated_analysis.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
        print(formatted_json)
        print("=" * 60)
        return True

    except APIError as api_err:
        print(f"\nError: Gemini API Error: {api_err.message}", file=sys.stderr)
        if hasattr(api_err, "code"):
            print(f"HTTP Status / Code: {api_err.code}", file=sys.stderr)
        return False
    except Exception as err:
        print(f"\nError: Unexpected error occurred: {err}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test VibeLingual Gemini Intelligence & Translation Component."
    )
    parser.add_argument(
        "text",
        type=str,
        help="Source text to analyze and translate (e.g. 'ഇന്നത്തെ കാലാവസ്ഥ വളരെ നല്ലതാണ്...')",
    )
    parser.add_argument(
        "--source-language",
        "-s",
        type=str,
        default="ml-IN",
        help="Source language code (default: ml-IN)",
    )
    parser.add_argument(
        "--target-language",
        "-t",
        type=str,
        default="en",
        help="Target language code (default: en)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help=f"Gemini model identifier (default: {DEFAULT_GEMINI_MODEL})",
    )

    args = parser.parse_args()
    success = analyze_text_with_gemini(
        text=args.text,
        source_language=args.source_language,
        target_language=args.target_language,
        model_name=args.model,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

