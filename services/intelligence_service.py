"""
Gemini Intelligence & Beyond Translation Service for VibeLingual.

Provides emotion-aware contextual understanding, cultural nuance analysis, intent & emotion
extraction, and dual (literal vs. natural) translation using Google Gemini 3.8 Flash.
Includes safe, automatic fallback to text-only translation when multimodal audio processing
encounters server unavailability, timeouts, or connection errors.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
import pydantic
from pydantic import BaseModel, Field

# Setup safe diagnostic logging without logging credentials or audio bytes
logger = logging.getLogger("vibelingual.intelligence")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [VibeLingual.Intelligence] %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# Primary and resilient fallback model identifiers
# gemini-3.5-flash-lite and gemini-3.5-flash are verified models with active Free Tier quota
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
RESILIENT_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-3.8-flash"]

# Maximum 30-second HTTP timeout in milliseconds (30,000 ms) for multimodal attempt
MULTIMODAL_TIMEOUT_MS = 30000


class ErrorCategory:
    """Classified error categories for safe UI presentation and diagnostics."""
    MULTIMODAL_AUDIO_FAILURE = "MULTIMODAL_AUDIO_FAILURE"
    TEXT_TRANSLATION_FAILURE = "TEXT_TRANSLATION_FAILURE"
    AUTH_OR_QUOTA_ERROR = "AUTH_OR_QUOTA_ERROR"
    NETWORK_OR_SERVER_ERROR = "NETWORK_OR_SERVER_ERROR"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class IntelligenceServiceError(Exception):
    """Raised when an error occurs during Gemini intelligence analysis."""
    def __init__(
        self,
        message: str,
        category: str = ErrorCategory.UNKNOWN_ERROR,
        safe_user_message: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.category = category
        self.safe_user_message = safe_user_message or message
        self.original_error = original_error


def classify_gemini_error(err: Exception) -> tuple[str, str]:
    """Classifies an error into a category and a safe user explanation without exposing secrets."""
    err_str = str(err).lower()
    err_code = getattr(err, "code", None)

    # 1. Authentication / Permission errors
    if err_code in [401, 403] or "unauthenticated" in err_str or "permission_denied" in err_str:
        return (
            ErrorCategory.AUTH_OR_QUOTA_ERROR,
            "API key was rejected or has insufficient permissions. Please check GEMINI_API_KEY in .env.",
        )

    # 2. Quota / Rate limit errors
    if err_code == 429 or "resource_exhausted" in err_str or "quota" in err_str or "rate limit" in err_str:
        return (
            ErrorCategory.AUTH_OR_QUOTA_ERROR,
            "Gemini API rate limit or quota exceeded. Please wait a moment before trying again.",
        )

    # 3. Server availability / Network timeouts
    if err_code in [500, 502, 503, 504] or any(
        kw in err_str for kw in ["deadline", "timeout", "timed out", "unavailable", "connection", "remote", "protocol"]
    ):
        status_suffix = f" (HTTP {err_code})" if err_code else ""
        return (
            ErrorCategory.NETWORK_OR_SERVER_ERROR,
            f"Google Gemini servers are temporarily busy or timed out{status_suffix}. Please try again.",
        )

    # 4. JSON decoding / Schema validation errors
    if isinstance(err, (json.JSONDecodeError, pydantic.ValidationError)) or "validation" in err_str:
        return (
            ErrorCategory.SCHEMA_VALIDATION_ERROR,
            "Gemini returned an invalid or malformed structured response.",
        )

    return (
        ErrorCategory.TEXT_TRANSLATION_FAILURE,
        f"Linguistic intelligence analysis failed: {type(err).__name__}.",
    )


class VibeLingualAnalysis(BaseModel):
    """Structured response schema enforced across Gemini linguistic outputs."""

    source_text: str = Field(
        description="The original spoken transcript in the source language."
    )
    source_language: str = Field(
        description="Human-readable name of the source language (e.g., 'Malayalam')."
    )
    target_language: str = Field(
        description="Human-readable name of the target language (e.g., 'English')."
    )
    literal_translation: str = Field(
        description="A close, direct translation preserving source grammar and structure."
    )
    natural_translation: str = Field(
        description="A fluent, idiomatic translation that conveys the message and emotional vibe naturally."
    )
    text_sentiment: str = Field(
        description="Linguistic sentiment interpreted strictly from the spoken transcript vocabulary and semantic context (e.g., 'Worried / Concerned', 'Joyful / Enthusiastic', 'Frustrated / Dissatisfied', 'Sad / Somber', 'Neutral / Factual', 'Matter-of-fact')."
    )
    sentiment_contrast_note: Optional[str] = Field(
        default=None,
        description="Observations regarding contrast or alignment between the semantic text sentiment and the physical vocal acoustic delivery, if any."
    )
    conversational_intent: str = Field(
        description="What the speaker is aiming to communicate or achieve in conversation."
    )
    cultural_context: str = Field(
        description="Relevant cultural meaning, context, or social nuance (or 'None' if factual/direct)."
    )
    idioms_or_expressions: List[str] = Field(
        default_factory=list,
        description="List of idioms, slang, or colloquial expressions detected, if any."
    )
    explanation: str = Field(
        description="Brief explanation of key linguistic choices or nuances between literal and natural translation."
    )
    suggested_delivery_style: str = Field(
        default="Natural and conversational",
        description="Recommended English speech delivery style matching the speaker's emotional state."
    )
    tts_speed_recommendation: float = Field(
        default=1.0,
        description="Recommended TTS playback speed factor (0.88 to 1.15). Default to 1.0 if delivery is calm, neutral, or uncertain."
    )


# System instructions enforcing linguistic, cultural, and sentiment analysis rules
SYSTEM_INSTRUCTION = """
You are VibeLingual, an expert multilingual linguist and cultural translator specializing in Indian languages.
Your purpose is to move "Beyond Translation" by analyzing the meaning, conversational intent, cultural nuances, and textual sentiment of the speaker.

Follow these strict analysis rules:
1. Meaning Fidelity:
   - Preserve the speaker's original meaning faithfully. The natural translation must accurately convey what was said.
2. Translation & Nuance:
   - Provide a literal_translation (direct grammatical phrasing).
   - Provide a natural_translation (fluent, native-sounding English capturing intended meaning and conversational vibe).
   - Explain cultural_context and idioms_or_expressions faithfully based on actual linguistic evidence without inventing meanings.
3. Text Sentiment Analysis (Distinct from Voice Acoustics):
   - Analyze text_sentiment strictly from the vocabulary, semantic phrasing, and contextual meaning of the words (e.g. 'Worried / Concerned', 'Joyful / Enthusiastic', 'Frustrated / Dissatisfied', 'Sad / Somber', 'Neutral / Factual').
   - Do NOT confuse text sentiment with physical vocal emotion.
   - If the speaker's observed acoustic voice delivery is provided, note any contrast in sentiment_contrast_note (for example: if the words express worry or difficulty but the observed voice delivery was calm and relaxed, note: 'Contrast: The speaker discussed a concerning topic with a calm, steady vocal demeanor').
4. Return strictly valid JSON conforming to the requested schema.
"""


from services.audio_emotion_service import (
    AudioEmotionAnalysis,
    analyze_audio_emotion,
    AudioEmotionServiceError,
)


def analyze_and_translate(
    text: str,
    source_language: str = "ml-IN",
    target_language: str = "en",
    audio_path: Optional[Union[str, Path]] = None,
    audio_emotion_data: Optional[Dict[str, Any]] = None,
    acoustic_data: Optional[AudioEmotionAnalysis] = None,
    model_name: str = DEFAULT_GEMINI_MODEL,
) -> Dict[str, Any]:
    """Analyzes and translates spoken text with Gemini, integrating pure acoustic emotion and textual sentiment.
    
    Decouples pure acoustic voice analysis (from audio alone) from linguistic translation & text sentiment
    (from transcript alone), completely eliminating lexical leakage into acoustic emotion detection.

    Args:
        text: Spoken transcript string to translate and analyze.
        source_language: Source language locale or name (default 'ml-IN').
        target_language: Target language locale or name (default 'en').
        audio_path: Optional audio path for acoustic emotion extraction if not pre-computed.
        audio_emotion_data: Optional DSP metrics dictionary.
        acoustic_data: Optional pre-computed AudioEmotionAnalysis from Stage 2.
        model_name: Gemini model identifier (defaults to 'gemini-3.8-flash').

    Returns:
        Dict conforming to complete VibeLingual analysis with literal/natural translations,
        pure acoustic emotion, separate text sentiment, cultural context, and delivery guidance.

    Raises:
        IntelligenceServiceError: If credentials, API calls, or schema validation fail.
    """
    # 1. Load credentials securely
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not api_key.strip():
        logger.error("GEMINI_API_KEY is not configured in environment or .env file.")
        raise IntelligenceServiceError(
            message="GEMINI_API_KEY is not configured.",
            category=ErrorCategory.AUTH_OR_QUOTA_ERROR,
            safe_user_message="API key is missing. Please set GEMINI_API_KEY in your .env file.",
        )

    # 2. Validate input text
    cleaned_text = text.strip()
    if not cleaned_text:
        logger.warning("Empty transcript provided to analyze_and_translate.")
        raise IntelligenceServiceError(
            message="Input text cannot be empty.",
            category=ErrorCategory.TEXT_TRANSLATION_FAILURE,
            safe_user_message="Transcribed speech was empty. Please speak clearly into the microphone.",
        )

    # 3. Step A: Obtain Pure Acoustic Emotion Analysis (NO TRANSCRIPT LEAKAGE)
    effective_acoustic: Optional[AudioEmotionAnalysis] = acoustic_data
    acoustic_error_notice: Optional[str] = None

    if effective_acoustic is None and audio_path:
        audio_p = Path(audio_path).resolve()
        if audio_p.is_file():
            logger.info(f"Extracting pure acoustic emotion from audio: {audio_p.name} (no transcript bias)...")
            try:
                effective_acoustic = analyze_audio_emotion(audio_p)
                logger.info(f"Acoustic analysis succeeded: Vibe='{effective_acoustic.detected_emotion}', Conf='{effective_acoustic.emotion_confidence}'.")
            except Exception as ac_err:
                logger.warning(f"Acoustic analysis failed: {ac_err}. Proceeding with text-only translation.", exc_info=True)
                effective_acoustic = None
                acoustic_error_notice = "Spoken voice acoustic analysis was temporarily unavailable; sentiment is interpreted strictly from transcript context."

    if effective_acoustic is None and not acoustic_error_notice:
        acoustic_error_notice = "Spoken voice acoustic analysis was temporarily unavailable; sentiment is interpreted strictly from transcript context."

    # 4. Import SDK modules dynamically
    try:
        from google import genai
        from google.genai import types
        from google.genai.errors import APIError
    except ImportError as imp_err:
        logger.exception("The google-genai SDK is not installed or accessible.")
        raise IntelligenceServiceError(
            message=f"google-genai SDK not available: {imp_err}",
            category=ErrorCategory.UNKNOWN_ERROR,
            safe_user_message="Google GenAI package is not properly installed.",
            original_error=imp_err,
        )

    try:
        client = genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(
                timeout=25000,
                retry_options=types.HttpRetryOptions(attempts=2),
            ),
        )
    except Exception as client_err:
        logger.exception("Failed to initialize Gemini Client.")
        raise IntelligenceServiceError(
            message=f"Failed to initialize Gemini Client: {client_err}",
            category=ErrorCategory.AUTH_OR_QUOTA_ERROR,
            safe_user_message="Failed to initialize Gemini client. Check credentials.",
            original_error=client_err,
        )

    # 5. Build user prompt for Text Translation & Linguistic Nuance
    prompt_parts = [
        f"Analyze the following {source_language} spoken statement and translate it into {target_language}.\n\n",
        f"Spoken Transcript: \"{cleaned_text}\"\n\n",
    ]

    if effective_acoustic:
        prompt_parts.append(
            f"Observed Acoustic Vocal Delivery (from physical audio signal):\n"
            f"- Detected Vocal Vibe: {effective_acoustic.detected_emotion}\n"
            f"- Emotional Intensity: {effective_acoustic.emotional_intensity}\n"
            f"- Speaking Style: {effective_acoustic.speaking_style}\n"
            f"- Vocal Energy: {effective_acoustic.vocal_energy}\n"
            f"- Speaking Pace: {effective_acoustic.speaking_pace}\n"
            f"- Vocal Tension: {effective_acoustic.voice_tension_or_softness}\n"
            f"- Acoustic Evidence: {', '.join(effective_acoustic.supporting_observations)}\n\n"
            "Translation & Sentiment Instructions:\n"
            "1. Translate faithfully into literal_translation.\n"
            "2. Translate naturally into fluent, idiomatic English capturing the intended meaning and spoken vibe.\n"
            "3. Identify text_sentiment strictly from the semantic meaning of the words (e.g. 'Worried / Concerned', 'Joyful / Enthusiastic', 'Frustrated / Dissatisfied', 'Sad / Somber', 'Neutral / Factual', 'Matter-of-fact').\n"
            "4. Contrast Analysis: If the words express worry, problems, or difficulties but the speaker sounded calm/relaxed, or vice versa, describe this in sentiment_contrast_note.\n"
            "5. Provide cultural_context, conversational_intent, and idioms without inventing meanings.\n"
        )
    else:
        prompt_parts.append(
            "Instructions:\n"
            "1. Translate faithfully into literal_translation and fluent natural_translation.\n"
            "2. Identify text_sentiment strictly from the vocabulary and semantic meaning of the transcript.\n"
            "3. Provide cultural_context, conversational_intent, and idioms without inventing meanings.\n"
        )

    user_prompt = "".join(prompt_parts)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        response_mime_type="application/json",
        response_schema=VibeLingualAnalysis,
    )

    # 6. Execute Text Translation & Sentiment with Resilient Fallback
    models_to_try = [model_name]
    for fb in RESILIENT_FALLBACK_MODELS:
        if fb not in models_to_try:
            models_to_try.append(fb)

    response = None
    last_error = None
    for current_model in models_to_try:
        logger.info(f"Executing translation & text sentiment using model '{current_model}'...")
        try:
            response = client.models.generate_content(
                model=current_model,
                contents=[user_prompt],
                config=config,
            )
            if response and response.text:
                logger.info(f"Translation & sentiment analysis succeeded with model '{current_model}'.")
                break
        except Exception as text_err:
            last_error = text_err
            cat, safe_msg = classify_gemini_error(text_err)
            logger.warning(
                f"Translation failed with model '{current_model}' [{cat}]: {type(text_err).__name__} - {safe_msg}",
                exc_info=True,
            )
            if current_model != models_to_try[-1]:
                logger.info(f"Attempting fallback model '{models_to_try[-1]}'...")
                continue
            logger.exception("All model attempts exhausted for translation.")
            raise IntelligenceServiceError(
                message=f"Gemini translation failed: {type(text_err).__name__} ({cat})",
                category=cat,
                safe_user_message=safe_msg,
                original_error=text_err,
            ) from text_err

    if not response or not response.text:
        logger.error("Gemini returned an empty response.")
        raise IntelligenceServiceError(
            message="Gemini returned an empty response.",
            category=ErrorCategory.SCHEMA_VALIDATION_ERROR,
            safe_user_message="Gemini returned an empty response. Please retry.",
        )

    # 7. Validate JSON response
    try:
        parsed_data = json.loads(response.text)
        validated_model = VibeLingualAnalysis.model_validate(parsed_data)
        result_dict = validated_model.model_dump()
    except Exception as val_err:
        logger.exception("Gemini translation response failed schema validation.")
        raise IntelligenceServiceError(
            message=f"Schema validation failed: {val_err}",
            category=ErrorCategory.SCHEMA_VALIDATION_ERROR,
            safe_user_message="Gemini translation response did not match the expected schema.",
            original_error=val_err,
        ) from val_err

    # 8. Merge Pure Acoustic Emotion with Text Sentiment (NO OVERWRITING)
    if effective_acoustic:
        result_dict["acoustic_emotion_available"] = True
        result_dict["acoustic_emotion"] = effective_acoustic.detected_emotion
        result_dict["emotion"] = effective_acoustic.detected_emotion
        result_dict["emotion_confidence"] = effective_acoustic.emotion_confidence
        result_dict["emotional_intensity"] = effective_acoustic.emotional_intensity
        result_dict["vocal_energy"] = effective_acoustic.vocal_energy
        result_dict["speaking_pace"] = effective_acoustic.speaking_pace
        result_dict["pitch_variation"] = effective_acoustic.pitch_variation
        result_dict["voice_tension"] = effective_acoustic.voice_tension_or_softness
        result_dict["speaking_style"] = effective_acoustic.speaking_style
        result_dict["acoustic_observations"] = effective_acoustic.supporting_observations
        result_dict["is_uncertain"] = effective_acoustic.is_uncertain
        result_dict["tts_speed_recommendation"] = effective_acoustic.suggested_tts_speed
        result_dict["fallback_notice"] = None

        # Detect Agreement / Disagreement contrast
        ac_vibe = effective_acoustic.detected_emotion.lower()
        txt_sent = result_dict.get("text_sentiment", "").lower()
        if any(c in ac_vibe for c in ["calm", "neutral"]) and any(k in txt_sent for k in ["worried", "concern", "anxious", "fear", "problem", "difficult", "sad", "frustrat", "grief", "pain", "somber"]):
            result_dict["sentiment_contrast_note"] = (
                "Contrast Detected: The speaker delivered this statement with a calm, neutral vocal demeanor "
                "despite the sad, concerning, or difficult semantic content of the words."
            )
        elif any(c in ac_vibe for c in ["calm", "neutral"]) and any(k in txt_sent for k in ["happy", "joy", "excit", "celebrat"]):
            result_dict["sentiment_contrast_note"] = (
                "Contrast Detected: The speaker delivered this positive statement with an understated, calm, and composed vocal delivery."
            )
        elif "worried" in ac_vibe and any(k in txt_sent for k in ["worried", "concern", "anxious", "fear"]):
            result_dict["sentiment_contrast_note"] = "Aligned: Audible vocal anxiety corresponds with the concerning semantic content."
        elif "happy" in ac_vibe and any(k in txt_sent for k in ["happy", "joy", "excit", "enthusias", "good", "great"]):
            result_dict["sentiment_contrast_note"] = "Aligned: Upbeat, animated vocal energy matches the positive semantic content."
        elif "sad" in ac_vibe and any(k in txt_sent for k in ["sad", "somber", "grief", "pain", "unhappy"]):
            result_dict["sentiment_contrast_note"] = "Aligned: Sorrowful vocal delivery corresponds with the sad semantic content."
    else:
        result_dict["acoustic_emotion_available"] = False
        result_dict["acoustic_emotion"] = None
        result_dict["emotion"] = result_dict.get("text_sentiment", "Calm / Neutral")
        result_dict["emotion_confidence"] = "low"
        result_dict["emotional_intensity"] = "moderate"
        result_dict["vocal_energy"] = "moderate"
        result_dict["speaking_pace"] = "moderate"
        result_dict["pitch_variation"] = "subtle variation"
        result_dict["voice_tension"] = "neutral"
        result_dict["acoustic_observations"] = []
        result_dict["is_uncertain"] = True
        result_dict["tts_speed_recommendation"] = 1.0
        result_dict["fallback_notice"] = acoustic_error_notice or (
            "Voice emotion analysis was temporarily unavailable; sentiment is interpreted strictly "
            "from transcript context with neutral 1.0x delivery."
        )

    return result_dict
