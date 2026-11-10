"""
VibeLingual Core Services Package.

Provides reusable modular client interfaces for:
- Speech-to-Text (Gnani Prisma v2.5)
- Acoustic Emotion & Speech Feature Analysis (Gemini 3.8 Flash Multimodal & Local DSP)
- Intelligence & Cultural Translation (Google Gemini 3.8 Flash)
- Text-to-Speech with Emotional Modulation (Gnani Timbre v2.5)
"""

from .stt_service import transcribe_audio, STTServiceError
from .audio_emotion_service import (
    analyze_audio_emotion,
    extract_local_acoustic_features,
    AudioEmotionAnalysis,
    AudioEmotionServiceError,
)
from .intelligence_service import (
    analyze_and_translate,
    IntelligenceServiceError,
    VibeLingualAnalysis,
)
from .tts_service import synthesize_speech, TTSServiceError, resolve_speech_speed

__all__ = [
    "transcribe_audio",
    "STTServiceError",
    "analyze_audio_emotion",
    "extract_local_acoustic_features",
    "AudioEmotionAnalysis",
    "AudioEmotionServiceError",
    "analyze_and_translate",
    "IntelligenceServiceError",
    "VibeLingualAnalysis",
    "synthesize_speech",
    "TTSServiceError",
    "resolve_speech_speed",
]
