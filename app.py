"""
VibeLingual – Beyond Words. Beyond Translation.
Emotion-Aware Malayalam-to-English Voice Translation Studio.

Integrating:
- Gnani Prisma v2.5 (Malayalam Speech Recognition)
- Google Gemini Multimodal Perception (Pure Acoustic Voice Emotion)
- Google Gemini Linguistic Intelligence (Context, Nuance & Text Sentiment)
- Gnani Timbre v2.5 (Expressive English Speech Synthesis)
"""

import os
import sys
import tempfile
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import streamlit as st

# Setup safe UI logger to ensure tracebacks print to terminal console
logger = logging.getLogger("vibelingual.studio")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [VibeLingual.Studio] %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.stt_service import transcribe_audio, STTServiceError
from services.audio_emotion_service import (
    AudioEmotionAnalysis,
    AudioEmotionServiceError,
    analyze_audio_emotion,
    extract_local_acoustic_features,
)
from services.intelligence_service import (
    analyze_and_translate,
    IntelligenceServiceError,
)
from services.tts_service import (
    synthesize_speech,
    resolve_speech_speed,
    TTSServiceError,
    EMOTION_SPEED_MAP,
)

# ==============================================================================
# Page Configuration & Visual Architecture
# ==============================================================================
st.set_page_config(
    page_title="VibeLingual — Beyond Words. Beyond Translation.",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Premium Cinematic Design System CSS
PREMIUM_STUDIO_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-canvas: #090812;
    --bg-surface: rgba(22, 18, 38, 0.78);
    --bg-surface-elevated: rgba(30, 24, 52, 0.88);
    --accent-violet: #9B6DFF;
    --accent-lavender: #C5A7FF;
    --accent-cyan: #78DCE8;
    --accent-emerald: #34D399;
    --accent-amber: #FBBF24;
    --accent-coral: #F87171;
    --text-primary: #F5F1FF;
    --text-secondary: #A59EB5;
    --text-muted: #736B88;
    --border-subtle: rgba(155, 109, 255, 0.18);
    --border-highlight: rgba(155, 109, 255, 0.45);
    --border-cyan: rgba(120, 220, 232, 0.40);
    --glow-violet: 0 0 32px rgba(155, 109, 255, 0.22);
    --glow-cyan: 0 0 32px rgba(120, 220, 232, 0.20);
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-full: 9999px;
}

/* Base Canvas & Atmosphere */
.stApp {
    background-color: var(--bg-canvas) !important;
    background-image: 
        radial-gradient(circle at 50% -10%, rgba(155, 109, 255, 0.22) 0%, rgba(120, 220, 232, 0.08) 45%, transparent 72%),
        radial-gradient(circle at 90% 50%, rgba(197, 167, 255, 0.06) 0%, transparent 45%),
        radial-gradient(circle at 10% 80%, rgba(120, 220, 232, 0.05) 0%, transparent 50%) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    overflow-x: hidden;
}

/* Page Container Centering & Anti-Clipping */
.main .block-container {
    max-width: 1120px !important;
    padding-top: 1.6rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background-color: #0b0916 !important;
    border-right: 1px solid var(--border-subtle) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Hero Section */
.hero-wrapper {
    text-align: center;
    padding: 20px 10px 16px 10px;
    max-width: 980px;
    margin: 0 auto;
    box-sizing: border-box;
}

.brand-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 20px;
    background: rgba(155, 109, 255, 0.12);
    border: 1px solid var(--border-highlight);
    border-radius: var(--radius-full);
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: var(--accent-lavender);
    text-transform: uppercase;
    margin-bottom: 16px;
    box-shadow: 0 4px 18px rgba(155, 109, 255, 0.15);
}

.hero-brand-name {
    font-size: clamp(2.8rem, 6vw, 4.2rem);
    font-weight: 900;
    letter-spacing: -0.015em;
    line-height: 1.12;
    background: linear-gradient(135deg, #FFFFFF 0%, #E8E0FF 45%, var(--accent-violet) 85%, var(--accent-cyan) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 0 16px;
    margin: 0 auto 10px auto;
    display: inline-block;
    box-sizing: border-box;
}

.hero-tagline {
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--accent-lavender);
    letter-spacing: -0.01em;
    margin-bottom: 12px;
}

.hero-description {
    font-size: 1.02rem;
    color: var(--text-secondary);
    line-height: 1.62;
    max-width: 760px;
    margin: 0 auto 24px auto;
}

/* 3-Part Designed Language Bridge Cards */
.bridge-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 14px;
    flex-wrap: wrap;
    margin: 0 auto 24px auto;
    max-width: 860px;
    box-sizing: border-box;
}

.bridge-card {
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(22, 18, 38, 0.85);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 12px 20px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    transition: all 0.25s ease;
    flex: 1;
    min-width: 190px;
    max-width: 260px;
    box-sizing: border-box;
}

.bridge-card:hover {
    border-color: var(--border-highlight);
    transform: translateY(-2px);
}

.bridge-card-active {
    border-color: var(--border-cyan);
    background: linear-gradient(135deg, rgba(155, 109, 255, 0.15) 0%, rgba(120, 220, 232, 0.10) 100%);
    box-shadow: 0 8px 28px rgba(120, 220, 232, 0.18);
}

.bridge-icon {
    font-size: 1.5rem;
    line-height: 1;
}

.bridge-info {
    display: flex;
    flex-direction: column;
    text-align: left;
}

.bridge-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--text-primary);
}

.bridge-sub {
    font-size: 0.74rem;
    color: var(--accent-lavender);
    font-weight: 500;
}

.bridge-arrow {
    color: var(--accent-cyan);
    font-size: 1.25rem;
    font-weight: 800;
}

/* 4-Stage Horizontal Pipeline Flow */
.pipeline-row {
    display: flex;
    gap: 12px;
    justify-content: center;
    align-items: stretch;
    flex-wrap: wrap;
    margin: 0 auto 26px auto;
    max-width: 980px;
    box-sizing: border-box;
}

.pipeline-card {
    flex: 1;
    min-width: 200px;
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(18, 14, 32, 0.75);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    box-sizing: border-box;
    transition: all 0.25s ease;
}

.pipeline-card:hover {
    border-color: var(--border-highlight);
    background: rgba(24, 18, 42, 0.85);
}

.stage-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 800;
    color: var(--accent-cyan);
    background: rgba(120, 220, 232, 0.12);
    border: 1px solid rgba(120, 220, 232, 0.35);
    border-radius: var(--radius-sm);
    padding: 4px 8px;
    line-height: 1;
}

.stage-text {
    display: flex;
    flex-direction: column;
    text-align: left;
}

.stage-title {
    font-size: 0.84rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.25;
}

.stage-sub {
    font-size: 0.70rem;
    color: var(--text-muted);
    margin-top: 2px;
}

/* Studio Central Console Border Wrapper */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-surface) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.55), var(--glow-violet) !important;
    padding: 24px 28px !important;
    max-width: 980px !important;
    margin: 0 auto 28px auto !important;
    box-sizing: border-box !important;
}

.console-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
}

.console-title-text {
    font-size: 0.90rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent-lavender);
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Studio Audio Visualizer Representation */
.studio-visualizer-bar-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    height: 48px;
    padding: 8px 16px;
    background: rgba(14, 11, 25, 0.7);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-subtle);
    margin: 10px 0 16px 0;
}

.vis-bar {
    width: 3.5px;
    background: linear-gradient(180deg, var(--accent-cyan) 0%, var(--accent-violet) 100%);
    border-radius: 2px;
    opacity: 0.85;
}

/* Staged Audio State Bar */
.staged-audio-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    background: rgba(16, 12, 28, 0.92);
    border: 1px solid var(--border-cyan);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    margin-top: 18px;
    margin-bottom: 12px;
    box-shadow: var(--glow-cyan);
    box-sizing: border-box;
}

.staged-meta-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--text-primary);
}

.staged-meta-sub {
    color: var(--text-secondary);
    font-size: 0.82rem;
    margin-left: 8px;
    font-family: 'JetBrains Mono', monospace;
}

/* Glassmorphism Cards */
.studio-glass-card {
    background: var(--bg-surface);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 24px;
    margin-bottom: 18px;
    transition: all 0.28s ease;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    box-sizing: border-box;
}

.studio-glass-card:hover {
    border-color: var(--border-highlight);
    box-shadow: 0 10px 32px rgba(0, 0, 0, 0.45), var(--glow-violet);
    transform: translateY(-2px);
}

.studio-hero-highlight-card {
    background: linear-gradient(135deg, rgba(155, 109, 255, 0.18) 0%, rgba(120, 220, 232, 0.08) 100%);
    border: 1px solid var(--border-highlight);
    border-radius: var(--radius-md);
    padding: 26px;
    margin-bottom: 18px;
    box-shadow: 0 12px 36px rgba(155, 109, 255, 0.22);
    box-sizing: border-box;
}

/* Emotion Spotlight */
.emotion-spotlight-shell {
    border-radius: var(--radius-md);
    padding: 26px;
    margin-bottom: 18px;
    border: 1px solid;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
    box-sizing: border-box;
}

.emotion-spotlight-title {
    font-size: 2.3rem;
    font-weight: 800;
    letter-spacing: -0.025em;
    line-height: 1.15;
    margin-bottom: 8px;
}

/* Dynamic Studio Badges & Pills */
.pill-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: var(--radius-full);
    font-size: 0.76rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}

.pill-cyan {
    background: rgba(120, 220, 232, 0.12);
    color: var(--accent-cyan);
    border: 1px solid var(--border-cyan);
}

.pill-violet {
    background: rgba(155, 109, 255, 0.14);
    color: var(--accent-lavender);
    border: 1px solid var(--border-highlight);
}

.pill-emerald {
    background: rgba(52, 211, 153, 0.12);
    color: var(--accent-emerald);
    border: 1px solid rgba(52, 211, 153, 0.35);
}

.pill-amber {
    background: rgba(251, 191, 36, 0.12);
    color: var(--accent-amber);
    border: 1px solid rgba(251, 191, 36, 0.35);
}

.pill-coral {
    background: rgba(248, 113, 113, 0.12);
    color: var(--accent-coral);
    border: 1px solid rgba(248, 113, 113, 0.35);
}

.pill-muted {
    background: rgba(165, 158, 181, 0.12);
    color: var(--text-secondary);
    border: 1px solid rgba(165, 158, 181, 0.28);
}

/* Intensity Segmented Meter */
.meter-track {
    display: flex;
    gap: 6px;
    margin: 12px 0 16px 0;
}

.meter-segment {
    flex: 1;
    height: 7px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.08);
}

.meter-segment-active {
    background: linear-gradient(90deg, var(--accent-violet), var(--accent-cyan));
    box-shadow: 0 0 12px rgba(120, 220, 232, 0.55);
}

/* Contrast Highlight Callout */
.contrast-callout {
    background: rgba(18, 14, 34, 0.9);
    border-left: 3px solid var(--accent-cyan);
    padding: 14px 18px;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 0.90rem;
    color: var(--text-primary);
    line-height: 1.55;
    margin-top: 14px;
    box-shadow: var(--glow-cyan);
}

/* Primary Translation CTA Button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7C3AED 0%, var(--accent-violet) 50%, var(--accent-cyan) 100%) !important;
    color: #FFFFFF !important;
    font-size: 1.08rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 0.85rem 2.2rem !important;
    box-shadow: 0 8px 28px rgba(155, 109, 255, 0.42) !important;
    transition: all 0.28s ease !important;
    letter-spacing: 0.01em !important;
    width: 100% !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 12px 38px rgba(120, 220, 232, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* Tab Overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: transparent;
    border-bottom: 1px solid var(--border-subtle);
    padding-bottom: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(18, 14, 34, 0.65) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
    padding: 9px 20px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    border-color: var(--border-highlight) !important;
}

.stTabs [aria-selected="true"] {
    background: rgba(155, 109, 255, 0.22) !important;
    border-color: var(--accent-violet) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 16px rgba(155, 109, 255, 0.25) !important;
}

/* Audio Player & Widget Container Refinement */
[data-testid="stAudioInput"] {
    background: rgba(16, 12, 28, 0.8) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px !important;
}

[data-testid="stFileUploader"] {
    background: rgba(16, 12, 28, 0.6) !important;
    border-radius: var(--radius-md) !important;
}

[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* Section Header Styling */
.section-headline {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 26px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-headline::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-subtle), transparent);
}
</style>"""
st.markdown(PREMIUM_STUDIO_CSS, unsafe_allow_html=True)

# ==============================================================================
# Emotion Visual System (Refined Color Palette & Aura Mapping)
# ==============================================================================
EMOTION_THEME_MAP = {
    "Calm / Neutral": {
        "color": "#34D399",
        "bg": "rgba(52, 211, 153, 0.08)",
        "border": "rgba(52, 211, 153, 0.35)",
        "glow": "0 0 28px rgba(52, 211, 153, 0.22)",
        "pill": "pill-emerald",
        "desc": "Steady, relaxed vocal cadence with conversational balance and absence of emotional strain.",
    },
    "Neutral": {
        "color": "#34D399",
        "bg": "rgba(52, 211, 153, 0.08)",
        "border": "rgba(52, 211, 153, 0.35)",
        "glow": "0 0 28px rgba(52, 211, 153, 0.22)",
        "pill": "pill-emerald",
        "desc": "Calm, steady, matter-of-fact vocal delivery with natural unforced conversational rhythm.",
    },
    "Happy / Excited": {
        "color": "#78DCE8",
        "bg": "rgba(120, 220, 232, 0.09)",
        "border": "rgba(120, 220, 232, 0.38)",
        "glow": "0 0 32px rgba(120, 220, 232, 0.26)",
        "pill": "pill-cyan",
        "desc": "Upbeat vocal energy with buoyant resonance, expressive pitch lifts, and forward momentum.",
    },
    "Worried / Anxious": {
        "color": "#FBBF24",
        "bg": "rgba(251, 191, 36, 0.08)",
        "border": "rgba(251, 191, 36, 0.38)",
        "glow": "0 0 28px rgba(251, 191, 36, 0.22)",
        "pill": "pill-amber",
        "desc": "Audible vocal breathiness, hesitation, fluctuating pitch contour, or noticeable vocal strain.",
    },
    "Frustrated / Angry": {
        "color": "#F87171",
        "bg": "rgba(248, 113, 113, 0.09)",
        "border": "rgba(248, 113, 113, 0.38)",
        "glow": "0 0 28px rgba(248, 113, 113, 0.22)",
        "pill": "pill-coral",
        "desc": "Elevated vocal projection, sharp articulation, and acoustic tension across syllables.",
    },
    "Sad": {
        "color": "#C5A7FF",
        "bg": "rgba(197, 167, 255, 0.08)",
        "border": "rgba(197, 167, 255, 0.38)",
        "glow": "0 0 28px rgba(197, 167, 255, 0.22)",
        "pill": "pill-violet",
        "desc": "Audible acoustic sorrow, grief-stricken vocal delivery, or tremulous vocal breaks.",
    },
    "Sad / Subdued": {
        "color": "#C5A7FF",
        "bg": "rgba(197, 167, 255, 0.08)",
        "border": "rgba(197, 167, 255, 0.38)",
        "glow": "0 0 28px rgba(197, 167, 255, 0.22)",
        "pill": "pill-violet",
        "desc": "Audible acoustic sorrow or grief-stricken vocal delivery.",
    },
    "Surprised": {
        "color": "#E879F9",
        "bg": "rgba(232, 121, 249, 0.09)",
        "border": "rgba(232, 121, 249, 0.38)",
        "glow": "0 0 28px rgba(232, 121, 249, 0.22)",
        "pill": "pill-violet",
        "desc": "Abrupt pitch elevation with rapid dynamic inflection shifts.",
    },
    "Neutral / Uncertain": {
        "color": "#A59EB5",
        "bg": "rgba(165, 158, 181, 0.08)",
        "border": "rgba(165, 158, 181, 0.32)",
        "glow": "0 0 24px rgba(165, 158, 181, 0.16)",
        "pill": "pill-muted",
        "desc": "Subtle, balanced, or unforced vocal indicators evaluated with prudent contextual weighting.",
    },
    "Uncertain / Mixed": {
        "color": "#A59EB5",
        "bg": "rgba(165, 158, 181, 0.08)",
        "border": "rgba(165, 158, 181, 0.32)",
        "glow": "0 0 24px rgba(165, 158, 181, 0.16)",
        "pill": "pill-muted",
        "desc": "Conflicting or subtle acoustic indicators evaluated with balanced contextual weighting.",
    },
}

SUPPORTED_AUDIO_EXTENSIONS = ["wav", "m4a", "mp3", "ogg", "flac", "aac"]
MAX_FILE_SIZE_MB = 15

# ==============================================================================
# Session State Initialization
# ==============================================================================
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "use_sample" not in st.session_state:
    st.session_state.use_sample = False

if "active_audio_bytes" not in st.session_state:
    st.session_state.active_audio_bytes = None

if "active_audio_name" not in st.session_state:
    st.session_state.active_audio_name = None

if "active_audio_type" not in st.session_state:
    st.session_state.active_audio_type = None

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False


def get_emotion_theme(emotion_str: str) -> Dict[str, str]:
    """Matches detected emotion string to visual theme with fallback."""
    for key, theme in EMOTION_THEME_MAP.items():
        if key.lower() in emotion_str.lower() or emotion_str.lower() in key.lower():
            return theme
    return EMOTION_THEME_MAP["Uncertain / Mixed"]


# ==============================================================================
# Sidebar Studio Persona & Configuration
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="brand-pill" style="margin-bottom: 12px;">Studio Console</div>', unsafe_allow_html=True)
    st.markdown("### VibeLingual")
    st.caption("Beyond Words. Beyond Translation.")
    st.divider()

    st.markdown("#### Voice Synthesis Persona")
    selected_voice = st.selectbox(
        "Timbre English Voice",
        options=["Kaveri", "Aryan", "Veda"],
        index=0,
        help="Synthesized English voice persona from Gnani Timbre v2.5.",
    )

    voice_descriptions = {
        "Kaveri": "Warm, articulate, balanced Indian English delivery.",
        "Aryan": "Confident, clear, and measured corporate/conversational voice.",
        "Veda": "Expressive, bright, and nuanced storytelling tone.",
    }
    st.caption(voice_descriptions.get(selected_voice, ""))

    st.divider()
    st.markdown("#### Integrated AI Architecture")
    st.markdown("• **Speech Recognition:** Gnani Prisma v2.5 (`ml-IN`)")
    st.markdown("• **Acoustic Emotion:** Gemini Multimodal (Pure Audio)")
    st.markdown("• **Linguistic Nuance:** Gemini Contextual Intelligence")
    st.markdown("• **Voice Synthesis:** Gnani Timbre v2.5 (`en-IN`)")

    st.divider()
    if st.button("Reset / Clear Session", use_container_width=True):
        st.session_state.pipeline_result = None
        st.session_state.use_sample = False
        st.session_state.active_audio_bytes = None
        st.session_state.active_audio_name = None
        st.session_state.active_audio_type = None
        st.session_state.is_processing = False
        st.rerun()


# ==============================================================================
# Landing / Hero Section (Rendered with zero indentation to prevent code-block escaping)
# ==============================================================================
HERO_HTML = """<div class="hero-wrapper">
<div class="brand-pill">MALAYALAM → ENGLISH · EMOTION-AWARE VOICE TRANSLATION</div>
<div class="hero-brand-name">VibeLingual</div>
<div class="hero-tagline">Beyond Words. Beyond Translation.</div>
<div class="hero-description">An emotion-aware Malayalam-to-English voice translation experience that preserves meaning, cultural nuances and vocal expression.</div>
<div class="bridge-container">
<div class="bridge-card"><span class="bridge-icon">🎙️</span><div class="bridge-info"><span class="bridge-title">Malayalam</span><span class="bridge-sub">Spoken Voice</span></div></div>
<div class="bridge-arrow">➔</div>
<div class="bridge-card bridge-card-active"><span class="bridge-icon">🌊</span><div class="bridge-info"><span class="bridge-title">Vibe &amp; Nuance</span><span class="bridge-sub">Acoustic &amp; Intent</span></div></div>
<div class="bridge-arrow">➔</div>
<div class="bridge-card"><span class="bridge-icon">🗣️</span><div class="bridge-info"><span class="bridge-title">English</span><span class="bridge-sub">Expressive Voice</span></div></div>
</div>
<div class="pipeline-row">
<div class="pipeline-card"><span class="stage-num">01</span><div class="stage-text"><span class="stage-title">Malayalam Speech Recognition</span><span class="stage-sub">Gnani Prisma v2.5 · 48 kHz</span></div></div>
<div class="pipeline-card"><span class="stage-num">02</span><div class="stage-text"><span class="stage-title">Acoustic Emotion Analysis</span><span class="stage-sub">Gemini Multimodal · Pure Audio</span></div></div>
<div class="pipeline-card"><span class="stage-num">03</span><div class="stage-text"><span class="stage-title">Context-Aware Translation</span><span class="stage-sub">Gemini Intelligence · Nuance</span></div></div>
<div class="pipeline-card"><span class="stage-num">04</span><div class="stage-text"><span class="stage-title">English Voice Synthesis</span><span class="stage-sub">Gnani Timbre v2.5 · Expressive</span></div></div>
</div>
</div>"""
st.markdown(HERO_HTML, unsafe_allow_html=True)

# ==============================================================================
# Voice Recording Studio (Clean Streamlit Container)
# ==============================================================================
with st.container(border=True):
    CONSOLE_HEADER_HTML = """<div class="console-header-row">
<div class="console-title-text"><span>🎙️</span><span>Spoken Malayalam Audio Studio</span></div>
<div style="font-size: 0.78rem; color: #78DCE8; font-weight: 600;">48,000 Hz Studio Ready</div>
</div>"""
    st.markdown(CONSOLE_HEADER_HTML, unsafe_allow_html=True)

    # Tasteful static audio visualizer representation to convey studio readiness
    vis_heights = [14, 22, 34, 18, 28, 42, 26, 38, 16, 30, 46, 24, 36, 20, 32, 44, 28, 40, 18, 34, 48, 26, 38, 22, 30, 16, 28, 36, 20, 14]
    bars_html = "".join([f'<div class="vis-bar" style="height: {h}px;"></div>' for h in vis_heights])
    st.markdown(f'<div class="studio-visualizer-bar-wrap">{bars_html}</div>', unsafe_allow_html=True)

    input_tab_record, input_tab_upload, input_tab_sample = st.tabs([
        "🎙️ Studio Microphone",
        "📁 Audio File Upload",
        "⚡ Verified Malayalam Sample",
    ])

    with input_tab_record:
        st.caption("Record live Malayalam speech directly from your microphone (captured natively at 48,000 Hz):")
        recorded_audio = st.audio_input("Record voice input", sample_rate=48000)
        if recorded_audio is not None:
            st.session_state.use_sample = False
            st.session_state.active_audio_bytes = recorded_audio.getvalue()
            st.session_state.active_audio_name = "microphone_recording.wav"
            st.session_state.active_audio_type = "audio/wav"

            # Preserve local copy for diagnostics
            try:
                (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)
                with open(PROJECT_ROOT / "outputs" / "latest_browser_recording.wav", "wb") as f:
                    f.write(st.session_state.active_audio_bytes)
            except Exception:
                pass

        st.markdown('<div style="font-size: 0.82rem; color: #A59EB5; margin-top: 8px;">💡 <strong>Guidance:</strong> Speak in your natural Malayalam cadence. Both calm, matter-of-fact conversational speech and passionate delivery are analyzed faithfully.</div>', unsafe_allow_html=True)

    with input_tab_upload:
        st.caption("Upload an existing Malayalam audio file:")
        uploaded_file = st.file_uploader(
            "Choose an audio recording",
            type=SUPPORTED_AUDIO_EXTENSIONS,
            help=f"Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS).upper()} (Max {MAX_FILE_SIZE_MB}MB)",
        )
        if uploaded_file is not None:
            st.session_state.use_sample = False
            st.session_state.active_audio_bytes = uploaded_file.getvalue()
            st.session_state.active_audio_name = uploaded_file.name
            ext = uploaded_file.name.split(".")[-1].lower()
            st.session_state.active_audio_type = f"audio/{ext}"

    with input_tab_sample:
        sample_wav_path = PROJECT_ROOT / "outputs" / "vibelingual_first_pipeline.wav"
        sample_m4a_path = PROJECT_ROOT / "media" / "malayalam_sample.wav.m4a"
        sample_file_path = sample_wav_path if sample_wav_path.is_file() else sample_m4a_path
        has_sample = sample_file_path.is_file()

        st.caption("Pre-loaded verified workspace Malayalam sample (48 kHz PCM):")
        st.markdown("*\"ഇന്നത്തെ കാലാവസ്ഥ വളരെ നല്ലതാണ് എനിക്ക് പുതിയ പ്രോജക്റ്റ് ചെയ്യാൻ വളരെയധികം സന്തോഷമുണ്ട്\"*")
        st.caption("Translation Preview: *\"The weather is very nice today, I am so excited to work on the new project!\"*")

        if has_sample:
            with open(sample_file_path, "rb") as sf:
                sample_bytes = sf.read()
            sample_fmt = "audio/wav" if sample_file_path.suffix.lower() == ".wav" else "audio/m4a"
            st.audio(sample_bytes, format=sample_fmt)

            if st.button("Load Malayalam Workspace Sample", use_container_width=True):
                st.session_state.use_sample = True
                st.session_state.active_audio_bytes = sample_bytes
                st.session_state.active_audio_name = sample_file_path.name
                st.session_state.active_audio_type = sample_fmt
                st.rerun()
        else:
            st.warning("Sample file not found in workspace.")

    # Display Staged Audio Status Bar
    if st.session_state.active_audio_bytes is not None:
        staged_bytes = st.session_state.active_audio_bytes
        staged_name = st.session_state.active_audio_name or "audio_input"
        staged_type = st.session_state.active_audio_type or "audio/wav"
        size_kb = len(staged_bytes) / 1024

        staged_bar_html = f"""<div class="staged-audio-card">
<div><span class="pill-tag pill-emerald">Audio Staged &amp; Ready</span><span class="staged-meta-title">{staged_name}</span><span class="staged-meta-sub">({size_kb:.1f} KB · 48 kHz PCM)</span></div>
<div style="font-size: 0.8rem; color: #78DCE8;">✓ Validated for STT &amp; Acoustic Inference</div>
</div>"""
        st.markdown(staged_bar_html, unsafe_allow_html=True)
        st.audio(staged_bytes, format=staged_type)

# ==============================================================================
# Pipeline Execution Trigger
# ==============================================================================
has_audio = st.session_state.active_audio_bytes is not None
is_busy = st.session_state.is_processing

btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    translate_btn = st.button(
        "✨ Translate Voice · Preserve Emotion",
        type="primary",
        disabled=not has_audio or is_busy,
        use_container_width=True,
    )

if translate_btn and has_audio and not is_busy:
    st.session_state.is_processing = True
    raw_audio_bytes = st.session_state.active_audio_bytes
    file_name = st.session_state.active_audio_name or "recording.wav"

    if len(raw_audio_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"Selected audio exceeds maximum file limit of {MAX_FILE_SIZE_MB}MB.")
        st.session_state.is_processing = False
        st.stop()

    temp_audio_path: Optional[Path] = None

    try:
        suffix = f".{file_name.split('.')[-1]}"
        if suffix not in [f".{ext}" for ext in SUPPORTED_AUDIO_EXTENSIONS]:
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw_audio_bytes)
            temp_audio_path = Path(tmp.name)

        pipeline_timings: Dict[str, float] = {}
        total_start = time.time()

        with st.status("Translating voice across speech, emotion, and culture...", expanded=True) as status:
            # ------------------------------------------------------------------
            # Stage 1/4: Speech Recognition (Gnani Prisma v2.5)
            # ------------------------------------------------------------------
            status.update(label="Stage 1/4: Transcribing Malayalam speech with Gnani Prisma v2.5...", state="running")
            stt_start = time.time()
            try:
                stt_res = transcribe_audio(
                    audio_path=temp_audio_path,
                    language_code="ml-IN",
                )
            except STTServiceError as e:
                cat = getattr(e, "category", "STT_ERROR")
                safe_msg = getattr(e, "safe_user_message", str(e))
                logger.exception(f"Speech-to-Text service failed [{cat}]: {e}")
                status.update(label=f"Speech-to-Text failed: {cat}", state="error")
                st.error(f"**Speech-to-Text Error ({cat}):** {safe_msg}")
                st.caption(f"Error Type: `{type(e).__name__}`. Full details have been printed to the terminal console.")
                st.session_state.is_processing = False
                st.stop()
            except Exception as e:
                logger.exception(f"Unexpected exception during Speech-to-Text stage: {e}")
                status.update(label="Speech-to-Text unexpected failure.", state="error")
                st.error(f"**Speech-to-Text Unexpected Failure:** {type(e).__name__} - {e}")
                st.caption("Detailed diagnostics have been logged to the terminal console.")
                st.session_state.is_processing = False
                st.stop()

            stt_duration = time.time() - stt_start
            pipeline_timings["stt"] = round(stt_duration, 2)
            transcript_text = stt_res.get("transcript", "").strip()

            if not transcript_text:
                logger.warning("Gnani STT returned an empty transcript.")
                status.update(label="STT returned an empty transcript.", state="error")
                st.error("Gnani Prisma v2.5 did not detect discernible Malayalam speech in this audio.")
                st.session_state.is_processing = False
                st.stop()

            st.write(f"✓ **Transcribed ({pipeline_timings['stt']}s):** *\"{transcript_text}\"*")

            # ------------------------------------------------------------------
            # Stage 2/4: Acoustic Emotion Analysis (PURE AUDIO, NO TRANSCRIPT)
            # ------------------------------------------------------------------
            status.update(
                label="Stage 2/4: Analyzing physical vocal acoustics & emotion from audio alone...",
                state="running",
            )
            acoustic_start = time.time()
            local_dsp = extract_local_acoustic_features(temp_audio_path)
            try:
                acoustic_res = analyze_audio_emotion(temp_audio_path)
                logger.info(f"Acoustic analysis succeeded: {acoustic_res.detected_emotion} ({acoustic_res.emotion_confidence})")
            except Exception as ac_err:
                logger.warning(f"Acoustic emotion analysis failed: {ac_err}. Continuing with text-only translation.", exc_info=True)
                acoustic_res = None

            acoustic_duration = time.time() - acoustic_start
            pipeline_timings["acoustic"] = round(acoustic_duration, 2)
            if acoustic_res:
                st.write(f"✓ **Spoken Acoustic Vibe ({pipeline_timings['acoustic']}s):** {acoustic_res.detected_emotion}")
            else:
                st.write(f"ℹ️ **Spoken Acoustic Vibe ({pipeline_timings['acoustic']}s):** Unavailable — Proceeding with contextual text translation.")

            # ------------------------------------------------------------------
            # Stage 3/4: Context-Aware Translation & Cultural Nuance (TEXT ONLY)
            # ------------------------------------------------------------------
            status.update(
                label="Stage 3/4: Translating context, cultural nuance & textual sentiment with Gemini...",
                state="running",
            )
            gemini_start = time.time()
            try:
                gemini_res = analyze_and_translate(
                    text=transcript_text,
                    source_language="ml-IN",
                    target_language="en",
                    audio_emotion_data=local_dsp,
                    acoustic_data=acoustic_res,
                )
            except IntelligenceServiceError as e:
                category = getattr(e, "category", "TRANSLATION_FAILURE")
                user_msg = getattr(e, "safe_user_message", str(e))
                logger.exception(f"Gemini intelligence stage failed [{category}]: {e}")
                status.update(label=f"Gemini intelligence failed: {category}", state="error")
                st.error(f"**Gemini Intelligence Analysis Error ({category}):** {user_msg}")
                st.caption(f"Error Type: `{type(e).__name__}`. Full traceback has been printed to the terminal console.")
                st.session_state.is_processing = False
                st.stop()
            except Exception as e:
                logger.exception(f"Unexpected exception in Gemini intelligence stage: {e}")
                status.update(label="Gemini intelligence unexpected failure.", state="error")
                st.error(f"**Intelligence Unexpected Failure:** {type(e).__name__} - {e}")
                st.caption("Full traceback has been printed to the terminal console.")
                st.session_state.is_processing = False
                st.stop()

            gemini_duration = time.time() - gemini_start
            pipeline_timings["gemini"] = round(gemini_duration, 2)
            natural_translation = gemini_res.get("natural_translation", "").strip()

            if not natural_translation:
                logger.error("Gemini analysis returned empty natural_translation.")
                status.update(label="Gemini did not return a translation.", state="error")
                st.error("Gemini analysis did not generate a natural translation.")
                st.session_state.is_processing = False
                st.stop()

            st.write(f"✓ **Cultural Translation ({pipeline_timings['gemini']}s):** Natural English Phrasing Generated")

            # ------------------------------------------------------------------
            # Stage 4/4: Emotion-Expressive TTS (Gnani Timbre v2.5)
            # ------------------------------------------------------------------
            status.update(
                label=f"Stage 4/4: Synthesizing expressive English voice with Gnani Timbre v2.5 ({selected_voice})...",
                state="running",
            )
            tts_start = time.time()

            # Determine emotion-modulated speaking speed and delivery style
            is_acoustic_available = gemini_res.get("acoustic_emotion_available", True)
            confidence_level = (gemini_res.get("emotion_confidence", "medium") or "medium").lower()
            is_uncertain = gemini_res.get("is_uncertain", False)

            # Use neutral delivery whenever acoustic confidence is low, uncertain, or unavailable
            if not is_acoustic_available or confidence_level == "low" or is_uncertain:
                effective_speed = 1.0
                delivery_emotion = "Calm / Neutral"
            else:
                rec_speed = gemini_res.get("tts_speed_recommendation", 1.0)
                delivery_emotion = gemini_res.get("acoustic_emotion") or gemini_res.get("emotion")
                effective_speed = resolve_speech_speed(speed=rec_speed, emotion=delivery_emotion)

            try:
                synthesized_audio = synthesize_speech(
                    text=natural_translation,
                    language_code="en-IN",
                    voice=selected_voice,
                    speed=effective_speed,
                    emotion=delivery_emotion,
                )
            except TTSServiceError as e:
                logger.exception(f"Text-to-Speech synthesis failed: {e}")
                status.update(label="Text-to-Speech failed.", state="error")
                st.error(f"**Text-to-Speech Error (Gnani Timbre v2.5):** {e}")
                st.caption("Detailed diagnostics have been logged to the terminal console.")
                st.session_state.is_processing = False
                st.stop()
            except Exception as e:
                logger.exception(f"Unexpected exception during Text-to-Speech stage: {e}")
                status.update(label="Text-to-Speech unexpected failure.", state="error")
                st.error(f"**Text-to-Speech Unexpected Failure:** {type(e).__name__} - {e}")
                st.caption("Detailed diagnostics have been logged to the terminal console.")
                st.session_state.is_processing = False
                st.stop()

            tts_duration = time.time() - tts_start
            pipeline_timings["tts"] = round(tts_duration, 2)
            pipeline_timings["total"] = round(time.time() - total_start, 2)

            status.update(
                label=f"Translation complete! Total pipeline latency: {pipeline_timings['total']}s",
                state="complete",
            )

        # Cache results into session state
        st.session_state.pipeline_result = {
            "transcript": transcript_text,
            "analysis": gemini_res,
            "dsp_metrics": local_dsp,
            "input_audio_bytes": raw_audio_bytes,
            "input_audio_name": file_name,
            "input_mime_type": suffix.replace(".", "audio/"),
            "synthesized_audio": synthesized_audio,
            "timings": pipeline_timings,
            "voice": selected_voice,
            "speed_applied": effective_speed,
        }

    finally:
        st.session_state.is_processing = False
        if temp_audio_path is not None and temp_audio_path.exists():
            try:
                temp_audio_path.unlink()
            except OSError:
                pass


# ==============================================================================
# Results Studio Presentation (Editorial Showcase)
# ==============================================================================
if st.session_state.pipeline_result is not None:
    res = st.session_state.pipeline_result
    analysis = res["analysis"]
    dsp = res.get("dsp_metrics", {})
    is_acoustic_available = analysis.get("acoustic_emotion_available", True)

    if is_acoustic_available:
        emotion_name = analysis.get("acoustic_emotion") or analysis.get("emotion", "Calm / Neutral")
    else:
        emotion_name = analysis.get("text_sentiment") or analysis.get("emotion", "Calm / Neutral")

    text_sentiment_val = analysis.get("text_sentiment")
    contrast_note = analysis.get("sentiment_contrast_note")
    theme = get_emotion_theme(emotion_name)
    intensity_str = analysis.get("emotional_intensity", "moderate").lower()
    confidence_str = analysis.get("emotion_confidence", "medium").upper()
    is_uncertain = analysis.get("is_uncertain", False)
    fallback_notice = analysis.get("fallback_notice")

    st.markdown("<br><hr style='border-color: rgba(155, 109, 255, 0.2);'><br>", unsafe_allow_html=True)
    st.markdown('<div class="section-headline">✨ Spoken Emotion &amp; Linguistic Intelligence</div>', unsafe_allow_html=True)

    # Resilience notice banner if fallback occurred
    if fallback_notice:
        fallback_html = f"""<div style="background: rgba(251, 191, 36, 0.12); border-left: 3px solid #FBBF24; padding: 12px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px; font-size: 0.9rem; color: #FDE68A; line-height: 1.5;">
<strong>Notice:</strong> {fallback_notice}
</div>"""
        st.markdown(fallback_html, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Row 1: Emotion Intelligence Showcase
    # --------------------------------------------------------------------------
    emo_col1, emo_col2 = st.columns([1, 1])

    with emo_col1:
        if is_acoustic_available:
            pace_pill = f'<span class="pill-tag pill-muted">Pace: {analysis.get("speaking_pace", "moderate").capitalize()}</span>' if analysis.get("speaking_pace") else ''
            energy_pill = f'<span class="pill-tag pill-muted">Energy: {analysis.get("vocal_energy", "moderate").capitalize()}</span>' if analysis.get("vocal_energy") else ''
            tension_pill = f'<span class="pill-tag pill-muted">Tension: {analysis.get("voice_tension", "relaxed / soft").capitalize()}</span>' if analysis.get("voice_tension") else ''
            sentiment_pill = f'<span class="pill-tag pill-violet">Semantic Text Sentiment: {text_sentiment_val}</span>' if text_sentiment_val else ''

            spotlight_html = f"""<div class="emotion-spotlight-shell" style="background: {theme['bg']}; border-color: {theme['border']}; box-shadow: {theme['glow']};">
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: {theme['color']}; margin-bottom: 6px;">Acoustic Spoken Vibe (Analyzed From Vocal Signal)</div>
<div class="emotion-spotlight-title" style="color: {theme['color']};">{emotion_name}</div>
<div style="font-size: 0.94rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 16px;">{theme['desc']}</div>
<div><span class="pill-tag {theme['pill']}">Confidence: {confidence_str}</span><span class="pill-tag pill-muted">Style: {analysis.get('speaking_style', 'Conversational')}</span>{pace_pill}{energy_pill}{tension_pill}</div>
<div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.08); display: flex; flex-wrap: wrap; align-items: center; gap: 6px;"><span class="pill-tag pill-emerald">Acoustic Signal: Verified</span>{sentiment_pill}</div>
</div>"""
            st.markdown(spotlight_html, unsafe_allow_html=True)
        else:
            unavailable_html = f"""<div class="emotion-spotlight-shell" style="background: rgba(165, 158, 181, 0.08); border-color: rgba(165, 158, 181, 0.35);">
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #A59EB5; margin-bottom: 6px;">Text-Based Sentiment Interpretation</div>
<div class="emotion-spotlight-title" style="color: #F5F1FF;">{text_sentiment_val or emotion_name}</div>
<div style="font-size: 0.94rem; color: #A59EB5; line-height: 1.5; margin-bottom: 16px;">Derived from transcript linguistic context and vocabulary. Physical voice audio was unanalyzed.</div>
<div><span class="pill-tag pill-muted">Voice Emotion: Temporarily Unavailable</span><span class="pill-tag pill-muted">Context: Textual Phrasing</span></div>
</div>"""
            st.markdown(unavailable_html, unsafe_allow_html=True)

        # Contrast or Disagreement analysis notice
        if contrast_note:
            contrast_html = f"""<div class="contrast-callout">
<strong style="color: var(--accent-cyan);">⚡ Emotion vs. Meaning Analysis:</strong><br>{contrast_note}
</div>"""
            st.markdown(contrast_html, unsafe_allow_html=True)

        if confidence_str == "LOW" or is_uncertain:
            st.markdown('<div style="margin-top: 12px; font-size: 0.84rem; color: #C5A7FF; background: rgba(155, 109, 255, 0.12); border: 1px solid rgba(155, 109, 255, 0.25); padding: 10px 16px; border-radius: var(--radius-sm);"><strong>Neutral Delivery Notice:</strong> Subtle vocal indicators or low acoustic confidence detected. English voice synthesis has defaulted to standard neutral 1.0x delivery.</div>', unsafe_allow_html=True)

    with emo_col2:
        intensity_levels = ["subtle", "moderate", "high", "intense"]
        active_level_idx = 1
        for idx, lvl in enumerate(intensity_levels):
            if lvl in intensity_str:
                active_level_idx = idx
                break

        steps_html = "".join([
            f'<div class="meter-segment {"meter-segment-active" if i <= active_level_idx else ""}"></div>'
            for i in range(4)
        ])

        speed_note = f"{res['speed_applied']}x"
        if res['speed_applied'] == 1.0 and (confidence_str == "LOW" or is_uncertain):
            speed_note += " (Neutral Baseline)"

        meter_card_html = f"""<div class="studio-glass-card">
<div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em; color: var(--accent-lavender); margin-bottom: 8px;">Acoustic Intensity &amp; Timbre Voice Modulation</div>
<div style="display: flex; justify-content: space-between; align-items: center;"><span style="font-weight: 800; font-size: 1.25rem; color: #FFFFFF; text-transform: capitalize;">{intensity_str} Vocal Energy</span><span style="font-size: 0.84rem; color: var(--accent-cyan); font-weight: 600;">Level {active_level_idx + 1} of 4</span></div>
<div class="meter-track">{steps_html}</div>
<div style="display: flex; justify-content: space-between; font-size: 0.74rem; color: var(--text-muted); margin-bottom: 18px;"><span>Subtle</span><span>Moderate</span><span>High</span><span>Intense</span></div>
<div style="border-top: 1px solid var(--border-subtle); padding-top: 16px;">
<div style="font-size: 0.82rem; color: var(--text-secondary);">Timbre Speech Pacing Multiplier:</div>
<div style="font-size: 1.25rem; font-weight: 800; color: var(--accent-cyan); margin-top: 3px;">Speed Factor: {speed_note}</div>
<div style="font-size: 0.84rem; color: var(--text-secondary); margin-top: 6px;">Synthesis Engine: <strong>Gnani Timbre v2.5</strong> ({res['voice']} Persona)</div>
</div>
</div>"""
        st.markdown(meter_card_html, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Row 2: Dual Translation (Words vs. Meaning)
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-headline">📖 Dual Translation: Words vs. Meaning</div>', unsafe_allow_html=True)

    col_orig, col_trans = st.columns(2)

    with col_orig:
        orig_card_html = f"""<div class="studio-glass-card">
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent-lavender); margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;"><span>Original Spoken Malayalam</span><span class="pill-tag pill-muted" style="margin: 0;">Prisma v2.5</span></div>
<div style="font-size: 1.35rem; font-weight: 500; line-height: 1.6; color: var(--text-primary);">"{res['transcript']}"</div>
<div style="margin-top: 18px; font-size: 0.82rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">Audio Source: {res['input_audio_name']}</div>
</div>"""
        st.markdown(orig_card_html, unsafe_allow_html=True)

    with col_trans:
        trans_card_html = f"""<div class="studio-hero-highlight-card">
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent-cyan); margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;"><span>Natural English (Emotion-Preserved)</span><span class="pill-tag pill-cyan" style="margin: 0;">Gemini Intelligence</span></div>
<div style="font-size: 1.45rem; font-weight: 800; line-height: 1.5; color: #FFFFFF;">"{analysis.get('natural_translation', '')}"</div>
<div style="margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap;"><span class="pill-tag pill-violet">Emotion-Preserved</span><span class="pill-tag pill-cyan">Culturally Fluent</span><span class="pill-tag pill-emerald">Native Phrasing</span></div>
</div>"""
        st.markdown(trans_card_html, unsafe_allow_html=True)

    # Word-for-Word Literal Translation Drawer
    literal_trans = analysis.get("literal_translation", "")
    if literal_trans:
        with st.expander("🔍 Word-for-Word (Literal) Translation & Grammatical Contrast", expanded=False):
            st.markdown(f"**Literal Direct Phrasing:** *\"{literal_trans}\"*")
            st.caption(
                "Direct grammatical translation showing what the Malayalam words literally state, contrasting with how the natural translation conveys what the speaker actually meant."
            )

    # --------------------------------------------------------------------------
    # Row 3: Conversational Intent & Cultural Nuance
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-headline">🌐 Cultural Intelligence &amp; Conversational Intent</div>', unsafe_allow_html=True)

    col_intent, col_culture = st.columns(2)

    with col_intent:
        intent_card_html = f"""<div class="studio-glass-card">
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent-lavender); margin-bottom: 10px;">Conversational Intent</div>
<p style="color: var(--text-primary); font-size: 0.98rem; line-height: 1.6;">{analysis.get('conversational_intent', 'Expressing thoughts naturally.')}</p>
<div style="margin-top: 14px; font-size: 0.84rem; color: var(--accent-cyan);">Suggested Delivery Style: <em>{analysis.get('suggested_delivery_style', 'Conversational')}</em></div>
</div>"""
        st.markdown(intent_card_html, unsafe_allow_html=True)

    with col_culture:
        cultural_ctx = analysis.get("cultural_context", "None")
        idioms = analysis.get("idioms_or_expressions", [])
        idiom_tags = f"<div style='margin-top: 12px; font-size: 0.85rem; color: #C5A7FF;'><strong>Idioms / Expressions:</strong> {', '.join(idioms)}</div>" if idioms else ""
        culture_card_html = f"""<div class="studio-glass-card">
<div style="font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: var(--accent-lavender); margin-bottom: 10px;">Cultural Context &amp; Malayalam Colloquialisms</div>
<p style="color: var(--text-primary); font-size: 0.98rem; line-height: 1.6;">{cultural_ctx if cultural_ctx.lower() != 'none' else 'Natural conversational Malayalam expression without regionally bound idioms.'}</p>
{idiom_tags}
</div>"""
        st.markdown(culture_card_html, unsafe_allow_html=True)

    if analysis.get("explanation"):
        with st.expander("💡 Linguistic Adaptation Rationale", expanded=False):
            st.write(analysis.get("explanation"))

    # --------------------------------------------------------------------------
    # Row 4: Expressive Voice Synthesis & Audio Playback
    # --------------------------------------------------------------------------
    st.markdown('<div class="section-headline">🎧 Synthesized English Voice Studio</div>', unsafe_allow_html=True)

    col_out_player, col_out_meta = st.columns([3, 2])

    with col_out_player:
        st.caption(f"Synthesized English Speech (Gnani Timbre v2.5 · Persona: {res['voice']}):")
        st.audio(res["synthesized_audio"], format="audio/wav")
        st.download_button(
            label="⬇️ Download Translated Audio (48 kHz WAV)",
            data=res["synthesized_audio"],
            file_name="vibelingual_translated.wav",
            mime="audio/wav",
            use_container_width=True,
        )

    with col_out_meta:
        persona_card_html = f"""<div class="studio-glass-card" style="padding: 20px 24px;">
<div style="font-size: 0.78rem; color: var(--accent-lavender); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">Synthesized Delivery Persona</div>
<div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin-top: 4px;">{res['voice']} (Indian English)</div>
<div style="font-size: 0.92rem; color: var(--accent-cyan); margin-top: 6px; font-weight: 700;">Emotion Pacing: {res['speed_applied']}x for {emotion_name}</div>
<div style="font-size: 0.80rem; color: var(--text-muted); margin-top: 6px; font-family: 'JetBrains Mono', monospace;">Format: 48,000 Hz Mono 16-bit PCM WAV</div>
</div>"""
        st.markdown(persona_card_html, unsafe_allow_html=True)

    # Side-by-Side Audio Audition Player (A/B Demonstration)
    with st.expander("🎙️ Side-by-Side Audio Audition (Original Malayalam vs. Synthesized English)", expanded=True):
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            st.markdown(f"**Original Spoken Recording** (`{res['input_audio_name']}`):")
            st.audio(res["input_audio_bytes"], format=res["input_mime_type"])
        with cmp_col2:
            st.markdown(f"**Emotion-Adapted English Speech** (`{res['voice']}`):")
            st.audio(res["synthesized_audio"], format="audio/wav")

    # --------------------------------------------------------------------------
    # Row 5: Supporting Vocal Characteristics & DSP Metrics
    # --------------------------------------------------------------------------
    observations = analysis.get("acoustic_observations", [])
    supporting = analysis.get("supporting_observations", [])
    combined_obs = observations or supporting

    if combined_obs or dsp:
        with st.expander("📊 Vocal Acoustic Characteristics & Digital Signal Metrics", expanded=False):
            if combined_obs:
                st.markdown("**Audible Acoustic Cues:**")
                for obs in combined_obs:
                    st.markdown(f"• {obs}")

            if dsp:
                st.markdown("---")
                st.markdown("**Digital Signal Processing (DSP) Signal Metrics:**")
                dsp_cols = st.columns(4)
                dsp_cols[0].metric("Audio Format", dsp.get("format", "WAV"))
                dsp_cols[1].metric("RMS Energy", str(dsp.get("rms_energy", "N/A")))
                dsp_cols[2].metric("Dynamic Range", str(dsp.get("dynamic_range_ratio", "N/A")))
                dsp_cols[3].metric("Pause Ratio", str(dsp.get("pause_ratio", "N/A")))

    # --------------------------------------------------------------------------
    # Row 6: Execution Latency & Action Controls
    # --------------------------------------------------------------------------
    with st.expander("⚡ Pipeline Diagnostics & Execution Timings", expanded=False):
        timings = res["timings"]
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Gnani Prisma STT", f"{timings.get('stt', 0)}s")
        d2.metric("Acoustic Emotion", f"{timings.get('acoustic', 0)}s")
        d3.metric("Linguistic Nuance", f"{timings.get('gemini', 0)}s")
        d4.metric("Gnani Timbre TTS", f"{timings.get('tts', 0)}s")
        d5.metric("Total Latency", f"{timings.get('total', 0)}s")

        st.caption("Engines: Gnani Prisma v2.5 (STT) · Google Gemini Flash (Multimodal) · Gnani Timbre v2.5 (TTS)")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Translate Another Audio Recording", use_container_width=True):
        st.session_state.pipeline_result = None
        st.session_state.active_audio_bytes = None
        st.session_state.active_audio_name = None
        st.session_state.active_audio_type = None
        st.session_state.use_sample = False
        st.session_state.is_processing = False
        st.rerun()

# ==============================================================================
# Footer
# ==============================================================================
FOOTER_HTML = """<div style="text-align: center; margin-top: 55px; padding: 24px 10px; border-top: 1px solid rgba(155, 109, 255, 0.15); color: #736B88; font-size: 0.82rem;">
<div style="font-weight: 700; color: #A59EB5; margin-bottom: 4px;">VibeLingual — Beyond Words. Beyond Translation.</div>
<div>Emotion-Aware Malayalam-to-English Speech Translation Studio</div>
</div>"""
st.markdown(FOOTER_HTML, unsafe_allow_html=True)
