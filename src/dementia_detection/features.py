"""Feature extraction for dementia detection."""

import re

import numpy as np
import pandas as pd


def extract_linguistic_features(text: str) -> dict:
    """Extract linguistic features from transcript text."""
    if pd.isna(text) or not text.strip():
        return {
            "word_count": 0,
            "unique_word_ratio": 0,
            "avg_word_length": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0,
            "hesitation_count": 0,
            "repetition_ratio": 0,
            "pause_count": 0,
        }

    # Basic text stats
    words = text.lower().split()
    word_count = len(words)
    unique_words = len(set(words))
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

    # Hesitation markers
    hesitations = len(re.findall(r"\b(um|uh|&uh|&um|er|ah|hm|mhm)\b", text.lower()))

    # Pauses and disfluencies
    pauses = len(re.findall(r"\(\.+\)", text))

    # Repetitions (simple detection of repeated words)
    repetitions = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])

    return {
        "word_count": word_count,
        "unique_word_ratio": unique_words / word_count if word_count > 0 else 0,
        "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        "sentence_count": len(sentences),
        "avg_sentence_length": word_count / len(sentences) if sentences else 0,
        "hesitation_count": hesitations,
        "repetition_ratio": repetitions / word_count if word_count > 0 else 0,
        "pause_count": pauses,
    }


def create_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create feature matrix from transcripts."""
    features = df["Transcript"].apply(extract_linguistic_features)
    return pd.DataFrame(features.tolist())
