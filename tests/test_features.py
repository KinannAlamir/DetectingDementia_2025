"""Tests for features module."""

import numpy as np
import pandas as pd

from dementia_detection.features import create_feature_matrix, extract_linguistic_features


def test_extract_linguistic_features_control(sample_transcript_control):
    """Test linguistic feature extraction for control transcript."""
    features = extract_linguistic_features(sample_transcript_control)

    assert isinstance(features, dict)
    assert len(features) == 8

    assert features["word_count"] > 0
    assert 0 < features["unique_word_ratio"] <= 1
    assert features["avg_word_length"] > 0
    assert features["sentence_count"] > 0
    assert features["avg_sentence_length"] > 0
    assert features["hesitation_count"] == 0  # No hesitations in control
    assert features["repetition_ratio"] >= 0
    assert features["pause_count"] >= 0


def test_extract_linguistic_features_dementia(sample_transcript_dementia):
    """Test linguistic feature extraction for dementia transcript."""
    features = extract_linguistic_features(sample_transcript_dementia)

    assert features["hesitation_count"] > 0  # Has um, uh, mhm
    assert features["repetition_ratio"] > 0  # Has "girl girl", "wipe wipe"
    assert features["unique_word_ratio"] < 1  # Has repetitions


def test_extract_linguistic_features_empty():
    """Test linguistic feature extraction for empty text."""
    features = extract_linguistic_features("")

    assert features["word_count"] == 0
    assert features["unique_word_ratio"] == 0
    assert features["avg_word_length"] == 0
    assert features["sentence_count"] == 0


def test_extract_linguistic_features_none():
    """Test linguistic feature extraction for None."""
    features = extract_linguistic_features(None)

    assert features["word_count"] == 0
    assert all(v == 0 for v in features.values())


def test_extract_linguistic_features_nan():
    """Test linguistic feature extraction for NaN."""
    features = extract_linguistic_features(np.nan)

    assert features["word_count"] == 0
    assert all(v == 0 for v in features.values())


def test_create_feature_matrix(sample_dataframe):
    """Test feature matrix creation from dataframe."""
    feature_df = create_feature_matrix(sample_dataframe)

    assert isinstance(feature_df, pd.DataFrame)
    assert len(feature_df) == 3
    assert len(feature_df.columns) == 8

    # Check column names
    expected_cols = [
        "word_count",
        "unique_word_ratio",
        "avg_word_length",
        "sentence_count",
        "avg_sentence_length",
        "hesitation_count",
        "repetition_ratio",
        "pause_count",
    ]
    assert all(col in feature_df.columns for col in expected_cols)


def test_hesitation_detection():
    """Test detection of various hesitation markers."""
    text_with_hesitations = "um well uh I think &um this is &uh correct mhm"
    features = extract_linguistic_features(text_with_hesitations)

    assert features["hesitation_count"] >= 4  # um, uh, &um, &uh, mhm


def test_pause_detection():
    """Test detection of pauses."""
    text_with_pauses = "The boy (.) is taking (.) cookies (.)"
    features = extract_linguistic_features(text_with_pauses)

    assert features["pause_count"] == 3


def test_repetition_detection():
    """Test detection of word repetitions."""
    text = "the the boy is is is taking cookies"
    features = extract_linguistic_features(text)

    assert features["repetition_ratio"] > 0
    # Should detect "the the" and "is is" (consecutive repetitions)


def test_unique_word_ratio():
    """Test unique word ratio calculation."""
    text_unique = "all different words here"
    text_repetitive = "word word word word"

    features_unique = extract_linguistic_features(text_unique)
    features_repetitive = extract_linguistic_features(text_repetitive)

    assert features_unique["unique_word_ratio"] > features_repetitive["unique_word_ratio"]
