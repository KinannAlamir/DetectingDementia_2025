"""Tests for data module."""

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from dementia_detection.data import load_data, prepare_features, split_data


def test_load_data(tmp_path):
    """Test data loading from CSV files."""
    # Create temporary CSV files
    control = pd.DataFrame(
        {"Transcript": ["Control text 1", "Control text 2"], "Category": [0, 0]}
    )
    dementia = pd.DataFrame(
        {"Transcript": ["Dementia text 1", "Dementia text 2"], "Category": [1, 1]}
    )
    testing = pd.DataFrame({"Transcript": ["Test text 1"], "Category": [0]})

    control.to_csv(tmp_path / "Control_db.csv", index=False)
    dementia.to_csv(tmp_path / "Dementia_db.csv", index=False)
    testing.to_csv(tmp_path / "Testing_db.csv", index=False)

    # Test loading
    train_df, test_df = load_data(tmp_path)

    assert len(train_df) == 4
    assert len(test_df) == 1
    assert "label" in train_df.columns
    assert list(train_df["label"].unique()) == [0, 1]


def test_prepare_features(sample_train_test_data):
    """Test feature preparation."""
    train_df, test_df = sample_train_test_data

    X_train, y_train, X_test, y_test, transformers = prepare_features(
        train_df, test_df, max_features=10
    )

    # Check sparse matrices
    assert issparse(X_train)
    assert issparse(X_test)

    # Check shapes
    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 2
    assert X_train.shape[1] == 18  # 10 TF-IDF + 8 linguistic

    # Check labels
    assert len(y_train) == 4
    assert list(y_train) == [0, 1, 0, 1]

    # Check transformers
    assert len(transformers) == 2


def test_split_data():
    """Test train/validation split."""
    X = np.random.rand(100, 10)
    y = np.array([0] * 50 + [1] * 50)

    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 80
    assert len(X_val) == 20
    assert len(y_train) == 80
    assert len(y_val) == 20

    # Check stratification
    assert np.isclose(y_train.mean(), 0.5, atol=0.1)
    assert np.isclose(y_val.mean(), 0.5, atol=0.1)


def test_prepare_features_with_missing_values():
    """Test feature preparation handles missing values."""
    train_df = pd.DataFrame(
        {"Transcript": ["Valid text", None, np.nan, ""], "label": [0, 1, 0, 1]}
    )
    test_df = pd.DataFrame({"Transcript": [None, "Valid"], "Category": [0, 1]})

    X_train, y_train, X_test, y_test, _ = prepare_features(
        train_df, test_df, max_features=5
    )

    # Should not raise errors
    assert X_train.shape[0] == 4
    assert X_test.shape[0] == 2
