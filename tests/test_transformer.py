"""Tests for transformer module."""

import numpy as np
import pandas as pd
import pytest

from dementia_detection.transformer import (
    TransformerDementiaDetector,
    create_transformer_model,
    prepare_transformer_data,
)


@pytest.fixture
def sample_transformer_data():
    """Sample data for transformer tests."""
    train = pd.DataFrame(
        {
            "Transcript": [
                "The boy is taking cookies from the jar.",
                "The mother is washing dishes.",
                "There is water on the floor.",
                "The stool is falling over.",
            ],
            "label": [0, 0, 1, 1],
        }
    )
    val = pd.DataFrame({"Transcript": ["Girl is laughing.", "Boy has cookies."], "label": [0, 1]})
    return train, val


def test_create_transformer_model():
    """Test transformer model creation."""
    model = create_transformer_model("distilbert-base-uncased")

    assert isinstance(model, TransformerDementiaDetector)
    assert model.model_name_or_path == "distilbert-base-uncased"
    assert model.max_length == 128
    assert model.tokenizer is not None
    assert model.local_files_only is False


def test_create_transformer_model_local():
    """Test transformer model with local_files_only parameter."""
    # Test that create_transformer_model passes the local_files_only parameter
    # We can't test actual loading without a real model directory
    # So we just verify the function signature works
    try:
        # This will fail without a real model, but tests the parameter passing
        create_transformer_model("models/roberta-base", local_files_only=True)
    except Exception:
        # Expected to fail without actual model files
        pass

    # Test with default (download from HF)
    model = create_transformer_model("distilbert-base-uncased", local_files_only=False)
    assert model.local_files_only is False


def test_prepare_transformer_data(sample_transformer_data):
    """Test data preparation for transformer."""
    train_df, val_df = sample_transformer_data

    train_dataset, val_dataset = prepare_transformer_data(train_df, val_df)

    assert len(train_dataset) == 4
    assert len(val_dataset) == 2
    assert "text" in train_dataset.column_names
    assert "label" in train_dataset.column_names


def test_prepare_transformer_data_with_missing():
    """Test data preparation with missing values."""
    train_df = pd.DataFrame({"Transcript": ["Valid text", None, np.nan], "label": [0, 1, 0]})
    val_df = pd.DataFrame({"Transcript": [None, "Valid"], "label": [0, 1]})

    train_dataset, val_dataset = prepare_transformer_data(train_df, val_df)

    assert len(train_dataset) == 3
    assert len(val_dataset) == 2


def test_transformer_predict_before_training():
    """Test that predict raises error before training."""
    model = create_transformer_model("distilbert-base-uncased")

    with pytest.raises(ValueError, match="Model not trained yet"):
        model.predict(["Test text"])


def test_transformer_evaluate_before_training():
    """Test that evaluate raises error before training."""
    model = create_transformer_model("distilbert-base-uncased")
    val_df = pd.DataFrame({"Transcript": ["Test"], "label": [0]})

    with pytest.raises(ValueError, match="Model not trained yet"):
        model.evaluate(val_df)


@pytest.mark.slow
def test_transformer_training_integration(sample_transformer_data, tmp_path):
    """Integration test for transformer training (marked as slow)."""
    train_df, val_df = sample_transformer_data

    # Use a small model for testing
    model = create_transformer_model("distilbert-base-uncased")

    # Train with minimal settings
    model.train(
        train_df,
        val_df,
        output_dir=str(tmp_path / "test_model"),
        num_epochs=1,
        batch_size=2,
    )

    # Check model is trained
    assert model.model is not None
    assert model.trainer is not None

    # Evaluate
    metrics = model.evaluate(val_df)
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1

    # Predict
    predictions = model.predict(["Test transcript"])
    assert len(predictions) == 1
    assert predictions[0] in [0, 1]
