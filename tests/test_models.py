"""Tests for models module."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from dementia_detection.models import (
    create_ensemble,
    get_models,
    get_tuned_models,
    train_and_evaluate,
)


def test_get_models():
    """Test that get_models returns expected models."""
    models = get_models()

    assert len(models) == 3
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "SVM" in models

    assert isinstance(models["Logistic Regression"], LogisticRegression)
    assert isinstance(models["Random Forest"], RandomForestClassifier)
    assert isinstance(models["SVM"], SVC)


def test_train_and_evaluate():
    """Test model training and evaluation."""
    # Create simple synthetic data
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.rand(20, 10)
    y_val = np.random.randint(0, 2, 20)

    model = LogisticRegression(random_state=42)
    metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val)

    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert "report" in metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert isinstance(metrics["report"], str)


def test_train_and_evaluate_without_fit():
    """Test evaluation of pre-trained model."""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.rand(20, 10)
    y_val = np.random.randint(0, 2, 20)

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate without fitting again
    metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val, fit=False)

    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_get_tuned_models():
    """Test hyperparameter tuning."""
    np.random.seed(42)
    X_train = np.random.rand(50, 10)
    y_train = np.random.randint(0, 2, 50)

    tuned_models = get_tuned_models(X_train, y_train)

    assert len(tuned_models) == 3
    assert "Logistic Regression (Tuned)" in tuned_models
    assert "Random Forest (Tuned)" in tuned_models
    assert "SVM (Tuned)" in tuned_models

    # Check that models are fitted
    for model in tuned_models.values():
        assert hasattr(model, "predict")


def test_create_ensemble():
    """Test ensemble creation."""
    models = {
        "LR": LogisticRegression(random_state=42),
        "RF": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    ensemble = create_ensemble(models)

    assert isinstance(ensemble, VotingClassifier)
    assert ensemble.voting == "soft"
    assert len(ensemble.estimators) == 3


def test_ensemble_prediction():
    """Test that ensemble can make predictions."""
    np.random.seed(42)
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)

    models = {
        "LR": LogisticRegression(random_state=42),
        "RF": RandomForestClassifier(random_state=42, n_estimators=10),
        "SVM": SVC(probability=True, random_state=42),
    }

    # Train models
    for model in models.values():
        model.fit(X_train, y_train)

    ensemble = create_ensemble(models)
    ensemble.fit(X_train, y_train)

    predictions = ensemble.predict(X_test)

    assert len(predictions) == 20
    assert all(p in [0, 1] for p in predictions)


def test_model_random_state():
    """Test that models use consistent random states."""
    models = get_models()

    assert models["Logistic Regression"].random_state == 42
    assert models["Random Forest"].random_state == 42
    assert models["SVM"].random_state == 42


def test_svm_probability():
    """Test that SVM has probability enabled for ensemble."""
    models = get_models()

    assert models["SVM"].probability is True
