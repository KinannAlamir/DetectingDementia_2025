"""Main entry point for dementia detection."""

from pathlib import Path

from dementia_detection.data import load_data, prepare_features, split_data
from dementia_detection.models import (
    create_ensemble,
    get_models,
    get_tuned_models,
    train_and_evaluate,
)


def main():
    """Train and evaluate multiple ML models for dementia detection."""
    print("Loading data...")
    train_df, test_df = load_data(Path("data/input"))
    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples\n")

    print("Preparing features (TF-IDF + Linguistic)...")
    X_full, y_full, X_test, y_test, transformers = prepare_features(train_df, test_df)
    X_train, X_val, y_train, y_val = split_data(X_full, y_full)
    print(
        f"Features: {X_train.shape[1]} (500 TF-IDF + 8 linguistic), "
        f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}\n"
    )

    # Baseline models
    print("=" * 70)
    print("BASELINE MODELS")
    print("=" * 70)
    baseline_results = {}
    for name, model in get_models().items():
        print(f"\n{name}")
        print("-" * 70)
        metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val)
        baseline_results[name] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1 Score: {metrics['f1_score']:.4f}")

    # Tuned models
    print("\n\n" + "=" * 70)
    print("TUNED MODELS (with GridSearchCV)")
    print("=" * 70)
    tuned_models = get_tuned_models(X_train, y_train)
    tuned_results = {}
    for name, model in tuned_models.items():
        print(f"\n{name}")
        print("-" * 70)
        metrics = train_and_evaluate(model, X_train, y_train, X_val, y_val, fit=False)
        tuned_results[name] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1 Score: {metrics['f1_score']:.4f}")

    # Ensemble
    print("\n\n" + "=" * 70)
    print("ENSEMBLE MODEL (Voting Classifier)")
    print("=" * 70)
    ensemble = create_ensemble(tuned_models)
    ensemble_metrics = train_and_evaluate(ensemble, X_train, y_train, X_val, y_val)
    print(f"Accuracy: {ensemble_metrics['accuracy']:.4f}")
    print(f"F1 Score: {ensemble_metrics['f1_score']:.4f}")
    print(f"\n{ensemble_metrics['report']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_results = {**baseline_results, **tuned_results, "Ensemble": ensemble_metrics}
    best_model = max(all_results.items(), key=lambda x: x[1]["f1_score"])
    print(f"Best model: {best_model[0]}")
    print(f"  Accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"  F1 Score: {best_model[1]['f1_score']:.4f}")


if __name__ == "__main__":
    main()
