"""Main entry point for dementia detection."""

import sys
from pathlib import Path

from dementia_detection.data import load_data, prepare_features, split_data
from dementia_detection.models import (
    create_ensemble,
    get_models,
    get_tuned_models,
    train_and_evaluate,
)
from dementia_detection.transformer import create_transformer_model


def main():
    """Train and evaluate multiple ML models for dementia detection."""
    # Parse arguments
    use_transformer = "--transformer" in sys.argv or "-t" in sys.argv

    # Default to local XLM-RoBERTa model
    model_path = "/Users/alamir/Documents/Travail/perso/Programmation/xlm-roberta-base"
    local_files_only = True

    # Allow override with command-line argument
    for i, arg in enumerate(sys.argv):
        if arg in ["--model-path", "-m"] and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
            local_files_only = True  # Use local files when path is specified
            break
        elif arg in ["--download-model"]:
            # Flag to force download instead of using local
            model_path = "xlm-roberta-base"
            local_files_only = False
            break

    print("Loading data...")
    train_df, test_df = load_data(Path("data/input"))
    print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples\n")

    if use_transformer:
        print("=" * 70)
        print("TRANSFORMER MODEL")
        print("=" * 70)
        if local_files_only:
            print(f"Using local model from: {model_path}")
        else:
            print(f"Using model: {model_path}")
        print("Training transformer model (this may take a few minutes)...\n")

        # Prepare data for transformer (using raw dataframe with split)
        from sklearn.model_selection import train_test_split

        train_data, val_data = train_test_split(
            train_df, test_size=0.2, random_state=42, stratify=train_df["label"]
        )

        # Create and train transformer
        transformer = create_transformer_model(model_path, local_files_only=local_files_only)
        transformer.train(
            train_data.reset_index(drop=True),
            val_data.reset_index(drop=True),
            num_epochs=3,
            batch_size=8,
        )

        # Evaluate
        metrics = transformer.evaluate(val_data.reset_index(drop=True))
        print("\nTransformer Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Loss: {metrics['loss']:.4f}")

        return

    # Traditional ML pipeline
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
