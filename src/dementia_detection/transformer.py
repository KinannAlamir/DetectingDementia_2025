"""Transformer-based models for dementia detection."""

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def prepare_transformer_data(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple:
    """Prepare data for transformer training."""
    train_dataset = Dataset.from_dict(
        {"text": train_df["Transcript"].fillna("").tolist(), "label": train_df["label"].tolist()}
    )
    val_dataset = Dataset.from_dict(
        {"text": val_df["Transcript"].fillna("").tolist(), "label": val_df["label"].tolist()}
    )
    return train_dataset, val_dataset


def tokenize_data(dataset: Dataset, tokenizer, max_length: int = 128):
    """Tokenize dataset for transformer model."""
    return dataset.map(
        lambda x: tokenizer(
            x["text"], truncation=True, padding="max_length", max_length=max_length
        ),
        batched=True,
    )


def compute_metrics(eval_pred):
    """Compute metrics for transformer evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary"),
    }


class TransformerDementiaDetector:
    """RoBERTa-based dementia detector."""

    def __init__(
        self,
        model_name_or_path: str = "roberta-base",
        max_length: int = 128,
        local_files_only: bool = False,
    ):
        """Initialize transformer model.

        Args:
            model_name_or_path: Hugging Face model name or local path to model directory
            max_length: Maximum sequence length for tokenization
            local_files_only: If True, only use local files (no downloading)
        """
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.local_files_only = local_files_only
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        self.model = None
        self.trainer = None

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        output_dir: str = "./results",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ):
        """Train the transformer model."""
        # Prepare datasets
        train_dataset, val_dataset = prepare_transformer_data(train_df, val_df)

        # Tokenize
        train_dataset = tokenize_data(train_dataset, self.tokenizer, self.max_length)
        val_dataset = tokenize_data(val_dataset, self.tokenizer, self.max_length)

        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path, num_labels=2, local_files_only=self.local_files_only
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            report_to="none",
            seed=42,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train
        self.trainer.train()

        return self

    def evaluate(self, val_df: pd.DataFrame) -> dict:
        """Evaluate the model."""
        if self.trainer is None:
            raise ValueError("Model not trained yet")

        # Prepare validation dataset
        val_dataset = Dataset.from_dict(
            {"text": val_df["Transcript"].fillna("").tolist(), "label": val_df["label"].tolist()}
        )
        val_dataset = tokenize_data(val_dataset, self.tokenizer, self.max_length)

        # Evaluate
        results = self.trainer.evaluate(val_dataset)

        return {
            "accuracy": results["eval_accuracy"],
            "f1_score": results["eval_f1"],
            "loss": results["eval_loss"],
        }

    def predict(self, texts: list) -> np.ndarray:
        """Predict labels for texts."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Get device from model
        device = next(self.model.parameters()).device

        # Tokenize
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        return predictions.cpu().numpy()


def create_transformer_model(
    model_name_or_path: str = "roberta-base", local_files_only: bool = False
) -> TransformerDementiaDetector:
    """Create a transformer-based dementia detector.

    Args:
        model_name_or_path: Hugging Face model name or local path to model directory
        local_files_only: If True, only use local files (no downloading)

    Examples:
        # Download from Hugging Face (default)
        model = create_transformer_model("roberta-base")

        # Use local model
        model = create_transformer_model("./models/roberta-base", local_files_only=True)
    """
    return TransformerDementiaDetector(
        model_name_or_path=model_name_or_path, local_files_only=local_files_only
    )
