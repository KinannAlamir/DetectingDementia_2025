"""Data loading and preparation for dementia detection."""

from pathlib import Path

import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dementia_detection.features import create_feature_matrix


def load_data(data_dir: Path = Path("data/input")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine training datasets with labels."""
    control = pd.read_csv(data_dir / "Control_db.csv").assign(label=0)
    dementia = pd.read_csv(data_dir / "Dementia_db.csv").assign(label=1)
    return pd.concat([control, dementia], ignore_index=True), pd.read_csv(
        data_dir / "Testing_db.csv"
    )


def prepare_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int = 500
) -> tuple:
    """Extract TF-IDF and linguistic features from transcripts."""
    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_train = vectorizer.fit_transform(train_df["Transcript"].fillna(""))
    tfidf_test = vectorizer.transform(test_df["Transcript"].fillna(""))

    # Linguistic features
    ling_train = create_feature_matrix(train_df)
    ling_test = create_feature_matrix(test_df)

    # Scale linguistic features
    scaler = StandardScaler()
    ling_train_scaled = scaler.fit_transform(ling_train)
    ling_test_scaled = scaler.transform(ling_test)

    # Combine features
    X_train = hstack([tfidf_train, ling_train_scaled])
    X_test = hstack([tfidf_test, ling_test_scaled])

    return (
        X_train,
        train_df["label"],
        X_test,
        test_df.get("Category", None),
        (vectorizer, scaler),
    )


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Split data into train and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
