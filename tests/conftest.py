"""Pytest configuration and fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_transcript_control():
    """Sample control transcript."""
    return (
        "well there's a mother standing there washing the dishes. "
        "and the window's open. and outside the window there's a curved walk. "
        "she's getting her feet wet from the overflow of the water."
    )


@pytest.fixture
def sample_transcript_dementia():
    """Sample dementia transcript with hesitations and repetitions."""
    return (
        "mhm there's a young boy um going in a cookie jar. "
        "and there's a girl girl. um I can't think of the uh. "
        "she has um she's trying to wipe wipe dishes."
    )


@pytest.fixture
def sample_dataframe():
    """Sample dataframe for testing."""
    return pd.DataFrame(
        {
            "Transcript": [
                "The boy is taking cookies from the jar.",
                "um the uh mother is washing washing dishes.",
                "There is water on the floor.",
            ],
            "label": [0, 1, 0],
            "Category": [0, 1, 0],
        }
    )


@pytest.fixture
def sample_train_test_data():
    """Sample train and test datasets."""
    train = pd.DataFrame(
        {
            "Transcript": [
                "The boy is taking cookies.",
                "um mother is washing dishes.",
                "Water is on the floor.",
                "uh the stool is falling.",
            ],
            "label": [0, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "Transcript": ["Girl is laughing.", "um cookies cookies."],
            "Category": [0, 1],
        }
    )
    return train, test
