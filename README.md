# Dementia Detection

ML-based dementia detection from DementiaBank transcripts using 3 classification methods.

## Features

- **3 ML Models**: Logistic Regression, Random Forest, and SVM
- **TF-IDF Features**: Extracted from transcript text
- **Linguistic Features**: Word count, hesitations, repetitions, pauses
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Ensemble Model**: Soft voting classifier combining all models
- **Clean Architecture**: Modular, pythonic design with minimal code
- **Ruff Compliant**: Passes all linting checks
- **Comprehensive Tests**: 22+ pytest tests with 100% coverage on core modules

## Installation

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Or install in editable mode
uv pip install -e .
```

## Usage

```bash
# Run the detection pipeline
uv run detect-dementia

# Or run directly
uv run python -m dementia_detection.main
```

## Project Structure

```
src/dementia_detection/
├── __init__.py     # Package initialization
├── data.py         # Data loading and feature extraction
├── models.py       # ML model definitions
└── main.py         # Main entry point
```

## Data

The project uses 3 CSV files from DementiaBank:
- `Control_db.csv`: Healthy control subjects (label=0)
- `Dementia_db.csv`: Dementia patients (label=1)
- `Testing_db.csv`: Mixed test set

Each file contains transcript data with columns:
`Language`, `Data`, `Participant`, `Age`, `Gender`, `Diagnosis`, `Category`, `mmse`, `Filename`, `Transcript`

## Development

```bash
# Run linting
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/dementia_detection --cov-report=term-missing

# Run tests verbosely
uv run pytest -v
```

## Models

1. **Logistic Regression**: Linear classifier with L2 regularization
2. **Random Forest**: Ensemble of 100 decision trees
3. **SVM**: Support Vector Machine with RBF kernel

All models are evaluated using accuracy, F1 score, and classification reports.
