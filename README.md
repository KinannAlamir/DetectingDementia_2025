# Dementia Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

ML-based dementia detection from DementiaBank transcripts using multiple classification methods.

## Features

- **4 ML Approaches**: Logistic Regression, Random Forest, SVM, and RoBERTa Transformer
- **TF-IDF Features**: Extracted from transcript text
- **Linguistic Features**: Word count, hesitations, repetitions, pauses
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Ensemble Model**: Soft voting classifier combining traditional ML models
- **Deep Learning**: Fine-tuned RoBERTa transformer for state-of-the-art performance
- **Clean Architecture**: Modular, pythonic design with minimal code
- **Ruff Compliant**: Passes all linting checks
- **Comprehensive Tests**: 30+ pytest tests with high coverage

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
# Run the traditional ML pipeline (default)
uv run detect-dementia

# Run with XLM-RoBERTa transformer (uses local model by default)
uv run detect-dementia --transformer
# or
uv run detect-dementia -t

# Run with custom local model path
uv run detect-dementia --transformer --model-path /path/to/your/model
# or
uv run detect-dementia -t -m /path/to/your/model

# Force download from Hugging Face instead of using local
uv run detect-dementia --transformer --download-model

# Or run directly
uv run python -m dementia_detection.main
```

## Project Structure

```
src/dementia_detection/
├── __init__.py        # Package initialization
├── data.py            # Data loading and feature extraction
├── features.py        # Linguistic feature engineering
├── models.py          # Traditional ML models
├── transformer.py     # RoBERTa transformer model
└── main.py            # Main entry point
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

### Traditional ML (Fast)
1. **Logistic Regression**: Linear classifier with L2 regularization
2. **Random Forest**: Ensemble of 100 decision trees
3. **SVM**: Support Vector Machine with RBF kernel
4. **Ensemble**: Soft voting combining all three

### Deep Learning (Best Performance)
5. **XLM-RoBERTa Transformer**: Fine-tuned pre-trained multilingual model
   - Uses local `xlm-roberta-base` by default (no download needed)
   - Path: `/Users/alamir/Documents/Travail/perso/Programmation/xlm-roberta-base`
   - Trained for 3 epochs with learning rate 2e-5
   - Best for capturing semantic patterns in transcripts
   - Supports 100+ languages (though this dataset is English)

All models are evaluated using accuracy, F1 score, and classification reports.
