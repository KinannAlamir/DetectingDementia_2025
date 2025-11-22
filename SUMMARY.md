# Project Summary

## Structure
```
DetectingDementia_2025/
├── pyproject.toml              # UV config & dependencies
├── README.md                   # Documentation
├── .gitignore                  # Git ignore rules
├── data/
│   └── input/
│       ├── Control_db.csv      # Healthy subjects (label=0)
│       ├── Dementia_db.csv     # Dementia patients (label=1)
│       └── Testing_db.csv      # Mixed test set
└── src/
    └── dementia_detection/
        ├── __init__.py         # Package init (4 lines)
        ├── data.py             # Data loading & features (23 lines)
        ├── models.py           # 3 ML models (23 lines)
        └── main.py             # Main pipeline (34 lines)
```

## Total Code: ~84 lines

## Features
- UV package manager with pyproject.toml
- Ruff linter passes (all checks green)
- 3 ML methods: Logistic Regression, Random Forest, SVM
- TF-IDF feature extraction (500 features)
- Clean pythonic code with type hints
- Modular architecture

## Results
- **Best Model**: Random Forest (81.82% accuracy)
- Logistic Regression: 77.27% accuracy
- SVM: 77.27% accuracy

## Usage
```bash
uv sync              # Install dependencies
uv run detect-dementia  # Run detection
uv run ruff check .     # Lint code
```
