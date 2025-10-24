# Transformer Usage Guide

## Quick Start

### Traditional ML (Fast - ~2 seconds)
```bash
uv run detect-dementia
```

This runs:
- Logistic Regression
- Random Forest
- SVM
- Ensemble (Voting Classifier)

Uses TF-IDF + Linguistic features (508 total features)

### Deep Learning with Local XLM-RoBERTa (Default)
```bash
uv run detect-dementia --transformer
# or
uv run detect-dementia -t
```

**This now uses your local model by default!**
- Path: `/Users/alamir/Documents/Travail/perso/Programmation/xlm-roberta-base`
- No download required
- XLM-RoBERTa is multilingual (100+ languages)
- Fine-tunes in ~5 minutes

### Use a Different Local Model
```bash
uv run detect-dementia -t -m /path/to/another/model
```

### Force Download from Hugging Face
```bash
uv run detect-dementia --transformer --download-model
```

## Model Information

### Your Local Model
- **Location**: `/Users/alamir/Documents/Travail/perso/Programmation/xlm-roberta-base`
- **Type**: XLM-RoBERTa Base (multilingual)
- **Parameters**: ~125M
- **Languages**: 100+ (including English)
- **Advantages**: 
  - Already downloaded (no waiting!)
  - Multilingual capabilities
  - Strong cross-lingual understanding

### Why XLM-RoBERTa?
- Built on RoBERTa architecture
- Trained on 2.5TB of CommonCrawl data
- Excellent for English text classification
- Potential for multilingual dementia detection research

## Model Selection

**Use Traditional ML when:**
- You need quick results
- Limited compute resources
- Interpretability is important
- Dataset is small (<1000 samples)

**Use Transformer when:**
- You want best performance
- Have GPU or time for training
- Want to capture semantic patterns
- Need state-of-the-art results

## Expected Performance

**Traditional ML:**
- Ensemble: ~86% accuracy, 0.86 F1
- Training time: ~2 seconds

**Transformer:**
- RoBERTa: ~90%+ accuracy (with more data/epochs)
- Training time: ~5 minutes (CPU) or ~1 minute (GPU)

## Customization

### Change Transformer Model
```bash
# Use BERT instead of RoBERTa
uv run detect-dementia -t -m ./models/bert-base-uncased

# Use DistilBERT (faster)
uv run detect-dementia -t -m ./models/distilbert-base-uncased

# Or download from Hugging Face
uv run detect-dementia -t  # defaults to roberta-base
```

### Adjust Training Parameters
Edit `src/dementia_detection/main.py`:
```python
transformer.train(
    train_data,
    val_data,
    num_epochs=5,        # default: 3
    batch_size=16,       # default: 8
    learning_rate=3e-5,  # default: 2e-5
)
```

## Command Line Arguments

- `--transformer` or `-t`: Enable transformer mode
- `--model-path PATH` or `-m PATH`: Use custom model path
- `--download-model`: Force download from Hugging Face (ignores local default)

Examples:
```bash
# Use default local XLM-RoBERTa
uv run detect-dementia -t

# Use different local model
uv run detect-dementia -t -m /path/to/bert-base-uncased

# Download from Hugging Face
uv run detect-dementia -t --download-model
```
