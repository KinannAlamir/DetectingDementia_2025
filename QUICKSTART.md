# Quick Reference

## Your Setup

**Local Model Path**: `/Users/alamir/Documents/Travail/perso/Programmation/xlm-roberta-base`

This model is now the **default** when using the transformer option.

## Running the Project

### Traditional ML (Fast)
```bash
uv run detect-dementia
```

### Transformer (Your XLM-RoBERTa)
```bash
uv run detect-dementia -t
```

That's it! No need to specify the path - it's already configured.

## Model Files Present
- config.json
- pytorch_model.bin (1.1 GB)
- tokenizer.json
- tokenizer_config.json
- sentencepiece.bpe.model
- model.safetensors
- All additional formats (ONNX, TensorFlow, Flax)

## Expected Behavior
When you run with `-t`, you'll see:
```
Using local model from: /Users/alamir/Documents/Travail/perso/Programmation/xlm-roberta-base
Training transformer model (this may take a few minutes)...
```

No download, no waiting - it loads directly from your local files!
