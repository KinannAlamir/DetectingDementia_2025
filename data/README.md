# Data Directory

This directory contains the datasets and models used for dementia detection.

## Input Data

The `input/` directory contains three CSV files from DementiaBank:

- `Control_db.csv` - Healthy control subjects (label=0)
- `Dementia_db.csv` - Dementia patients (label=1)  
- `Testing_db.csv` - Mixed test set

### Data Format

Each CSV file contains the following columns:
- `Language` - Language of the transcript
- `Data` - Date of recording
- `Participant` - Participant ID
- `Age` - Age of participant
- `Gender` - Gender of participant
- `Diagnosis` - Clinical diagnosis
- `Category` - Category of participant
- `mmse` - Mini-Mental State Examination score
- `Filename` - Original filename
- `Transcript` - The transcript text (main feature)

## Model Files

### XLM-RoBERTa Base Model

The `xlm-roberta-base/` directory contains the pre-trained transformer model. This is a large directory (~2GB) and is **not included in the repository**.

To use the transformer functionality:

**Option 1: Use your local model (recommended)**
```bash
uv run detect-dementia --transformer --model-path /path/to/your/xlm-roberta-base
```

**Option 2: Download from Hugging Face**
```bash
uv run detect-dementia --transformer --download-model
```

The model will be automatically downloaded from Hugging Face Hub if not found locally.

## Data Source

The DementiaBank dataset is from the TalkBank project:
- Website: https://dementia.talkbank.org/
- Citation: Becker, J. T., Boiler, F., Lopez, O. L., Saxton, J., & McGonigle, K. L. (1994). The natural history of Alzheimer's disease: description of study cohort and accuracy of diagnosis. Archives of Neurology, 51(6), 585-594.

## Privacy Note

Please ensure you have proper authorization to use this data and follow all applicable privacy regulations and research ethics guidelines.
