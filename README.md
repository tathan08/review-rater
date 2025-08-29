# Review Classification Pipeline

A review classification system for detecting policy violations in Google reviews.

## 🚀 **Google Colab (Recommended)**

The easiest way to run this pipeline is in Google Colab:

1. Upload `notebooks/00_colab_complete_pipeline.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Add your Gemini API key to Colab secrets (🔑 icon in sidebar)
3. Run all cells - everything is pre-configured!

## 💻 **Local Setup (Mac/Windows/Linux)**

### **Prerequisites**

- Python 3.9.X
- Git

### **1. Clone and Setup**

```bash
# Clone the repository
git clone <your-repo-url>
cd review-rater

# Create virtual environment
# On Windows:
python -m venv .venv
.venv\Scripts\activate

# On Mac/Linux:
python3 -m venv .venv
source .venv/bin/activate

# Verify Python version
python --version
```

### **2. Install Dependencies**

```bash
# Install requirements
pip install -r requirements.txt

# For Ollama users (optional):
# Install Ollama from https://ollama.com/
ollama pull mistral:7b-instruct
```

### **3. Run Pipeline Commands**

**Test the setup:**

```bash
python test_pipeline_status.py
```

#### **Option 1: Ollama Classification (Local LLM)**

```bash
# Requires Ollama installation
python -m src.prompt_runner --model mistral:7b-instruct --csv data/sample/sample_reviews.csv --out results/predictions/predictions.csv
python -m src.evaluate_prompts --pred results/predictions/predictions.csv
```

#### **Option 2: HuggingFace Models**

```bash
python -m src.hf_pipeline --csv data/sample/sample_reviews.csv --out results/predictions/predictions_hf.csv
python -m src.evaluate_prompts --pred results/predictions/predictions_hf.csv
```

#### **Option 3: Ensemble (Best Results)**

```bash
python -m src.ensemble --csv data/sample/sample_reviews.csv --out results/predictions/predictions_ens.csv --model mistral:7b-instruct --tau 0.55
python -m src.evaluate_prompts --pred results/predictions/predictions_ens.csv
```

## 🔬 **Pseudo-labeling for HuggingFace Training**

Generate training data for HuggingFace models using Google Gemini:

```python
from src.pseudo_labelling.gemini_labeller import GeminiPseudoLabeler
from src.config.pipeline_config import config
import pandas as pd

# Set your Gemini API key
config.gemini_api_key = "your-gemini-api-key"

# Initialize labeler
labeler = GeminiPseudoLabeler(config)

# Generate pseudo labels for training data
df = pd.read_csv("your_unlabeled_reviews.csv")
labeled_df = labeler.generate_pseudo_labels(df, sample_size=1000)

# Save for HuggingFace training
labeled_df.to_csv("training_data_with_pseudo_labels.csv", index=False)
```

## **Features**

- **Ollama Integration**: Local LLM classification (no API needed)
- **HuggingFace Models**: Pre-trained transformer models  
- **Ensemble Classification**: Combines multiple approaches for best results
- **Gemini Pseudo-labeling**: Generate training data (requires API key)
- **Policy Detection**: No_Ads, Irrelevant, Rant_No_Visit categories
- **Evaluation Metrics**: Complete performance analysis

## **Troubleshooting**

**Environment Issues (Windows/Mac):**

- Use Google Colab instead (zero setup required)
- Or create a fresh virtual environment

**HuggingFace Library Issues:**

- Run: `pip install --upgrade transformers torch`
- Google Colab has pre-installed compatible versions

**Ollama Not Working:**

- Install from [Ollama website](https://ollama.com/)
- Run `ollama serve` in separate terminal  
- Use HuggingFace pipeline instead

**Platform-Specific Notes:**

- **Windows**: Use `python` and `pip` commands
- **Mac/Linux**: May need `python3` and `pip3` commands
- **All Platforms**: Google Colab recommended for hassle-free setup

## **Directory Structure Implementation**

``` bash
review-rater
├── src/
│   ├── config/pipeline_config.py       # Centralized configuration
│   ├── core/                           # Core utilities and constants
│   ├── pseudo_labelling/               # Gemini pseudo-labeling system  
│   ├── pipeline/                       # Pipeline orchestration
│   ├── integration/                    # Component integration
├── notebooks/                          # Notebook to run Google Colab
├── data/
│   ├── raw/                            # For raw input data
│   ├── clean/                          # For cleaned/processed data (renamed from processed)
│   ├── pseudo-label/                   # For pseudo-labeled data from Gemini
│   ├── training/                       # For training data split
│   ├── testing/                        # For testing data split
│   └── sample/sample_reviews.csv       # Moved from root
├── models/
│   ├── saved_models/                   # For trained models
│   └── cache/                          # For model cache
├── results/
│   ├── predictions/                    # All predictions moved here
│   ├── evaluations/                    # For evaluation results
│   └── reports/                        # For generated reports
└── logs/pipeline_logs/                 # For pipeline logs
├── docs/policy_prompts.md              # Logic for categorising reviews
├── prompts/                            # Prompt Engineering
```
