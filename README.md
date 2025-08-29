# clone/open the repo root (e.g., review-rater/)

cd review-rater

# create a clean env on Python 3.11

/opt/homebrew/bin/python3.11 -m venv .venv311
source .venv311/bin/activate
python -V

# install deps

pip install -U pip
pip install -r requirements.txt

# default for prompts

ollama pull mistral:7b-instruct

# Prompts (Ollama)

python -m src.prompt_runner --model mistral:7b-instruct --csv data/sample_reviews.csv --out predictions.csv
python -m src.evaluate_prompts --pred predictions.csv

# HF baseline (no prompts)

python -m src.hf_pipeline --csv data/sample_reviews.csv --out predictions_hf.csv
python -m src.evaluate_prompts --pred predictions_hf.csv

# Ensemble (recommended)

python -m src.ensemble --csv data/sample_reviews.csv --out predictions_ens.csv --model mistral:7b-instruct --tau 0.55
python -m src.evaluate_prompts --pred predictions_ens.csv

## Spam Detection System

Multi-method baseline spam detection with TF-IDF + Logistic Regression, keyword detection, pattern analysis, and sentiment analysis.

### Quick Demo

```bash
# Interactive demonstration of all detection methods
python demo_spam_detection.py
```

### Training & Evaluation

```bash
# Train composite detector on sample data
python train_spam_detector.py --train_data data/sample_reviews.csv --output_dir outputs

# View results
cat outputs/evaluation_results.json
head outputs/predictions_composite.csv
```

### Custom Configuration

```bash
# Use custom detection weights
python train_spam_detector.py --weights example_weights.json

# Train on different dataset
python train_spam_detector.py --train_data data/reviews_clean.csv
```

### What It Detects

- **Promotional Spam**: Promo codes, discounts, contact information
- **Financial Spam**: Crypto, investment schemes, "make money fast"
- **Pattern Anomalies**: Repetitive text, template reviews, formatting issues
- **Sentiment Anomalies**: Fake positive reviews, extreme rants, mismatches

### Detection Methods

| Method                       | Weight | Purpose                                |
| ---------------------------- | ------ | -------------------------------------- |
| TF-IDF + Logistic Regression | 40%    | ML-based content pattern learning      |
| Keyword Detection            | 25%    | Rule-based promotional/financial terms |
| Pattern Analysis             | 20%    | Repetition, templates, formatting      |
| Sentiment Analysis (VADER)   | 15%    | Emotional anomalies and mismatches     |

**Output**: Final label (APPROVE/REJECT), confidence score, violation category, confidence intervals, and component breakdowns.
