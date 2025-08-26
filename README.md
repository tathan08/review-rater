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
