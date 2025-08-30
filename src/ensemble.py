import argparse
import pandas as pd

# Flow A (prompts via Ollama)
from .prompt_runner import classify_one

# Flow B (HF zero-shot)
from .hf_pipeline import load_pipes, policy_zero_shot

# Consolidated utilities
from .core.utils import standardize_columns, find_text_column, ensure_id_column, save_results_with_diagnostics
from .core.constants import POLICY_CATEGORIES, LABELS, CONFIDENCE_THRESHOLDS

def run(csv_in: str, csv_out: str, model_name: str = "mistral:7b-instruct",
        device=None, tau: float = CONFIDENCE_THRESHOLDS['DEFAULT'], agreement_only: bool = False):
    """
    - tau: minimum confidence required to trust a single-source REJECT
    - agreement_only: if True, only REJECT when BOTH flows reject
    """
    df = pd.read_csv(csv_in)
    df = standardize_columns(df)
    df = ensure_id_column(df)

    text_col = find_text_column(df)

    # HF pipes (we only need zero-shot here)
    _, _, zshot = load_pipes(device)

    rows = []
    for _, r in df.iterrows():
        txt = str(r[text_col])

        # --- Flow A: prompts/LLM (Ollama)
        pr = classify_one(txt, model_name)  # {'label','category','raw': {cat: {confidence,...}}}
        prompt_label = pr.get("label", LABELS['APPROVE'])
        prompt_cat   = pr.get("category", POLICY_CATEGORIES['NONE'])
        prompt_conf  = 0.0
        # try to read the confidence the chosen category reported
        raw = pr.get("raw", {})
        if isinstance(raw, dict) and prompt_cat in raw and isinstance(raw[prompt_cat], dict):
            prompt_conf = float(raw[prompt_cat].get("confidence", 0.0))

        # --- Flow B: HF zero-shot (always get best & its score; we'll apply tau here)
        # call with tau=0.0 to get the best label + raw score back; threshold later
        hf_cat_raw, hf_conf = policy_zero_shot(zshot, txt, tau=0.0)
        hf_label = LABELS['REJECT'] if hf_cat_raw != POLICY_CATEGORIES['NONE'] else LABELS['APPROVE']
        hf_cat   = hf_cat_raw

        # apply threshold to HF decision
        if hf_label == LABELS['REJECT'] and hf_conf < tau:
            hf_label, hf_cat = LABELS['APPROVE'], POLICY_CATEGORIES['NONE']

        # --- Arbitration
        if agreement_only:
            if prompt_label == LABELS['REJECT'] and hf_label == LABELS['REJECT']:
                final_label = LABELS['REJECT']
                # if disagree on category, pick the higher-confidence one
                if prompt_cat == hf_cat:
                    final_cat = prompt_cat
                else:
                    final_cat = prompt_cat if prompt_conf >= hf_conf else hf_cat
            else:
                final_label, final_cat = LABELS['APPROVE'], POLICY_CATEGORIES['NONE']
        else:
            # highest-confidence reject rule
            if prompt_label == LABELS['REJECT'] and hf_label == LABELS['REJECT']:
                if prompt_cat == hf_cat:
                    final_label, final_cat = LABELS['REJECT'], prompt_cat
                else:
                    final_label, final_cat = (LABELS['REJECT'], prompt_cat) if prompt_conf >= hf_conf else (LABELS['REJECT'], hf_cat)
            elif prompt_label == LABELS['REJECT']:
                final_label, final_cat = (LABELS['REJECT'], prompt_cat) if prompt_conf >= tau else (LABELS['APPROVE'], POLICY_CATEGORIES['NONE'])
            elif hf_label == LABELS['REJECT']:
                final_label, final_cat = LABELS['REJECT'], hf_cat
            else:
                final_label, final_cat = LABELS['APPROVE'], POLICY_CATEGORIES['NONE']

        rows.append({
            "id": r["id"],
            "text": txt,
            "pred_label": final_label,
            "pred_category": final_cat,
            # diagnostics (keep these to understand decisions)
            "prompt_label": prompt_label, "prompt_category": prompt_cat, "prompt_conf": round(prompt_conf, 4),
            "hf_label": hf_label, "hf_category": hf_cat, "hf_conf": round(float(hf_conf), 4),
        })

    # Use consolidated save function
    save_results_with_diagnostics(rows, csv_out, include_diagnostics=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_reviews.csv")
    ap.add_argument("--out", default="predictions_ens.csv")
    ap.add_argument("--model", default="mistral:7b-instruct", help="Ollama model for the prompt flow")
    ap.add_argument("--device", default=None, help="HF device (None/-1 CPU; 0 for GPU)")
    ap.add_argument("--tau", type=float, default=CONFIDENCE_THRESHOLDS['DEFAULT'], help="min confidence to accept a single-source reject")
    ap.add_argument("--agreement_only", action="store_true", help="reject only when both flows reject")
    args = ap.parse_args()
    run(args.csv, args.out, args.model, args.device, args.tau, args.agreement_only)