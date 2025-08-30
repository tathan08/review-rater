import argparse
import pandas as pd
from transformers import pipeline

from .core.constants import DEFAULT_MODELS, ZERO_SHOT_LABELS, ZERO_SHOT_TO_POLICY, POLICY_CATEGORIES, LABELS
from .core.utils import extract_toxicity_result, save_results_with_diagnostics, standardize_columns, find_text_column, ensure_id_column

def load_pipes(device=None):
    """Load the HuggingFace pipelines"""
    # sequential
    # positive / negative
    sentiment = pipeline("sentiment-analysis", model=DEFAULT_MODELS['SENTIMENT'], device=device)
    # salty / toxicity severe OR threats
    toxic = pipeline("text-classification", model=DEFAULT_MODELS['TOXIC'], top_k=None, device=device)
    # zero-shot classification for policy violations
    zshot = pipeline("zero-shot-classification", model=DEFAULT_MODELS['ZERO_SHOT'], device=device)
    return sentiment, toxic, zshot

def policy_zero_shot(zshot, text: str, tau: float = 0.5):
    """Run zero-shot classification for policy violations"""
    # Score all labels independently
    res = zshot(
        text,
        candidate_labels=ZERO_SHOT_LABELS,
        hypothesis_template="This review is {}.",
        multi_label=True,   # <— important
    )
    # Build a dict: phrase -> score
    scores = {lab: float(scr) for lab, scr in zip(res["labels"], res["scores"])}

    # Consider only the 3 rejecting policies
    reject_scores = {
        ZERO_SHOT_TO_POLICY[ZERO_SHOT_LABELS[0]]: scores[ZERO_SHOT_LABELS[0]],  # No_Ads
        ZERO_SHOT_TO_POLICY[ZERO_SHOT_LABELS[1]]: scores[ZERO_SHOT_LABELS[1]],  # Irrelevant
        ZERO_SHOT_TO_POLICY[ZERO_SHOT_LABELS[2]]: scores[ZERO_SHOT_LABELS[2]],  # Rant_No_Visit
    }

    # Pick the strongest rejecting policy
    best_cat, best_score = max(reject_scores.items(), key=lambda kv: kv[1])

    # Reject only if best rejecting score clears the threshold
    if best_score >= tau:
        return best_cat, best_score

    # Otherwise approve
    return POLICY_CATEGORIES['NONE'], scores.get(ZERO_SHOT_LABELS[3], 1.0 - best_score)  # confidence optional

def run(csv_in: str, csv_out: str, device=None, mode="policy_only"):
    """Run HF pipeline classification on CSV input"""
    df = pd.read_csv(csv_in)
    df = standardize_columns(df)
    df = ensure_id_column(df)
    
    # Find text column
    text_col = find_text_column(df)

    sentiment, toxic, zshot = load_pipes(device)

    rows = []
    for _, r in df.iterrows():
        txt = str(r[text_col])

        # Zero-shot policy decision (LLM-only baseline)
        policy, conf = policy_zero_shot(zshot, txt)
        pred_label = LABELS['REJECT'] if policy != POLICY_CATEGORIES['NONE'] else LABELS['APPROVE']

        # Get sentiment and toxicity for diagnostics (simplified to avoid API issues)
        try:
            s_result = sentiment(txt)
            s = s_result[0] if isinstance(s_result, list) and len(s_result) > 0 else {"label": "NEUTRAL", "score": 0.5}
            
            tox_result = toxic(txt)
            # Simple extraction for toxicity - just get the first result
            if isinstance(tox_result, list) and len(tox_result) > 0:
                if isinstance(tox_result[0], dict):
                    tox_label = tox_result[0].get("label", "NONE")
                    tox_score = float(tox_result[0].get("score", 0.0))
                else:
                    tox_label, tox_score = "NONE", 0.0
            else:
                tox_label, tox_score = "NONE", 0.0
        except Exception as e:
            # Fallback values if pipeline fails
            s = {"label": "NEUTRAL", "score": 0.5}
            tox_label, tox_score = "NONE", 0.0

        rows.append({
            "id": r['id'],
            "text": txt,
            "pred_label": pred_label,
            "pred_category": policy,
            "policy_confidence": round(float(conf), 4),
            "sentiment_label": s.get("label", "NEUTRAL"),
            "sentiment_score": round(float(s.get("score", 0.5)), 4),
            "tox_top": tox_label,
            "tox_top_score": round(float(tox_score), 4),
        })

    # Use consolidated save function
    save_results_with_diagnostics(rows, csv_out, include_diagnostics=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_reviews.csv")
    ap.add_argument("--out", default="predictions_hf.csv")
    ap.add_argument("--device", default=None, help="e.g., 0 for CUDA GPU; leave None for CPU")
    ap.add_argument("--mode", default="policy_only", choices=["policy_only"], help="kept for future modes")
    args = ap.parse_args()
    run(args.csv, args.out, device=args.device, mode=args.mode)
