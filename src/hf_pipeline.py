import argparse
import pandas as pd
from transformers import pipeline

# --- Models ---
# 1) Sentiment (binary)
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# 2) Toxicity (multi-label; optional diagnostics)
TOXIC_MODEL = "unitary/toxic-bert"
# 3) Zero-shot for policies
ZSHOT_MODEL = "facebook/bart-large-mnli"

CANDIDATE_LABELS = [
    "an advertisement or promotional solicitation for this business (promo code, referral, links, contact to buy)",
    "off-topic or unrelated to this business (e.g., politics, crypto, chain messages, personal stories not about this place)",
    "a generic negative rant about this business without evidence of a visit (short insults, 'scam', 'overpriced', 'worst ever')",
    "a relevant on-topic description of a visit or experience at this business"
]

MAP_TO_POLICY = {
    "an advertisement or promotional solicitation for this business (promo code, referral, links, contact to buy)": "No_Ads",
    "off-topic or unrelated to this business (e.g., politics, crypto, chain messages, personal stories not about this place)": "Irrelevant",
    "a generic negative rant about this business without evidence of a visit (short insults, 'scam', 'overpriced', 'worst ever')": "Rant_No_Visit",
    "a relevant on-topic description of a visit or experience at this business": "None",
}

def extract_tox_top(tox_output):
    """
    Accepts either:
      - [{label, score}, ...]
      - [[{label, score}, ...]]
      - {label, score}
    Returns (label, score) or ("", 0.0).
    """
    def top_of(lst):
        if not lst:
            return "", 0.0
        best = max(lst, key=lambda d: float(d.get("score", 0.0)))
        return best.get("label", ""), float(best.get("score", 0.0))

    if isinstance(tox_output, dict):
        return tox_output.get("label", ""), float(tox_output.get("score", 0.0))

    if isinstance(tox_output, list):
        first = tox_output[0] if tox_output else None
        if isinstance(first, dict):
            # shape A
            return top_of(tox_output)
        if isinstance(first, list):
            # shape B
            return top_of(first)

    return "", 0.0


def load_pipes(device=None):
    sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=device)
    toxic = pipeline("text-classification", model=TOXIC_MODEL, top_k=None, device=device)
    zshot = pipeline("zero-shot-classification", model=ZSHOT_MODEL, device=device)
    return sentiment, toxic, zshot

def policy_zero_shot(zshot, text: str, tau: float = 0.5):
    # Score all labels independently
    res = zshot(
        text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This review is {}.",
        multi_label=True,   # <— important
    )
    # Build a dict: phrase -> score
    scores = {lab: float(scr) for lab, scr in zip(res["labels"], res["scores"])}

    # Consider only the 3 rejecting policies
    reject_scores = {
        MAP_TO_POLICY[CANDIDATE_LABELS[0]]: scores[CANDIDATE_LABELS[0]],  # No_Ads
        MAP_TO_POLICY[CANDIDATE_LABELS[1]]: scores[CANDIDATE_LABELS[1]],  # Irrelevant
        MAP_TO_POLICY[CANDIDATE_LABELS[2]]: scores[CANDIDATE_LABELS[2]],  # Rant_No_Visit
    }

    # Pick the strongest rejecting policy
    best_cat, best_score = max(reject_scores.items(), key=lambda kv: kv[1])

    # Reject only if best rejecting score clears the threshold
    if best_score >= tau:
        return best_cat, best_score

    # Otherwise approve
    return "None", scores.get(CANDIDATE_LABELS[3], 1.0 - best_score)  # confidence optional

def run(csv_in: str, csv_out: str, device=None, mode="policy_only"):
    df = pd.read_csv(csv_in)
    # normalize headers
    df.columns = df.columns.str.strip().str.lower()

    # choose text column
    text_col = next((c for c in ["text", "review", "content", "body"] if c in df.columns), None)
    if text_col is None:
        raise SystemExit(f"No text column found in {csv_in}")

    # id column or synthesize
    id_col = "id" if "id" in df.columns else None
    if id_col is None:
        df["id"] = range(1, len(df) + 1)
        id_col = "id"

    sentiment, toxic, zshot = load_pipes(device)

    rows = []
    for _, r in df.iterrows():
        txt = str(r[text_col])

        # Zero-shot policy decision (LLM-only baseline)
        policy, conf = policy_zero_shot(zshot, txt)
        pred_label = "REJECT" if policy != "None" else "APPROVE"

       
        s = sentiment(txt)[0]  # {'label': 'NEGATIVE'/'POSITIVE', 'score': ...}
        tox = toxic(txt)
        tox_label, tox_score = extract_tox_top(tox)       # list of labels with scores

        rows.append({
            "id": r[id_col],
            "text": txt,
            "pred_label": pred_label,
            "pred_category": policy,
            "policy_confidence": round(conf, 4),
            "sentiment_label": s["label"],
            "sentiment_score": round(float(s["score"]), 4),
            "tox_top": tox_label,
            "tox_top_score": round(tox_score, 4),
        })

    out = pd.DataFrame(rows)

    out[["id","text","pred_label","pred_category"]].to_csv(csv_out, index=False)
    out.to_csv(csv_out.replace(".csv", "_diagnostics.csv"), index=False)
    print(f"Wrote {csv_out} and {csv_out.replace('.csv','_diagnostics.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_reviews.csv")
    ap.add_argument("--out", default="predictions_hf.csv")
    ap.add_argument("--device", default=None, help="e.g., 0 for CUDA GPU; leave None for CPU")
    ap.add_argument("--mode", default="policy_only", choices=["policy_only"], help="kept for future modes")
    args = ap.parse_args()
    run(args.csv, args.out, device=args.device, mode=args.mode)
