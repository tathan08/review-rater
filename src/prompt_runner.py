import argparse
import pandas as pd

from prompts.policy_prompts import (
    build_prompt,
    NO_ADS_SYSTEM, FEW_SHOTS_NO_ADS,
    IRRELEVANT_SYSTEM, FEW_SHOTS_IRRELEVANT,
    RANT_NO_VISIT_SYSTEM, FEW_SHOTS_RANT,
)
from .core.utils import run_ollama, extract_json, preclassify_review, save_results_with_diagnostics
from .core.constants import POLICY_CATEGORIES, LABELS

def preclassify(text: str):
    """Hard guardrails:
       - Ads only when solicitation/coupon/link present
       - Irrelevant when clearly off-topic w/o solicitation
       - Rant_No_Visit when about this place, negative-ish, and no visit markers
    """
    if AD_SOLICIT_PAT.search(text) or LINK_PAT.search(text):
        return "No_Ads"

    if OFFTOPIC_PAT.search(text):
        # Off-topic crypto/politics etc. → Irrelevant unless it’s also ads (handled above)
        return "Irrelevant"

    negish = any(w in text.lower() for w in ["scam", "scammers", "terrible", "awful", "worst", "ripoff", "overpriced"])
    if negish and not VISIT_MARKERS.search(text) and PLACE_REFERENCES.search(text):
        return "Rant_No_Visit"

    return None


def classify_one(text: str, model: str):

    rule = preclassify_review(text)
    if rule:
        return {"label": LABELS['REJECT'], "category": rule, "raw": {"rule": True}}

    outputs = {}
    for cat, system, shots in [
        (POLICY_CATEGORIES['NO_ADS'], NO_ADS_SYSTEM, FEW_SHOTS_NO_ADS),
        (POLICY_CATEGORIES['IRRELEVANT'], IRRELEVANT_SYSTEM, FEW_SHOTS_IRRELEVANT),
        (POLICY_CATEGORIES['RANT_NO_VISIT'], RANT_NO_VISIT_SYSTEM, FEW_SHOTS_RANT),
    ]:
        prompt = build_prompt(system, text, shots)
        raw = run_ollama(model, prompt)
        try:
            js = extract_json(raw)
        except Exception as e:
            js = {"label":LABELS['APPROVE'],"category":POLICY_CATEGORIES['NONE'],"rationale":f"ParseFail:{e}","confidence":0.0,"flags":{}}
        # ensure defaults
        js.setdefault("confidence", 0.0)
        outputs[cat] = js

    # Collect only rejects
    rejects = [(cat, js) for cat, js in outputs.items() if js.get("label") == LABELS['REJECT']]

    if len(rejects) == 0:
        return {"label":LABELS['APPROVE'],"category":POLICY_CATEGORIES['NONE'],"raw":outputs}

    # choose the reject with highest confidence
    cat_best, js_best = max(rejects, key=lambda kv: kv[1].get("confidence", 0.0))

    # Optional: tiny tie-breaker if equal confidence
    ties = [kv for kv in rejects if abs(kv[1].get("confidence",0.0) - js_best.get("confidence",0.0)) < 1e-6]
    if len(ties) > 1:
        # lightweight semantic tie-breakers
        txt = text.lower()
        if any(w in txt for w in ["promo", "code", "referral", "dm", "whatsapp", "telegram", "coupon", "use code", "http", "www"]):
            cat_best = POLICY_CATEGORIES['NO_ADS']
        elif any(w in txt for w in ["scam", "scammers", "ripoff", "rip-off", "overpriced", "terrible", "awful", "worst"]):
            cat_best = POLICY_CATEGORIES['RANT_NO_VISIT']
        else:
            cat_best = POLICY_CATEGORIES['IRRELEVANT']

    return {"label":LABELS['REJECT'],"category":cat_best,"raw":outputs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral:7b-instruct")
    ap.add_argument("--csv", default="data/sample_reviews.csv")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip().str.lower()
    
    # Find text column
    text_col = next((c for c in ["text", "review", "content", "body"] if c in df.columns), None)
    if not text_col:
        raise SystemExit(f"No text column found in {args.csv}. Have: {list(df.columns)}")
    
    # Ensure ID column
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    preds = []
    for _, row in df.iterrows():
        res = classify_one(row[text_col], args.model)
        preds.append({
            "id": row["id"],
            "text": row[text_col],
            "pred_label": res["label"],
            "pred_category": res["category"]
        })
        print(f"[{row['id']}] -> {res['label']} / {res['category']}")

    # Use consolidated save function
    save_results_with_diagnostics(preds, args.out, include_diagnostics=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
