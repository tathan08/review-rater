import argparse
import pandas as pd
import re 

from prompts.policy_prompts import (
    build_prompt,
    NO_ADS_SYSTEM, FEW_SHOTS_NO_ADS,
    IRRELEVANT_SYSTEM, FEW_SHOTS_IRRELEVANT,
    RANT_NO_VISIT_SYSTEM, FEW_SHOTS_RANT,
)
from .utils import run_ollama, extract_json


AD_SOLICIT_PAT = re.compile(r"(promo\s*code|referr?al|use\s+code|discount\s*code|coupon|dm\s+me|whats?app|telegram|wa\.me|t\.me|contact\s+me|order\s+via|book\s+now|call\s+\+?\d)", re.I)
LINK_PAT = re.compile(r"https?://|www\.", re.I)
OFFTOPIC_PAT = re.compile(r"\b(crypto|bitcoin|btc|politics|election|stock tips|forex|nft|blockchain)\b", re.I)
VISIT_MARKERS = re.compile(r"\b(i|we)\s+(visited|went|came|ordered|ate|bought|tried|dined)\b", re.I)
PLACE_REFERENCES = re.compile(r"\b(this (place|shop|restaurant|cafe|store|hotel)|here)\b", re.I)

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

    rule = preclassify(text)
    if rule:
        return {"label": "REJECT", "category": rule, "raw": {"rule": True}}

    outputs = {}
    for cat, system, shots in [
        ("No_Ads", NO_ADS_SYSTEM, FEW_SHOTS_NO_ADS),
        ("Irrelevant", IRRELEVANT_SYSTEM, FEW_SHOTS_IRRELEVANT),
        ("Rant_No_Visit", RANT_NO_VISIT_SYSTEM, FEW_SHOTS_RANT),
    ]:
        prompt = build_prompt(system, text, shots)
        raw = run_ollama(model, prompt)
        try:
            js = extract_json(raw)
        except Exception as e:
            js = {"label":"APPROVE","category":"None","rationale":f"ParseFail:{e}","confidence":0.0,"flags":{}}
        # ensure defaults
        js.setdefault("confidence", 0.0)
        outputs[cat] = js

    # Collect only rejects
    rejects = [(cat, js) for cat, js in outputs.items() if js.get("label") == "REJECT"]

    if len(rejects) == 0:
        return {"label":"APPROVE","category":"None","raw":outputs}

    # choose the reject with highest confidence
    cat_best, js_best = max(rejects, key=lambda kv: kv[1].get("confidence", 0.0))

    # Optional: tiny tie-breaker if equal confidence
    ties = [kv for kv in rejects if abs(kv[1].get("confidence",0.0) - js_best.get("confidence",0.0)) < 1e-6]
    if len(ties) > 1:
        # lightweight semantic tie-breakers
        txt = text.lower()
        if any(w in txt for w in ["promo", "code", "referral", "dm", "whatsapp", "telegram", "coupon", "use code", "http", "www"]):
            cat_best = "No_Ads"
        elif any(w in txt for w in ["scam", "scammers", "ripoff", "rip-off", "overpriced", "terrible", "awful", "worst"]):
            cat_best = "Rant_No_Visit"
        else:
            cat_best = "Irrelevant"

    return {"label":"REJECT","category":cat_best,"raw":outputs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistral:7b-instruct")
    ap.add_argument("--csv", default="data/sample_reviews.csv")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    preds = []
    for _, row in df.iterrows():
        res = classify_one(row["text"], args.model)
        preds.append({
            "id": row["id"],
            "text": row["text"],
            "pred_label": res["label"],
            "pred_category": res["category"]
        })
        print(f"[{row['id']}] -> {res['label']} / {res['category']}")

    pd.DataFrame(preds).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
