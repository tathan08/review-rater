#!/usr/bin/env python3
import argparse
import pandas as pd
from transformers import pipeline
import re

# ===== Models =====
TOXIC_MODEL = "unitary/toxic-bert"
ZSHOT_MODEL = "facebook/bart-large-mnli"

# Zero-shot labels (keep wording stable for consistent scores)
CANDIDATE_LABELS = [
    "an advertisement or promotional solicitation for this business (promo code, referral, links, contact to buy)",
    "off-topic or unrelated to this business (e.g., politics, crypto, chain messages, personal stories not about this place)",
    "a generic negative rant about this business without evidence of a visit (short insults, 'scam', 'overpriced', 'worst ever')",
    "a relevant on-topic description of a visit or experience at this business",
]
POLICY_MAP = {
    CANDIDATE_LABELS[0]: "No_Ads",
    CANDIDATE_LABELS[1]: "Irrelevant",
    CANDIDATE_LABELS[2]: "Rant_No_Visit",
    CANDIDATE_LABELS[3]: "None",
}

# Toxicity → helper buckets for fusion
TOX_TO_RANT = {"toxic", "severe_toxic", "obscene", "threat", "insult"}
TOX_TO_IRRELEVANT = {"identity_hate"}

# ===== Safer Ad evidence regex =====
# - URLs / domains
# - Phone numbers
# - Explicit promo keywords
# - WhatsApp
# - Explicit "dm me/us" only (not generic "dm")
AD_PATTERNS = [
    r"https?://", r"\bwww\.", r"\.[a-z]{2,6}\b",                  # URLs / domains
    r"\b(?:\+?\d[\s\-()]*){7,}\b",                                # phone numbers
    r"\bpromo(?:\s*code)?\b", r"\bdiscount\b", r"\bcoupon\b",
    r"\breferral\b", r"\buse\s*code\b", r"\benter\s*code\b",
    r"\bwhatsapp\b",
    r"\bdm\s+(?:me|us)\b",                                        # explicit DM ask
    r"\bcontact\s+(?:us|me)\b", r"\bcall\s+(?:us|me)\b",
]
AD_REGEX = re.compile("|".join(AD_PATTERNS), flags=re.IGNORECASE)

def ad_evidence(text: str):
    """Return (bool_found, matched_pattern) for debugging."""
    t = text or ""
    m = AD_REGEX.search(t)
    return (bool(m), m.group(0) if m else "")

# ===== Pipelines =====
def load_pipes(device=None):
    """
    device:
      None -> CPU
      0    -> first CUDA GPU (if available)
    """
    toxic = pipeline(
        "text-classification",
        model=TOXIC_MODEL,
        top_k=None,            # return all labels with scores (multi-label)
        device=device,
    )
    zshot = pipeline(
        "zero-shot-classification",
        model=ZSHOT_MODEL,
        device=device,
    )
    return toxic, zshot

# ===== Helpers =====
def extract_tox_top(tox_output):
    """
    Normalizes various HF shapes into (label, score) of the top toxicity.
    Accepts dict, list[dict], or list[list[dict]].
    """
    def top_of(lst):
        if not lst:
            return ("", 0.0)
        best = max(lst, key=lambda d: float(d.get("score", 0.0)))
        return best.get("label", ""), float(best.get("score", 0.0))

    if isinstance(tox_output, dict):
        return tox_output.get("label", ""), float(tox_output.get("score", 0.0))
    if isinstance(tox_output, list):
        first = tox_output[0] if tox_output else None
        if isinstance(first, dict):   # flat list of dicts
            return top_of(tox_output)
        if isinstance(first, list):   # nested list
            return top_of(first)
    return ("", 0.0)

def zero_shot_scores(zshot, text):
    """
    Returns a dict with normalized keys:
      {"No_Ads": x, "Irrelevant": y, "Rant_No_Visit": z, "None": w}
    """
    res = zshot(
        text,
        candidate_labels=ZERO_SHOT_LABELS,
        hypothesis_template="This review is {}.",
        multi_label=True,
    )
    label2score = {POLICY_MAP[lab]: float(scr) for lab, scr in zip(res["labels"], res["scores"])}
    for k in ("No_Ads", "Irrelevant", "Rant_No_Visit", "None"):
        label2score.setdefault(k, 0.0)
    return label2score

def decide_category(
    zs, tox_label, tox_score, *, text,
    tau_irrelevant=0.55, tau_rant=0.55, tau_ads=0.70, tox_tau=0.50, ads_margin=0.10
):
    """
    Fusion (final):
      A) If toxicity high:
           - toxic/insult/obscene/threat -> REJECT: Rant_No_Visit
           - identity_hate               -> REJECT: Irrelevant
      B) Else if Irrelevant or Rant >= thresholds -> REJECT that one
      C) Else if Ads >= tau_ads AND has strong ad evidence AND (Ads >= max(Irr,Rant)+ads_margin) -> REJECT: No_Ads
      D) Else APPROVE
    """
    reasons = []

    # A) Toxicity first
    if tox_label and tox_score >= tox_tau:
        if tox_label in TOX_TO_RANT:
            reasons.append(f"Toxic:{tox_label}")
            return "REJECT", "Rant_No_Visit", reasons
        if tox_label in TOX_TO_IRRELEVANT:
            reasons.append(f"Toxic:{tox_label}")
            return "REJECT", "Irrelevant", reasons

    # B) Irrelevant/Rant by zero-shot confidence
    irr = zs["Irrelevant"]; rant = zs["Rant_No_Visit"]; ads = zs["No_Ads"]
    if max(irr, rant) >= min(tau_irrelevant, tau_rant):
        cat = "Irrelevant" if irr >= rant else "Rant_No_Visit"
        reasons.append(f"Policy:{cat}")
        return "REJECT", cat, reasons

    # C) Ads needs BOTH: high score + real evidence + margin over other policies
    has_ads, matched = ad_evidence(text)
    if has_ads and (ads >= tau_ads) and (ads >= max(irr, rant) + ads_margin):
        reasons.append("Policy:No_Ads")
        return "REJECT", "No_Ads", reasons

    # D) Otherwise approve
    return "APPROVE", "None", reasons

# ===== Main =====
def run(csv_in, csv_out, device=None, policy_tau=0.55, tox_tau=0.50):
    # Load input
    df = pd.read_csv(csv_in)
    df.columns = df.columns.str.strip().str.lower()

    # Pick text column
    text_col = next((c for c in ("text", "review", "content", "body") if c in df.columns), None)
    if text_col is None:
        raise SystemExit(f"No text column found in {csv_in}")

    # Ensure an id
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    # Models
    toxic, zshot = load_pipes(device)

    # Output rows
    main_rows, diag_rows = [], []

    for _, r in df.iterrows():
        txt = str(r[text_col])

        # Zero-shot (scores for Ads/Irrelevant/Rant/None)
        zs = zero_shot_scores(zshot, txt)

        # Toxicity (top label & score)
        tox_out = toxic(txt)
        tox_label, tox_score = extract_tox_top(tox_out)

        # Decision
        final_label, pred_category, reasons = decide_category(
            zs, tox_label, tox_score,
            text=txt,
            tau_irrelevant=0.55,
            tau_rant=0.55,
            tau_ads=0.70,
            tox_tau=tox_tau,
            ads_margin=0.10,
        )

        # Main (no numeric scores)
        main_rows.append({
            "id": r["id"],
            "text": txt,
            "final_label": final_label,      # APPROVE or REJECT
            "pred_category": pred_category,  # No_Ads / Irrelevant / Rant_No_Visit / None
            "reasons": "; ".join(reasons),   # label-only tags
        })

        # Diagnostics (with numeric scores for tuning)
        has_ads_ev, matched_ev = ad_evidence(txt)
        diag_rows.append({
            "id": r["id"],
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
            "final_label": final_label,
            "pred_category": pred_category,
            "zs_no_ads": round(zs["No_Ads"], 4),
            "zs_irrelevant": round(zs["Irrelevant"], 4),
            "zs_rant": round(zs["Rant_No_Visit"], 4),
            "zs_none": round(zs["None"], 4),
            "tox_top": tox_label,
            "tox_top_score": round(tox_score, 4),
            "ad_evidence": has_ads_ev,
            "ad_match": matched_ev,
        })

    # Use consolidated save function
    save_results_with_diagnostics(rows, csv_out, include_diagnostics=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_reviews.csv", help="Input CSV with a text/review/content/body column")
    ap.add_argument("--out", default="predictions.csv", help="Output CSV path")
    ap.add_argument("--device", default=None, help="None for CPU, or e.g. 0 for CUDA GPU")
    ap.add_argument("--policy_tau", type=float, default=0.55, help="(kept for compatibility; not used directly)")
    ap.add_argument("--tox_tau", type=float, default=0.50, help="Toxicity threshold to aid policy split")
    args = ap.parse_args()

    run(args.csv, args.out, device=args.device, policy_tau=args.policy_tau, tox_tau=args.tox_tau)
