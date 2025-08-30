#!/usr/bin/env python3
import argparse
import pandas as pd
from transformers import pipeline
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# ===== Configuration =====
@dataclass
class ModelConfig:
    """Model configurations"""
    TOXIC_MODEL: str = "unitary/toxic-bert"
    ZSHOT_MODEL: str = "facebook/bart-large-mnli"

@dataclass  
class ThresholdConfig:
    """Decision thresholds for classification"""
    TAU_IRRELEVANT: float = 0.55
    TAU_RANT: float = 0.55  
    TAU_ADS: float = 0.70
    TOX_TAU: float = 0.50
    ADS_MARGIN: float = 0.10

# Initialize configs
MODELS = ModelConfig()
THRESHOLDS = ThresholdConfig()

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

# ===== Ad evidence detection =====
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

def ad_evidence(text: str) -> Tuple[bool, str]:
    """Return (bool_found, matched_pattern) for debugging."""
    t = text or ""
    m = AD_REGEX.search(t)
    return (bool(m), m.group(0) if m else "")

# ===== Pipeline Management =====
def load_pipes(device: Optional[int] = None) -> Tuple[Any, Any]:
    """
    Load toxicity and zero-shot classification pipelines.
    
    Args:
        device: None for CPU, 0 for first CUDA GPU (if available)
        
    Returns:
        Tuple of (toxic_pipeline, zshot_pipeline)
    """
    toxic = pipeline(
        "text-classification",
        model=MODELS.TOXIC_MODEL,
        top_k=None,            # return all labels with scores (multi-label)
        device=device,
    )
    zshot = pipeline(
        "zero-shot-classification", 
        model=MODELS.ZSHOT_MODEL,
        device=device,
    )
    return toxic, zshot

# ===== Helper Functions =====
def extract_tox_top(tox_output: Any) -> Tuple[str, float]:
    """
    Normalizes various HF shapes into (label, score) of the top toxicity.
    Accepts dict, list[dict], or list[list[dict]].
    
    Args:
        tox_output: Output from toxicity pipeline
        
    Returns:
        Tuple of (top_label, top_score)
    """
    def _top_of_list(lst: List[Dict[str, Any]]) -> Tuple[str, float]:
        if not lst:
            return ("", 0.0)
        best = max(lst, key=lambda d: float(d.get("score", 0.0)))
        return best.get("label", ""), float(best.get("score", 0.0))

    if isinstance(tox_output, dict):
        return tox_output.get("label", ""), float(tox_output.get("score", 0.0))
    
    if isinstance(tox_output, list):
        first = tox_output[0] if tox_output else None
        if isinstance(first, dict):   # flat list of dicts
            return _top_of_list(tox_output)
        if isinstance(first, list):   # nested list
            return _top_of_list(first)
    
    return ("", 0.0)

def zero_shot_scores(zshot: Any, text: str) -> Dict[str, float]:
    """
    Get zero-shot classification scores for policy categories.
    
    Args:
        zshot: Zero-shot classification pipeline
        text: Input text to classify
        
    Returns:
        Dict with normalized keys: {"No_Ads": x, "Irrelevant": y, "Rant_No_Visit": z, "None": w}
    """
    res = zshot(
        text,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template="This review is {}.",
        multi_label=True,
    )
    
    label2score = {
        POLICY_MAP[lab]: float(scr) 
        for lab, scr in zip(res["labels"], res["scores"])
    }
    
    # Ensure all expected keys are present
    for category in ("No_Ads", "Irrelevant", "Rant_No_Visit", "None"):
        label2score.setdefault(category, 0.0)
        
    return label2score

def decide_category(
    zs: Dict[str, float], 
    tox_label: str, 
    tox_score: float, 
    *, 
    text: str,
    thresholds: Optional[ThresholdConfig] = None
) -> Tuple[str, str, List[str]]:
    """
    Apply fusion logic to determine final category.
    
    Args:
        zs: Zero-shot scores dict
        tox_label: Top toxicity label
        tox_score: Top toxicity score
        text: Input text for ad evidence checking
        thresholds: Custom thresholds (uses defaults if None)
        
    Returns:
        Tuple of (final_label, pred_category, reasons)
        
    Decision logic:
      A) If toxicity high:
           - toxic/insult/obscene/threat -> REJECT: Rant_No_Visit
           - identity_hate               -> REJECT: Irrelevant
      B) Else if Irrelevant or Rant >= thresholds -> REJECT that one
      C) Else if Ads >= tau_ads AND has strong ad evidence AND (Ads >= max(Irr,Rant)+ads_margin) -> REJECT: No_Ads
      D) Else APPROVE
    """
    if thresholds is None:
        thresholds = THRESHOLDS
        
    reasons = []

    # A) Toxicity-based decisions
    if tox_label and tox_score >= thresholds.TOX_TAU:
        if tox_label in TOX_TO_RANT:
            reasons.append(f"Toxic:{tox_label}")
            return "REJECT", "Rant_No_Visit", reasons
        if tox_label in TOX_TO_IRRELEVANT:
            reasons.append(f"Toxic:{tox_label}")
            return "REJECT", "Irrelevant", reasons

    # B) Policy-based decisions by zero-shot confidence
    irr, rant, ads = zs["Irrelevant"], zs["Rant_No_Visit"], zs["No_Ads"]
    
    if max(irr, rant) >= min(thresholds.TAU_IRRELEVANT, thresholds.TAU_RANT):
        category = "Irrelevant" if irr >= rant else "Rant_No_Visit"
        reasons.append(f"Policy:{category}")
        return "REJECT", category, reasons

    # C) Ad-specific decision (requires evidence + high confidence + margin)
    has_ads, _ = ad_evidence(text)
    if (has_ads and 
        ads >= thresholds.TAU_ADS and 
        ads >= max(irr, rant) + thresholds.ADS_MARGIN):
        reasons.append("Policy:No_Ads")
        return "REJECT", "No_Ads", reasons

    # D) Default approval
    return "APPROVE", "None", reasons

# ===== Data Processing Helpers =====
def prepare_dataframe(csv_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load and prepare input DataFrame.
    
    Args:
        csv_path: Path to input CSV file
        
    Returns:
        Tuple of (prepared_dataframe, text_column_name)
        
    Raises:
        SystemExit: If no suitable text column is found
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # Find text column
    text_col = next(
        (c for c in ("text", "review", "content", "body") if c in df.columns), 
        None
    )
    if text_col is None:
        raise SystemExit(f"No text column found in {csv_path}")

    # Ensure ID column exists
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    return df, text_col

def create_main_row(row_id: Any, text: str, final_label: str, pred_category: str, reasons: List[str]) -> Dict[str, Any]:
    """Create main output row (without numeric scores)."""
    return {
        "id": row_id,
        "text": text,
        "final_label": final_label,      # APPROVE or REJECT
        "pred_category": pred_category,  # No_Ads / Irrelevant / Rant_No_Visit / None
        "reasons": "; ".join(reasons),   # label-only tags
    }

def create_diag_row(
    row_id: Any, text: str, final_label: str, pred_category: str,
    zs: Dict[str, float], tox_label: str, tox_score: float,
    has_ads_ev: bool, matched_ev: str
) -> Dict[str, Any]:
    """Create diagnostics output row (with numeric scores for tuning)."""
    return {
        "id": row_id,
        "text": text,
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
    }

def save_outputs(main_rows: List[Dict], diag_rows: List[Dict], csv_out: str) -> None:
    """Save main and diagnostics CSV files."""
    pd.DataFrame(main_rows).to_csv(csv_out, index=False)
    diag_path = csv_out.replace(".csv", "_diagnostics.csv")
    pd.DataFrame(diag_rows).to_csv(diag_path, index=False)
    print(f"Wrote {csv_out} and {diag_path}")

# ===== Main Processing =====
def run(csv_in: str, csv_out: str, device: Optional[int] = None, policy_tau: float = 0.55, tox_tau: float = 0.50) -> None:
    """
    Main processing function for review classification.
    
    Args:
        csv_in: Path to input CSV file
        csv_out: Path to output CSV file
        device: Device for model inference (None for CPU, 0 for first GPU)
        policy_tau: Policy threshold (kept for compatibility; not used directly)
        tox_tau: Toxicity threshold for classification
    """
    # Prepare input data
    df, text_col = prepare_dataframe(csv_in)
    
    # Load models
    toxic, zshot = load_pipes(device)
    
    # Configure thresholds
    thresholds = ThresholdConfig()
    thresholds.TOX_TAU = tox_tau

    # Process each row
    main_rows, diag_rows = [], []

    for _, row in df.iterrows():
        text = str(row[text_col])

        # Get zero-shot classification scores
        zs = zero_shot_scores(zshot, text)

        # Get toxicity classification
        tox_out = toxic(text)
        tox_label, tox_score = extract_tox_top(tox_out)

        # Make final decision
        final_label, pred_category, reasons = decide_category(
            zs, tox_label, tox_score,
            text=text,
            thresholds=thresholds
        )

        # Create output rows
        main_rows.append(create_main_row(
            row["id"], text, final_label, pred_category, reasons
        ))

        has_ads_ev, matched_ev = ad_evidence(text)
        diag_rows.append(create_diag_row(
            row["id"], text, final_label, pred_category,
            zs, tox_label, tox_score, has_ads_ev, matched_ev
        ))

    # Save results
    save_outputs(main_rows, diag_rows, csv_out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample/sample_reviews.csv", help="Input CSV with a text/review/content/body column")
    ap.add_argument("--out", default="results/predictions/predictions.csv", help="Output CSV path")
    ap.add_argument("--device", default=None, help="None for CPU, or e.g. 0 for CUDA GPU")
    ap.add_argument("--policy_tau", type=float, default=0.55, help="(kept for compatibility; not used directly)")
    ap.add_argument("--tox_tau", type=float, default=0.50, help="Toxicity threshold to aid policy split")
    args = ap.parse_args()

    run(args.csv, args.out, device=args.device, policy_tau=args.policy_tau, tox_tau=args.tox_tau)
