"""
Consolidated Utilities
Common functions used across the review rater system
"""

import json
import re
import subprocess
from typing import Dict, List, Optional, Any, Union
import pandas as pd

from .constants import POLICY_CATEGORIES, LABELS, REGEX_PATTERNS

def run_ollama(model: str, prompt: str) -> str:
    """Call Ollama model with prompt; ensure 'ollama pull {model}' was done beforehand"""
    try:
        res = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=300  # 5 minute timeout
        )
        txt = res.stdout.decode("utf-8", errors="ignore").strip()
        return txt
    except subprocess.TimeoutExpired:
        return '{"label":"APPROVE","category":"None","rationale":"Timeout error","confidence":0.0,"flags":{}}'
    except Exception as e:
        return f'{{"label":"APPROVE","category":"None","rationale":"Error: {str(e)}","confidence":0.0,"flags":{{}}}}'

def extract_json(txt: str) -> Dict[str, Any]:
    """Extract and parse JSON from model output"""
    # Be robust: find the first {...} block
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    block = m.group(0)
    try:
        parsed = json.loads(block)
        # Ensure required fields with defaults
        parsed.setdefault("label", LABELS['APPROVE'])
        parsed.setdefault("category", POLICY_CATEGORIES['NONE'])
        parsed.setdefault("rationale", "")
        parsed.setdefault("confidence", 0.0)
        parsed.setdefault("flags", {})
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")

def preclassify_review(text: str) -> Optional[str]:
    """
    Apply hard guardrails for quick classification:
    - Ads only when solicitation/coupon/link present
    - Irrelevant when clearly off-topic w/o solicitation
    - Rant_No_Visit when about this place, negative-ish, and no visit markers
    """
    # Compile patterns once
    ad_pattern = re.compile(REGEX_PATTERNS['AD_SOLICIT'], re.I)
    link_pattern = re.compile(REGEX_PATTERNS['LINK'], re.I)
    offtopic_pattern = re.compile(REGEX_PATTERNS['OFFTOPIC'], re.I)
    visit_pattern = re.compile(REGEX_PATTERNS['VISIT_MARKERS'], re.I)
    place_pattern = re.compile(REGEX_PATTERNS['PLACE_REFERENCES'], re.I)
    
    # Check for ads
    if ad_pattern.search(text) or link_pattern.search(text):
        return POLICY_CATEGORIES['NO_ADS']

    # Check for off-topic content
    if offtopic_pattern.search(text):
        # Off-topic crypto/politics etc. → Irrelevant unless it's also ads (handled above)
        return POLICY_CATEGORIES['IRRELEVANT']

    # Check for negative rants without visit evidence
    text_lower = text.lower()
    negish = any(w in text_lower for w in REGEX_PATTERNS['NEGATIVE_WORDS'])
    if negish and not visit_pattern.search(text) and place_pattern.search(text):
        return POLICY_CATEGORIES['RANT_NO_VISIT']

    return None

def find_text_column(df: pd.DataFrame) -> str:
    """Find the text column in a dataframe"""
    possible_names = ["text", "review", "content", "body", "comment", "message"]
    df_columns_lower = [col.lower().strip() for col in df.columns]
    
    for name in possible_names:
        if name in df_columns_lower:
            # Return the original column name
            return df.columns[df_columns_lower.index(name)]
    
    raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has an 'id' column"""
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = range(1, len(df) + 1)
    return df

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names by stripping whitespace and converting to lowercase"""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df

def create_standard_result(
    review_id: Union[str, int],
    text: str,
    label: str = LABELS['APPROVE'],
    category: str = POLICY_CATEGORIES['NONE'],
    confidence: float = 0.0,
    rationale: str = "",
    flags: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    """Create a standardized result dictionary"""
    if flags is None:
        flags = {"links": False, "coupon": False, "visit_claimed": False}
    
    return {
        "id": review_id,
        "text": text,
        "pred_label": label,
        "pred_category": category,
        "confidence": confidence,
        "rationale": rationale,
        "flags": flags
    }

def extract_toxicity_result(tox_output: Union[Dict, List]) -> tuple[str, float]:
    """
    Extract toxicity label and score from various output formats:
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
            # shape A: list of dicts
            return top_of(tox_output)
        if isinstance(first, list):
            # shape B: list of lists
            return top_of(first)

    return "", 0.0

def save_results_with_diagnostics(
    results: List[Dict[str, Any]], 
    output_path: str,
    include_diagnostics: bool = True
) -> None:
    """Save results to CSV with optional diagnostics file"""
    df = pd.DataFrame(results)
    
    # Main results file with core columns
    core_columns = ["id", "text", "pred_label", "pred_category"]
    main_df = df[[col for col in core_columns if col in df.columns]]
    main_df.to_csv(output_path, index=False)
    
    # Full diagnostics file if requested
    if include_diagnostics:
        diagnostics_path = output_path.replace(".csv", "_diagnostics.csv")
        df.to_csv(diagnostics_path, index=False)
        print(f"Wrote {output_path} and {diagnostics_path}")
    else:
        print(f"Wrote {output_path}")

def validate_model_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix model output to ensure consistent format"""
    # Ensure required fields
    if "label" not in output or output["label"] not in [LABELS['APPROVE'], LABELS['REJECT']]:
        output["label"] = LABELS['APPROVE']
    
    if "category" not in output or output["category"] not in POLICY_CATEGORIES.values():
        output["category"] = POLICY_CATEGORIES['NONE']
    
    if "confidence" not in output:
        output["confidence"] = 0.0
    else:
        try:
            output["confidence"] = float(output["confidence"])
            output["confidence"] = max(0.0, min(1.0, output["confidence"]))  # Clamp to [0,1]
        except (ValueError, TypeError):
            output["confidence"] = 0.0
    
    if "rationale" not in output:
        output["rationale"] = ""
    
    if "flags" not in output:
        output["flags"] = {"links": False, "coupon": False, "visit_claimed": False}
    
    return output
