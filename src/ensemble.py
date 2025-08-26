import argparse
import pandas as pd

# Flow A (prompts via Ollama)
from src.prompt_runner import classify_one

# Flow B (HF zero-shot)
from src.hf_pipeline import load_pipes, policy_zero_shot

def run(csv_in: str, csv_out: str, model_name: str = "mistral:7b-instruct",
        device=None, tau: float = 0.55, agreement_only: bool = False):
    """
    - tau: minimum confidence required to trust a single-source REJECT
    - agreement_only: if True, only REJECT when BOTH flows reject
    """
    df = pd.read_csv(csv_in)
    df.columns = df.columns.str.strip().str.lower()

    text_col = next((c for c in ["text", "review", "content", "body"] if c in df.columns), None)
    if not text_col:
        raise SystemExit(f"No text column found in {csv_in}. Have: {list(df.columns)}")
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    # HF pipes (we only need zero-shot here)
    _, _, zshot = load_pipes(device)

    rows = []
    for _, r in df.iterrows():
        txt = str(r[text_col])

        # --- Flow A: prompts/LLM (Ollama)
        pr = classify_one(txt, model_name)  # {'label','category','raw': {cat: {confidence,...}}}
        prompt_label = pr.get("label", "APPROVE")
        prompt_cat   = pr.get("category", "None")
        prompt_conf  = 0.0
        # try to read the confidence the chosen category reported
        raw = pr.get("raw", {})
        if isinstance(raw, dict) and prompt_cat in raw and isinstance(raw[prompt_cat], dict):
            prompt_conf = float(raw[prompt_cat].get("confidence", 0.0))

        # --- Flow B: HF zero-shot (always get best & its score; we'll apply tau here)
        # call with tau=0.0 to get the best label + raw score back; threshold later
        hf_cat_raw, hf_conf = policy_zero_shot(zshot, txt, tau=0.0)
        hf_label = "REJECT" if hf_cat_raw != "None" else "APPROVE"
        hf_cat   = hf_cat_raw

        # apply threshold to HF decision
        if hf_label == "REJECT" and hf_conf < tau:
            hf_label, hf_cat = "APPROVE", "None"

        # --- Arbitration
        if agreement_only:
            if prompt_label == "REJECT" and hf_label == "REJECT":
                final_label = "REJECT"
                # if disagree on category, pick the higher-confidence one
                if prompt_cat == hf_cat:
                    final_cat = prompt_cat
                else:
                    final_cat = prompt_cat if prompt_conf >= hf_conf else hf_cat
            else:
                final_label, final_cat = "APPROVE", "None"
        else:
            # highest-confidence reject rule
            if prompt_label == "REJECT" and hf_label == "REJECT":
                if prompt_cat == hf_cat:
                    final_label, final_cat = "REJECT", prompt_cat
                else:
                    final_label, final_cat = ("REJECT", prompt_cat) if prompt_conf >= hf_conf else ("REJECT", hf_cat)
            elif prompt_label == "REJECT":
                final_label, final_cat = ("REJECT", prompt_cat) if prompt_conf >= tau else ("APPROVE", "None")
            elif hf_label == "REJECT":
                final_label, final_cat = "REJECT", hf_cat
            else:
                final_label, final_cat = "APPROVE", "None"

        rows.append({
            "id": r["id"],
            "text": txt,
            "pred_label": final_label,
            "pred_category": final_cat,
            # diagnostics (keep these to understand decisions)
            "prompt_label": prompt_label, "prompt_category": prompt_cat, "prompt_conf": round(prompt_conf, 4),
            "hf_label": hf_label, "hf_category": hf_cat, "hf_conf": round(float(hf_conf), 4),
        })

    out = pd.DataFrame(rows)
    out[["id","text","pred_label","pred_category"]].to_csv(csv_out, index=False)
    out.to_csv(csv_out.replace(".csv","_diagnostics.csv"), index=False)
    print(f"Wrote {csv_out} and {csv_out.replace('.csv','_diagnostics.csv')}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/sample_reviews.csv")
    ap.add_argument("--out", default="predictions_ens.csv")
    ap.add_argument("--model", default="mistral:7b-instruct", help="Ollama model for the prompt flow")
    ap.add_argument("--device", default=None, help="HF device (None/-1 CPU; 0 for GPU)")
    ap.add_argument("--tau", type=float, default=0.55, help="min confidence to accept a single-source reject")
    ap.add_argument("--agreement_only", action="store_true", help="reject only when both flows reject")
    args = ap.parse_args()
    run(args.csv, args.out, args.model, args.device, args.tau, args.agreement_only)