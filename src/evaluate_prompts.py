import argparse, os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def pick_pred_label_column(df: pd.DataFrame) -> str:
    """
    Prefer old 'pred_label' if present; else use new 'final_label'.
    Raises a clear error if neither exists.
    """
    cols = {c.strip(): c for c in df.columns}
    if "pred_label" in cols:
        return cols["pred_label"]
    if "final_label" in cols:
        # Create a compat alias so the rest of the code can use 'pred_label'
        df["pred_label"] = df[cols["final_label"]]
        return "pred_label"
    raise SystemExit("Neither 'pred_label' nor 'final_label' found in predictions CSV.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="predictions.csv",
                    help="Predictions CSV; accepts columns from old (pred_label) or new (final_label) pipelines.")
    ap.add_argument("--gold", default="data/sample_reviews.csv",
                    help="Gold CSV with columns: id,text,gold_label,gold_category")
    args = ap.parse_args()

    if not os.path.exists(args.pred):
        raise SystemExit(f"Predictions file not found: {args.pred}")
    if not os.path.exists(args.gold):
        raise SystemExit(f"Gold file not found: {args.gold}")

    gold = pd.read_csv(args.gold)
    pred = pd.read_csv(args.pred)

    # Normalize column headers
    gold.columns = gold.columns.str.strip()
    pred.columns = pred.columns.str.strip()

    # Merge on id
    df = gold.merge(pred, on="id", how="left", suffixes=("_gold", "_pred"))

    # Pick/alias prediction label column
    pred_label_col = pick_pred_label_column(df)

    # Normalize label/category strings to uppercase canonical for labels and plain string for categories
    # (Keeps your 'REJECT'/'APPROVE' exactly as-is if already uppercase)
    def norm_label(s):
        return (str(s).strip().upper()) if pd.notna(s) else s

    def norm_cat(s):
        return str(s).strip() if pd.notna(s) else s

    if "gold_label" in df.columns:
        df["gold_label"] = df["gold_label"].map(norm_label)
    else:
        raise SystemExit("Gold CSV must contain 'gold_label' column.")

    if pred_label_col in df.columns:
        df[pred_label_col] = df[pred_label_col].map(norm_label)
    else:
        raise SystemExit(f"Predictions CSV must contain '{pred_label_col}' column after compatibility aliasing.")

    # Category columns may be missing on either side
    if "gold_category" in df.columns:
        df["gold_category"] = df["gold_category"].map(norm_cat)
    else:
        # Keep a column for downstream logic even if missing
        df["gold_category"] = None

    if "pred_category" in df.columns:
        df["pred_category"] = df["pred_category"].map(norm_cat)
    else:
        df["pred_category"] = None

    # ==== Binary approve/reject ====
    print("== APPROVE/REJECT ==")
    print(classification_report(df["gold_label"], df[pred_label_col], zero_division=0))
    print(confusion_matrix(df["gold_label"], df[pred_label_col]))

    # ==== Category metrics ====
    # View A: On GOLD rejects only (penalizes false negatives as 'None')
    rej_gold = df[df["gold_label"] == "REJECT"].copy()
    # If model approved or is NaN, treat category as "None"
    rej_gold["pred_category"] = rej_gold["pred_category"].where(rej_gold["pred_category"].notna(), "None")
    rej_gold["gold_category"] = rej_gold["gold_category"].astype(str)
    rej_gold["pred_category"] = rej_gold["pred_category"].astype(str)

    print("\n== Category (on GOLD rejects; 'None' if model approved) ==")
    print(classification_report(rej_gold["gold_category"], rej_gold["pred_category"], zero_division=0))

    # View B: On cases where BOTH gold and pred are REJECT (measures pure category assignment)
    rej_both = df[(df["gold_label"] == "REJECT") & (df[pred_label_col] == "REJECT")].copy()
    if len(rej_both):
        rej_both["gold_category"] = rej_both["gold_category"].astype(str)
        rej_both["pred_category"] = rej_both["pred_category"].astype(str)
        print("\n== Category (on cases where BOTH are REJECT) ==")
        print(classification_report(rej_both["gold_category"], rej_both["pred_category"], zero_division=0))
    else:
        print("\n== Category (on cases where BOTH are REJECT) ==")
        print("No rows where both gold and prediction are REJECT.")

    # Sample disagreements
    errs = df[
        (df["gold_label"] != df[pred_label_col]) |
        ((df["gold_label"] == "REJECT") & (df["gold_category"] != df["pred_category"]))
    ]
    print("\nSample errors:")
    text_col = "text_pred" if "text_pred" in errs.columns else ("text_gold" if "text_gold" in errs.columns else None)
    cols = ["id", "gold_label", pred_label_col, "gold_category", "pred_category"]
    if text_col:
        cols.insert(1, text_col)
    # Only print columns that exist
    cols = [c for c in cols if c in errs.columns]
    print(errs[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
