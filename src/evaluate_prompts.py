import argparse, os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="predictions.csv", help="Predictions CSV (id,text,pred_label,pred_category)")
ap.add_argument("--gold", default="data/sample_reviews.csv", help="Gold CSV (id,text,gold_label,gold_category)")
args = ap.parse_args()

if not os.path.exists(args.pred):
    raise SystemExit(f"Predictions file not found: {args.pred}")

gold = pd.read_csv(args.gold)
pred = pd.read_csv(args.pred)

gold.columns = gold.columns.str.strip()
pred.columns = pred.columns.str.strip()

df = gold.merge(pred, on="id", how="left", suffixes=("_gold", "_pred"))

# ==== Binary approve/reject ====
print("== APPROVE/REJECT ==")
print(classification_report(df["gold_label"], df["pred_label"], zero_division=0))
print(confusion_matrix(df["gold_label"], df["pred_label"]))

# ==== Category metrics ====
# View A: On GOLD rejects only (penalizes false negatives as 'None')
rej_gold = df[df["gold_label"] == "REJECT"].copy()
rej_gold["pred_category"] = rej_gold["pred_category"].astype(object).where(rej_gold["pred_category"].notna(), "None")
rej_gold["gold_category"] = rej_gold["gold_category"].astype(str)
rej_gold["pred_category"] = rej_gold["pred_category"].astype(str)

print("\n== Category (on GOLD rejects; 'None' if model approved) ==")
print(classification_report(rej_gold["gold_category"], rej_gold["pred_category"], zero_division=0))

# View B: On cases where BOTH gold and pred are REJECT (measures pure category assignment)
rej_both = df[(df["gold_label"] == "REJECT") & (df["pred_label"] == "REJECT")].copy()
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
    (df["gold_label"] != df["pred_label"]) |
    ((df["gold_label"] == "REJECT") & (df["gold_category"] != df["pred_category"]))
]
print("\nSample errors:")
text_col = "text_pred" if "text_pred" in errs.columns else ("text_gold" if "text_gold" in errs.columns else None)
cols = ["id","gold_label","pred_label","gold_category","pred_category"]
if text_col: cols.insert(1, text_col)
print(errs[[c for c in cols if c in errs.columns]].head(10).to_string(index=False))