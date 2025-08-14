import json
import numpy as np
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent
preds = root / "data" / "processed" / "train_predictions.csv"
out   = root / "models" / "calibration.json"
out.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(preds)
if {"prob_up", "label"}.issubset(df.columns):
    corr = np.corrcoef(df["prob_up"].astype(float), df["label"].astype(float))[0, 1]
    invert = bool(corr < 0)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"invert": invert}, f, indent=2)
    print(f"corr={corr:.3f} → invert={invert} | wrote {out}")
else:
    print("train_predictions.csv missing columns prob_up/label")
