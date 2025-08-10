
# demo/predict_today.py
import argparse, sys, math, warnings, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import yfinance as yf
import joblib

# ---------- Locate news_pipeline ----------
HERE = Path(__file__).resolve().parent           # .../demo
ROOT = HERE.parent                               # project root

for p in [HERE, ROOT]:
    if (p / "news_pipeline.py").exists():
        sys.path.insert(0, str(p))
        break

try:
    from news_pipeline import run_news_pipeline, NEWYORK
except Exception as e:
    raise SystemExit(f"[predict_today] Could not import news_pipeline.py from {HERE} or {ROOT}: {e}")

# ---------- Config ----------
MODEL_DIR_CANDS = [
    ROOT / "models" / "finbert_multi_input",     # SavedModel dir
    ROOT / "models" / "finbert_multi_input.h5",  # H5 file
]
SCALER_CANDS = [
    ROOT / "models" / "numerical_scaler.pkl",
    ROOT / "models" / "feature_scaler.pkl",
]

OUT_CSV = (HERE / "data" / "processed" / "predict_today.csv").resolve()
MAX_LEN = 128
FINBERT_NAME = "ProsusAI/finbert"   # for tokenizer and (if needed) HF classifier

# ---------- Utilities ----------
def load_model_any():
    for path in MODEL_DIR_CANDS:
        if path.is_dir():
            try:
                m = tf.keras.models.load_model(str(path))
                return m, str(path)
            except Exception:
                pass
        elif path.is_file():
            try:
                m = tf.keras.models.load_model(str(path))
                return m, str(path)
            except Exception:
                pass
    return None, None

def load_scaler_any():
    for path in SCALER_CANDS:
        if path.exists():
            try:
                sc = joblib.load(str(path))
                return sc, str(path)
            except Exception:
                pass
    return None, None

def build_tech_features():
    hist = yf.download("^IXIC", period="6mo", interval="1d", auto_adjust=True, progress=False).dropna()
    hist["MA_5"] = hist["Close"].rolling(5).mean()
    hist["MA_20"] = hist["Close"].rolling(20).mean()
    hist["Momentum_1d"] = hist["Close"].diff()
    hist["Momentum_5d"] = hist["Close"].diff(5)
    row = hist.iloc[-1:][["MA_5","MA_20","Momentum_1d","Momentum_5d"]].fillna(0.0)
    return row.reset_index(drop=True)

def select_today_top_news(detailed_df: pd.DataFrame) -> pd.Series:
    df = detailed_df.copy()
    df["pub_time_et"] = pd.to_datetime(df["pub_time_et"], errors="coerce")
    try:
        tz = df["pub_time_et"].dt.tz
    except Exception:
        tz = None
    if tz is None:
        df["pub_time_et"] = df["pub_time_et"].dt.tz_localize(NEWYORK)
    today_et = pd.Timestamp.now(tz=NEWYORK).normalize()
    same_day = df[df["pub_time_et"].dt.normalize() == today_et]
    if not same_day.empty:
        df = same_day
    if "w" in df.columns:
        df = df.sort_values("w", ascending=False)
    else:
        df = df.sort_values("pub_time_et", ascending=False)
    return df.iloc[0]

@torch.no_grad()
def finbert_pos_neg(texts):
    """Return arrays (prob_pos, prob_neg) using HF FinBERT for the provided texts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(FINBERT_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(FINBERT_NAME).to(device).eval()
    probs_all = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i+16], padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        logits = mdl(**enc).logits
        probs = torch.softmax(logits, dim=1)  # [neg, neu, pos]
        probs_all.append(probs.detach().cpu().numpy())
    import numpy as np
    P = np.vstack(probs_all)
    return P[:,2], P[:,0]  # pos, neg

def prepare_inputs(headline_text: str, sent_score: float, scaler):
    tech = build_tech_features()
    # numerical vector order used in training:
    cols = ["sentiment_score_z","MA_5","MA_20","Momentum_1d","Momentum_5d"]
    num_vec = pd.DataFrame([[sent_score] + tech.iloc[0].tolist()], columns=cols).astype(float)

    if scaler is not None:
        try:
            use_cols = getattr(scaler, "feature_names_in_", cols)
            Xnum = pd.DataFrame(scaler.transform(num_vec[use_cols]), columns=use_cols)
        except Exception as e:
            warnings.warn(f"Scaler transform failed ({e}); using raw features.")
            Xnum = num_vec.copy()
    else:
        warnings.warn("No scaler found; using raw numerical features (assuming already standardized).")
        Xnum = num_vec.copy()

    # Tokenize for model's text input
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_NAME)
    enc = tokenizer(
        [headline_text],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="tf"
    )
    token_inputs = {
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc.get("token_type_ids", tf.zeros_like(enc["input_ids"])),
    }
    numerical_input = tf.convert_to_tensor(Xnum.values.astype("float32"))
    return token_inputs, numerical_input, Xnum

def predict(title_text: str, model, scaler):
    prob_pos, prob_neg = finbert_pos_neg([title_text])
    sent_score = float(prob_pos[0] - prob_neg[0])
    token_inputs, numerical_input, Xnum = prepare_inputs(title_text, sent_score, scaler)
    prob_up = float(model.predict([token_inputs, numerical_input], verbose=0).flatten()[0])
    y_hat = int(prob_up >= 0.5)
    return sent_score, Xnum, prob_up, y_hat

def auto_pick_and_predict(model, scaler):
    cutoff = dt.datetime.now(NEWYORK).replace(hour=9, minute=29, second=59, microsecond=0)
    metrics, detailed = run_news_pipeline(cutoff_et=cutoff, batch_size=8)
    if detailed is None or detailed.empty:
        raise SystemExit("[predict_today] No news found before cutoff; cannot run demo.")
    top = select_today_top_news(detailed)
    title_text = str(top.get("title", top.get("text_clean","")))
    sent_score, Xnum, prob_up, y_hat = predict(title_text, model, scaler)
    rec = {
        "mode": "auto",
        "pub_time_et": str(top.get("pub_time_et","")),
        "title": title_text,
        "sentiment_score": sent_score,
        "prob_up": prob_up,
        "y_hat": y_hat,
    }
    for k in Xnum.columns:
        rec[k] = float(Xnum.iloc[0][k])
    return rec

def manual_predict(title_text: str, model, scaler):
    sent_score, Xnum, prob_up, y_hat = predict(title_text, model, scaler)
    rec = {
        "mode": "manual",
        "pub_time_et": "",
        "title": title_text,
        "sentiment_score": sent_score,
        "prob_up": prob_up,
        "y_hat": y_hat,
    }
    for k in Xnum.columns:
        rec[k] = float(Xnum.iloc[0][k])
    return rec

def main():
    parser = argparse.ArgumentParser(description="Predict NASDAQ up/down (1/0) using one headline + tech indicators.")
    parser.add_argument("--title", type=str, default=None, help="If provided, use this headline instead of auto-picking today's top news (before 09:29 ET).")
    args = parser.parse_args()

    model, model_path = load_model_any()
    if model is None:
        raise SystemExit(
            "[predict_today] Could not find saved model.\n"
            "Save your trained model as either:\n"
            "  models/finbert_multi_input  (SavedModel dir)\n"
            "or\n"
            "  models/finbert_multi_input.h5\n"
        )
    scaler, scaler_path = load_scaler_any()

    if args.title:
        rec = manual_predict(args.title, model, scaler)
    else:
        rec = auto_pick_and_predict(model, scaler)

    # Pretty print
    print("\\n=== PREDICTION (1=UP, 0=DOWN) ===")
    print(f"Mode: {rec['mode']}")
    if rec.get("pub_time_et"): print(f"pub_time_et: {rec['pub_time_et']}")
    print(f"Headline: {rec['title']}")
    print(f"sentiment_score (pos-neg): {rec['sentiment_score']:.4f}")
    print("Numerical vector: [sentiment_score_z, MA_5, MA_20, Momentum_1d, Momentum_5d]")
    print(f"{rec['sentiment_score']:.6f}, {rec['MA_5']:.6f}, {rec['MA_20']:.6f}, {rec['Momentum_1d']:.6f}, {rec['Momentum_5d']:.6f}")
    print(f"Output (y_hat): {rec['y_hat']}   |   prob_up={rec['prob_up']:.3f}")

    # Save CSV
    df = pd.DataFrame([rec])
    OUT = (HERE / "data" / "processed" / "predict_today.csv").resolve()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\\nSaved → {OUT}")

import os
from datetime import datetime
# === Save prediction log ===
log_dir = os.path.join("demo", "data")
log_path = os.path.join(log_dir, "prediction_log.csv")
os.makedirs(log_dir, exist_ok=True)

# Prepare the log entry
today_str = datetime.now().strftime("%Y-%m-%d")
log_entry = pd.DataFrame([{
    "Date": today_str,
    "Predicted": int(pred_label),            # 1 = up, 0 = down
    "Confidence": float(pred_prob),          # model predicted probability for "up"
    "Sentiment_Score": float(sentiment_score),# today's average sentiment score
    "Actual": None                           # to be filled after market close
}])

# Append to CSV
if os.path.exists(log_path):
    log_entry.to_csv(log_path, mode='a', header=False, index=False)
else:
    log_entry.to_csv(log_path, index=False)

print(f"[LOG] Saved prediction to {log_path}")

if __name__ == "__main__":
    main()
