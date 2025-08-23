# demo/make_train_predictions.py
# -*- coding: utf-8 -*-
"""
Generate model prediction probabilities (prob_up) on historical data for threshold optimization.
Workflow:
1) Read your historical CSV (must contain: title_clean, Date or market_date, label, sentiment_score_z)
2) Use yfinance to fetch ^IXIC over the same period, compute technical indicators (MA_5, MA_20, Momentum_1d, Momentum_5d)
3) Merge technical indicators into each news entry
4) Load tokenizer + your trained model (.keras preferred, .h5 weights as fallback)
5) Batch inference, output prob_up to CSV
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Lambda
import keras  # enable_unsafe_deserialization

from transformers import AutoTokenizer, TFAutoModel

# ------------------ PATH SETTINGS ------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOTS = [SCRIPT_DIR, SCRIPT_DIR.parent]

def _find(path_rel: Path) -> Path:
    for r in ROOTS:
        p = (r / path_rel).resolve()
        if p.exists():
            return p
    return (ROOTS[0] / path_rel).resolve()

# Modify these two paths to your actual data files
SRC_CSV_REL   = Path("data/processed/Merged_News_and_NASDAQ_Data_Extended_With_Sentiment.csv")
OUT_CSV_REL   = Path("data/demo/processed/train_predictions.csv")

KERAS_REL     = Path("models/finbert_multi_input.keras")    # Your saved .keras model
H5_REL        = Path("models/finbert_multi_input.h5")       # Your saved .h5 weights (fallback)

SRC_CSV_PATH  = _find(SRC_CSV_REL)
OUT_CSV_PATH  = _find(OUT_CSV_REL)
KERAS_PATH    = _find(KERAS_REL)
H5_PATH       = _find(H5_REL)

TOKENIZER_NAME = "ProsusAI/finbert"
MAX_LENGTH = 128
BATCH_SIZE = 32

# Your numeric feature order (must match training)
NUMERIC_ORDER = ["sentiment_score_z", "MA_5", "MA_20", "Momentum_1d", "Momentum_5d"]

# ------------------ UTILITY FUNCTIONS ------------------
def _to_date_col(df: pd.DataFrame) -> pd.Series:
    # Your file may have 'Date' or 'market_date'
    date_col = None
    for c in ["Date", "market_date", "date"]:
        if c in df.columns:
            date_col = c; break
    if date_col is None:
        raise ValueError("No date column found. Requires either 'Date' or 'market_date'")
    return pd.to_datetime(df[date_col])

def _fetch_ixic_with_indicators(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    # Fetch extra days to avoid NaN in rolling calculations
    start_pad = start_dt - timedelta(days=30)
    df = yf.download("^IXIC", start=start_pad, end=end_dt + timedelta(days=1), auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("Failed to fetch ^IXIC price data via yfinance")
    df = df.reset_index()
    df = df.sort_values("Date")

    # Technical indicators
    df["MA_5"]       = df["Close"].rolling(5).mean()
    df["MA_20"]      = df["Close"].rolling(20).mean()
    df["Momentum_1d"]= df["Close"].diff(1)
    df["Momentum_5d"]= df["Close"].diff(5)
    df = df.ffill().fillna(0.0)

    # Keep only required date range
    mask = (df["Date"] >= start_dt.normalize()) & (df["Date"] <= end_dt.normalize())
    return df.loc[mask].copy()

def _tokenize_batch(texts, tokenizer):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )

def _build_model_lambda_wrapped_bert() -> Model:
    """Rebuild the same structure as training for .h5 weights (BERT pooler + numeric features → Dense)."""
    bert = TFAutoModel.from_pretrained(TOKENIZER_NAME)

    input_ids      = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
    token_type_ids = Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="token_type_ids")
    numeric_input  = Input(shape=(5,),          dtype=tf.float32, name="numerical_input")

    def call_bert(tensors):
        ids, mask, types = tensors
        out = bert(
            input_ids=tf.cast(ids, tf.int32),
            attention_mask=tf.cast(mask, tf.int32),
            token_type_ids=tf.cast(types, tf.int32),
        )
        return out.pooler_output  # (batch, 768)

    pooled = Lambda(call_bert, name="bert_pooled")([input_ids, attention_mask, token_type_ids])
    x = Concatenate()([pooled, numeric_input])
    x = Dense(128, activation="relu", name="dense")(x)
    x = Dropout(0.3, name="dropout")(x)
    out = Dense(1, activation="sigmoid", name="dense_1")(x)

    model = Model(
        inputs=[input_ids, attention_mask, token_type_ids, numeric_input],
        outputs=out
    )
    return model

def _load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # First try .keras (can deserialize Lambda directly using unsafe + provide base_bert_model)
    if KERAS_PATH.exists():
        try:
            keras.config.enable_unsafe_deserialization()
            # Ensure Lambda can find base_bert_model (some save formats use this name)
            import builtins as _bi
            global base_bert_model
            base_bert_model = TFAutoModel.from_pretrained(TOKENIZER_NAME)
            _bi.base_bert_model = base_bert_model

            model = load_model(
                KERAS_PATH,
                custom_objects={"base_bert_model": base_bert_model},
                safe_mode=False
            )
            print(f"[OK] Loaded .keras model: {KERAS_PATH}")
            return model, tokenizer
        except Exception as e:
            print(f"[warn] Failed to read .keras: {e} → using .h5 weights")

    # Fallback: use .h5 weights
    if not H5_PATH.exists():
        raise FileNotFoundError(f"Model not found:\n  .keras: {KERAS_PATH}\n  .h5: {H5_PATH}")
    model = _build_model_lambda_wrapped_bert()
    # by_name=True allows matching weights by layer name
    model.load_weights(H5_PATH, by_name=True)
    print(f"[OK] Loaded .h5 weights with rebuilt architecture: {H5_PATH}")
    return model, tokenizer

def _predict_in_batches(model: Model, tokenizer, titles, X_num_arr) -> np.ndarray:
    """Run predictions on all samples in batches, return prob_up (N,)"""
    probs = []
    N = len(titles)
    for i in range(0, N, BATCH_SIZE):
        batch_titles = titles[i:i+BATCH_SIZE]
        enc = _tokenize_batch(batch_titles, tokenizer)
        x_num = X_num_arr[i:i+BATCH_SIZE].astype(np.float32)

        # Some Keras versions require different input formats; try both
        try:
            preds = model.predict(
                [
                    {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "token_type_ids": enc["token_type_ids"],
                    },
                    x_num
                ],
                verbose=0
            )
        except Exception:
            preds = model.predict(
                [
                    enc["input_ids"],
                    enc["attention_mask"],
                    enc["token_type_ids"],
                    x_num
                ],
                verbose=0
            )
        probs.append(preds.reshape(-1))
    return np.concatenate(probs, axis=0)

# ------------------ MAIN PROCESS ------------------
if __name__ == "__main__":
    print(f"[in] Reading historical data: {SRC_CSV_PATH}")
    df = pd.read_csv(SRC_CSV_PATH)
    if "title_clean" not in df.columns or "label" not in df.columns or "sentiment_score_z" not in df.columns:
        raise ValueError("CSV must contain 'title_clean', 'label', 'sentiment_score_z' (and also a date column)")

    # Date column
    dates = _to_date_col(df)
    df["_date_"] = dates.dt.normalize()

    # Fetch technical indicators
    start_dt, end_dt = df["_date_"].min(), df["_date_"].max()
    print(f"[yfinance] Fetching ^IXIC {start_dt.date()} ~ {end_dt.date()} and computing technical indicators…")
    ixic = _fetch_ixic_with_indicators(start_dt, end_dt)
    ixic = ixic.rename(columns={"Date": "_date_"})[["_date_", "MA_5", "MA_20", "Momentum_1d", "Momentum_5d"]]

    # --- Ensure both sides have a clean _date_ column (no timezone, normalized to day) ---
def _to_naive_date_col(s):
    s = pd.to_datetime(s, errors="coerce")
    # Remove timezone, normalize to midnight
    s = s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
    return s.dt.normalize()

# Left df: create single-level _date_ column
date_col = None
for c in ["_date_", "Date", "market_date", "date"]:
    if c in df.columns:
        date_col = c
        break
if date_col is None:
    raise ValueError("Historical data missing a date column (requires Date/market_date)")

df = df.copy()
df["_date_"] = _to_naive_date_col(df[date_col])

# Right ixic: ensure _date_ column exists (from index or Date column)
ixic = ixic.copy()
if "_date_" not in ixic.columns:
    if "Date" in ixic.columns:
        ixic["_date_"] = _to_naive_date_col(ixic["Date"])
    elif isinstance(ixic.index, pd.DatetimeIndex):
        ixic["_date_"] = _to_naive_date_col(ixic.index.to_series())
    else:
        raise ValueError("ixic lacks Date column or DatetimeIndex, cannot create _date_")

# Flatten MultiIndex columns from yfinance if present
if isinstance(ixic.columns, pd.MultiIndex):
    ixic.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in ixic.columns.to_list()
    ]

# Keep only required columns for merging
keep_cols = ["_date_", "MA_5", "MA_20", "Momentum_1d", "Momentum_5d"]
missing = [c for c in keep_cols if c not in ixic.columns]
if missing:
    raise ValueError(f"ixic missing required technical columns: {missing}")
ixic = ixic[keep_cols]

# Final merge (both sides have single-level _date_)
dfm = df.merge(ixic, on="_date_", how="left").ffill().fillna(0.0)

    # Prepare input
titles = dfm["title_clean"].astype(str).fillna("").tolist()
if not set(NUMERIC_ORDER).issubset(dfm.columns):
    missing = [c for c in NUMERIC_ORDER if c not in dfm.columns]
    raise ValueError(f"Missing numeric feature columns: {missing}")
X_num = dfm[NUMERIC_ORDER].values.astype(np.float32)

print("[model] Loading tokenizer + model…")
model, tokenizer = _load_model_and_tokenizer()

print("[predict] Running batch inference…")
prob_up = _predict_in_batches(model, tokenizer, titles, X_num)

    # Output
out = dfm.copy()
out["prob_up"] = prob_up
out_path = OUT_CSV_PATH
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False, encoding="utf-8")
print(f"[out] Saved to: {out_path}")
print("[hint] Next, run find_best_threshold.py to scan for the optimal threshold.")
