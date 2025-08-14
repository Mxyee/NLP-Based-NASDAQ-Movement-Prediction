"""
End-to-end demo:
- Fetch recent NASDAQ news (last N days, UK time) via Google News RSS
- Compute sentiment with FinBERT (fallback to Loughran–McDonald, negation-aware)
- Fetch NASDAQ (^IXIC) prices via yfinance and compute tech indicators
- Load fine-tuned model (.keras preferred, .h5 fallback)
- Predict today's direction (1=Up / 0=Down)

Numeric feature order (must match training):
["sentiment_score_z", "MA_5", "MA_20", "Momentum_1d", "Momentum_5d"]
"""

from __future__ import annotations

import os  # CHANGED: Moved import to top
# CHANGED: Set environment variables before imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Suppress TensorFlow optimization warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress Hugging Face cache warnings

import sys
import json
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse
import pytz
import yfinance as yf

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
import keras  # for enable_unsafe_deserialization
from transformers import AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification, pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import joblib

# sentiment module (ensure demo/sentiment_lm.py exists)
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from sentiment_lm import add_lm_sentiment  # noqa

# ------------------------- Robust paths -------------------------
ROOTS = [SCRIPT_DIR, SCRIPT_DIR.parent]

def find(path_rel: Path) -> Path:
    for r in ROOTS:
        p = (r / path_rel).resolve()
        if p.exists():
            return p
    return (ROOTS[0] / path_rel).resolve()

LM_REL = Path("resources_lexicons/Loughran-McDonald_MasterDictionary_1993-2024.csv")
KERAS_REL = Path("models/finbert_multi_input.keras")
H5_REL = Path("models/finbert_multi_input.h5")
SCALER_REL = Path("models/num_scaler.pkl")
META_REL = Path("models/demo_metadata.json")
THRESH_PATH = find(Path("models/threshold.json"))
CALIB_PATH = find(Path("models/calibration.json"))
TRAIN_PREDS_REL = Path("data/processed/train_predictions.csv")
TRAIN_PREDS_PATH = find(TRAIN_PREDS_REL)

LM_DICT_PATH = find(LM_REL)
KERAS_MODEL_PATH = find(KERAS_REL)
H5_WEIGHTS_PATH = find(H5_REL)
SCALER_PATH = find(SCALER_REL)

# ------------------------- Config -------------------------
MAX_LENGTH = 128
TOKENIZER_NAME = "ProsusAI/finbert"
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q=nasdaq&hl=en-GB&gl=GB&ceid=GB:en"

# ------------------------- 1) Fetch news -------------------------
def fetch_recent_news(days: int = 1) -> pd.DataFrame:
    uk_tz = pytz.timezone("Europe/London")
    cutoff_time = datetime.now(uk_tz) - timedelta(days=days)

    r = requests.get(GOOGLE_NEWS_RSS, timeout=20)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    items = []
    for item in root.findall(".//item"):
        title = (item.find("title").text or "").strip()
        pub_date_str = item.find("pubDate").text
        pub_dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=pytz.UTC).astimezone(uk_tz)
        if pub_dt >= cutoff_time and title:
            items.append({"title": title, "pub_time_uk": pub_dt})

    return pd.DataFrame(items).drop_duplicates(subset="title")

# ------------------------- 2) NASDAQ + tech indicators -------------------------
def fetch_nasdaq_data(period_days: int = 30) -> pd.DataFrame:
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=period_days)
    df = yf.download("^IXIC", start=start_date, end=end_date, auto_adjust=False, progress=False)
    if df.empty:
        return df
    df = df.reset_index()
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["Momentum_1d"] = df["Close"].diff(1)
    df["Momentum_5d"] = df["Close"].diff(5)
    df = df.ffill().fillna(0.0)
    return df

# ------------------------- 3) Sentiment (FinBERT) -------------------------
def add_finbert_sentiment(news_df, text_col='title'):
    try:
        sentiment_pipeline = pipeline('sentiment-analysis', model='ProsusAI/finbert')
        results = sentiment_pipeline(news_df[text_col].tolist())
        news_df['finbert_label'] = [r['label'] for r in results]
        news_df['finbert_score'] = [1 if r['label'] == 'positive' else -1 if r['label'] == 'negative' else 0 for r in results]
        news_df['finbert_conf'] = [r['score'] for r in results]
    except Exception as e:
        print(f"[warn] FinBERT sentiment failed: {e}, falling back to LM")
        news_df = add_lm_sentiment(
            news_df,
            text_col=text_col,
            lm_csv_path=LM_DICT_PATH,
            time_col="pub_time_uk",
            use_negation=True,
            neg_window=3,
            zscore_mode="within_day"
        )
        news_df['finbert_score'] = news_df['lm_score']
        news_df['finbert_conf'] = news_df.get('lm_hit_ratio', 0.0)
        news_df['finbert_label'] = news_df['lm_score'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')
    return news_df

# ------------------------- 4) Model builders / loaders -------------------------
def build_model_lambda_wrapped_bert():
    clf = TFAutoModelForSequenceClassification.from_pretrained(
        TOKENIZER_NAME,
        output_hidden_states=True,
        output_attentions=False
    )
    bert = clf.bert

    input_ids = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="attention_mask")
    token_type_ids = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="token_type_ids")
    numeric_input = tf.keras.Input(shape=(5,), dtype=tf.float32, name="numerical_input")

    def call_bert(tensors):
        ids, mask, types = tensors
        out = bert(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=types
        )
        return out.pooler_output

    pooled = Lambda(call_bert, output_shape=(768,), name="lambda")([input_ids, attention_mask, token_type_ids])
    x = Concatenate()([pooled, numeric_input])
    x = Dense(128, activation="relu", name="dense")(x)
    x = Dropout(0.3, name="dropout")(x)
    out = Dense(1, activation="sigmoid", name="dense_1")(x)

    model = Model(inputs=[input_ids, attention_mask, token_type_ids, numeric_input], outputs=out)
    return model

def load_trained_model():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    FORCE_H5 = True  # Prefer fine-tuned .keras

    if not FORCE_H5 and KERAS_MODEL_PATH.exists():
        try:
            keras.config.enable_unsafe_deserialization()
            def call_bert(tensors):
                ids, mask, types = tensors
                bert = TFAutoModel.from_pretrained(TOKENIZER_NAME)
                out = bert(input_ids=ids, attention_mask=mask, token_type_ids=types)
                return out.pooler_output
            model = load_model(
                KERAS_MODEL_PATH,
                custom_objects={"Lambda": Lambda, "call_bert": call_bert},
                safe_mode=False
            )
            print(f"[OK] Loaded .keras model: {KERAS_MODEL_PATH}")
            return model, tokenizer
        except Exception as e:
            print(f"[warn] .keras load failed: {e} → trying .h5 fallback")

    if not H5_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"No model file found.\nTried .keras: {KERAS_MODEL_PATH}\nTried .h5: {H5_WEIGHTS_PATH}"
        )
    model = build_model_lambda_wrapped_bert()
    try:
        model.load_weights(H5_WEIGHTS_PATH, by_name=True, skip_mismatch=False)
        print("[OK] Loaded .h5 weights (strict by_name)")
    except ValueError as e:
        print(f"[warn] Strict load failed: {e} → trying skip_mismatch=True...")
        model.load_weights(H5_WEIGHTS_PATH, by_name=True, skip_mismatch=True)
        print(f"[OK] Loaded .h5 weights with skip_mismatch=True")
    return model, tokenizer



def load_threshold_and_calibration(default_threshold=0.01):
    threshold = default_threshold
    invert = False

    if THRESH_PATH.exists():
        try:
            with open(THRESH_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg.get('threshold'), (int, float)):
                threshold = float(cfg['threshold'])
                print(f"[cfg] threshold loaded: {threshold}")
        except Exception as e:
            print(f"[warn] failed to read threshold.json: {e}")

    if CALIB_PATH.exists():
        try:
            with open(CALIB_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg.get("invert"), bool):
                invert = cfg["invert"]
                print(f"[cfg] invert loaded: {invert}")
        except Exception as e:
            print(f"[warn] failed to read calibration.json: {e}")
    else:
        if TRAIN_PREDS_PATH.exists():
            try:
                dfp = pd.read_csv(TRAIN_PREDS_PATH)
                if {"prob_up", "label"}.issubset(dfp.columns):
                    probs = dfp["prob_up"].values
                    labels = dfp["label"].values
                    thresholds = np.arange(0.01, 0.99, 0.01)
                    best_f1, best_thresh = 0, default_threshold
                    for t in thresholds:
                        preds = (probs >= t).astype(int)
                        f1 = f1_score(labels, preds)
                        if f1 > best_f1:
                            best_f1, best_thresh = f1, t
                    threshold = best_thresh
                    corr = np.corrcoef(probs, labels)[0, 1]
                    invert = bool(corr < 0)
                    print(f"[auto] Optimal threshold: {threshold} (F1={best_f1:.3f}), invert={invert}")
                    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with open(THRESH_PATH, "w", encoding="utf-8") as f:
                        json.dump({"threshold": threshold}, f)
                    with open(CALIB_PATH, "w", encoding="utf-8") as f:
                        json.dump({"invert": invert}, f)
                else:
                    print("[auto] train_predictions.csv missing prob_up/label, skipping calibration")
            except Exception as e:
                print(f"[auto] Failed to compute threshold: {e}")
        else:
            print("[auto] No train_predictions.csv, using default threshold=0.01")
    return threshold, invert

def try_load_scaler_and_threshold():
    scaler = StandardScaler()
    threshold = 0.01
    if SCALER_PATH.exists():
        try:
            scaler = joblib.load(SCALER_PATH)
            print(f"[OK] Loaded scaler: {SCALER_PATH}")
        except Exception as e:
            print(f"[warn] Failed to load scaler: {e}, using default scaler")
            # CHANGED: Fit default scaler with dummy data
            dummy_data = np.array([
                [0.0, 21000.0, 21000.0, 0.0, 0.0],  # Typical NASDAQ values
                [1.0, 21500.0, 21200.0, 50.0, 500.0],
                [-1.0, 20500.0, 20800.0, -50.0, -500.0]
            ])
            scaler.fit(dummy_data)
            print(f"[info] Default scaler fitted with dummy data: mean={scaler.mean_}, scale={scaler.scale_}")
    else:
        print(f"[warn] No scaler file found, using default scaler")
        # CHANGED: Fit default scaler with dummy data
        dummy_data = np.array([
            [0.0, 21000.0, 21000.0, 0.0, 0.0],
            [1.0, 21500.0, 21200.0, 50.0, 500.0],
            [-1.0, 20500.0, 20800.0, -50.0, -500.0]
        ])
        scaler.fit(dummy_data)
        print(f"[info] Default scaler fitted with dummy data: mean={scaler.mean_}, scale={scaler.scale_}")
    return scaler, threshold

# ------------------------- 5) Build inputs & predict -------------------------
def build_encodings_per_title(news_df: pd.DataFrame, tokenizer: AutoTokenizer):
    enc_list = []
    for t in news_df["title"].astype(str).fillna(""):
        enc_list.append(
            tokenizer(
                t,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="tf"
            )
        )
    return enc_list

def prepare_inputs(
    news_df: pd.DataFrame,
    nasdaq_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    sentiment_value: float,
    scaler=StandardScaler|None):
    if news_df.empty or nasdaq_df.empty:
        raise ValueError("Empty news or NASDAQ data — cannot predict.")

    ma5 = float(nasdaq_df["MA_5"].iloc[-1])
    ma20 = float(nasdaq_df["MA_20"].iloc[-1])
    mom1 = float(nasdaq_df["Momentum_1d"].iloc[-1])
    mom5 = float(nasdaq_df["Momentum_5d"].iloc[-1])
    print(f"[info] Technical indicators: MA_5={ma5:.2f}, MA_20={ma20:.2f}, Momentum_1d={mom1:.2f}, Momentum_5d={mom5:.2f}")

    X_num_raw = np.asarray([[sentiment_value, ma5, ma20, mom1, mom5]], dtype=np.float32)
    X_num_raw[:, 0] *= arg.sent_w
    try:
        X_num = scaler.transform(X_num_raw.astype(np.float32))
        # CHANGED: Log finbert_score instead of sentiment_score_z
        print(f"[info] finbert_score(before scale)={X_num_raw[0, 0]:.4f} (sent_w={arg.sent_w})")
        print(f"[info] Scaled features: finbert_score={X_num[0, 0]:.4f}, MA_5={X_num[0, 1]:.4f}, "
              f"MA_20={X_num[0, 2]:.4f}, Momentum_1d={X_num[0, 3]:.4f}, Momentum_5d={X_num[0, 4]:.4f}")
    except Exception as e:
        print(f"[warn] Scaler transform failed: {e}, using raw numeric features")
        X_num = X_num_raw.astype(np.float32)
    enc_list = build_encodings_per_title(news_df, tokenizer)
    return enc_list, X_num

def run_prediction(model: Model, enc_list, X_num: np.ndarray, threshold: float = 0.01, invert: bool = False) -> tuple[float, int]:
    batch = {
        "input_ids": tf.concat([e["input_ids"] for e in enc_list], axis=0),
        "attention_mask": tf.concat([e["attention_mask"] for e in enc_list], axis=0),
        "token_type_ids": tf.concat([e["token_type_ids"] for e in enc_list], axis=0),
    }
    batch_size = int(batch["input_ids"].shape[0])
    X_num_batched = np.repeat(X_num.astype(np.float32), batch_size, axis=0)

    inputs = [
         batch["input_ids"],
         batch["attention_mask"],
         batch["token_type_ids"],
        X_num_batched
    ]

    preds = model.predict(
        [batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], X_num_batched],
        verbose=0
    ).reshape(-1)

    print(f"[diag] batch preds: min={preds.min():.4f}, p25={np.percentile(preds,25):.4f}, "
          f"median={np.median(preds):.4f}, p75={np.percentile(preds,75):.4f}, max={preds.max():.4f}")
    try:
        samp = np.asarray(preds[:5]).reshape(-1)
        print(f"[diag] first 5 preds: {', '.join(f'{p:.4f}' for p in samp)}")
    except Exception:
        pass
    avg_prob = float(np.mean(preds))
    if invert:
        avg_prob = 1.0 - avg_prob
    label = int(avg_prob >= threshold)
    return avg_prob, label

def _probe(model, tokenizer, X_num):
    tests = [
        "Stocks plunge on recession fears, bankruptcy risks and weak earnings.",
        "Markets rally to record highs as tech leads gains and strong earnings."
    ]
    enc = tokenizer(tests, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="tf")
    Xb = np.repeat(X_num, repeats=len(tests), axis=0)
    # CHANGED: Format inputs as [dict, numerical_input]
    inputs = [
            enc["input_ids"],
            enc["attention_mask"],
             enc["token_type_ids"],
        Xb
    ]
    p = model.predict(inputs, verbose=0).reshape(-1)
    print(f"[probe] neg headline prob_up={p[0]:.4f} | pos headline prob_up={p[1]:.4f}")
def fine_tune_model(train_news_df, train_nasdaq_df, train_labels, epochs=3, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = build_model_lambda_wrapped_bert()

    # Prepare sentiment
    train_news_df = add_finbert_sentiment(train_news_df)
    # Group by date if multiple news per day (adjust 'date' column name as needed)
    sentiment_values = train_news_df.groupby('date')['finbert_score'].mean().values

    # Prepare technical indicators
    train_nasdaq_df["MA_5"] = train_nasdaq_df["Close"].rolling(5).mean()
    train_nasdaq_df["MA_20"] = train_nasdaq_df["Close"].rolling(20).mean()
    train_nasdaq_df["Momentum_1d"] = train_nasdaq_df["Close"].diff(1)
    train_nasdaq_df["Momentum_5d"] = train_nasdaq_df["Close"].diff(5)
    train_nasdaq_df = train_nasdaq_df.ffill().fillna(0.0)
    X_num_raw = np.array([
        sentiment_values,
        train_nasdaq_df["MA_5"].values,
        train_nasdaq_df["MA_20"].values,
        train_nasdaq_df["Momentum_1d"].values,
        train_nasdaq_df["Momentum_5d"].values
    ]).T

    # Scale numerical features
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num_raw)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[OK] Saved new scaler: {SCALER_PATH}")

    # Tokenize titles
    enc = tokenizer(
        train_news_df['title'].tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="tf"
    )
    X_text = [enc['input_ids'], enc['attention_mask'], enc['token_type_ids']]

    # Fine-tune
    model.compile(optimizer=Adam(1e-5), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(
        [X_text[0], X_text[1], X_text[2], X_num],
        np.array(train_labels),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )

    # Save fine-tuned model
    model.save(KERAS_MODEL_PATH)
    print(f"[OK] Fine-tuned model saved to {KERAS_MODEL_PATH}")

    # Generate training predictions for threshold calibration
    train_preds = model.predict([X_text[0], X_text[1], X_text[2], X_num], verbose=0).reshape(-1)
    pd.DataFrame({"prob_up": train_preds, "label": train_labels}).to_csv(TRAIN_PREDS_PATH, index=False)
    print(f"[OK] Saved training predictions to {TRAIN_PREDS_PATH}")



# ------------------------- Main -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sent_w", type=float, default=1.0, 
                        help="Multiplier for sentiment feature before scaling (default=1.0)")
    parser.add_argument("--sweep_sent_w", action="store_true",
                        help="Sweep a set of sentiment weights and print results (no file outputs)")
    arg = parser.parse_args()
    print(f"[paths] LM: {LM_DICT_PATH}")
    print(f"[paths] .keras: {KERAS_MODEL_PATH}")
    print(f"[paths] .h5: {H5_WEIGHTS_PATH}")

    # 1) Data
    print("[step] Fetch news (1 day, UK tz)...")
    news_df = fetch_recent_news(days=1)
    print(f"[info] News fetched: {len(news_df)} rows")

    print("[step] Fetch NASDAQ (^IXIC) + indicators...")
    nasdaq_df = fetch_nasdaq_data()
    print(f"[info] NASDAQ rows: {len(nasdaq_df)}")

    # 2) Sentiment (FinBERT)
    print("[step] Compute FinBERT sentiment + within-day z-score...")
    news_df = add_finbert_sentiment(news_df, text_col="title")
    finbert_score = float(news_df['finbert_score'].mean()) if not news_df.empty else 0.0
    if not news_df.empty and news_df['finbert_score'].std() > 0:
        news_df['finbert_score_z'] = (news_df['finbert_score'] - news_df['finbert_score'].mean()) / news_df['finbert_score'].std()
    else:
        news_df['finbert_score_z'] = 0.0
    sentiment_score_z = float(news_df["finbert_score_z"].mean()) if "finbert_score_z" in news_df.columns else 0.0
    print(f"[info] sentiment_feature (mean finbert_score): {finbert_score:.4f}  |  sentiment_score_z (mean): {sentiment_score_z:.4f}")
    print(f"[debug] mean(finbert_score)={news_df['finbert_score'].mean():.4f}, "
          f"mean(finbert_conf)={news_df['finbert_conf'].mean():.3f}, any_hits={int((news_df['finbert_score'] != 0).sum() > 0)}")
    # 3) Model + tokenizer (+ scaler/threshold)
    print("[step] Load model & tokenizer...")
    model, tokenizer = load_trained_model()
    scaler, threshold = try_load_scaler_and_threshold()

    # 4) Inputs
    print("[step] Build model inputs...")
    enc_list, X_num = prepare_inputs(
        news_df=news_df,
        nasdaq_df=nasdaq_df,
        tokenizer=tokenizer,
        sentiment_value=finbert_score,  # Use finbert_score instead of sentiment_score_z
        scaler=scaler
    )

    # 5) Predict
    print("[step] Predict...")
    threshold, invert = load_threshold_and_calibration(default_threshold=0.01)
    avg_prob, label = run_prediction(model, enc_list, X_num, threshold=threshold, invert=invert)
    print("\n" + "="*60)
    print("  PREDICTION")
    print("="*60)
    print(f"Avg probability of UP: {avg_prob:.4f}")
    print(f"Predicted direction:  {'UP (1) 📈' if label==1 else 'DOWN (0) 📉'}")
    print("="*60 + "\n")

    # 6) Save log
    out_dir = (SCRIPT_DIR / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prediction_log.csv"
    log = news_df.copy()
    log["predicted_prob_up"] = avg_prob
    log["predicted_direction"] = "Up" if label == 1 else "Down"
    keep_cols = ["pub_time_uk", "title", "finbert_label", "finbert_score", "finbert_conf", "finbert_score_z",
                 "predicted_prob_up", "predicted_direction"]
    for c in keep_cols:
        if c not in log.columns:
            log[c] = np.nan
    
    if out_path.exists():
        log[keep_cols].to_csv(out_path, mode = 'a', header = False,index = False, encoding = "utf-8")
    else:
        log[keep_cols].to_csv(out_path, mode = "w", header = False,index=False, encoding="utf-8")
    print(f"[saved] {out_path}")

    # 7) Probe
    _probe(model, tokenizer, X_num)

