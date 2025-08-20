# demo/make_train_predictions.py
# -*- coding: utf-8 -*-
"""
產生歷史資料的模型預測機率 (prob_up) 以便做 threshold 最佳化。
流程：
1) 讀入你的歷史資料 CSV（要含 title_clean, Date 或 market_date, label, sentiment_score_z）
2) 用 yfinance 抓取同區間的 ^IXIC，計算技術指標 (MA_5, MA_20, Momentum_1d, Momentum_5d)
3) 合併技術指標到每筆新聞
4) 載入 tokenizer + 你的已訓練模型（.keras 優先，.h5 權重為備案）
5) 批量推論，輸出 prob_up 到 CSV
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

# ------------------ 路徑設定 ------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ROOTS = [SCRIPT_DIR, SCRIPT_DIR.parent]

def _find(path_rel: Path) -> Path:
    for r in ROOTS:
        p = (r / path_rel).resolve()
        if p.exists():
            return p
    return (ROOTS[0] / path_rel).resolve()

# 修改這兩個路徑成你的實際資料檔名
SRC_CSV_REL   = Path("data/processed/Merged_News_and_NASDAQ_Data_Extended_With_Sentiment.csv")
OUT_CSV_REL   = Path("data/demo/processed/train_predictions.csv")

KERAS_REL     = Path("models/finbert_multi_input.keras")    # 你保存的 .keras 完整模型
H5_REL        = Path("models/finbert_multi_input.h5")       # 你保存的 .h5 權重（備案）

SRC_CSV_PATH  = _find(SRC_CSV_REL)
OUT_CSV_PATH  = _find(OUT_CSV_REL)
KERAS_PATH    = _find(KERAS_REL)
H5_PATH       = _find(H5_REL)

TOKENIZER_NAME = "ProsusAI/finbert"
MAX_LENGTH = 128
BATCH_SIZE = 32

# 你的數值特徵順序（跟訓練時一致）
NUMERIC_ORDER = ["sentiment_score_z", "MA_5", "MA_20", "Momentum_1d", "Momentum_5d"]

# ------------------ 工具函數 ------------------
def _to_date_col(df: pd.DataFrame) -> pd.Series:
    # 你的檔案可能是 'Date' 或 'market_date'
    date_col = None
    for c in ["Date", "market_date", "date"]:
        if c in df.columns:
            date_col = c; break
    if date_col is None:
        raise ValueError("找不到日期欄位，至少需要 'Date' 或 'market_date' 其中之一")
    return pd.to_datetime(df[date_col])

def _fetch_ixic_with_indicators(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    # 多抓幾天避免 rolling 的 NaN
    start_pad = start_dt - timedelta(days=30)
    df = yf.download("^IXIC", start=start_pad, end=end_dt + timedelta(days=1), auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError("yfinance 抓不到 ^IXIC 價格資料")
    df = df.reset_index()
    df = df.sort_values("Date")

    # 技術指標
    df["MA_5"]       = df["Close"].rolling(5).mean()
    df["MA_20"]      = df["Close"].rolling(20).mean()
    df["Momentum_1d"]= df["Close"].diff(1)
    df["Momentum_5d"]= df["Close"].diff(5)
    df = df.ffill().fillna(0.0)

    # 只留你需要的日期範圍
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
    """為 .h5 權重重建與訓練時等價的結構（BERT pooler + 數值特徵接 Dense）。"""
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

    # 先嘗試 .keras（可直接反序列化 Lambda，用 unsafe + 提供 base_bert_model）
    if KERAS_PATH.exists():
        try:
            keras.config.enable_unsafe_deserialization()
            # 讓 Lambda 找得到 base_bert_model（有些存檔版本會用這個名字）
            import builtins as _bi
            global base_bert_model
            base_bert_model = TFAutoModel.from_pretrained(TOKENIZER_NAME)
            _bi.base_bert_model = base_bert_model

            model = load_model(
                KERAS_PATH,
                custom_objects={"base_bert_model": base_bert_model},
                safe_mode=False
            )
            print(f"[OK] 載入 .keras 模型：{KERAS_PATH}")
            return model, tokenizer
        except Exception as e:
            print(f"[warn] .keras 讀取失敗：{e} → 改用 .h5 權重")

    # 備案：用 .h5 權重
    if not H5_PATH.exists():
        raise FileNotFoundError(f"找不到模型：\n  .keras: {KERAS_PATH}\n  .h5: {H5_PATH}")
    model = _build_model_lambda_wrapped_bert()
    # by_name=True 允許權重依名稱對齊
    model.load_weights(H5_PATH, by_name=True)
    print(f"[OK] 以重建結構載入 .h5 權重：{H5_PATH}")
    return model, tokenizer

def _predict_in_batches(model: Model, tokenizer, titles, X_num_arr) -> np.ndarray:
    """把所有樣本分批跑完，回傳 prob_up (N,)"""
    probs = []
    N = len(titles)
    for i in range(0, N, BATCH_SIZE):
        batch_titles = titles[i:i+BATCH_SIZE]
        enc = _tokenize_batch(batch_titles, tokenizer)
        x_num = X_num_arr[i:i+BATCH_SIZE].astype(np.float32)

        # 有些 Keras 版本比較挑剔 input 結構，包兩種都試一次
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

# ------------------ 主流程 ------------------
if __name__ == "__main__":
    print(f"[in] 讀取歷史資料：{SRC_CSV_PATH}")
    df = pd.read_csv(SRC_CSV_PATH)
    if "title_clean" not in df.columns or "label" not in df.columns or "sentiment_score_z" not in df.columns:
        raise ValueError("CSV 需包含 'title_clean', 'label', 'sentiment_score_z' 欄位（另外還需要日期欄位）")

    # 日期欄位
    dates = _to_date_col(df)
    df["_date_"] = dates.dt.normalize()

    # 補齊技術指標
    start_dt, end_dt = df["_date_"].min(), df["_date_"].max()
    print(f"[yfinance] 抓取 ^IXIC {start_dt.date()} ~ {end_dt.date()} 並計算技術指標…")
    ixic = _fetch_ixic_with_indicators(start_dt, end_dt)
    ixic = ixic.rename(columns={"Date": "_date_"})[["_date_", "MA_5", "MA_20", "Momentum_1d", "Momentum_5d"]]
    # --- 保證左右兩邊都有一個乾淨的 _date_ 欄位（無時區、日期對齊到日）---
def _to_naive_date_col(s):
    s = pd.to_datetime(s, errors="coerce")
    # 去掉時區，壓成日期(00:00)
    s = s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s
    return s.dt.normalize()

# 左邊 df：建立單一層級的 _date_ 欄位
date_col = None
for c in ["_date_", "Date", "market_date", "date"]:
    if c in df.columns:
        date_col = c
        break
if date_col is None:
    raise ValueError("歷史資料缺少日期欄位（需有 Date/market_date）")

df = df.copy()
df["_date_"] = _to_naive_date_col(df[date_col])

# 右邊 ixic：無論是 index 或欄位，統一生出 _date_
ixic = ixic.copy()
if "_date_" not in ixic.columns:
    if "Date" in ixic.columns:
        ixic["_date_"] = _to_naive_date_col(ixic["Date"])
    elif isinstance(ixic.index, pd.DatetimeIndex):
        ixic["_date_"] = _to_naive_date_col(ixic.index.to_series())
    else:
        raise ValueError("ixic 沒有 Date 欄或 DatetimeIndex，無法建立 _date_")

# 若 yfinance 產生了 MultiIndex 欄位，攤平成單層欄位名稱
if isinstance(ixic.columns, pd.MultiIndex):
    ixic.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        for col in ixic.columns.to_list()
    ]

# 只保留要 merge 的欄位
keep_cols = ["_date_", "MA_5", "MA_20", "Momentum_1d", "Momentum_5d"]
missing = [c for c in keep_cols if c not in ixic.columns]
if missing:
    raise ValueError(f"ixic 缺少必要技術欄位：{missing}")
ixic = ixic[keep_cols]

# 最後再做 merge（兩邊都是單一層欄位 _date_）
dfm = df.merge(ixic, on="_date_", how="left").ffill().fillna(0.0)

    # 合併
dfm = df.merge(ixic, on="_date_", how="left").ffill().fillna(0.0)

    # 準備輸入
titles = dfm["title_clean"].astype(str).fillna("").tolist()
if not set(NUMERIC_ORDER).issubset(dfm.columns):
    missing = [c for c in NUMERIC_ORDER if c not in dfm.columns]
    raise ValueError(f"缺少數值特徵欄位：{missing}")
X_num = dfm[NUMERIC_ORDER].values.astype(np.float32)

print("[model] 載入 tokenizer + 模型…")
model, tokenizer = _load_model_and_tokenizer()

print("[predict] 批量推論…")
prob_up = _predict_in_batches(model, tokenizer, titles, X_num)

    # 輸出
out = dfm.copy()
out["prob_up"] = prob_up
out_path = OUT_CSV_PATH
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False, encoding="utf-8")
print(f"[out] 已輸出：{out_path}")
print("[hint] 接著用 find_best_threshold.py 掃描最佳 threshold 就可以了。")
