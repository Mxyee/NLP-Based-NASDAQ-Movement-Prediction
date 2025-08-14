import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path 

from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SCALER_PATH = MODEL_DIR / "scaler.pkl"

# 1) 讀你訓練用的合併檔（如果有）
#    檔名請換成你實際的訓練 CSV；沒有就先用空的
TRAIN_CSV = ROOT.parent / "data" / "processed" / "Merged_News_and_NASDAQ_Data_Extended_With_Sentiment.csv"
sent_z = None
if TRAIN_CSV.exists():
    df_train = pd.read_csv(TRAIN_CSV)
    if "sentiment_score_z" in df_train.columns:
        sent_z = df_train["sentiment_score_z"].astype(float).to_numpy().reshape(-1, 1)

# 2) 用 yfinance 補齊技術指標的歷史尺度（近兩年）
px = yf.download("^IXIC", period="2y", interval="1d", auto_adjust=False, progress=False)
px = px.dropna().copy()
px["MA_5"] = px["Close"].rolling(5).mean()
px["MA_20"] = px["Close"].rolling(20).mean()
px["Momentum_1d"] = px["Close"].diff(1)
px["Momentum_5d"] = px["Close"].diff(5)
px = px.dropna()

tech = px[["MA_5","MA_20","Momentum_1d","Momentum_5d"]].to_numpy()

# 3) 組合成與模型一致的 5 維（sent_z + 4 個技術指標）
#    若沒有訓練 CSV，就用 sentiment=0 當填補，重點是讓 scaler 有數值範圍
if sent_z is None or len(sent_z) < 30:
    sent_z = np.zeros((tech.shape[0], 1), dtype=np.float32)

min_len = min(len(sent_z), len(tech))
X_hist = np.hstack([sent_z[:min_len], tech[:min_len]]).astype(np.float32)

scaler = StandardScaler().fit(X_hist)

# 4) 保存
import joblib
joblib.dump(scaler, SCALER_PATH)
print(f"[OK] scaler saved → {SCALER_PATH}")