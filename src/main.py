import pandas as pd

# 嘗試讀取 nasdaq.csv（與 main.py 同一層）
df = pd.read_csv("nasdaq_2025.csv")

# 加入 label 欄位（Close > Open 為 1，否則為 0）
df["label"] = (df["Close"] > df["Open"]).astype(int)

# 儲存新的 CSV
df.to_csv("nasdaq_labels.csv", index=False)

# 顯示前幾筆資料驗證
print(df.head())
