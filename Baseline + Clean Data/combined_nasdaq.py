import pandas as pd

# --- 處理 nasdaq_2025.csv ---
print("正在處理 nasdaq_2025.csv...")
df_2025 = pd.read_csv("nasdaq_2025.csv")
# 確保 'Close' 和 'Open' 是數字類型，並處理逗號和 NaN
df_2025['Close'] = df_2025['Close'].astype(str).str.replace(',', '', regex=False)
df_2025['Open'] = df_2025['Open'].astype(str).str.replace(',', '', regex=False)
df_2025['Close'] = pd.to_numeric(df_2025['Close'], errors='coerce')
df_2025['Open'] = pd.to_numeric(df_2025['Open'], errors='coerce')
df_2025 = df_2025.dropna(subset=['Close', 'Open']) # 移除轉換失敗的行

# 加入 label 欄位 (Close > Open 為 1，否則為 0)
df_2025["label"] = (df_2025["Close"] > df_2025["Open"]).astype(int)
print("nasdaq_2025.csv 處理完成。")
print(df_2025.head())
print("-" * 30)


# --- 處理 nasdaq_2008-2016.csv ---
print("正在處理 nasdaq_2008-2016.csv...")
df_2008_2016 = pd.read_csv("nasdaq_2008-2016.csv")
# 確保 'Close' 和 'Open' 是數字類型，並處理逗號和 NaN
df_2008_2016['Close'] = df_2008_2016['Close'].astype(str).str.replace(',', '', regex=False)
df_2008_2016['Open'] = df_2008_2016['Open'].astype(str).str.replace(',', '', regex=False)
df_2008_2016['Close'] = pd.to_numeric(df_2008_2016['Close'], errors='coerce')
df_2008_2016['Open'] = pd.to_numeric(df_2008_2016['Open'], errors='coerce')
df_2008_2016 = df_2008_2016.dropna(subset=['Close', 'Open']) # 移除轉換失敗的行

# 加入 label 欄位 (Close > Open 為 1，否則為 0)
df_2008_2016["label"] = (df_2008_2016["Close"] > df_2008_2016["Open"]).astype(int)
print("nasdaq_2008-2016.csv 處理完成。")
print(df_2008_2016.head())
print("-" * 30)


# --- 合併兩個 DataFrame ---
print("正在合併處理後的資料...")
# 確保兩個 DataFrame 有相同的欄位順序，以便正確合併
# 這裡假設兩個 CSV 檔案的欄位結構相同
combined_df = pd.concat([df_2025, df_2008_2016], ignore_index=True)
print("資料合併完成。")
print(f"合併後的總行數: {len(combined_df)}")
print(combined_df.head())
print(combined_df.tail())
print("-" * 30)

# --- 儲存新的 CSV ---
# 將合併後的資料儲存到 nasdaq_labels.csv
# 這將包含所有年份的帶有 label 的 NASDAQ 數據
print("正在儲存合併後的資料到 nasdaq_labels.csv...")
combined_df.to_csv("nasdaq_labels.csv", index=False)
print("nasdaq_labels.csv 儲存成功！")
