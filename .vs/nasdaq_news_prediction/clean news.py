import pandas as pd

# 輸入與輸出檔案名稱
input_file = "nasdaq_news_gnews.csv"
output_file = "nasdaq_news_cleaned.csv"

# 讀入原始資料
try:
    df = pd.read_csv(input_file)
    print(f"📥 原始筆數：{len(df)}")

    # 保留原始版本備查
    df_raw = df.copy()

    # 清除無效標題（NaN 或純空白）
    df = df[df['title'].astype(str).str.strip() != ""]
    df = df[df['title'].notnull()]

    print(f"🧹 移除空白或缺失標題後：{len(df)}")

    # 軟性去重：只用 title 做唯一性（比較保守）
    df_cleaned = df.drop_duplicates(subset=['title'])

    print(f"📊 去重後剩下：{len(df_cleaned)}（移除 {len(df) - len(df_cleaned)} 筆重複標題）")

    # 日期欄轉換與排序
    df_cleaned['date'] = pd.to_datetime(df_cleaned['date'], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['date'])
    df_cleaned = df_cleaned.sort_values(by='date')

    # 儲存結果
    df_cleaned.to_csv(output_file, index=False)
    print(f"✅ 輸出結果：{output_file}，共 {len(df_cleaned)} 筆")

except Exception as e:
    print("❌ 發生錯誤：", e)