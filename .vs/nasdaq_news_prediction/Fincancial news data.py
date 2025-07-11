import requests
import pandas as pd
from datetime import datetime, timedelta
import time

api_key = "c32488a50396c4ce46716b7edca4f66a"
query = "NASDAQ"
lang = "en"
max_results_per_call = 100

# 設定抓取時間區間
start_date = datetime.strptime("2024-03-21", "%Y-%m-%d")
end_date = datetime.strptime("2024-06-21", "%Y-%m-%d")
step = timedelta(days=7)

all_articles = []

# 抓每週新聞
current = start_date
while current < end_date:
    from_day = current.strftime("%Y-%m-%d")
    to_day = (current + step).strftime("%Y-%m-%d")
    print(f"📆 正在抓取：{from_day} ~ {to_day}")

    url = (
        f"https://gnews.io/api/v4/search?q={query}&lang={lang}"
        f"&from={from_day}&to={to_day}&max={max_results_per_call}&token={api_key}"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json().get("articles", [])
        print(f"    ✅ 抓到 {len(data)} 則")

        for article in data:
            all_articles.append({
                "date": article["publishedAt"][:10],
                "title": article["title"],
                "source": article["source"]["name"],
                "url": article["url"],
                "range": f"{from_day}~{to_day}"
            })

    except Exception as e:
        print(f"❌ 錯誤：{e}")

    current += step
    time.sleep(1)  # 避免太快觸發限制

# 建立 DataFrame 並存檔
df = pd.DataFrame(all_articles).drop_duplicates().sort_values(by="date")
df.to_csv("nasdaq_news_gnews.csv", index=False)
print(f"\n📊 完成！共收集 {len(df)} 筆新聞，已輸出 nasdaq_news_gnews.csv")
