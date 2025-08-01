import feedparser
import pandas as pd
from urllib.parse import quote

# Set Google News RSS Keywords
keywords = ["Nasdaq", "tech stocks", "stock market"]

# Building RSS URL
rss_urls = [
    "https://news.google.com/rss/search?q=" + quote(f"{k} after:2025-03-01 before:2025-06-22") +
    "&hl=en-US&gl=US&ceid=US:en"
    for k in keywords
]

records = []
for url in rss_urls:
    feed = feedparser.parse(url)
    for entry in feed.entries:
        records.append({
            "date": entry.published if 'published' in entry else "",
            "title": entry.title,
            "source": entry.source.title if 'source' in entry else ""
        })

# Sort out DataFrame and remove duplicates
df = pd.DataFrame(records).drop_duplicates(subset=["date", "title"])
df.sort_values("date", inplace=True)
df.to_csv("google_news_rss.csv", index=False, encoding="utf-8-sig")
print("Finished，Total caught", len(df), "news data")
