
# demo/run_demo_gnews.py
import re, html, string, sys
from urllib.parse import quote
from datetime import datetime, timedelta, timezone

import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler

from _paths import RAW, PROC

# -------- Parameters --------
KEYWORDS = ["Nasdaq", "tech stocks", "stock market"]
WINDOW_DAYS = 3  # lookback window for news

# -------- Fetch Google News RSS --------
def fetch_news():
    today = datetime.now(timezone.utc).date()
    after = (today - timedelta(days=WINDOW_DAYS)).isoformat()
    before = (today + timedelta(days=1)).isoformat()  # include today

    rss_urls = [
        "https://news.google.com/rss/search?q="
        + quote(f"{k} after:{after} before:{before}")
        + "&hl=en-US&gl=US&ceid=US:en"
        for k in KEYWORDS
    ]

    rows = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for e in feed.entries:
            rows.append({
                "date": e.published if "published" in e else "",
                "title": e.title,
                "source": e.source.title if "source" in e else ""
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["date","title"])
    df.sort_values("date", inplace=True)
    out = RAW / "google_news_rss.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[A1] Saved news RSS → {out} ({len(df)} rows)")
    return df

# -------- Clean & dedup titles --------
BRACKETED = re.compile(r"\[[^\]]*\]")
TICKER_PAREN = re.compile(r"\(([A-Z]{1,5})(?:\.[A-Z]{1,3})?\)")
MULTI_DOTS = re.compile(r"\.\.+")
WS = re.compile(r"\s+")

def strip_html(t: str) -> str:
    if not t: return ""
    soup = BeautifulSoup(t, "html5lib")
    for tag in soup(["script","style","noscript"]): tag.extract()
    return soup.get_text(" ", strip=True)

def normalize_title(t: str) -> str:
    t = html.unescape(t or "")
    t = strip_html(t)
    t = BRACKETED.sub("", t)
    t = TICKER_PAREN.sub("", t)
    t = MULTI_DOTS.sub(".", t)
    t = WS.sub(" ", t).strip().lower()
    return t

def clean_news(df):
    if df.empty:
        out = PROC / "all_news_cleaned_sorted.csv"
        pd.DataFrame(columns=["date","title","title_clean","source"]).to_csv(out, index=False)
        print(f"[A2] No news; wrote empty {out}")
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    df["title_clean"] = df["title"].astype(str).map(normalize_title)
    df = df.sort_values("date").drop_duplicates(subset="title_clean", keep="last")
    df = df[["date","title","title_clean","source"]].sort_values("date")
    out = PROC / "all_news_cleaned_sorted.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[A2] Saved cleaned news → {out} ({len(df)} rows)")
    return df

# -------- Download Nasdaq (^IXIC) and label --------
def fetch_ixic():
    import yfinance as yf
    ixic = yf.download("^IXIC", period="20y", interval="1d", auto_adjust=True, progress=False).dropna()
    ixic = ixic.rename_axis("Date").reset_index()
    ixic["label"] = (ixic["Close"] > ixic["Open"]).astype(int)
    out = PROC / "nasdaq_labels.csv"
    ixic.to_csv(out, index=False)
    print(f"[A3] Saved IXIC → {out} ({len(ixic)} rows)")
    return ixic

# -------- Merge news to next trading day --------
def merge_news_market(news, mkt):
    news = news.sort_values("date")
    mkt  = mkt.sort_values("Date")
    merged = pd.merge_asof(news, mkt, left_on="date", right_on="Date", direction="forward")
    final = merged[["Date","title_clean","title","source","Open","Close","label"]].dropna(subset=["Date"])
    out = PROC / "news_market_merged.csv"
    final.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[A4] Saved merged → {out} ({len(final)} rows)")
    return final

# -------- Dictionary sentiment with MinMax; fallback to FinBERT --------
def load_lm_dicts():
    import os
    import pandas as pd
    cands = [
        ("LM_positive.csv","LM_negative.csv"),
        (RAW/"LM_positive.csv", RAW/"LM_negative.csv"),
    ]
    for p_pos, p_neg in cands:
        p_pos = str(p_pos); p_neg = str(p_neg)
        if os.path.exists(p_pos) and os.path.exists(p_neg):
            pos = set(pd.read_csv(p_pos, header=None)[0].astype(str).str.upper())
            neg = set(pd.read_csv(p_neg, header=None)[0].astype(str).str.upper())
            return pos, neg
    return None, None

TOKEN = re.compile(r"[A-Za-z]+")
def toks(s: str):
    s = (s or "").upper().translate(str.maketrans("","", string.punctuation))
    return TOKEN.findall(s)

def score_lm(title_clean: str, pos_set, neg_set) -> int:
    ts = toks(title_clean)
    pos = sum(1 for t in ts if t in pos_set)
    neg = sum(1 for t in ts if t in neg_set)
    return pos - neg

def finbert_scores(texts):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch, numpy as np
    model = "ProsusAI/finbert"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = texts[i:i+16]
            enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs_all.append(probs)
    import numpy as np
    return np.vstack(probs_all) if probs_all else None  # [neg,neu,pos]

def sentiment(merged):
    pos_set, neg_set = load_lm_dicts()
    if pos_set is not None:
        print("[A5] Using Loughran–McDonald dictionaries")
        merged["sentiment_score"] = merged["title_clean"].astype(str).apply(lambda s: score_lm(s, pos_set, neg_set))
    else:
        print("[A5] LM dict not found → FinBERT fallback")
        texts = merged["title"].fillna(merged["title_clean"]).astype(str).tolist()
        probs = finbert_scores(texts)
        if probs is None or len(probs) != len(merged):
            raise SystemExit("FinBERT scoring failed.")
        import numpy as np
        merged["prob_neg"], merged["prob_neu"], merged["prob_pos"] = probs[:,0], probs[:,1], probs[:,2]
        merged["sentiment_score"] = merged["prob_pos"] - merged["prob_neg"]

    scaler = MinMaxScaler()
    merged["sentiment_score_norm"] = scaler.fit_transform(merged[["sentiment_score"]]).round(4)

    out = PROC / "Merged_News_and_NASDAQ_Data_Extended_With_Sentiment.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[A6] Saved final → {out} ({len(merged)} rows)")

    # Small console summary
    print("\n[A6] Preview:")
    print(merged.tail(5)[["Date","title_clean","sentiment_score","sentiment_score_norm"]])

def main():
    news = fetch_news()
    news_clean = clean_news(news)
    ixic = fetch_ixic()
    merged = merge_news_market(news_clean, ixic)
    sentiment(merged)
    print("\n[Route A] Done.")

if __name__ == "__main__":
    main()
