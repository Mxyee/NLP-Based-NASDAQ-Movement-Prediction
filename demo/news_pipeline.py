"""
news_pipeline.py
----------------
Fetch finance news from RSS → Clean → Dedup → FinBERT scoring → Aggregate daily sentiment metrics.

Default sources:
- Reuters business news
- WSJ markets
- Yahoo Finance top news

Usage example:
--------------
from news_pipeline import run_news_pipeline, NEWYORK
import datetime as dt

cutoff = dt.datetime.now(NEWYORK).replace(hour=9, minute=29, second=59, microsecond=0)
metrics, detailed = run_news_pipeline(cutoff_et=cutoff, batch_size=8)
print(metrics)
print(detailed.head())
"""

import re, html, hashlib, datetime as dt
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import feedparser
from bs4 import BeautifulSoup
from dateutil.tz import gettz

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- Timezones ----
NEWYORK = gettz("America/New_York")
LONDON = gettz("Europe/London")

# ---- Regex patterns ----
_WS = re.compile(r"\s+")
_TICKER_PAREN = re.compile(r"\(([A-Z]{1,5})(?:\.[A-Z]{1,3})?\)")
_BRACKETED = re.compile(r"\[[^\]]*\]")
_URL_UTM = re.compile(r"(utm_[^=&]+=[^&]+&?)", re.I)
_MULTI_DOTS = re.compile(r"\.\.+")

def _strip_html(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html5lib")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(" ", strip=True)

def _norm_space(text: str) -> str:
    return _WS.sub(" ", text or "").strip()

def _canonicalize_url(url: Optional[str]) -> str:
    if not url:
        return ""
    url = re.sub(_URL_UTM, "", url)
    url = url.split("#")[0]
    return url.rstrip("?.!,;")

def _normalize_title(t: str) -> str:
    t = html.unescape(t or "")
    t = _strip_html(t)
    t = _norm_space(t)
    t = _BRACKETED.sub("", t)
    t = _TICKER_PAREN.sub("", t)
    t = _MULTI_DOTS.sub(".", t)
    return t.strip().lower()

def _hash_row(title_norm: str, url: str) -> str:
    return hashlib.md5(f"{title_norm}||{url}".encode("utf-8")).hexdigest()

# ---- 1) Fetch RSS ----
def fetch_news_rss(
    sources: Iterable[str],
    cutoff_et: dt.datetime,
    max_items_per_feed: int = 200,
) -> pd.DataFrame:
    rows = []
    for url in sources:
        try:
            feed = feedparser.parse(url)
        except Exception:
            continue
        cnt = 0
        for e in feed.entries:
            if cnt >= max_items_per_feed:
                break
            pub = None
            for key in ("published_parsed","updated_parsed"):
                if hasattr(e, key) and getattr(e, key):
                    pub = dt.datetime(*getattr(e, key)[:6], tzinfo=dt.timezone.utc).astimezone(NEWYORK)
                    break
            if pub is None or pub > cutoff_et:
                continue

            title = html.unescape(getattr(e, "title", "")).strip()
            summary = html.unescape(getattr(e, "summary", "")).strip()
            link = _canonicalize_url(getattr(e, "link", ""))
            raw_html = getattr(e, "summary", "") or getattr(e, "content", [{}])[0].get("value", "")

            text_raw = _norm_space(f"{title}. {_strip_html(summary)}")[:2000]
            title_norm = _normalize_title(title)

            if len(title_norm) < 5: 
                continue

            rows.append({
                "source": url,
                "url": link,
                "pub_time_et": pub,
                "title": title,
                "summary": summary,
                "raw_html": raw_html,
                "text_raw": text_raw,
                "title_norm": title_norm,
            })
            cnt += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["row_id"] = [_hash_row(t,u) for t,u in zip(df["title_norm"], df["url"])]
    df = df.sort_values("pub_time_et").drop_duplicates(subset=["row_id"], keep="last")
    return df.reset_index(drop=True)

# ---- 2) Clean ----
def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["source","url","pub_time_et","title","text_clean"])
    df = df.copy()
    df["title_norm"] = df["title"].map(_normalize_title)
    df = df.sort_values("pub_time_et").drop_duplicates(subset=["title_norm"], keep="last")
    cleaned = []
    for _, r in df.iterrows():
        title = _norm_space(_strip_html(html.unescape(r.get("title",""))))
        summary = _norm_space(_strip_html(html.unescape(r.get("summary",""))))
        txt = f"{title}. {summary}".strip()
        txt = _norm_space(txt).rstrip(".")
        cleaned.append(txt)
    df["text_clean"] = cleaned
    cols = ["source","url","pub_time_et","title","text_clean","title_norm"]
    return df.loc[:, cols].reset_index(drop=True)

# ---- 3) FinBERT sentiment ----
class SentimentScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert", device: Optional[str] = None, batch_size: int = 16):
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.labels = ["negative","neutral","positive"]

    @torch.no_grad()
    def score(self, texts: List[str]) -> Dict[str, np.ndarray]:
        if not texts:
            return {k: np.array([]) for k in ["prob_neg","prob_neu","prob_pos","label_idx"]}
        probs_all, label_idx = [], []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            enc = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
            label_idx.append(probs.argmax(axis=1))
        probs = np.vstack(probs_all)
        label_idx = np.concatenate(label_idx)
        return {
            "prob_neg": probs[:,0],
            "prob_neu": probs[:,1],
            "prob_pos": probs[:,2],
            "label_idx": label_idx,
        }

def score_news_sentiment(df: pd.DataFrame, scorer: Optional[SentimentScorer] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["source","url","pub_time_et","title","text_clean","prob_neg","prob_neu","prob_pos","label_idx"])
    if scorer is None:
        scorer = SentimentScorer()
    out = scorer.score(df["text_clean"].tolist())
    df = df.copy()
    for k,v in out.items():
        df[k] = v
    return df

# ---- 4) Aggregate daily sentiment ----
def aggregate_daily_sentiment(df: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    if df is None or df.empty:
        return {k: 0.0 for k in ["sent_mean_pos","sent_mean_neg","sent_pos_ratio","sent_score_w"]}, df
    latest = df["pub_time_et"].max()
    hrs = (latest - df["pub_time_et"]).dt.total_seconds() / 3600.0 + 1e-6
    w = 1.0 / (1.0 + hrs)
    df = df.copy()
    df["w"] = w
    sent_mean_pos = float(np.average(df["prob_pos"], weights=w))
    sent_mean_neg = float(np.average(df["prob_neg"], weights=w))
    sent_pos_ratio = float((df["prob_pos"] > 0.5).mean())
    sent_score_w = float(np.average(df["prob_pos"] - df["prob_neg"], weights=w))
    metrics = {
        "sent_mean_pos": sent_mean_pos,
        "sent_mean_neg": sent_mean_neg,
        "sent_pos_ratio": sent_pos_ratio,
        "sent_score_w": sent_score_w,
    }
    return metrics, df

# ---- 5) End-to-end ----
DEFAULT_SOURCES = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://finance.yahoo.com/news/rssindex",
]

def run_news_pipeline(
    sources: Iterable[str] = tuple(DEFAULT_SOURCES),
    cutoff_et: Optional[dt.datetime] = None,
    model_name: str = "ProsusAI/finbert",
    device: Optional[str] = None,
    batch_size: int = 16,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    if cutoff_et is None:
        now_et = dt.datetime.now(NEWYORK)
        cutoff_et = now_et.replace(hour=9, minute=29, second=59, microsecond=0)
    fetched = fetch_news_rss(sources, cutoff_et=cutoff_et)
    cleaned = clean_news(fetched)
    scorer = SentimentScorer(model_name=model_name, device=device, batch_size=batch_size)
    scored = score_news_sentiment(cleaned, scorer=scorer)
    metrics, detailed = aggregate_daily_sentiment(scored)
    return metrics, detailed
