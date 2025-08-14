# demo/sentiment_lm.py
import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt", quiet=True)

def load_sentiment_data(filepath):
    df = pd.read_csv(filepath)
    pos_words = set(df[df["Positive"] != 0]["Word"].str.upper())
    neg_words = set(df[df["Negative"] != 0]["Word"].str.upper())
    return pos_words, neg_words

def preprocess_for_sentiments(text):
    if pd.isna(text):
        return []
    text = text.upper()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in word_tokenize(text) if w.isalpha()]
    
    tokens = [re.sub(r"(S|ES)$", "", w) if len(w) > 3 else w for w in tokens]
    return tokens

def add_lm_sentiment(df, text_col, lm_csv_path, time_col=None, use_negation=False, neg_window=3, zscore_mode=None):
    pos_words, neg_words = load_sentiment_data(lm_csv_path)

    lm_scores = []
    lm_hits = []
    hit_ratios = []

    for txt in df[text_col]:
        tokens = preprocess_for_sentiments(txt)
        hits = 0
        score = 0
        i = 0
        while i < len(tokens):
            token = tokens[i]
            negated = False
            if use_negation and token in {"NOT", "NO", "NEVER"}:
                negated = True
                window_tokens = tokens[i+1:i+1+neg_window]
                for wt in window_tokens:
                    if wt in pos_words:
                        score -= 1
                        hits += 1
                    elif wt in neg_words:
                        score += 1
                        hits += 1
                i += neg_window
            else:
                if token in pos_words:
                    score += 1
                    hits += 1
                elif token in neg_words:
                    score -= 1
                    hits += 1
            i += 1
        lm_scores.append(score)
        lm_hits.append(hits)
        hit_ratios.append(hits / len(tokens) if tokens else 0.0)

    df = df.copy()
    df["lm_score"] = lm_scores
    df["lm_hits"] = lm_hits
    df["lm_hit_ratio"] = hit_ratios

    if zscore_mode == "within_day" and time_col and time_col in df.columns:
        df["date_only"] = pd.to_datetime(df[time_col]).dt.date
        df["lm_score_z"] = df.groupby("date_only")["lm_score"].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) != 0 else 1)
        )
    else:
        df["lm_score_z"] = (df["lm_score"] - df["lm_score"].mean()) / (
            df["lm_score"].std(ddof=0) if df["lm_score"].std(ddof=0) != 0 else 1
        )

    return df
