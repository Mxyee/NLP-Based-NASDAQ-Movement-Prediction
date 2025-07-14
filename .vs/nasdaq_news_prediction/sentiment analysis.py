# sentiment analysis
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize

def load_sentiment_data(filepath):
    df = pd.read_csv(filepath)
    positive_words = set(df[df['Positive']!=0]['Word'].str.upper())
    negative_words = set(df[df['Negative']!=0]['Word'].str.upper())
    return positive_words, negative_words
master_dict_path = 'sentiments_lexicon/Loughran-McDonald_MasterDictionary_1993-2024.csv'

positive_words, negative_words = load_sentiment_data(master_dict_path)
print(f"Positive words count: {len(positive_words)}")
print(f"Negative words count: {len(negative_words)}")
# --- 2. 為情感分析準備的文本預處理函數 ---
# 這個函數與你 TF-IDF 的 preprocess 不同，它不需要詞形還原和停用詞移除
# 只需要轉大寫和移除標點，以便與情感詞典匹配
def preprocess_for_sentiments(text):
    if pd.isna(text):
        return ""
    text = text.upper()
    text = text.translate(str.maketrans('','', string.punctuation))
    tokens = word_tokenize(text)
    return [word for word in tokens if word.isalpha()]

# --- 3. 計算情感分數 ---
def calculate_sentiment_score(text, pos_words, neg_words):
    tokens = preprocess_for_sentiments(text)
    positive_count = 0
    negative_count = 0

    for word in tokens:
        if word in pos_words:
            positive_count += 1
        elif word in neg_words:
            negative_count += 1
    sentiment_score = positive_count - negative_count
    return sentiment_score

test_text_pos = "Apple's stock gains strongly after positive earnings report."
test_text_neg = "Market plunged due to financial crisis and recession fears."
print(f"'{test_text_pos}' 情感分數: {calculate_sentiment_score(test_text_pos, positive_words, negative_words)}")
print(f"'{test_text_neg}' 情感分數: {calculate_sentiment_score(test_text_neg, positive_words, negative_words)}")