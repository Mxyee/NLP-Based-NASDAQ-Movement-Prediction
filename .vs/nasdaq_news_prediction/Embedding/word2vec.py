import pandas as pd
import nltk
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
#1. Read data
df = pd.read_csv("Merged_News_and_NASDAQ_Data_Extended_With_Sentiment.csv")
df = df.dropna(subset=['title_clean', 'sentiment_score_z', 'label'])

print("DataFrame Columns:", df.columns.tolist())
print(f'Cleaned data size: {len(df)}')
# 2. Text preprocessing
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text_for_embedding(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords]
    return tokens

sentences = list(filter(None, df['title_clean'].apply(preprocess_text_for_embedding)))
if len(sentences) > 0:
    print(f'Number of sentences after preprocessing: {len(sentences)}')
else:
    print('No valid sentences found after preprocessing.')

# 3. Train Word2Vec model
# vector_size: 詞向量的維度 (例如 100 或 300)
# window: 訓練時考慮的上下文窗口大小
# min_count: 忽略出現頻率低於此值的單詞 (減少噪音詞，加快訓練)
# workers: 並行訓練的線程數
# sg: 0 代表 CBOW (Continuous Bag of Words)，1 代表 Skip-gram (Skip-gram 通常在大型語料庫上效果更好，但 CBOW 訓練更快)
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)

if len(sentences) > 0 :
    w2v_model.save("word2vec_model.model")
    print("Word2Vec model trained and saved successfully.")
else:
    print("No valid sentences to train Word2Vec model.")
# 4. Create document vectors
def sentence_vector(tokens, model, vector_size=100):
    vectors = []
    for word in tokens:
        if word in model.wv:
            vectors.append(model.wv[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

if 'w2v_model' in locals():
    df['title_embedding'] = df['title_clean'].apply(
        lambda x: sentence_vector(preprocess_text_for_embedding(x), w2v_model, vector_size=100) 
    )
    print("Document vectors created successfully.")
    if not df.empty and 'title_embedding' in df.columns:
        print(f"Sample document vector: {df['title_embedding'].iloc[0].shape}")
    else:
        print('No document vectors created.')
else:
    print("Word2Vec model not found. Cannot create document vectors.")

