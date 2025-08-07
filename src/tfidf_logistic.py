import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack # 導入 hstack 用於合併稀疏和稠密特徵
from collections import Counter # 用於檢查類別分佈
from sklearn.preprocessing import StandardScaler # 假設 sentiment_score_z 是用這個處理的

# 確保 NLTK 資源已下載
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- 1. 讀取數據 ---
# 確保讀取的是包含 'sentiment_score_z' 欄位的 CSV 文件
df = pd.read_csv("Merged_News_and_NASDAQ_Data_Extended_With_Sentiment.csv")

# 檢查欄位，確認 'title_clean' 和 'sentiment_score_z' 存在
print("DataFrame Columns:", df.columns.tolist())

# 確保沒有缺失值，特別是對於我們將要使用的特徵和標籤
df = df.dropna(subset=['title_clean', 'sentiment_score_z', 'label'])
print(f"數據清洗後剩餘樣本數: {len(df)}")

# --- 2. 文本預處理 (用於 TF-IDF) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text_for_tfidf(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

df["title_cleaned_for_tfidf"] = df["title_clean"].apply(preprocess_text_for_tfidf)

# --- 3. TF-IDF 特徵提取 ---
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # 可以調整 max_features
X_tfidf = tfidf_vectorizer.fit_transform(df["title_cleaned_for_tfidf"])

# --- 4. 準備情感分數特徵 ---
# 這裡直接使用 sentiment_score_z，假設它已經被 StandardScaler 處理過
# 如果沒有，你需要在這裡添加 StandardScaler 的步驟
# 例如：
# scaler_z = StandardScaler()
# df['sentiment_score_z'] = scaler_z.fit_transform(df[['sentiment_score']])
X_sentiment_z = df['sentiment_score_z'].values.reshape(-1, 1) # 轉換為2D數組

# --- 5. 合併所有特徵 ---
# 使用 hstack 將稀疏的 TF-IDF 特徵和稠密的縮放情感分數特徵水平堆疊
X_combined = hstack([X_tfidf, X_sentiment_z])

# --- 6. 定義目標變數 ---
y = df['label']

# --- 7. 檢查類別分佈 (非常重要！) ---
print("\n原始標籤分佈:")
print(y.value_counts())
print(f"多數類佔比: {y.value_counts().max() / len(y):.2f}")

# --- 8. 劃分訓練集和測試集 ---
# 使用 train_test_split 作為基線的快速測試。
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# --- 9. 建立並訓練 Logistic Regression 模型 ---
# 解決類別不平衡問題：使用 class_weight='balanced'
print("\n訓練帶有 class_weight='balanced' 的 Logistic Regression 模型...")
model = LogisticRegression(max_iter=1000, class_weight='balanced') # 加入 class_weight='balanced'
model.fit(X_train, y_train)

# --- 10. 預測與評估 ---
y_pred = model.predict(X_test)

print("\n=== Logistic Regression (TF-IDF + Scaled Sentiment) Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Accuracy Score ===")
print(accuracy_score(y_test, y_pred))

# 額外檢查：模型預測的類別分佈
print("\n模型預測的測試集標籤分佈:")
print(pd.Series(y_pred).value_counts())