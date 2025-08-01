import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from collections import Counter
import xgboost as xgb
from scipy.stats import uniform, randint # 導入分佈函數，用於 RandomizedSearchCV

# --- 下載 NLTK 資源 (只在第一次運行時需要) ---
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
X_sentiment_z = df['sentiment_score_z'].values.reshape(-1, 1)

# --- 5. 合併所有特徵 ---
X_combined = hstack([X_tfidf, X_sentiment_z])

# --- 6. 定義目標變數 ---
y = df['label']

# --- 7. 檢查類別分佈 ---
print("\n原始標籤分佈:")
print(y.value_counts())
print(f"多數類佔比: {y.value_counts().max() / len(y):.2f}")

# --- 8. 劃分訓練集和測試集 ---
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# --- 9. 設定 XGBoost 模型和參數分佈用於 RandomizedSearchCV ---

# 基礎 XGBoost 分類器實例
# 注意：這裡的 'use_label_encoder=False' 在較新版本可能不再需要
# 但為了兼容性，保留它通常是安全的，警告無害。
# 如果你在 RandomSearch 外部直接設置 scale_pos_weight，這裡就不用在 param_distributions 裡面再放了
base_xgb_model = xgb.XGBClassifier(objective='binary:logistic',
                                   eval_metric='logloss',
                                   use_label_encoder=False,
                                   random_state=42)

# 計算 scale_pos_weight，通常是 (負樣本數 / 正樣本數)
# 根據你的 y_train.value_counts()，0 是 1023 (負類)，1 是 1224 (正類)
# 所以 scale_pos_weight = 1023 / 1224
pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"計算得出的 scale_pos_weight: {pos_weight_value:.4f}")

# 定義要搜索的超參數分佈
# 這裡提供了一些建議的範圍，你可以根據你的數據和計算資源進行調整
param_dist = {
    'n_estimators': randint(100, 500),         # 樹的數量，在 100 到 500 之間隨機
    'learning_rate': uniform(0.01, 0.3),       # 學習率，在 0.01 到 0.3 之間隨機
    'max_depth': randint(3, 10),               # 樹的最大深度，在 3 到 10 之間隨機
    'subsample': uniform(0.6, 0.4),            # 訓練樣本採樣比例 (0.6 到 1.0)
    'colsample_bytree': uniform(0.6, 0.4),     # 訓練特徵採樣比例 (0.6 到 1.0)
    'gamma': uniform(0, 0.5),                  # 葉節點分裂所需的最小損失減少量 (0 到 0.5)
    # 如果你希望 scale_pos_weight 也參與隨機搜索，可以這樣設置一個範圍
    # 但通常它會基於數據集的固有不平衡比例固定下來，所以這裡作為參考。
    'scale_pos_weight': [pos_weight_value] # 將計算出的值作為一個單一值的列表
    # 'scale_pos_weight': uniform(0.7, 1.2) # 或者像這樣給一個範圍
}

# 創建 RandomizedSearchCV 對象
# n_iter: 隨機採樣的參數組合數量，越大越好，但計算時間越長
# scoring: 評估指標。'f1_macro' 適用於類別平衡或略微不平衡的情況，它會平均兩個類別的 F1-score
# cv: 交叉驗證折數
# verbose: 打印詳細進度
# n_jobs: 並行運行任務的數量 (-1 表示使用所有可用核心)
print("\n開始運行 RandomizedSearchCV 進行 XGBoost 超參數調優...")
random_search = RandomizedSearchCV(
    estimator=base_xgb_model,
    param_distributions=param_dist,
    n_iter=100, # 嘗試 100 種不同的參數組合，可以根據你的計算能力調整
    scoring='f1_macro',
    cv=5, # 5 折交叉驗證
    verbose=2, # 打印更詳細的進度
    random_state=42,
    n_jobs=-1
)
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    scale_pos_weight=pos_weight_value,
    # 添加這一行來啟用 GPU 加速
    tree_method='gpu_hist' # 對於大多數 NVIDIA GPU 這是最佳選擇
    # 或者 'cuda' (舊版本)
)
# 執行搜索
random_search.fit(X_train, y_train)

# --- 10. 獲取最佳模型和評估 ---
print("\n=== RandomizedSearchCV 調優結果 ===")
print("最佳參數:", random_search.best_params_)
print("最佳交叉驗證 F1-macro (在訓練集上):", random_search.best_score_)

# 使用最佳模型在測試集上進行最終評估
best_xgb_model = random_search.best_estimator_
y_pred_best = best_xgb_model.predict(X_test)

print("\n=== 最佳 XGBoost 模型 (經 RandomizedSearchCV 調優) Classification Report ===")
print(classification_report(y_test, y_pred_best))

print("\n=== 最佳 XGBoost 模型 (經 RandomizedSearchCV 調優) Accuracy Score ===")
print(accuracy_score(y_test, y_pred_best))

print("\n最佳 XGBoost 模型預測的測試集標籤分佈:")
print(pd.Series(y_pred_best).value_counts())