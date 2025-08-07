import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack
from collections import Counter
import xgboost as xgb
from scipy.stats import uniform, randint # For defining distributions in RandomizedSearchCV

# --- Download NLTK resources (only needed on first run) ---
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- 1. Load Data ---
# Make sure the CSV contains 'sentiment_score_z' column
df = pd.read_csv("data/processed/Merged_News_and_NASDAQ_Data_Extended_With_Sentiment.csv")

# Verify that 'title_clean' and 'sentiment_score_z' exist
print("DataFrame Columns:", df.columns.tolist())

# Remove rows with missing values in key columns
df = df.dropna(subset=['title_clean', 'sentiment_score_z', 'label'])
print(f"Cleaned data size: {len(df)}")

# --- 2. Text Preprocessing (for TF-IDF) ---
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

# --- 3. TF-IDF Feature Extraction ---
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df["title_cleaned_for_tfidf"])

# --- 4. Sentiment Score Feature ---
X_sentiment_z = df['sentiment_score_z'].values.reshape(-1, 1)

# --- 5. Combine TF-IDF and Sentiment Features ---
X_combined = hstack([X_tfidf, X_sentiment_z])

# --- 6. Define Target Variable ---
y = df['label']

# --- 7. Examine Class Distribution ---
print("\nOriginal label distribution:")
print(y.value_counts())
print(f"Majority class ratio: {y.value_counts().max() / len(y):.2f}")

# --- 8. Split into Training and Test Sets ---
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# --- 9. Set up XGBoost Model and Random Search Parameters ---
base_xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42
)

# Calculate scale_pos_weight = (# negative samples / # positive samples)
pos_weight_value = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"Calculated scale_pos_weight: {pos_weight_value:.4f}")

# Define hyperparameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 500),
    'learning_rate': uniform(0.01, 0.3),
    'max_depth': randint(3, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'scale_pos_weight': [pos_weight_value]
}

# Create RandomizedSearchCV object
print("\nRunning RandomizedSearchCV for XGBoost hyperparameter tuning...")
random_search = RandomizedSearchCV(
    estimator=base_xgb_model,
    param_distributions=param_dist,
    n_iter=100,
    scoring='f1_macro',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Optional: manually defined XGBoost model with GPU support
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    scale_pos_weight=pos_weight_value,
    tree_method='gpu_hist'  # GPU acceleration if available
)

# Perform hyperparameter search
random_search.fit(X_train, y_train)

# --- 10. Evaluation ---
print("\n=== RandomizedSearchCV Tuning Results ===")
print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation F1-macro (on training set):", random_search.best_score_)

# Evaluate best model on test set
best_xgb_model = random_search.best_estimator_
y_pred_best = best_xgb_model.predict(X_test)

print("\n=== Classification Report of Best XGBoost Model ===")
print(classification_report(y_test, y_pred_best))

print("\n=== Accuracy Score of Best XGBoost Model ===")
print(accuracy_score(y_test, y_pred_best))

print("\nPredicted label distribution:")
print(pd.Series(y_pred_best).value_counts())