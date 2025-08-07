import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack  # For combining sparse and dense features
from collections import Counter  # To inspect label distribution
from sklearn.preprocessing import StandardScaler  # Assuming sentiment_score_z is scaled with this

# Ensure NLTK resources are downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- 1. Load data ---
# Make sure the CSV file includes the 'sentiment_score_z' column
df = pd.read_csv("data/processed/Merged_News_and_NASDAQ_Data_Extended_With_Sentiment.csv")

# Check if 'title_clean' and 'sentiment_score_z' columns exist
print("DataFrame Columns:", df.columns.tolist())

# Ensure no missing values in the key features and target label
df = df.dropna(subset=['title_clean', 'sentiment_score_z', 'label'])
print(f"Number of samples after data cleaning: {len(df)}")

# --- 2. Text preprocessing (for TF-IDF) ---
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

# --- 3. TF-IDF feature extraction ---
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features
X_tfidf = tfidf_vectorizer.fit_transform(df["title_cleaned_for_tfidf"])

# --- 4. Prepare sentiment score feature ---
# Directly use sentiment_score_z assuming it has already been standardized
# If not, add the StandardScaler step here like:
# scaler_z = StandardScaler()
# df['sentiment_score_z'] = scaler_z.fit_transform(df[['sentiment_score']])
X_sentiment_z = df['sentiment_score_z'].values.reshape(-1, 1)  # Convert to 2D array

# --- 5. Combine all features ---
# Use hstack to combine sparse TF-IDF features with dense scaled sentiment score feature
X_combined = hstack([X_tfidf, X_sentiment_z])

# --- 6. Define target variable ---
y = df['label']

# --- 7. Check class distribution (very important!) ---
print("\nOriginal label distribution:")
print(y.value_counts())
print(f"Majority class ratio: {y.value_counts().max() / len(y):.2f}")

# --- 8. Split training and test sets ---
# Use train_test_split for quick baseline testing
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# --- 9. Build and train Logistic Regression model ---
# Handle class imbalance using class_weight='balanced'
print("\nTraining Logistic Regression model with class_weight='balanced'...")
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# --- 10. Prediction and evaluation ---
y_pred = model.predict(X_test)

print("\n=== Logistic Regression (TF-IDF + Scaled Sentiment) Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Accuracy Score ===")
print(accuracy_score(y_test, y_pred))

# Additional check: Distribution of predicted labels
print("\nPredicted label distribution on the test set:")
print(pd.Series(y_pred).value_counts())
