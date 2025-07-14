import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the resource for the first time use
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Read the dataset
df = pd.read_csv("Merged_News_and_NASDAQ_Data.csv")

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Processing function: cleaning text
def preprocess(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return ' '.join(tokens)

# Added post-processing column
print(df.columns)
df["title_cleaned"] = df["Title"].apply(preprocess)

# TF-IDF feature conversion
vectorizer = TfidfVectorizer(max_features=1000)  # Adjustable maximum number of features
X = vectorizer.fit_transform(df["title_cleaned"])

# Corresponding label field
y = df["label"]  

# View Results
print("TF-IDF feature matrix shape：", X.shape)
print("Label distribution：")
print(y.value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
X_tfidf = vectorizer.fit_transform(df["title_cleaned"])
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, df["label"], test_size=0.2, random_state=42)

# Build and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Forecasts and reports
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

