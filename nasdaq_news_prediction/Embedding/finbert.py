import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Load data ---
df = pd.read_csv("Merged_News_and_NASDAQ_Data_Extended_With_Sentiment.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Close'] = df['Close'].astype(str).str.replace(',', '')
#Change Close to numeric, errors='coerce' will convert non-numeric values to NaN
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['title_clean', 'sentiment_score_z', 'label', 'Close'])
#Calculate moving average for the last 5 days
df['MA_5'] = df['Close'].rolling(window=5).mean()
#Calculate moving average for the last 20 days
df['MA_20'] = df['Close'].rolling(window=20).mean()
#Calculate momentum as the difference between the current close and the close 1 day ago
df['Momentum_1d'] = df['Close'].diff()
df['Momentum_5d'] = df['Close'].diff(periods=5)

df = df.dropna()

X_text = df['title_clean'].tolist()
X_numerical = df[['sentiment_score_z', 'MA_5', 'MA_20', 'Momentum_1d', 'Momentum_5d']].values.astype(np.float32)
y = df['label'].tolist()

# 2. Train-test split
X_train_text, X_test_text, \
X_train_numerical, X_test_numerical, \
y_train, y_test = train_test_split(
    X_text, X_numerical, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Initialize FinBERT Tokenizer and TFAutoModelForSequenceClassification
model_name = 'ProsusAI/finbert'
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the classification model and extract the base BERT model
base_bert_model_for_features = TFAutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, output_hidden_states=True, output_attentions=False
)
base_bert_model = base_bert_model_for_features.bert  # Extract base BERT

max_length = 128

def tokenize_texts(texts, tokenizer, max_length):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='tf'
    )

def create_multi_input_tf_dataset(texts, numerical_features, labels, tokenizer, batch_size=16, shuffle=True):
    encoding = tokenize_texts(texts, tokenizer, max_length)
    numerical_tf = tf.convert_to_tensor(numerical_features, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.float32)
    #ensure labels are tf.float32
    dataset = tf.data.Dataset.from_tensor_slices((
        ({"input_ids": encoding["input_ids"],
          "attention_mask": encoding["attention_mask"],
          "token_type_ids": encoding["token_type_ids"]},
         numerical_tf),
        labels_tf
    ))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(texts))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_dataset = create_multi_input_tf_dataset(X_train_text, X_train_numerical, y_train, tokenizer, batch_size=16, shuffle=True)
test_dataset = create_multi_input_tf_dataset(X_test_text, X_test_numerical, y_test, tokenizer, batch_size=16, shuffle=False)

# 6. Build multi-input Keras model
input_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
token_type_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name='token_type_ids')

# Use Lambda layer to call the base BERT model
bert_output_lambda = tf.keras.layers.Lambda(
    lambda x: base_bert_model(
        input_ids=x[0],
        attention_mask=x[1],
        token_type_ids=x[2]
    ).pooler_output,
    output_shape=(768,)
)([input_ids, attention_mask, token_type_ids])

numerical_input = tf.keras.Input(shape=(X_numerical.shape[1],), dtype=tf.float32, name='numerical_input')
concatenated_features = tf.keras.layers.concatenate([bert_output_lambda, numerical_input])
x = tf.keras.layers.Dense(128, activation='relu')(concatenated_features)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(
    inputs=[{'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}, numerical_input],
    outputs=output
)
for layer in base_bert_model.encoder.layer:
    layer.trainable = False  # Freeze all layers initially

num_unfrozen_layers = 3
for layer in base_bert_model.encoder.layer[-num_unfrozen_layers:]:
    layer.trainable = True
# Freeze the base BERT model

if base_bert_model.pooler is not None:
    base_bert_model.pooler.trainable = True  # Ensure pooler is trainable


# 7. Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# 8. Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.BinaryCrossentropy()
metrics = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
]
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

model.summary()

# 9. Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
print("\nStarting FinBERT + Sentiment multi-input model training...")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=20, #Early stopping will handle the actual number of epochs
    class_weight=class_weight_dict,
    callbacks=[early_stopping]
)

# 10. Evaluate model
print(f"\nFinBERT + Sentiment multi-input model training complete. Evaluating on test set...")
results = model.evaluate(test_dataset, return_dict=True)
print(f"Test Loss: {results['loss']:.4f}, Test Accuracy: {results['accuracy']:.4f}")
print(f"Test Precision: {results['precision']:.4f}, Test Recall: {results['recall']:.4f}")

# 11. Predict and report
test_text_encoding = tokenize_texts(X_test_text, tokenizer, max_length)
test_sentiment_tf = tf.expand_dims(tf.convert_to_tensor(X_test_numerical, dtype=tf.float32), axis=1)
test_inputs_for_predict = [
    {
        'input_ids': test_text_encoding['input_ids'],
        'attention_mask': test_text_encoding['attention_mask'],
        'token_type_ids': test_text_encoding['token_type_ids']
        
    },
    tf.convert_to_tensor(X_test_numerical, dtype=tf.float32)
]
y_pred_probs = model.predict(test_inputs_for_predict)
y_pred = (y_pred_probs.flatten() > 0.5).astype(int)
y_true = np.array(y_test)

print("\n=== FinBERT (partially unfrozen) + Sentiment multi-input model Classification Report ===")
print(classification_report(y_true, y_pred))
print("\n=== FinBERT (partially unfrozen) + Sentiment multi-input model Accuracy Score ===")
print(accuracy_score(y_true, y_pred))
print("\nPredicted label distribution:")
print(pd.Series(y_pred.flatten()).value_counts())

# ecdsa-SK-tp256-
# Using early stopping to prevent overfitting, so the epoch ended earlier than 20