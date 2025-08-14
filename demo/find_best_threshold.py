import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. load the historical predictions CSV
csv_path = "C:/Users/Harry/Desktop/nasdaq_news_prediction/demo/data/processed/train_predictions.csv"
df = pd.read_csv(csv_path)

# ensure the necessary columns are present
assert 'prob_up' in df.columns, "CSV lost prob_up "
assert 'label' in df.columns, "CSV lost label "

best_acc = 0
best_thresh_acc = 0
best_f1 = 0
best_thresh_f1 = 0

# 2. scan different threshold
for thresh in np.arange(0.0, 1.01, 0.01):
    preds = (df['prob_up'] >= thresh).astype(int)
    acc = accuracy_score(df['label'], preds)
    f1 = f1_score(df['label'], preds)

    if acc > best_acc:
        best_acc = acc
        best_thresh_acc = thresh

    if f1 > best_f1:
        best_f1 = f1
        best_thresh_f1 = thresh

# 3. show the results
print(f"Best Accuracy Threshold: {best_thresh_acc:.2f} | Accuracy={best_acc:.4f}")
print(f"Best F1-score Threshold: {best_thresh_f1:.2f} | F1-score={best_f1:.4f}")
