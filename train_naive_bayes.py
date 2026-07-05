"""
Naive Bayes (MultinomialNB) model eğitimi.

Mevcut TF-IDF vectorizer ve scaler'ı kullanarak MultinomialNB modelini eğitir.
MultinomialNB negatif değer kabul etmediği için, combined matristeki negatif
değerler sıfıra çekilir (MaxAbsScaler alternatifi yerine basit clipping).

Üretilenler:
  models/nb_model.pkl
  results/final_evaluation.csv  (NB satırı eklenir/güncellenir)
  results/model_comparison.csv  (NB satırı eklenir/güncellenir)
"""
import os
import sys
import time
import csv
import io

# Windows konsolunda Türkçe karakter desteği
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
import scipy.sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

ROOT = os.path.dirname(os.path.abspath(__file__))
NUMERIC = ["review_length", "word_count", "exclamation_count", "question_count",
           "avg_word_length", "uppercase_ratio", "sentiment_polarity",
           "sentiment_subjectivity"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall":    recall_score(y_true, y_pred, average="macro"),
        "F1-Macro":  f1_score(y_true, y_pred, average="macro"),
    }


# --------------------------------------------------------------- veri yükle
log("Veri yükleniyor...")
df = pd.read_csv(os.path.join(ROOT, "data", "reviews_preprocessed.csv"),
                 usecols=["summary", "cleaned_text", "label"] + NUMERIC)
df = df.dropna(subset=["cleaned_text"]).reset_index(drop=True)
log(f"Satır: {len(df):,}")

idx_train = np.load(os.path.join(ROOT, "features", "train_idx.npy"))
idx_val   = np.load(os.path.join(ROOT, "features", "val_idx.npy"))
idx_test  = np.load(os.path.join(ROOT, "features", "test_idx.npy"))
y = df["label"].values

# --------------------------------------------------------------- özellik matrisi
log("TF-IDF vectorizer ve scaler yükleniyor...")
tfidf_vectorizer = joblib.load(os.path.join(ROOT, "models", "tfidf_vectorizer.pkl"))
scaler = joblib.load(os.path.join(ROOT, "models", "scaler.pkl"))

# Summary'li combined text (build_enhancements.py ile aynı)
sys.path.insert(0, os.path.join(ROOT, "webapp"))
from pipeline import clean_text  # noqa: E402

summary_clean = df["summary"].fillna("").map(clean_text)
text_clean = df["cleaned_text"].astype(str)
combined = (summary_clean + " " + text_clean).str.strip()

log("TF-IDF + numeric dönüşümü...")
X_tfidf = tfidf_vectorizer.transform(combined)
X_num = scaler.transform(df[NUMERIC])

# MultinomialNB negatif değer kabul etmez.
# StandardScaler çıktısında negatif değerler olabilir; bunları clip ediyoruz.
X_num_clipped = np.clip(X_num, 0, None)
X_combined = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(X_num_clipped)]).tocsr()

X_train = X_combined[idx_train]
X_test  = X_combined[idx_test]
y_train = y[idx_train]
y_test  = y[idx_test]

log(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# --------------------------------------------------------------- NB eğitimi
log("MultinomialNB eğitimi...")
t0 = time.time()
nb_model = MultinomialNB(alpha=0.1)  # Laplace smoothing
nb_model.fit(X_train, y_train)
train_time = time.time() - t0
log(f"Eğitim süresi: {train_time:.2f} s")

# --------------------------------------------------------------- değerlendirme
y_pred_test = nb_model.predict(X_test)
m = metrics(y_test, y_pred_test)
log(f"Test Accuracy:  {m['Accuracy']:.4f}")
log(f"Test Precision: {m['Precision']:.4f}")
log(f"Test Recall:    {m['Recall']:.4f}")
log(f"Test F1-Macro:  {m['F1-Macro']:.4f}")

# --------------------------------------------------------------- kaydet
log("Model kaydediliyor...")
joblib.dump(nb_model, os.path.join(ROOT, "models", "nb_model.pkl"))

# --------------------------------------------------------------- final_evaluation.csv güncelle
log("final_evaluation.csv güncelleniyor...")
eval_path = os.path.join(ROOT, "results", "final_evaluation.csv")
rows = []
if os.path.exists(eval_path):
    with open(eval_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["Model"] != "Naive Bayes":  # eski NB varsa çıkar
                rows.append(r)

# NB satırını ekle (Ensemble'dan sonra, DL'den önce)
nb_row = {"Model": "Naive Bayes", "Accuracy": m["Accuracy"],
          "Precision": m["Precision"], "Recall": m["Recall"],
          "F1-Macro": m["F1-Macro"]}

# Sıralama: LR, SVM, LightGBM, NB, Ensemble, LSTM, BiLSTM
order = ["Logistic Regression", "SVM", "LightGBM", "Naive Bayes",
         "Ensemble (Soft Voting)", "LSTM", "BiLSTM"]
ordered = []
existing = {r["Model"]: r for r in rows}
existing["Naive Bayes"] = nb_row
for name in order:
    if name in existing:
        ordered.append(existing[name])

with open(eval_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["Model", "Accuracy", "Precision", "Recall", "F1-Macro"])
    w.writeheader()
    for r in ordered:
        w.writerow(r)

# model_comparison.csv de güncelle
comp_df = pd.DataFrame(ordered).set_index("Model")
for col in ["Accuracy", "Precision", "Recall", "F1-Macro"]:
    comp_df[col] = comp_df[col].astype(float).round(4)
comp_df.to_csv(os.path.join(ROOT, "results", "model_comparison.csv"))

# best_model.txt güncelle
best_name = comp_df["F1-Macro"].astype(float).idxmax()
best_f1 = comp_df["F1-Macro"].astype(float).max()
with open(os.path.join(ROOT, "results", "best_model.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best Model: {best_name}\nF1-Macro: {best_f1}")

log(f"\nBİTTİ. NB F1-Macro: {m['F1-Macro']:.4f}")
log(f"En iyi model: {best_name} (F1={best_f1:.4f})")
log("Kaydedilenler: models/nb_model.pkl, results/final_evaluation.csv, model_comparison.csv")
