"""
Geliştirme paketi: (1) summary/başlık alanını TF-IDF'e ekleme, (2) Ensemble
(soft voting) ve aynı zamanda metodoloji düzeltmesi olarak TF-IDF + scaler'ı
YALNIZ eğitim setinde fit etme (önceki sürümde tüm veride fit ediliyordu).

Üretilenler:
  models/  -> tfidf_vectorizer.pkl, scaler.pkl, lr/svm/lgbm_model.pkl (summary'li)
  results/ -> final_evaluation.csv, model_comparison.csv, best_model.txt,
              ablation_summary.csv, ablation_summary.png

Derin modeller (LSTM/BiLSTM) sequence tabanlı ayrı bir yoldan geldiği için
yeniden eğitilmez; test metrikleri mevcut tablodan korunur.
"""
import os
import sys
import time
import csv
import gc

import numpy as np
import pandas as pd
import joblib
import scipy.sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
import lightgbm as lgb

# Eğitim ile çıkarımın bire bir aynı temizliği kullanması için webapp/pipeline
# içindeki clean_text fonksiyonunu yeniden kullanıyoruz.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "webapp"))
from pipeline import clean_text  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
NUMERIC = ["review_length", "word_count", "exclamation_count", "question_count",
           "avg_word_length", "uppercase_ratio", "sentiment_polarity",
           "sentiment_subjectivity"]
TFIDF_PARAMS = dict(max_features=50000, ngram_range=(1, 2), sublinear_tf=True,
                    min_df=2, max_df=0.95, dtype=np.float32)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall":    recall_score(y_true, y_pred, average="macro"),
        "F1-Macro":  f1_score(y_true, y_pred, average="macro"),
    }


# --------------------------------------------------------------- veri + split
log("Veri yükleniyor...")
df = pd.read_csv(os.path.join(ROOT, "data", "reviews_preprocessed.csv"),
                 usecols=["summary", "cleaned_text", "label"] + NUMERIC)
df = df.dropna(subset=["cleaned_text"]).reset_index(drop=True)
log(f"Satır: {len(df):,}")

idx_train = np.load(os.path.join(ROOT, "features", "train_idx.npy"))
idx_val   = np.load(os.path.join(ROOT, "features", "val_idx.npy"))
idx_test  = np.load(os.path.join(ROOT, "features", "test_idx.npy"))
y = df["label"].values

log("Summary alanı temizleniyor (clean_text)...")
summary_clean = df["summary"].fillna("").map(clean_text)
text_clean = df["cleaned_text"].astype(str)
combined = (summary_clean + " " + text_clean).str.strip()


def build_matrix(text_series, fit_idx):
    """TF-IDF + ölçeklenmiş sayısal özellikler; ikisi de YALNIZ fit_idx'te fit."""
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    vec.fit(text_series.iloc[fit_idx])
    X_text = vec.transform(text_series)
    scaler = StandardScaler()
    scaler.fit(df[NUMERIC].iloc[fit_idx])
    X_num = scaler.transform(df[NUMERIC])
    X = scipy.sparse.hstack([X_text, scipy.sparse.csr_matrix(X_num)]).tocsr()
    return X, vec, scaler


def train_trio(X):
    Xtr, Xte = X[idx_train], X[idx_test]
    ytr = y[idx_train]
    lr = LogisticRegression(C=1, solver="lbfgs", max_iter=500,
                            class_weight="balanced")
    lr.fit(Xtr, ytr)
    svm = CalibratedClassifierCV(
        LinearSVC(C=0.1, max_iter=2000, dual=False, class_weight="balanced"),
        cv=3)
    svm.fit(Xtr, ytr)
    lgbm = lgb.LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.1,
                              class_weight="balanced", n_jobs=-1, random_state=42,
                              verbose=-1)
    lgbm.fit(Xtr, ytr)
    proba = {
        "lr":   lr.predict_proba(Xte),
        "svm":  svm.predict_proba(Xte),
        "lgbm": lgbm.predict_proba(Xte),
    }
    return {"lr": lr, "svm": svm, "lgbm": lgbm}, proba


yte = y[idx_test]

# --------------------------------------------------------------- ABLATION
log("ABLATION 1/2: TEXT ONLY (yalnız yorum metni) eğitiliyor...")
X_text_only, _, _ = build_matrix(text_clean, idx_train)
_, proba_text = train_trio(X_text_only)
del X_text_only; gc.collect()

log("ABLATION 2/2: TEXT + SUMMARY eğitiliyor (üretim modelleri)...")
X_comb, vec_comb, scaler_comb = build_matrix(combined, idx_train)
models_comb, proba_comb = train_trio(X_comb)

# --------------------------------------------------------------- ENSEMBLE
log("Ensemble (soft voting) hesaplanıyor...")
ens_text = np.argmax((proba_text["lr"] + proba_text["svm"] + proba_text["lgbm"]) / 3, axis=1)
ens_comb = np.argmax((proba_comb["lr"] + proba_comb["svm"] + proba_comb["lgbm"]) / 3, axis=1)

# --------------------------------------------------------------- tablolar
NAMES = {"lr": "Logistic Regression", "svm": "SVM", "lgbm": "LightGBM"}
ablation_rows = []
for k in ["lr", "svm", "lgbm"]:
    f_text = f1_score(yte, np.argmax(proba_text[k], axis=1), average="macro")
    f_comb = f1_score(yte, np.argmax(proba_comb[k], axis=1), average="macro")
    ablation_rows.append((NAMES[k], f_text, f_comb, f_comb - f_text))
f_text_ens = f1_score(yte, ens_text, average="macro")
f_comb_ens = f1_score(yte, ens_comb, average="macro")
ablation_rows.append(("Ensemble (Soft Voting)", f_text_ens, f_comb_ens, f_comb_ens - f_text_ens))

log("\n==== ABLATION (F1-Macro) ====")
log(f"{'Model':24} {'TextOnly':>9} {'+Summary':>9} {'Delta':>8}")
for name, ft, fc, d in ablation_rows:
    log(f"{name:24} {ft:9.4f} {fc:9.4f} {d:+8.4f}")

# --------------------------------------------------------------- kaydet: modeller
log("Üretim modelleri kaydediliyor (summary'li, train-only fit)...")
joblib.dump(vec_comb, os.path.join(ROOT, "models", "tfidf_vectorizer.pkl"))
joblib.dump(scaler_comb, os.path.join(ROOT, "models", "scaler.pkl"))
joblib.dump(models_comb["lr"], os.path.join(ROOT, "models", "lr_model.pkl"))
joblib.dump(models_comb["svm"], os.path.join(ROOT, "models", "svm_model.pkl"))
joblib.dump(models_comb["lgbm"], os.path.join(ROOT, "models", "lgbm_model.pkl"))

# --------------------------------------------------------------- kaydet: sonuçlar
# Üretim leaderboard'u: klasikler + ensemble (summary'li) yeniden hesaplanır,
# derin modeller mevcut tablodan korunur.
log("Sonuç tabloları yazılıyor...")
prod = {}
for k in ["lr", "svm", "lgbm"]:
    prod[NAMES[k]] = metrics(yte, np.argmax(proba_comb[k], axis=1))
prod["Ensemble (Soft Voting)"] = metrics(yte, ens_comb)

# derin modelleri eski tablodan al
deep = {}
old_path = os.path.join(ROOT, "models", "backup_text_only", "final_evaluation.csv")
if os.path.exists(old_path):
    old = pd.read_csv(old_path)
    for _, r in old.iterrows():
        if r["Model"] in ("LSTM", "BiLSTM"):
            deep[r["Model"]] = {"Accuracy": r["Accuracy"], "Precision": r["Precision"],
                                "Recall": r["Recall"], "F1-Macro": r["F1-Macro"]}

order = ["Logistic Regression", "SVM", "LightGBM", "Ensemble (Soft Voting)", "LSTM", "BiLSTM"]
final = {m: (prod.get(m) or deep.get(m)) for m in order if (prod.get(m) or deep.get(m))}

with open(os.path.join(ROOT, "results", "final_evaluation.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Model", "Accuracy", "Precision", "Recall", "F1-Macro"])
    for m, d in final.items():
        w.writerow([m, d["Accuracy"], d["Precision"], d["Recall"], d["F1-Macro"]])

# compare sayfası için yuvarlanmış kopya
comp = pd.DataFrame(final).T.round(4)
comp.index.name = ""
comp.to_csv(os.path.join(ROOT, "results", "model_comparison.csv"))

# en iyi model
best_name = max(final, key=lambda m: final[m]["F1-Macro"])
best_f1 = final[best_name]["F1-Macro"]
with open(os.path.join(ROOT, "results", "best_model.txt"), "w", encoding="utf-8") as f:
    f.write(f"Best Model: {best_name}\nF1-Macro: {best_f1}")

# ablation csv
with open(os.path.join(ROOT, "results", "ablation_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Model", "F1_TextOnly", "F1_TextSummary", "Delta"])
    for name, ft, fc, d in ablation_rows:
        w.writerow([name, round(ft, 4), round(fc, 4), round(d, 4)])

# ablation grafiği
labels = [r[0].replace(" (Soft Voting)", "\n(Ensemble)") for r in ablation_rows]
ft_vals = [r[1] for r in ablation_rows]
fc_vals = [r[2] for r in ablation_rows]
x = np.arange(len(labels)); wbar = 0.38
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - wbar/2, ft_vals, wbar, label="Yalnız metin", color="#9aa7b8")
b2 = ax.bar(x + wbar/2, fc_vals, wbar, label="Metin + Başlık (summary)", color="#2e7d32")
ax.set_ylabel("F1-Macro"); ax.set_title("Başlık (summary) alanının katkısı — Ablation")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.65, max(fc_vals) + 0.03); ax.legend()
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "results", "ablation_summary.png"), dpi=120)

log(f"\nBİTTİ. En iyi model: {best_name} (F1={best_f1:.4f})")
log("Kaydedilenler: models/*.pkl, results/final_evaluation.csv, model_comparison.csv, "
    "ablation_summary.csv, ablation_summary.png, best_model.txt")
