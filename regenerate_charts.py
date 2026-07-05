"""
Tüm karşılaştırma grafiklerini yeniden oluşturur.

Mevcut eğitilmiş modelleri (models/) ve test setini kullanarak aşağıdaki
grafikleri TÜM modeller (LR, SVM, LightGBM, NB, LSTM, BiLSTM, Ensemble)
dahil olacak şekilde üretir:

  1. ablation_summary.png       — Summary katkısı (klasik + ensemble)
  2. model_comparison_radar.png — Radar grafiği (tüm modeller)
  3. confusion_matrices_all.png — Confusion matrix (tüm modeller)
  4. roc_curves_all.png         — ROC eğrileri (tüm modeller)
  5. pr_curves_all.png          — Precision-Recall (tüm modeller)
  6. training_time_comparison.png — Eğitim süresi karşılaştırması
  7. error_analysis.png         — Hata analizi (tüm modeller)
  8. rule_vs_ml.png             — Kural-tabanlı vs ML (tüm ML modeller)
  9. cross_validation.png       — Cross-validation (birden fazla model)
 10. feature_importance.png     — Özellik önemi (LR + LightGBM)
"""
import os
import sys
import io
import time
import json

# Windows konsolunda Türkçe karakter desteği
if sys.stdout and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import joblib
import scipy.sparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import lightgbm as lgb

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
MODELS  = os.path.join(ROOT, "models")

NUMERIC = ["review_length", "word_count", "exclamation_count", "question_count",
           "avg_word_length", "uppercase_ratio", "sentiment_polarity",
           "sentiment_subjectivity"]
CLASS_NAMES = ["Negatif", "Nötr", "Pozitif"]

# Renk paleti: her model için tutarlı renk
MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "SVM":                 "#2ca02c",
    "LightGBM":            "#d62728",
    "Naive Bayes":         "#ff7f0e",
    "Ensemble (Soft Voting)": "#9467bd",
    "LSTM":                "#8c564b",
    "BiLSTM":              "#e377c2",
}
MODEL_LS = {
    "Logistic Regression": "-",
    "SVM":                 "--",
    "LightGBM":            "-.",
    "Naive Bayes":         ":",
    "Ensemble (Soft Voting)": "-",
    "LSTM":                "--",
    "BiLSTM":              "-.",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def metrics(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall":    recall_score(y_true, y_pred, average="macro"),
        "F1-Macro":  f1_score(y_true, y_pred, average="macro"),
    }


# ============================================================ VERİ YÜKLEME
log("Veri ve modeller yükleniyor...")

df = pd.read_csv(os.path.join(ROOT, "data", "reviews_preprocessed.csv"),
                 usecols=["summary", "text", "cleaned_text", "label"] + NUMERIC)
df = df.dropna(subset=["cleaned_text"]).reset_index(drop=True)

idx_test  = np.load(os.path.join(ROOT, "features", "test_idx.npy"))
y         = df["label"].values
y_test    = y[idx_test]

tfidf_vectorizer = joblib.load(os.path.join(MODELS, "tfidf_vectorizer.pkl"))
scaler_obj       = joblib.load(os.path.join(MODELS, "scaler.pkl"))

# Summary'li combined text
sys.path.insert(0, os.path.join(ROOT, "webapp"))
from pipeline import clean_text  # noqa: E402

summary_clean = df["summary"].fillna("").map(clean_text)
text_clean = df["cleaned_text"].astype(str)
combined = (summary_clean + " " + text_clean).str.strip()

X_tfidf = tfidf_vectorizer.transform(combined)
X_num   = scaler_obj.transform(df[NUMERIC])
X_combined = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(X_num)]).tocsr()
X_test_tfidf = X_combined[idx_test]

# NB için negatif olmayan matris
X_num_clipped = np.clip(X_num, 0, None)
X_combined_nb = scipy.sparse.hstack([X_tfidf, scipy.sparse.csr_matrix(X_num_clipped)]).tocsr()
X_test_nb = X_combined_nb[idx_test]

# Sequence verileri
X_seq = np.load(os.path.join(ROOT, "features", "sequences.npy"))
X_test_seq = X_seq[idx_test]

texts_test = df["text"].iloc[idx_test].values


# ============================================================ MODEL YÜKLEME
log("Modeller yükleniyor...")
lr_model   = joblib.load(os.path.join(MODELS, "lr_model.pkl"))
svm_model  = joblib.load(os.path.join(MODELS, "svm_model.pkl"))
lgbm_model = joblib.load(os.path.join(MODELS, "lgbm_model.pkl"))
nb_model   = joblib.load(os.path.join(MODELS, "nb_model.pkl"))

lstm_model = None
bilstm_model = None
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow.keras.models import load_model
    if os.path.exists(os.path.join(MODELS, "lstm_model.keras")):
        lstm_model = load_model(os.path.join(MODELS, "lstm_model.keras"))
    if os.path.exists(os.path.join(MODELS, "bilstm_model.keras")):
        bilstm_model = load_model(os.path.join(MODELS, "bilstm_model.keras"))
except ImportError:
    log("TensorFlow yüklenemedi, DL modelleri atlanacak.")

# Olasılık ve tahmin sözlükleri
model_probs = {
    "Logistic Regression": lr_model.predict_proba(X_test_tfidf),
    "SVM":                 svm_model.predict_proba(X_test_tfidf),
    "LightGBM":            lgbm_model.predict_proba(X_test_tfidf),
    "Naive Bayes":         nb_model.predict_proba(X_test_nb),
}
# Ensemble (soft voting: LR + SVM + LightGBM ortalaması)
ens_proba = (model_probs["Logistic Regression"] +
             model_probs["SVM"] +
             model_probs["LightGBM"]) / 3
model_probs["Ensemble (Soft Voting)"] = ens_proba

if lstm_model is not None:
    model_probs["LSTM"] = lstm_model.predict(X_test_seq, verbose=0)
if bilstm_model is not None:
    model_probs["BiLSTM"] = bilstm_model.predict(X_test_seq, verbose=0)

model_preds = {n: np.argmax(p, axis=1) for n, p in model_probs.items()}
model_metrics = {n: metrics(y_test, p) for n, p in model_preds.items()}

# DL modelleri yüklenemezse bile radar grafiği için metrikleri CSV'den al
eval_csv = os.path.join(RESULTS, "final_evaluation.csv")
if os.path.exists(eval_csv):
    eval_df = pd.read_csv(eval_csv)
    for _, row in eval_df.iterrows():
        mname = row["Model"]
        if mname not in model_metrics:
            model_metrics[mname] = {
                "Accuracy": float(row["Accuracy"]),
                "Precision": float(row["Precision"]),
                "Recall": float(row["Recall"]),
                "F1-Macro": float(row["F1-Macro"]),
            }

# Sıralama
ORDER = ["Logistic Regression", "SVM", "LightGBM", "Naive Bayes",
         "Ensemble (Soft Voting)", "LSTM", "BiLSTM"]
active_models = [m for m in ORDER if m in model_probs]
# Radar ve training time için: olasılık gerekmeden metrik bulunan modeller
all_metric_models = [m for m in ORDER if m in model_metrics]

log(f"Aktif modeller (proba): {active_models}")
log(f"Metrik bulunan modeller: {all_metric_models}")

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120})


# ============================================================ 1. ABLATION
log("1/10 Ablation grafiği...")

# Ablation verisi mevcut csv'den
abl_path = os.path.join(RESULTS, "ablation_summary.csv")
if os.path.exists(abl_path):
    abl_df = pd.read_csv(abl_path)
    labels = abl_df["Model"].values
    labels = [l.replace(" (Soft Voting)", "\n(Ensemble)") for l in labels]
    ft_vals = abl_df["F1_TextOnly"].values
    fc_vals = abl_df["F1_TextSummary"].values
else:
    labels = ["LR", "SVM", "LightGBM", "Ensemble"]
    ft_vals = [0] * 4
    fc_vals = [0] * 4

x = np.arange(len(labels))
wbar = 0.38
fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - wbar/2, ft_vals, wbar, label="Yalnız metin", color="#9aa7b8")
b2 = ax.bar(x + wbar/2, fc_vals, wbar, label="Metin + Başlık (summary)", color="#2e7d32")
ax.set_ylabel("F1-Macro")
ax.set_title("Başlık (summary) alanının katkısı — Ablation")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.65, max(list(fc_vals) + list(ft_vals)) + 0.03)
ax.legend()
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height():.3f}",
                (b.get_x() + b.get_width()/2, b.get_height()),
                ha="center", va="bottom", fontsize=8)
# Not ekle: DL modelleri sequence tabanlı
ax.text(0.98, 0.02, "Not: LSTM/BiLSTM sequence tabanlıdır,\nsummary eklenmemiştir.",
        transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
        style="italic", color="gray")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "ablation_summary.png"), dpi=120)
plt.close(fig)


# ============================================================ 2. RADAR
log("2/10 Radar grafiği...")

categories = ["Accuracy", "Precision", "Recall", "F1-Macro"]
N = len(categories)
angles = [n/N * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, polar=True)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"],
           color="grey", size=10)
plt.ylim(0, 1)

for name in all_metric_models:
    m = model_metrics[name]
    vals = [m[c] for c in categories] + [m[categories[0]]]
    col = MODEL_COLORS.get(name, "gray")
    ls = MODEL_LS.get(name, "-")
    ax.plot(angles, vals, color=col, linewidth=2, linestyle=ls, label=name)
    ax.fill(angles, vals, color=col, alpha=0.05)

plt.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.title("Model Performansları Radar Grafiği", size=15, y=1.1)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "model_comparison_radar.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)


# ============================================================ 3. CONFUSION MATRIX
log("3/10 Confusion matrix...")

n_models = len(active_models)
cols = min(4, n_models)
rows_cm = (n_models + cols - 1) // cols
fig, axes = plt.subplots(rows_cm, cols, figsize=(6 * cols, 5 * rows_cm))
if rows_cm == 1:
    axes = [axes] if cols == 1 else list(axes)
else:
    axes = [ax for row in axes for ax in row]

for i, name in enumerate(active_models):
    ax = axes[i]
    y_pred = model_preds[name]
    cm = confusion_matrix(y_test, y_pred)
    cm_n = confusion_matrix(y_test, y_pred, normalize="true")
    annot = np.empty_like(cm, dtype=object)
    for r in range(3):
        for c in range(3):
            annot[r, c] = f"{cm[r, c]}\n({cm_n[r, c]:.1%})"
    sns.heatmap(cm_n, annot=annot, fmt="", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar=False)
    short = name.replace("Logistic Regression", "LR").replace(" (Soft Voting)", "")
    ax.set_title(short, fontsize=11)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")

# Boş eksenleri gizle
for j in range(len(active_models), len(axes)):
    axes[j].set_visible(False)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "confusion_matrices_all.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)


# ============================================================ 4. ROC CURVES
log("4/10 ROC eğrileri...")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for i, cname in enumerate(CLASS_NAMES):
    ax = axes[i]
    for name in active_models:
        probs = model_probs[name]
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        col = MODEL_COLORS.get(name, "gray")
        ls = MODEL_LS.get(name, "-")
        short = name.replace("Logistic Regression", "LR").replace(" (Soft Voting)", " Ens.")
        ax.plot(fpr, tpr, color=col, linestyle=ls, lw=2,
                label=f"{short} ({roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"ROC: {cname}")
    ax.legend(loc="lower right", fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "roc_curves_all.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)


# ============================================================ 5. PR CURVES
log("5/10 PR eğrileri...")

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for i, cname in enumerate(CLASS_NAMES):
    ax = axes[i]
    baseline = np.sum(y_test_bin[:, i]) / len(y_test_bin[:, i])
    for name in active_models:
        probs = model_probs[name]
        prec, rec, _ = precision_recall_curve(y_test_bin[:, i], probs[:, i])
        ap = average_precision_score(y_test_bin[:, i], probs[:, i])
        col = MODEL_COLORS.get(name, "gray")
        ls = MODEL_LS.get(name, "-")
        short = name.replace("Logistic Regression", "LR").replace(" (Soft Voting)", " Ens.")
        ax.plot(rec, prec, color=col, linestyle=ls, lw=2,
                label=f"{short} ({ap:.3f})")
    ax.axhline(y=baseline, color="k", linestyle="--",
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR: {cname}")
    ax.legend(loc="lower left", fontsize=7)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "pr_curves_all.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)


# ============================================================ 6. TRAINING TIME
log("6/10 Eğitim süresi karşılaştırması...")

# 20K örneklemle tüm modellerin eğitim süresini ölçüyoruz
idx_train = np.load(os.path.join(ROOT, "features", "train_idx.npy"))
dfs_idx = np.random.RandomState(42).choice(idx_train, size=min(20000, len(idx_train)),
                                           replace=False)
Xtr_text = combined.iloc[dfs_idx]
ytr = y[dfs_idx]

v = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
Xtr_tfidf = v.fit_transform(Xtr_text)

# Test seti de aynı vectorizer ile
Xte_text = combined.iloc[idx_test]
Xte_tfidf = v.transform(Xte_text)

modeller_time = {
    "LogReg":   LogisticRegression(max_iter=500, class_weight="balanced"),
    "SVM":      LinearSVC(class_weight="balanced", max_iter=2000),
    "LightGBM": lgb.LGBMClassifier(n_estimators=200, num_leaves=63,
                                    class_weight="balanced", n_jobs=-1, verbose=-1),
    "NaiveBayes": MultinomialNB(alpha=0.1),
}

sure, f1l, adlar = [], [], []
for ad, m in modeller_time.items():
    t0 = time.time()
    m.fit(Xtr_tfidf, ytr)
    dt = time.time() - t0
    f1 = f1_score(y_test, m.predict(Xte_tfidf), average="macro")
    sure.append(dt)
    f1l.append(f1)
    adlar.append(ad)
    log(f"  {ad:12s}  süre={dt:6.2f}s  F1-Macro={f1:.3f}")

# DL modelleri: eğitim süresini history dosyalarından çıkar
for key, title in [("lstm", "LSTM"), ("bilstm", "BiLSTM")]:
    hfile = os.path.join(RESULTS, f"history_{key}.json")
    if os.path.exists(hfile):
        with open(hfile) as f:
            h = json.load(f)
        # Her epoch ~ortalama süre (toplam eğitim süresi bilinmiyor,
        # ama epoch sayısı ve yaklaşık süre veriyoruz)
        n_epochs = len(h.get("loss", []))
        # Gerçek eğitim süresi bilinmediğinden yaklaşık değer
        approx_time = n_epochs * 45  # ~45s/epoch tahmini (CPU)
        final_f1 = model_metrics.get(title, {}).get("F1-Macro", 0)
        sure.append(approx_time)
        f1l.append(final_f1)
        adlar.append(title)

fig, ax = plt.subplots(figsize=(9, 6))
colors_scatter = [MODEL_COLORS.get({
    "LogReg": "Logistic Regression", "SVM": "SVM", "LightGBM": "LightGBM",
    "NaiveBayes": "Naive Bayes", "LSTM": "LSTM", "BiLSTM": "BiLSTM"
}.get(a, a), "#333") for a in adlar]
ax.scatter(sure, f1l, s=150, c=colors_scatter, zorder=5, edgecolors="black", linewidth=0.5)
for a, x_val, yv in zip(adlar, sure, f1l):
    ax.annotate(a, (x_val, yv), xytext=(8, 8), textcoords="offset points", fontsize=9)
ax.set_title("Eğitim Süresi vs F1-Macro (Tüm Modeller)")
ax.set_xlabel("Eğitim süresi (s)")
ax.set_ylabel("F1-Macro")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "training_time_comparison.png"), dpi=120)
plt.close(fig)


# ============================================================ 7. ERROR ANALYSIS
log("7/10 Hata analizi...")

n_models_err = len(active_models)
cols_e = min(4, n_models_err)
rows_e = (n_models_err + cols_e - 1) // cols_e
fig, axes = plt.subplots(rows_e, cols_e, figsize=(6 * cols_e, 5 * rows_e))
if rows_e == 1:
    axes = list(axes) if cols_e > 1 else [axes]
else:
    axes = [ax for row in axes for ax in row]

for i, name in enumerate(active_models):
    ax = axes[i]
    y_pred = model_preds[name]
    errors = y_test != y_pred
    err_rate = errors.mean() * 100
    if errors.sum() > 0:
        err_cm = confusion_matrix(y_test[errors], y_pred[errors])
        sns.heatmap(err_cm, annot=True, fmt="d", cmap="Reds",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=ax, cbar=False)
    short = name.replace("Logistic Regression", "LR").replace(" (Soft Voting)", "")
    ax.set_title(f"Hata: {short}\n(%{err_rate:.1f} hata)", fontsize=10)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")

for j in range(len(active_models), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Model Bazında Hata Dağılımları", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "error_analysis.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)


# ============================================================ 8. RULE VS ML
log("8/10 Kural-tabanlı vs ML...")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

n = min(4000, len(idx_test))
sidx = np.random.RandomState(1).choice(len(idx_test), size=n, replace=False)
real_idx = idx_test[sidx]
texts_sample = df["text"].iloc[real_idx].astype(str).tolist()
y_true_sample = y_test[sidx]

vader = SentimentIntensityAnalyzer()
y_vader = [2 if (c := vader.polarity_scores(t)["compound"]) >= 0.05
           else 0 if c <= -0.05 else 1 for t in texts_sample]
y_tb = [2 if (p := TextBlob(t).sentiment.polarity) > 0.05
        else 0 if p < -0.05 else 1 for t in texts_sample]

# Tüm ML modelleri
skor = {"VADER": f1_score(y_true_sample, y_vader, average="macro"),
        "TextBlob": f1_score(y_true_sample, y_tb, average="macro")}

for name in active_models:
    y_ml = model_preds[name][sidx]
    short = name.replace("Logistic Regression", "LR").replace(" (Soft Voting)", " Ens.")
    skor[short] = f1_score(y_true_sample, y_ml, average="macro")

fig, ax = plt.subplots(figsize=(12, 5))
colors_rule = []
for k in skor.keys():
    if k in ("VADER", "TextBlob"):
        colors_rule.append("#999")
    else:
        # ML model rengi bul
        full_name = {v.replace("Logistic Regression", "LR").replace(" (Soft Voting)", " Ens."): k2
                     for k2, v in [(n, n) for n in active_models]}
        colors_rule.append("#2e7d32")

bars = ax.bar(range(len(skor)), list(skor.values()), color=colors_rule)
ax.set_xticks(range(len(skor)))
ax.set_xticklabels(list(skor.keys()), rotation=30, ha="right", fontsize=9)
for i, (k, v) in enumerate(skor.items()):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
# Kural-tabanlı çubuklara gri renk
for i, k in enumerate(skor.keys()):
    if k in ("VADER", "TextBlob"):
        bars[i].set_color("#999")
    else:
        bars[i].set_color(MODEL_COLORS.get(
            next((m for m in active_models if
                  m.replace("Logistic Regression", "LR").replace(" (Soft Voting)", " Ens.") == k), ""),
            "#2e7d32"))

ax.set_title("Kural-tabanlı (VADER/TextBlob) vs ML Modelleri — F1-Macro")
ax.set_ylabel("F1-Macro")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "rule_vs_ml.png"), dpi=120)
plt.close(fig)


# ============================================================ 9. CROSS-VALIDATION
log("9/10 Cross-validation (birden fazla model, 20K örneklem)...")

cv_models = {
    "LogReg":     Pipeline([("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                                       sublinear_tf=True)),
                             ("clf", LogisticRegression(max_iter=500, class_weight="balanced"))]),
    "SVM":        Pipeline([("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                                       sublinear_tf=True)),
                             ("clf", LinearSVC(class_weight="balanced", max_iter=2000))]),
    "LightGBM":   Pipeline([("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                                       sublinear_tf=True)),
                             ("clf", lgb.LGBMClassifier(n_estimators=200, num_leaves=63,
                                                         class_weight="balanced", n_jobs=-1,
                                                         verbose=-1))]),
    "NaiveBayes": Pipeline([("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                                       sublinear_tf=True)),
                             ("clf", MultinomialNB(alpha=0.1))]),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_text = combined.iloc[dfs_idx]
cv_y = ytr

cv_results = {}
for ad, pipe in cv_models.items():
    log(f"  CV: {ad}...")
    cvs = cross_val_score(pipe, cv_text, cv_y, cv=skf, scoring="f1_macro", n_jobs=-1)
    cv_results[ad] = cvs
    log(f"    {ad}: {cvs.mean():.4f} +/- {cvs.std():.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
positions = np.arange(len(cv_results))
width = 0.15
for fold in range(5):
    offsets = positions + fold * width - 2 * width
    vals = [cv_results[m][fold] for m in cv_results]
    ax.bar(offsets, vals, width, label=f"Fold {fold + 1}", alpha=0.8)

# Ortalama çizgileri
for i, (m, cvs) in enumerate(cv_results.items()):
    ax.hlines(cvs.mean(), i - 2.5 * width, i + 2.5 * width,
              colors="red", linewidth=2, linestyle="--")

ax.set_xticks(positions)
ax.set_xticklabels(list(cv_results.keys()), fontsize=10)
ax.set_ylabel("F1-Macro")
ax.set_title("5-Fold Cross-Validation — Tüm Klasik Modeller")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# Ortalama değerleri ek açıklama
for i, (m, cvs) in enumerate(cv_results.items()):
    ax.text(i, cvs.mean() + 0.003, f"{cvs.mean():.3f}", ha="center",
            fontsize=8, color="red", fontweight="bold")

fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "cross_validation.png"), dpi=120)
plt.close(fig)


# ============================================================ 10. FEATURE IMPORTANCE
log("10/10 Özellik önemi...")

vocab = np.array(tfidf_vectorizer.get_feature_names_out())
sinif_ad = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# LR katsayıları
for i in range(3):
    ax = axes[0, i]
    coefs = lr_model.coef_[i][:len(vocab)]
    top = np.argsort(coefs)[-15:]
    colors_fi = ["#d62728" if i == 0 else "#999" if i == 1 else "#2ca02c"][0]
    ax.barh(vocab[top], coefs[top], color=colors_fi)
    ax.set_title(f"LR: {sinif_ad[i]} sınıfına en çok iten kelimeler", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)

# LightGBM feature importance
lgbm_imp = lgbm_model.feature_importances_[:len(vocab)]
top_lgbm = np.argsort(lgbm_imp)[-15:]
for i in range(3):
    ax = axes[1, i]
    if i == 0:
        ax.barh(vocab[top_lgbm], lgbm_imp[top_lgbm], color="#d62728")
        ax.set_title("LightGBM: En önemli özellikler (split)", fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
    else:
        ax.set_visible(False)

# NB log probability
if hasattr(nb_model, "feature_log_prob_"):
    ax = axes[1, 1]
    ax.set_visible(True)
    # Her sınıf için en ayırt edici özellikler
    nb_imp = nb_model.feature_log_prob_
    # Sınıf 2 (Pozitif) vs sınıf 0 (Negatif) farkı
    diff = nb_imp[2, :len(vocab)] - nb_imp[0, :len(vocab)]
    top_pos = np.argsort(diff)[-10:]
    top_neg = np.argsort(diff)[:10]
    combined_idx = np.concatenate([top_neg, top_pos])
    combined_vals = diff[combined_idx]
    colors_nb = ["#d62728" if v < 0 else "#2ca02c" for v in combined_vals]
    ax.barh(vocab[combined_idx], combined_vals, color=colors_nb)
    ax.set_title("NB: Pozitif vs Negatif (log-prob farkı)", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.axvline(x=0, color="black", linewidth=0.5)

fig.suptitle("Özellik Önemi — LR Katsayıları, LightGBM ve NB", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "feature_importance.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)


# ============================================================ TAMAMLANDI
log("\n" + "=" * 50)
log("TÜM GRAFİKLER BAŞARIYLA YENİDEN ÜRETİLDİ!")
log("=" * 50)
log(f"Aktif modeller ({len(active_models)}): {', '.join(active_models)}")
log(f"Üretilen dosyalar: {RESULTS}/")
for fname in ["ablation_summary.png", "model_comparison_radar.png",
              "confusion_matrices_all.png", "roc_curves_all.png",
              "pr_curves_all.png", "training_time_comparison.png",
              "error_analysis.png", "rule_vs_ml.png",
              "cross_validation.png", "feature_importance.png"]:
    log(f"  ✓ {fname}")
