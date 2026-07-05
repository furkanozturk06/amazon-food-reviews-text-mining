"""
Yorum Duygu Analizi - web arayüzü.

Projede eğitilen modelleri (klasikler, topluluklar, LSTM/BiLSTM ve ince ayarlı
DistilBERT) tek bir panelde toplar; tekli tahmin, toplu analiz, model
karşılaştırması ve keşifsel analiz sayfaları sunar.
"""
import os
import csv
import warnings

from flask import (Flask, render_template, request, jsonify,
                   send_from_directory, abort)

import pipeline

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
# Şablon/statik değişiklikleri sunucuyu yeniden başlatmadan yansısın
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

ENGINE = pipeline.Engine()

# --- sayfalarda kullanılan görsel kümeleri (results/ altındaki gerçek dosyalar) ---
EDA_IMAGES = [
    ("eda_class_distribution.png", "Sınıf dağılımı"),
    ("eda_star_distribution.png", "Yıldız puanı dağılımı"),
    ("eda_review_length.png", "Yorum uzunluğu"),
    ("eda_word_count.png", "Kelime sayısı"),
    ("eda_top_words.png", "En sık kelimeler"),
    ("eda_bigrams.png", "İkili kelime grupları"),
    ("eda_wordcloud_negatif.png", "Kelime bulutu - Negatif"),
    ("eda_wordcloud_notr.png", "Kelime bulutu - Nötr"),
    ("eda_wordcloud_pozitif.png", "Kelime bulutu - Pozitif"),
    ("eda_correlation.png", "Özellik korelasyonu"),
    ("eda_helpfulness.png", "Faydalılık analizi"),
    ("eda_temporal_hour_day.png", "Ay ve güne göre duygu"),
    ("eda_time_trend.png", "Yıllara göre eğilim"),
    ("eda_top_users.png", "En aktif kullanıcılar"),
    ("eda_zipf.png", "Zipf yasası"),
    ("eda_summary_vs_text.png", "Özet ve metin karşılaştırması"),
    ("preprocessing_features.png", "Ön işleme etkisi"),
    ("lemma_vs_stem.png", "Lemmatization - stemming"),
    ("tsne.png", "t-SNE görselleştirmesi"),
    ("aspect_based_sentiment.png", "Varlık bazlı duygu"),
    ("aspect_avg_score.png", "Varlıklara göre ortalama skor"),
]

COMPARE_IMAGES = [
    ("model_comparison_final.png", "Model karşılaştırması (F1 & Accuracy)"),
    ("model_comparison_radar.png", "Radar karşılaştırması"),
    ("ablation_summary.png", "Başlık (summary) katkısı — Ablation"),
    ("confusion_matrices_all.png", "Karışıklık matrisleri"),
    ("roc_curves_all.png", "ROC eğrileri"),
    ("pr_curves_all.png", "Precision-Recall eğrileri"),
    ("training_histories.png", "Eğitim geçmişi (derin modeller)"),
    ("training_time_comparison.png", "Eğitim süresi karşılaştırması"),
    ("cross_validation.png", "Çapraz doğrulama"),
    ("error_analysis.png", "Hata analizi"),
    ("feature_importance.png", "Özellik önemi"),
    ("rule_vs_ml.png", "Kural tabanlı ve makine öğrenmesi"),
]


# ----------------------------------------------------------- başlangıç verileri
def read_metrics():
    """Güncel karşılaştırma tablosu. Önce final_comparison_all.csv (BERT dahil,
    model adı = indeks sütunu), yoksa eski final_evaluation.csv."""
    final = os.path.join(pipeline.RESULTS_DIR, "final_comparison_all.csv")
    legacy = os.path.join(pipeline.RESULTS_DIR, "final_evaluation.csv")
    rows = []
    path = final if os.path.exists(final) else legacy
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # final_comparison_all.csv'de model adı başlıksız ilk sütundadır ("")
            name = r.get("Model") or r.get("") or next(iter(r.values()))
            try:
                rows.append({
                    "model": name,
                    "accuracy": round(float(r["Accuracy"]), 4),
                    "precision": round(float(r["Precision"]), 4),
                    "recall": round(float(r["Recall"]), 4),
                    "f1": round(float(r["F1-Macro"]), 4),
                })
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda x: x["f1"], reverse=True)
    return rows


def read_best_model():
    """En iyi modeli güncel metrik tablosundan türet."""
    rows = read_metrics()
    if rows:
        b = rows[0]
        return f"Best Model: {b['model']}\nF1-Macro: {b['f1']:.4f}"
    return ""


def class_distribution():
    """features/labels.npy -> sınıf başına örnek sayısı."""
    import numpy as np
    path = os.path.join(pipeline.FEATURES_DIR, "labels.npy")
    if not os.path.exists(path):
        return {}
    y = np.load(path)
    return {pipeline.SLUGS[i]: {"label": pipeline.LABELS[i], "count": int((y == i).sum())}
            for i in (0, 1, 2)}


METRICS = read_metrics()
BEST_MODEL = read_best_model()
CLASS_DIST = class_distribution()
TOP_WORDS = ENGINE.top_words_per_class(15)


def model_options():
    """Arayüzde seçilebilir modeller (sadece başarıyla yüklenenler)."""
    return [{"key": m, "title": pipeline.MODEL_TITLES[m], "kind": pipeline.MODEL_KIND[m]}
            for m in ENGINE.available()]


@app.context_processor
def inject_globals():
    return {
        "nav_models": model_options(),
        "model_count": len(ENGINE.available()),
    }


# ----------------------------------------------------------- sayfalar
@app.route("/")
def index():
    return render_template("index.html", models=model_options())


@app.route("/toplu")
def bulk():
    return render_template("bulk.html", models=model_options())


@app.route("/karsilastirma")
def compare():
    return render_template("compare.html", metrics=METRICS,
                           best_model=BEST_MODEL, images=COMPARE_IMAGES)


@app.route("/kesif")
def eda():
    return render_template("eda.html", images=EDA_IMAGES,
                           distribution=CLASS_DIST, top_words=TOP_WORDS)


@app.route("/hakkinda")
def about():
    return render_template("about.html", metrics=METRICS, best_model=BEST_MODEL)


# ----------------------------------------------------------- api
@app.route("/api/tahmin", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    summary = (data.get("summary") or "").strip()
    model = data.get("model") or "lr"
    if not text:
        return jsonify({"error": "Metin boş olamaz."}), 400
    if model not in ENGINE.available():
        return jsonify({"error": "Seçilen model yüklü değil."}), 400

    result = ENGINE.predict(text, model, summary=summary)

    # tüm modelleri aynı metinle çalıştırıp karşılaştırma satırı üret
    others = []
    for m in ENGINE.available():
        r = ENGINE.predict(text, m, summary=summary)
        others.append({
            "key": m,
            "title": pipeline.MODEL_TITLES[m],
            "kind": pipeline.MODEL_KIND[m],
            "label": r["label"],
            "slug": r["slug"],
            "confidence": r["confidence"],
        })
    result["all_models"] = others
    return jsonify(result)


@app.route("/api/toplu", methods=["POST"])
def api_bulk():
    data = request.get_json(force=True, silent=True) or {}
    texts = data.get("texts") or []
    model = data.get("model") or "lr"
    texts = [str(t).strip() for t in texts if str(t).strip()]
    if not texts:
        return jsonify({"error": "Analiz edilecek metin bulunamadı."}), 400
    if model not in ENGINE.available():
        return jsonify({"error": "Seçilen model yüklü değil."}), 400

    texts = texts[:2000]   # makul üst sınır
    counts = {"negatif": 0, "notr": 0, "pozitif": 0}
    results = []
    for t in texts:
        r = ENGINE.predict(t, model)
        counts[r["slug"]] += 1
        results.append({
            "text": t,
            "label": r["label"],
            "slug": r["slug"],
            "confidence": r["confidence"],
        })
    return jsonify({
        "model": model,
        "model_title": pipeline.MODEL_TITLES[model],
        "total": len(results),
        "counts": counts,
        "results": results,
    })


@app.route("/gorsel/<path:name>")
def serve_image(name):
    """results/ klasöründeki grafik dosyalarını güvenli biçimde sunar."""
    name = os.path.basename(name)
    allowed = {f for f, _ in EDA_IMAGES} | {f for f, _ in COMPARE_IMAGES}
    if name not in allowed:
        abort(404)
    return send_from_directory(pipeline.RESULTS_DIR, name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
