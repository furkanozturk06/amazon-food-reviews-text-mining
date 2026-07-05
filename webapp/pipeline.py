"""
Tahmin motoru (v2) — klasik v2 modeller + füzyon (tfidf+BERT) + fine-tune DistilBERT.

Ön işleme ve özellik üretimi tune_classical.py / train_bert.py / train_distilbert.py
ile BİREBİR aynıdır; aksi halde modeller eğitildikleri dağılımdan farklı girdi alır.

Ağır bağımlılıklar (sentence-transformers, tensorflow, fine-tune DistilBERT) yalnızca
ilgili model ilk kez kullanıldığında TEMBEL yüklenir; webapp açılışı hızlı kalır.
"""
import os
import re
import json

# ÖNEMLİ: torch EN BAŞTA import edilmeli; aksi halde scipy/sklearn/lightgbm kendi
# OpenMP DLL'lerini önce yükler ve torch'un c10.dll'i WinError 1114 ile çöker.
try:
    import torch  # noqa: F401
except Exception:
    torch = None

import joblib
import numpy as np
import scipy.sparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

# ---------------------------------------------------------------- sabitler
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
V2_DIR = os.path.join(MODELS_DIR, "v2")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
DISTILBERT_DIR = os.path.join(V2_DIR, "distilbert")

MAX_LEN = 200          # eski LSTM/BiLSTM sequence uzunluğu
BERT_MAXLEN = 192      # DistilBERT fine-tune ile aynı

LABELS = {0: "Negatif", 1: "Nötr", 2: "Pozitif"}
SLUGS = {0: "negatif", 1: "notr", 2: "pozitif"}

# 14 sayısal özellik — tune_classical.py'deki sıra ile birebir aynı
NUMERIC_FEATURES = [
    "review_length", "word_count", "exclamation_count", "question_count",
    "avg_word_length", "uppercase_ratio", "sentiment_polarity", "sentiment_subjectivity",
    "negation_count", "all_caps_ratio", "sentence_count", "avg_sentence_length",
    "unique_word_ratio", "punctuation_density",
]

CLASSICAL = {"lr", "svm", "lgbm", "nb"}
SEQUENTIAL = {"lstm", "bilstm"}
ENSEMBLE_MEMBERS = ["lr", "svm", "lgbm"]

# Arayüzdeki gösterim/öncelik sırası
# Not: "fusion" (TF-IDF+BERT) ve "bert_emb" (donmuş BERT+LR) artefaktları üretiliyor
# ama net katkı sağlamadıkları için arayüzde GÖSTERİLMİYOR. "BERT" = fine-tune DistilBERT.
MODEL_ORDER = ["distilbert", "stacking", "ensemble",
               "lr", "svm", "lgbm", "nb", "lstm", "bilstm"]

MODEL_TITLES = {
    "distilbert": "DistilBERT (fine-tuned)",
    "fusion": "Füzyon (TF-IDF + BERT)",
    "stacking": "Stacking (Blend)",
    "ensemble": "Ensemble (Weighted)",
    "lr": "Logistic Regression",
    "svm": "SVM",
    "lgbm": "LightGBM",
    "nb": "ComplementNB",
    "bert_emb": "BERT embedding + LR",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
}
MODEL_KIND = {
    "distilbert": "Transformer",
    "fusion": "Hibrit",
    "stacking": "Topluluk",
    "ensemble": "Topluluk",
    "lr": "Klasik", "svm": "Klasik", "lgbm": "Klasik", "nb": "Klasik",
    "bert_emb": "BERT",
    "lstm": "Derin", "bilstm": "Derin",
}

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------- nltk hazırlığı
def _ensure_nltk():
    for res, pkg in [("corpora/stopwords", "stopwords"),
                     ("corpora/wordnet", "wordnet"),
                     ("tokenizers/punkt", "punkt"),
                     ("tokenizers/punkt_tab", "punkt_tab")]:
        try:
            nltk.data.find(res)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


_ensure_nltk()

NEGATION_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "neither", "without", "cannot",
    "don", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn",
    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn", "ain",
}
_STOP = set(stopwords.words("english")) - NEGATION_WORDS
_LEMMA = WordNetLemmatizer()
# negation_count için (train script'leriyle aynı küme)
_NEG_COUNT = {"not", "no", "never", "don", "didn", "wouldn", "shouldn", "couldn", "won",
              "isn", "aren", "wasn", "weren", "nor", "neither", "without", "cannot"}


# ---------------------------------------------------------------- ön işleme
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    return " ".join(_LEMMA.lemmatize(w) for w in tokens if w not in _STOP and len(w) >= 2)


def numeric_features(text):
    """14 sayısal özellik — ham metinden, tune_classical.py ile aynı tanımlar."""
    text = str(text)
    length = len(text)
    words = text.split()
    wc = len(words)
    avg_word = sum(len(w) for w in words) / wc if wc else 0.0
    upper = sum(1 for c in text if c.isupper()) / length if length else 0.0
    blob = TextBlob(text)
    negation = sum(1 for w in text.lower().split() if w in _NEG_COUNT)
    all_caps = sum(1 for w in words if w.isupper() and len(w) > 1) / max(wc, 1)
    sent_count = max(len(re.split(r"[.!?]+", text)) - 1, 1)
    avg_sent_len = wc / max(sent_count, 1)
    uniq = len(set(text.lower().split())) / max(wc, 1)
    punct = sum(1 for c in text if c in ".,;:!?-") / max(length, 1)
    return [
        length, wc, text.count("!"), text.count("?"), avg_word, upper,
        blob.sentiment.polarity, blob.sentiment.subjectivity,
        negation, all_caps, sent_count, avg_sent_len, uniq, punct,
    ]


def bert_raw(text, summary=""):
    """DistilBERT/MiniLM için ham giriş — train script'leriyle aynı biçim."""
    s = str(summary).strip()
    t = str(text).strip()
    return (s + ". " + t).strip(". ").strip()


def encode_sequence(cleaned, word_index):
    seq = np.zeros((1, MAX_LEN), dtype=np.int32)
    for i, w in enumerate(cleaned.split()[:MAX_LEN]):
        seq[0, i] = word_index.get(w, 0)
    return seq


# ---------------------------------------------------------------- model deposu
class Engine:
    def __init__(self):
        self.vectorizer = None
        self.scaler = None
        self.config = {}
        self.weights = {"lr": 1/3, "svm": 1/3, "lgbm": 1/3}
        self.word_index = {}
        self.models = {}      # ad -> model
        self.status = {}      # ad -> bool
        self.feature_names = None
        self.lr_coef = None
        self._embedder = None     # tembel: SentenceTransformer
        self._bert_tok = None     # tembel: DistilBERT tokenizer
        self._bert_model = None   # tembel: DistilBERT model
        self._load()

    def _v2(self, name):
        return os.path.join(V2_DIR, name)

    def _load(self):
        # ortak bileşenler (v2)
        if os.path.exists(self._v2("tfidf.pkl")):
            self.vectorizer = joblib.load(self._v2("tfidf.pkl"))
            self.feature_names = np.asarray(self.vectorizer.get_feature_names_out())
        if os.path.exists(self._v2("scaler.pkl")):
            self.scaler = joblib.load(self._v2("scaler.pkl"))
        if os.path.exists(self._v2("config.json")):
            try:
                self.config = json.load(open(self._v2("config.json"), encoding="utf-8"))
                self.weights = self.config.get("weights", self.weights)
            except Exception:
                pass

        # klasik v2 modeller + blender
        for name, fname in [("lr", "lr.pkl"), ("svm", "svm.pkl"),
                            ("lgbm", "lgbm.pkl"), ("nb", "nb.pkl"),
                            ("_blender", "blender.pkl"),
                            ("_bert_lr", "bert_lr.pkl"), ("_fusion_lr", "fusion_lr.pkl")]:
            p = self._v2(fname)
            if os.path.exists(p):
                try:
                    self.models[name] = joblib.load(p)
                    self.status[name] = True
                except Exception:
                    self.status[name] = False
            else:
                self.status[name] = False

        # türetilmiş modeller hazır mı?
        self.status["ensemble"] = all(self.status.get(m) for m in ENSEMBLE_MEMBERS)
        self.status["stacking"] = self.status["ensemble"] and self.status.get("_blender", False)
        self.status["fusion"] = self.status.get("_fusion_lr", False)
        self.status["bert_emb"] = self.status.get("_bert_lr", False)
        self.status["distilbert"] = os.path.isdir(DISTILBERT_DIR) and (
            os.path.exists(os.path.join(DISTILBERT_DIR, "model.safetensors"))
            or os.path.exists(os.path.join(DISTILBERT_DIR, "model_int8.pt")))

        # eski derin modeller (.keras) — opsiyonel, varsa sun
        self.word_index = {}
        tok_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
        if os.path.exists(tok_path):
            try:
                tok = joblib.load(tok_path)
                self.word_index = tok.get("word_index", {}) if isinstance(tok, dict) else {}
            except Exception:
                self.word_index = {}
        for name, fname in [("lstm", "lstm_model.keras"), ("bilstm", "bilstm_model.keras")]:
            self.status[name] = os.path.exists(os.path.join(MODELS_DIR, fname))

        # LR katsayıları (kelime önemi / vurgu için)
        lr = self.models.get("lr")
        if lr is not None and hasattr(lr, "coef_"):
            self.lr_coef = np.atleast_2d(lr.coef_)

    # ---- tembel yükleyiciler ----
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(EMB_MODEL)
        return self._embedder

    def _embed(self, raw):
        emb = self._get_embedder().encode([raw], convert_to_numpy=True,
                                          normalize_embeddings=True)
        return emb.astype(np.float32)   # (1, 384)

    def _get_distilbert(self):
        if self._bert_model is None:
            from transformers import AutoTokenizer
            self._bert_tok = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
            safet = os.path.join(DISTILBERT_DIR, "model.safetensors")
            int8 = os.path.join(DISTILBERT_DIR, "model_int8.pt")
            if os.path.exists(safet):
                # tam (fp32) model
                from transformers import AutoModelForSequenceClassification
                self._bert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
            else:
                # küçültülmüş (int8) model — teslim paketinde bu kullanılır
                self._bert_model = torch.load(int8, map_location="cpu", weights_only=False)
            self._bert_model.eval()
        return self._bert_tok, self._bert_model

    def _get_keras(self, name):
        m = self.models.get(name)
        if m is None:
            from tensorflow.keras.models import load_model
            m = load_model(os.path.join(MODELS_DIR, f"{name}_model.keras"))
            self.models[name] = m
        return m

    # ---- yardımcılar ----
    def available(self):
        return [m for m in MODEL_ORDER if self.status.get(m)]

    def _classical_matrix(self, cleaned, num):
        tfidf = self.vectorizer.transform([cleaned])
        scaled = self.scaler.transform([num])
        return scipy.sparse.hstack([tfidf, scipy.sparse.csr_matrix(scaled)]).tocsr()

    def _proba(self, model_name, cleaned, num, raw):
        """Seçilen model için 3-sınıf olasılık vektörü döndürür."""
        if model_name == "nb":
            return self.models["nb"].predict_proba(self.vectorizer.transform([cleaned]))[0]

        if model_name in ("lr", "svm", "lgbm"):
            X = self._classical_matrix(cleaned, num)
            return self.models[model_name].predict_proba(X)[0]

        if model_name in ("ensemble", "stacking"):
            X = self._classical_matrix(cleaned, num)
            P = {m: self.models[m].predict_proba(X)[0] for m in ENSEMBLE_MEMBERS}
            if model_name == "ensemble":
                w = self.weights
                return sum(w.get(m, 1/3) * P[m] for m in ENSEMBLE_MEMBERS)
            meta_x = np.hstack([P["lr"], P["svm"], P["lgbm"]]).reshape(1, -1)
            return self.models["_blender"].predict_proba(meta_x)[0]

        if model_name == "bert_emb":
            return self.models["_bert_lr"].predict_proba(self._embed(raw))[0]

        if model_name == "fusion":
            tfidf = self.vectorizer.transform([cleaned])
            scaled = scipy.sparse.csr_matrix(self.scaler.transform([num]))
            emb = scipy.sparse.csr_matrix(self._embed(raw))
            X = scipy.sparse.hstack([tfidf, scaled, emb]).tocsr()
            return self.models["_fusion_lr"].predict_proba(X)[0]

        if model_name == "distilbert":
            tok, mdl = self._get_distilbert()
            enc = tok([raw], truncation=True, max_length=BERT_MAXLEN, return_tensors="pt")
            with torch.no_grad():
                logits = mdl(**enc).logits
            return torch.softmax(logits, dim=-1)[0].numpy()

        if model_name in SEQUENTIAL:
            m = self._get_keras(model_name)
            return m.predict(encode_sequence(cleaned, self.word_index), verbose=0)[0]

        raise ValueError(f"Bilinmeyen model: {model_name}")

    # ---- açıklama yardımcıları (v2 LR katsayıları) ----
    def _highlights(self, cleaned, pred_idx):
        if self.vectorizer is None or self.lr_coef is None or not cleaned:
            return []
        tokens = cleaned.split()
        vocab = self.vectorizer.vocabulary_
        row = self.vectorizer.transform([cleaned])
        coef = self.lr_coef[pred_idx]
        n = coef.shape[0]
        contrib = {}
        for tok in set(tokens):
            col = vocab.get(tok)
            contrib[tok] = (float(row[0, col]) * float(coef[col]), True) if (col is not None and col < n) else (0.0, False)
        peak = max((abs(c[0]) for c in contrib.values() if c[1]), default=0.0) or 1.0
        out = []
        for tok in tokens:
            score, known = contrib[tok]
            out.append({"token": tok, "known": known,
                        "intensity": round(score / peak, 3) if known else 0.0,
                        "direction": (1 if score > 0 else (-1 if score < 0 else 0))})
        return out

    def _influential_words(self, cleaned, pred_idx, limit=8):
        if self.vectorizer is None or not cleaned:
            return []
        row = self.vectorizer.transform([cleaned]).tocoo()
        items = sorted(zip(row.col, row.data), key=lambda kv: kv[1], reverse=True)[:limit]
        out = []
        for col, weight in items:
            direction = 0
            if self.lr_coef is not None and col < self.lr_coef.shape[1]:
                direction = 1 if self.lr_coef[pred_idx, col] >= 0 else -1
            out.append({"word": str(self.feature_names[col]),
                        "weight": round(float(weight), 3), "direction": direction})
        return out

    def top_words_per_class(self, limit=15):
        if self.lr_coef is None or self.feature_names is None:
            return {}
        n_text = len(self.feature_names)
        result = {}
        for idx in (0, 1, 2):
            row = self.lr_coef[idx][:n_text]
            order = np.argsort(row)[::-1]
            words = []
            for j in order:
                name = self.feature_names[j]
                if " " in name:
                    continue
                words.append([str(name), round(float(row[j]), 3)])
                if len(words) >= limit:
                    break
            result[SLUGS[idx]] = {"label": LABELS[idx], "words": words}
        return result

    # ---- tahmin ----
    def predict(self, text, model_name, summary=""):
        if not self.status.get(model_name):
            raise ValueError(f"Model yüklü değil: {model_name}")

        body_cleaned = clean_text(text)
        sum_cleaned = clean_text(summary) if summary else ""
        cleaned = (sum_cleaned + " " + body_cleaned).strip() if sum_cleaned else body_cleaned
        num = numeric_features(text)
        raw = bert_raw(text, summary)

        proba = np.asarray(self._proba(model_name, cleaned, num, raw), dtype=float)
        pred_idx = int(np.argmax(proba))

        return {
            "model": model_name,
            "model_title": MODEL_TITLES[model_name],
            "prediction": pred_idx,
            "label": LABELS[pred_idx],
            "slug": SLUGS[pred_idx],
            "confidence": round(float(proba[pred_idx]) * 100, 2),
            "probabilities": {
                "negatif": round(float(proba[0]) * 100, 2),
                "notr": round(float(proba[1]) * 100, 2),
                "pozitif": round(float(proba[2]) * 100, 2),
            },
            "influential_words": self._influential_words(cleaned, pred_idx),
            "highlights": self._highlights(cleaned, pred_idx),
            "stats": {
                "char_count": num[0], "word_count": num[1],
                "polarity": round(num[6], 3), "subjectivity": round(num[7], 3),
            },
            "cleaned_text": cleaned,
        }
