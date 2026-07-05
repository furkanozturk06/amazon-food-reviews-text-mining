#!/usr/bin/env python3
"""Amazon Fine Food Reviews - Metin Madenciligi Proje Notebook Olusturucu.

7 moduler Jupyter Notebook uretir:
  01 Veri Hazirlama
  02 Kesfsel Veri Analizi (EDA)
  03 Metin On Isleme
  04 Ozellik Cikarimi
  05 Model Egitimi
  06 Model Degerlendirme
  07 Varlik Bazli Duygu Analizi (ABSA)
"""

import json, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))


def _write(name, cells):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13.0"},
        },
        "cells": cells,
    }
    with open(os.path.join(BASE, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"  [OK] {name}")


def md(src):
    lines = src.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
    }


def code(src):
    lines = src.split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
    }


# ----------
#  NOTEBOOK 1 — Veri Hazirlama
# ----------
def nb01():
    _write("01_veri_hazirlama.ipynb", [

        md("# 1. Amazon Veri Hazırlama ve Temizleme\nBu notebook, Amazon Fine Food Reviews ham CSV verisinden filtrelenmiş ve dengelenmiş bir CSV oluşturur."),

        code(
"import os\n"
"import pandas as pd\n"
"import numpy as np\n"
"from sklearn.utils import resample\n"
"import config\n"
"\n"
"print('=== Klasörler Oluşturuluyor ===')\n"
"for d in ['data', 'models', 'features', 'results']:\n"
"    os.makedirs(d, exist_ok=True)"
        ),

        code(
"print('\\n=== ADIM 1: Veriyi Okuma ===')\n"
"df = pd.read_csv(config.DATA_FILE)\n"
"toplam_ham = len(df)\n"
"print(f'Toplam satır sayısı: {len(df):,}')\n"
"print(f'Kolonlar: {list(df.columns)}')\n"
"df.head(3)"
        ),

        code(
"print('\\n=== ADIM 2: Veri Temizliği ===')\n"
"df = df.dropna(subset=['Text'])\n"
"df = df[df['Text'].astype(str).str.strip() != '']\n"
"df = df[df['Text'].str.len() >= 10]\n"
"\n"
"initial = len(df)\n"
"df = df.drop_duplicates(subset=['UserId', 'Text'])\n"
"print(f'Duplikat kaldırma: {initial:,} -> {len(df):,} ({initial - len(df):,} silindi)')\n"
"print(f'Temizlik sonrası kalan: {len(df):,}')"
        ),

        code(
"print('\\n=== ADIM 3: Yıldız -> 3-Sınıf Dönüşümü ===')\n"
"print('Orijinal Score dağılımı:')\n"
"print(df['Score'].value_counts().sort_index())\n"
"\n"
"def map_score(s):\n"
"    if s in [1, 2]: return 0\n"
"    elif s == 3:    return 1\n"
"    elif s in [4, 5]: return 2\n"
"    return -1\n"
"\n"
"df['label'] = df['Score'].apply(map_score)\n"
"df = df[df['label'] != -1]\n"
"\n"
"df = df.rename(columns={'Text': 'text', 'Summary': 'summary'})\n"
"df = df[['Id','ProductId','UserId','HelpfulnessNumerator',\n"
"         'HelpfulnessDenominator','Score','Time','summary','text','label']]\n"
"print('Dönüşüm tamamlandı.')"
        ),

        code(
"print('\\n=== ADIM 4: Undersampling ===')\n"
"print('Dengeleme Öncesi:')\n"
"cc = df['label'].value_counts()\n"
"for lbl, cnt in cc.items():\n"
"    print(f'  {config.CLASS_NAMES[lbl]} (Sınıf {lbl}): {cnt:,} (%{cnt/len(df)*100:.2f})')\n"
"\n"
"min_c = cc.min()\n"
"print(f'\\nEn az sınıf ({min_c:,}) baz alınarak undersampling...')\n"
"\n"
"parts = []\n"
"for lbl in cc.index:\n"
"    parts.append(resample(df[df['label']==lbl], replace=False,\n"
"                          n_samples=min_c, random_state=config.RANDOM_STATE))\n"
"df_balanced = pd.concat(parts).sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)\n"
"\n"
"print('\\nDengeleme Sonrası:')\n"
"for lbl, cnt in df_balanced['label'].value_counts().items():\n"
"    print(f'  {config.CLASS_NAMES[lbl]} (Sınıf {lbl}): {cnt:,} (%{cnt/len(df_balanced)*100:.2f})')"
        ),

        code(
"print('\\n=== ADIM 5: Kaydetme ===')\n"
"df_balanced.to_csv('data/reviews_cleaned.csv', index=False)\n"
"print('Kaydedildi: data/reviews_cleaned.csv')"
        ),

        code(
"print('\\n' + '='*50)\n"
"print('VERİ SETİ İSTATİSTİK RAPORU'.center(50))\n"
"print('='*50)\n"
"print(f'Ham veri satır sayısı           : {toplam_ham:,}')\n"
"print(f'Dengelenmiş veri satır sayısı   : {len(df_balanced):,}')\n"
"print(f'Benzersiz Ürün Sayısı           : {df_balanced[\"ProductId\"].nunique():,}')\n"
"print(f'Benzersiz Kullanıcı Sayısı      : {df_balanced[\"UserId\"].nunique():,}')\n"
"print(f'Ortalama Yorum Uzunluğu         : {df_balanced[\"text\"].str.len().mean():.0f} karakter')\n"
"print('='*50)"
        ),
    ])


# ----------
#  NOTEBOOK 2 — Kesfsel Veri Analizi (EDA)
# ----------
def nb02():
    _write("02_kesfsel_analiz.ipynb", [

        md("# 2. Keşifsel Veri Analizi (EDA)\nBu notebook, temizlenmiş Amazon yorum verileri üzerinde görselleştirme ve istatistiksel analiz yapar."),

        code(
"import os\n"
"import pandas as pd\n"
"import numpy as np\n"
"%matplotlib inline\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"from wordcloud import WordCloud, STOPWORDS\n"
"from sklearn.feature_extraction.text import CountVectorizer\n"
"from datetime import datetime\n"
"\n"
"os.makedirs('results', exist_ok=True)\n"
"print('Veri yükleniyor...')\n"
"df = pd.read_csv('data/reviews_cleaned.csv')\n"
"df['text'] = df['text'].astype(str)\n"
"# Amazon yorumlarındaki HTML etiketlerini (<br /> vb.) temizle; aksi halde\n"
"# kelime bulutu ve en sık kelimelerde 'br' gibi gürültü öne çıkar.\n"
"df['text'] = df['text'].str.replace(r'<[^>]+>', ' ', regex=True)\n"
"df['text'] = df['text'].str.replace(r'\\s+', ' ', regex=True).str.strip()\n"
"df['review_length'] = df['text'].str.len()\n"
"df['word_count'] = df['text'].apply(lambda x: len(x.split()))\n"
"df['timestamp'] = pd.to_datetime(df['Time'], unit='s')\n"
"df['year'] = df['timestamp'].dt.year\n"
"\n"
"class_names = {0: 'Negatif', 1: 'Nötr', 2: 'Pozitif'}\n"
"df['class_name'] = df['label'].map(class_names)\n"
"\n"
"sns.set_theme(style='whitegrid')\n"
"plt.rcParams.update({'figure.figsize': (12, 8), 'figure.dpi': 150})"
        ),

        md("## 2.1 Genel Dağılım Grafikleri"),

        code(
"plt.figure()\n"
"sns.countplot(data=df, x='class_name', order=['Negatif','Nötr','Pozitif'],\n"
"             palette='Set2', hue='class_name', legend=False)\n"
"plt.title('Sınıf Dağılımı (Dengelenmiş)')\n"
"plt.xlabel('Sınıf')\n"
"plt.ylabel('Yorum Sayısı')\n"
"plt.tight_layout()\n"
"plt.savefig('results/eda_class_distribution.png')\n"
"plt.show()"
        ),

        code(
"plt.figure()\n"
"sns.countplot(data=df, x='Score', palette='viridis', hue='Score', legend=False)\n"
"plt.title('Orijinal Yıldız Dağılımı')\n"
"plt.xlabel('Yıldız Puanı')\n"
"plt.ylabel('Yorum Sayısı')\n"
"plt.tight_layout()\n"
"plt.savefig('results/eda_star_distribution.png')\n"
"plt.show()"
        ),

        code(
"plt.figure()\n"
"sns.histplot(data=df, x='review_length', hue='class_name', bins=50,\n"
"             kde=True, palette='Set2', alpha=0.5)\n"
"plt.xlim(0, 3000)\n"
"plt.title('Yorum Uzunluğu Dağılımı (Karakter)')\n"
"plt.xlabel('Karakter Sayısı')\n"
"plt.ylabel('Frekans')\n"
"plt.tight_layout()\n"
"plt.savefig('results/eda_review_length.png')\n"
"plt.show()"
        ),

        code(
"plt.figure()\n"
"sns.boxplot(data=df, x='class_name', y='word_count', palette='Set2',\n"
"            showfliers=False, hue='class_name', legend=False)\n"
"plt.title('Sınıflara Göre Kelime Sayısı Dağılımı')\n"
"plt.xlabel('Sınıf')\n"
"plt.ylabel('Kelime Sayısı')\n"
"plt.tight_layout()\n"
"plt.savefig('results/eda_word_count.png')\n"
"plt.show()"
        ),

        md("## 2.2 Metin Analizi"),

        code(
"sample_size = min(100000, len(df))\n"
"df_sample = df.sample(n=sample_size, random_state=42)\n"
"\n"
"# Ham frekansta en sık kelimeler her sınıfta aynı çıkıyor (taste, flavor, coffee...)\n"
"# ve bulutlar birbirine benziyordu. Bunun yerine 'sınıfa özgü' kelimeleri buluyoruz:\n"
"# bir kelimenin o sınıftaki görülme oranı ile diğer sınıflardaki oranının farkı.\n"
"ek_stop = {'food','good','great','one','will','just','like','really','product',\n"
"           'buy','get','would','also','much','even','br','don','ve','time','use'}\n"
"_cv = CountVectorizer(stop_words='english', max_features=4000, min_df=20)\n"
"_X = _cv.fit_transform(df_sample['text'])\n"
"_vocab = np.array(_cv.get_feature_names_out())\n"
"_ys = df_sample['label'].values\n"
"\n"
"def sinifa_ozgu(label, n=100):\n"
"    ic = np.asarray(_X[_ys == label].mean(axis=0)).ravel()\n"
"    dis = np.asarray(_X[_ys != label].mean(axis=0)).ravel()\n"
"    skor = ic - dis\n"
"    sonuc = {}\n"
"    for i in skor.argsort()[::-1]:\n"
"        if skor[i] <= 0:\n"
"            break\n"
"        w = _vocab[i]\n"
"        if w in ek_stop:\n"
"            continue\n"
"        sonuc[w] = float(skor[i])\n"
"        if len(sonuc) >= n:\n"
"            break\n"
"    return sonuc"
        ),

        code(
"# Sınıfa göre renk: Negatif kırmızımsı, Nötr sarımsı, Pozitif yeşilimsi\n"
"renk_haritasi = {0: 'Reds', 1: 'Blues', 2: 'Greens'}\n"
"for label, name in class_names.items():\n"
"    freqs = sinifa_ozgu(label, 100)\n"
"    wc = WordCloud(width=800, height=400, background_color='white',\n"
"                   colormap=renk_haritasi[label], max_words=100)\n"
"    wc.generate_from_frequencies(freqs)\n"
"\n"
"    plt.figure()\n"
"    plt.imshow(wc, interpolation='bilinear')\n"
"    plt.axis('off')\n"
"    plt.title(f'Sınıfa Özgü Kelimeler - {name}', fontsize=20)\n"
"    plt.tight_layout()\n"
"    fname = f'eda_wordcloud_{name.lower().replace(\"ö\",\"o\").replace(\"ü\",\"u\")}.png'\n"
"    plt.savefig(os.path.join('results', fname))\n"
"    plt.show()"
        ),

        code(
"plt.figure(figsize=(18, 6))\n"
"for i, (label, name) in enumerate(class_names.items(), 1):\n"
"    plt.subplot(1, 3, i)\n"
"    d = sinifa_ozgu(label, 15)\n"
"    kelimeler = list(d.keys())[::-1]\n"
"    degerler = [d[w] for w in kelimeler]\n"
"    plt.barh(kelimeler, degerler, color=sns.color_palette('Set2')[i-1])\n"
"    plt.title(f'Sınıfa Özgü 15 Kelime - {name}')\n"
"    plt.xlabel('Ayırt edicilik (oran farkı)')\n"
"plt.tight_layout()\n"
"plt.savefig('results/eda_top_words.png')\n"
"plt.show()"
        ),

        code(
"# Bigram'larda da sınıfa özgü (ayırt edici) ikilileri gösteriyoruz.\n"
"_cvb = CountVectorizer(stop_words='english', ngram_range=(2, 2), max_features=3000, min_df=10)\n"
"_Xb = _cvb.fit_transform(df_sample['text'])\n"
"_vb = np.array(_cvb.get_feature_names_out())\n"
"\n"
"plt.figure(figsize=(18, 6))\n"
"for i, (label, name) in enumerate(class_names.items(), 1):\n"
"    plt.subplot(1, 3, i)\n"
"    ic = np.asarray(_Xb[_ys == label].mean(axis=0)).ravel()\n"
"    dis = np.asarray(_Xb[_ys != label].mean(axis=0)).ravel()\n"
"    skor = ic - dis\n"
"    order = skor.argsort()[-10:]\n"
"    plt.barh([_vb[j] for j in order], [skor[j] for j in order],\n"
"             color=sns.color_palette('Set2')[i-1])\n"
"    plt.title(f'Sınıfa Özgü 10 Bigram - {name}')\n"
"    plt.xlabel('Ayırt edicilik')\n"
"plt.tight_layout()\n"
"plt.savefig('results/eda_bigrams.png')\n"
"plt.show()"
        ),

        md("## 2.3 Amazon'a Özgü Analizler"),

        code(
"# Helpfulness (Faydalılık) Analizi\n"
"df['help_ratio'] = df.apply(\n"
"    lambda r: r['HelpfulnessNumerator']/r['HelpfulnessDenominator']\n"
"              if r['HelpfulnessDenominator'] > 0 else None, axis=1)\n"
"\n"
"df_help = df[df['HelpfulnessDenominator'] >= 5].copy()\n"
"df_help['help_bucket'] = pd.cut(df_help['help_ratio'],\n"
"    bins=[-0.01, 0.0, 0.5, 0.99, 1.01],\n"
"    labels=['0%', '0-50%', '50-99%', '100%'])\n"
"\n"
"help_sent = df_help.groupby('help_bucket', observed=True)['label'].mean().reset_index()\n"
"\n"
"plt.figure(figsize=(10, 6))\n"
"sns.barplot(x='help_bucket', y='label', data=help_sent, palette='coolwarm')\n"
"plt.title('Faydalılık Oranına Göre Ortalama Duygu Skoru')\n"
"plt.xlabel('Helpful Oy Yüzdesi')\n"
"plt.ylabel('Ortalama Sınıf (0=Negatif, 2=Pozitif)')\n"
"plt.tight_layout()\n"
"plt.savefig('results/eda_helpfulness.png')\n"
"plt.show()"
        ),

        code(
"# Summary vs Text Uzunluk Karşılaştırması\n"
"df['summary'] = df['summary'].astype(str)\n"
"df['summary_len'] = df['summary'].str.len()\n"
"\n"
"plt.figure(figsize=(10, 6))\n"
"summary_means = df.groupby('class_name')[['review_length','summary_len']].mean()\n"
"summary_means.plot(kind='bar', rot=0, color=['steelblue','coral'])\n"
"plt.title('Sınıflara Göre Ortalama Metin ve Özet Uzunluğu')\n"
"plt.ylabel('Karakter Sayısı')\n"
"plt.legend(['Yorum Metni', 'Özet (Summary)'])\n"
"plt.tight_layout()\n"
"plt.savefig('results/eda_summary_vs_text.png')\n"
"plt.show()"
        ),

        code(
"# Yıllara Göre Yorum Trendi\n"
"plt.figure()\n"
"yearly = df.groupby(['year','class_name']).size().reset_index(name='count')\n"
"sns.lineplot(data=yearly, x='year', y='count', hue='class_name',\n"
"             marker='o', palette='Set2')\n"
"plt.title('Yıllara Göre Yorum Sayısı Trendi')\n"
"plt.xlabel('Yıl')\n"
"plt.ylabel('Yorum Sayısı')\n"
"plt.tight_layout()\n"
"plt.savefig('results/eda_time_trend.png')\n"
"plt.show()"
        ),

        md("## 2.4 Özet İstatistikler"),

        code(
"print('\\n' + '='*50)\n"
"print('SINIFLARA GÖRE ÖZET İSTATİSTİKLER'.center(50))\n"
"print('='*50)\n"
"summary = df.groupby('class_name')['word_count'].agg(['mean','std','min','max'])\n"
"print(summary)\n"
"print('='*50)"
        ),

        md("## 2.5 Konu Modellemesi (LDA)\nDenetimsiz (unsupervised) **Latent Dirichlet Allocation** ile yorumlardaki gizli konuları keşfediyoruz."),

        code(
"from sklearn.decomposition import LatentDirichletAllocation\n"
"\n"
"print('LDA için 100.000 rastgele örnek seçiliyor...')\n"
"df_lda = df.sample(n=min(100000, len(df)), random_state=42)\n"
"texts = df_lda['text'].astype(str).values\n"
"\n"
"print('CountVectorizer oluşturuluyor...')\n"
"tf_vec = CountVectorizer(max_df=0.95, min_df=2, max_features=10000,\n"
"                         stop_words='english')\n"
"tf_matrix = tf_vec.fit_transform(texts)\n"
"\n"
"print('LDA modeli eğitiliyor...')\n"
"lda = LatentDirichletAllocation(n_components=5, max_iter=10,\n"
"    learning_method='online', random_state=42, n_jobs=-1)\n"
"lda.fit(tf_matrix)\n"
"\n"
"print('\\n--- Bulunan Konuların En Önemli 10 Kelimesi ---')\n"
"feat_names = tf_vec.get_feature_names_out()\n"
"for idx, topic in enumerate(lda.components_):\n"
"    top = topic.argsort()[:-11:-1]\n"
"    words = [feat_names[i] for i in top]\n"
"    print(f'Topic {idx+1}: {\" | \".join(words)}')\n"
"\n"
"try:\n"
"    import pyLDAvis\n"
"    import pyLDAvis.lda_model\n"
"    pyLDAvis.enable_notebook()\n"
"    panel = pyLDAvis.lda_model.prepare(lda, tf_matrix, tf_vec)\n"
"    pyLDAvis.save_html(panel, 'results/lda_interactive_plot.html')\n"
"    print('\\nLDA grafiği results/lda_interactive_plot.html olarak kaydedildi!')\n"
"    display(panel)\n"
"except ImportError:\n"
"    print('pyLDAvis kurulu değil, interaktif grafik atlanıyor.')"
        ),
    ])


# ----------
#  NOTEBOOK 3 — Metin On Isleme
# ----------
def nb03():
    _write("03_metin_on_isleme.ipynb", [

        md("# 3. Metin Ön İşleme\nBu notebook, NLP modelleri için metin temizleme ve ekstra özellik çıkarımı adımlarını içerir."),

        code(
"import pandas as pd\n"
"import numpy as np\n"
"import re\n"
"import os\n"
"import joblib\n"
"%matplotlib inline\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"from tqdm.auto import tqdm\n"
"import nltk\n"
"from textblob import TextBlob\n"
"from nltk.corpus import stopwords\n"
"from nltk.tokenize import word_tokenize\n"
"from nltk.stem import WordNetLemmatizer\n"
"\n"
"nltk.download('punkt')\n"
"nltk.download('punkt_tab')\n"
"nltk.download('stopwords')\n"
"nltk.download('wordnet')\n"
"nltk.download('averaged_perceptron_tagger')\n"
"\n"
"tqdm.pandas()\n"
"\n"
"print('Veri yükleniyor...')\n"
"df = pd.read_csv('data/reviews_cleaned.csv')\n"
"print(f'Veri yüklendi, satır sayısı: {len(df):,}')"
        ),

        md("## 3.1 Ek Feature Üretimi"),

        code(
"print('Feature\\'lar çıkarılıyor...')\n"
"df['text'] = df['text'].astype(str)\n"
"\n"
"df['review_length'] = df['text'].str.len()\n"
"df['word_count'] = df['text'].apply(lambda x: len(x.split()))\n"
"df['exclamation_count'] = df['text'].str.count('!')\n"
"df['question_count'] = df['text'].str.count(r'\\?')\n"
"\n"
"def avg_word_len(text):\n"
"    words = text.split()\n"
"    if len(words) == 0: return 0\n"
"    return sum(len(w) for w in words) / len(words)\n"
"\n"
"df['avg_word_length'] = df['text'].progress_apply(avg_word_len)\n"
"\n"
"def upper_ratio(text):\n"
"    if len(text) == 0: return 0\n"
"    return sum(1 for c in text if c.isupper()) / len(text)\n"
"\n"
"df['uppercase_ratio'] = df['text'].progress_apply(upper_ratio)\n"
"\n"
"def get_sentiment(text):\n"
"    blob = TextBlob(text)\n"
"    return pd.Series([blob.sentiment.polarity, blob.sentiment.subjectivity])\n"
"\n"
"print('TextBlob sentiment hesaplanıyor...')\n"
"df[['sentiment_polarity','sentiment_subjectivity']] = df['text'].progress_apply(get_sentiment)\n"
"\n"
"print('\\nSınıflara göre feature ortalamaları:')\n"
"features = ['review_length','word_count','exclamation_count','question_count',\n"
"            'avg_word_length','uppercase_ratio','sentiment_polarity','sentiment_subjectivity']\n"
"print(df.groupby('label')[features].mean())"
        ),

        md("## 3.2 Metin Temizleme Fonksiyonu"),

        code(
"stop_words = set(stopwords.words('english'))\n"
"# Olumsuzlama kelimelerini KORU: 'not good' temizlikte 'good'a dönüşmesin diye\n"
"# bu kelimeleri stopword listesinden çıkarıyoruz. Böylece TF-IDF bigram'ları\n"
"# ('not good', 'didn like' vb.) olumsuzluğu yakalayabiliyor.\n"
"negation_words = {'no','not','nor','never','none','nothing','neither','without','cannot',\n"
"                  'don','aren','couldn','didn','doesn','hadn','hasn','haven','isn',\n"
"                  'mightn','mustn','needn','shan','shouldn','wasn','weren','won','wouldn','ain'}\n"
"stop_words = stop_words - negation_words\n"
"lemmatizer = WordNetLemmatizer()\n"
"\n"
"def clean_text(text):\n"
"    text = text.lower()\n"
"    text = re.sub(r'<.*?>', '', text)\n"
"    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text, flags=re.MULTILINE)\n"
"    text = re.sub(r'&[a-z]+;', ' ', text)\n"
"    text = text.encode('ascii', 'ignore').decode('ascii')\n"
"    text = re.sub(r'[^\\w\\s]', ' ', text)\n"
"    text = re.sub(r'\\d+', '', text)\n"
"    text = re.sub(r'\\s+', ' ', text).strip()\n"
"    tokens = word_tokenize(text)\n"
"    cleaned = [lemmatizer.lemmatize(w) for w in tokens\n"
"               if w not in stop_words and len(w) >= 2]\n"
"    return ' '.join(cleaned)"
        ),

        md("## 3.3 Önce / Sonra Karşılaştırması"),

        code(
"samples = df['text'].sample(5, random_state=42).tolist()\n"
"cleaned = [clean_text(t) for t in samples]\n"
"\n"
"comp = pd.DataFrame({'Orijinal Metin': samples, 'Temizlenmiş Metin': cleaned})\n"
"pd.set_option('display.max_colwidth', None)\n"
"display(comp)"
        ),

        md("## 3.4 Tüm Veriye Uygulama"),

        code(
"print('Tüm metinler temizleniyor (bu adım uzun sürebilir)...')\n"
"df['cleaned_text'] = df['text'].progress_apply(clean_text)\n"
"\n"
"initial_len = len(df)\n"
"df = df[df['cleaned_text'].str.strip() != '']\n"
"df = df.dropna(subset=['cleaned_text'])\n"
"dropped = initial_len - len(df)\n"
"print(f'\\nBoş kalan satır: {dropped}')\n"
"print(f'Kalan veri boyutu: {len(df):,}')"
        ),

        md("## 3.5 Feature İstatistikleri"),

        code(
"print('Feature istatistikleri:')\n"
"stats = df.groupby('label')[features].agg(['mean','std']).round(3)\n"
"display(stats)\n"
"\n"
"plt.figure(figsize=(10, 6))\n"
"sns.violinplot(data=df, x='label', y='sentiment_polarity', palette='Set2')\n"
"plt.title('Sınıflara Göre Sentiment Polarity')\n"
"plt.xlabel('Sınıf (0=Negatif, 1=Nötr, 2=Pozitif)')\n"
"plt.ylabel('Polarity')\n"
"plt.tight_layout()\n"
"os.makedirs('results', exist_ok=True)\n"
"plt.savefig('results/preprocessing_features.png')\n"
"plt.show()"
        ),

        md("## 3.6 Kaydet"),

        code(
"output_csv = 'data/reviews_preprocessed.csv'\n"
"df.to_csv(output_csv, index=False)\n"
"\n"
"os.makedirs('models', exist_ok=True)\n"
"joblib.dump(clean_text, 'models/preprocessor.pkl')\n"
"\n"
"print(f'Kaydedildi: {output_csv}')\n"
"print('Kaydedildi: models/preprocessor.pkl')\n"
"display(df.head())"
        ),
    ])


# ----------
#  NOTEBOOK 4 — Ozellik Cikarimi (Feature Extraction)
# ----------
def nb04():
    _write("04_ozellik_cikarimi.ipynb", [

        md("# 4. Özellik Çıkarımı (Feature Extraction)\nBu notebook, temizlenmiş veriden ML ve DL modelleri için özellik vektörleri (TF-IDF ve Sequence) oluşturur."),

        code(
"import pandas as pd\n"
"import numpy as np\n"
"import joblib\n"
"import os\n"
"import gc\n"
"from collections import Counter\n"
"from sklearn.feature_extraction.text import TfidfVectorizer\n"
"from sklearn.preprocessing import StandardScaler\n"
"from sklearn.model_selection import train_test_split\n"
"import scipy.sparse\n"
"\n"
"print('Veri yükleniyor...')\n"
"use_cols = ['label','cleaned_text','review_length','word_count',\n"
"            'exclamation_count','question_count','avg_word_length',\n"
"            'uppercase_ratio','sentiment_polarity','sentiment_subjectivity']\n"
"df = pd.read_csv('data/reviews_preprocessed.csv', usecols=use_cols)\n"
"df = df.dropna(subset=['cleaned_text'])\n"
"print(f'Veri yüklendi, Shape: {df.shape}')"
        ),

        md("## 4.1 TF-IDF (Klasik Modeller İçin)"),

        code(
"print('TF-IDF dönüşümü uygulanıyor...')\n"
"tfidf_vectorizer = TfidfVectorizer(\n"
"    max_features=50000,\n"
"    ngram_range=(1, 2),\n"
"    sublinear_tf=True,\n"
"    min_df=2,\n"
"    max_df=0.95,\n"
"    dtype=np.float32\n"
")\n"
"X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_text'])\n"
"print(f'TF-IDF Shape: {X_tfidf.shape}')\n"
"\n"
"print('\\nEn Sık 20 TF-IDF Özelliği:')\n"
"feat_names = np.array(tfidf_vectorizer.get_feature_names_out())\n"
"sums = X_tfidf.sum(axis=0).A1\n"
"top_idx = sums.argsort()[-20:][::-1]\n"
"for i, idx in enumerate(top_idx, 1):\n"
"    print(f'{i}. {feat_names[idx]} ({sums[idx]:.2f})')"
        ),

        code(
"print('Sayısal özellikler normalize ediliyor...')\n"
"numeric_features = [\n"
"    'review_length','word_count','exclamation_count','question_count',\n"
"    'avg_word_length','uppercase_ratio','sentiment_polarity','sentiment_subjectivity'\n"
"]\n"
"scaler = StandardScaler()\n"
"X_numeric = scaler.fit_transform(df[numeric_features])\n"
"\n"
"print('TF-IDF ve sayısal özellikler birleştiriliyor...')\n"
"X_combined = scipy.sparse.hstack([X_tfidf, X_numeric]).tocsr()\n"
"print(f'Final Shape: {X_combined.shape}')\n"
"\n"
"del X_tfidf, X_numeric, X_combined\n"
"gc.collect()"
        ),

        code(
"os.makedirs('models', exist_ok=True)\n"
"joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')\n"
"joblib.dump(scaler, 'models/scaler.pkl')\n"
"print('TF-IDF ve Scaler kaydedildi.')"
        ),

        md("## 4.2 Sequence Encoding (Derin Öğrenme İçin)"),

        code(
"print('Tokenizer eğitiliyor...')\n"
"NUM_WORDS = 50000\n"
"MAX_LEN = 200\n"
"\n"
"word_counter = Counter()\n"
"for text in df['cleaned_text']:\n"
"    word_counter.update(str(text).split())\n"
"\n"
"most_common = word_counter.most_common(NUM_WORDS - 1)\n"
"word_index = {word: i+1 for i, (word, _) in enumerate(most_common)}\n"
"vocab_size = len(word_index) + 1\n"
"print(f'Vocab Size: {vocab_size}')\n"
"\n"
"tokenizer = {'word_index': word_index, 'num_words': NUM_WORDS}"
        ),

        code(
"print('Metinler sequence\\'lara dönüştürülüyor...')\n"
"X_seq = np.zeros((len(df), MAX_LEN), dtype=np.int32)\n"
"\n"
"for i, text in enumerate(df['cleaned_text']):\n"
"    words = str(text).split()\n"
"    seq = [word_index.get(w, 0) for w in words]\n"
"    length = min(len(seq), MAX_LEN)\n"
"    X_seq[i, :length] = seq[:length]\n"
"\n"
"print(f'Sequence Shape: {X_seq.shape}')\n"
"for i in range(3):\n"
"    print(f'Örnek {i+1}: {X_seq[i][:15]}... (uzunluk: {len(X_seq[i])})')"
        ),

        code(
"os.makedirs('features', exist_ok=True)\n"
"joblib.dump(tokenizer, 'models/tokenizer.pkl')\n"
"np.save('features/sequences.npy', X_seq)\n"
"print('Sequence ve Tokenizer kaydedildi.')"
        ),

        md("## 4.3 Train / Validation / Test Split"),

        code(
"print('Veri seti bölünüyor (%70 Train, %15 Val, %15 Test)...')\n"
"indices = np.arange(len(df))\n"
"y = df['label'].values\n"
"\n"
"idx_train, idx_temp, y_train, y_temp = train_test_split(\n"
"    indices, y, test_size=0.30, random_state=42, stratify=y)\n"
"\n"
"idx_val, idx_test, y_val, y_test = train_test_split(\n"
"    idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)\n"
"\n"
"print(f'\\nEğitim (Train)     : {len(idx_train):,}')\n"
"print(f'Doğrulama (Val)    : {len(idx_val):,}')\n"
"print(f'Test               : {len(idx_test):,}')\n"
"\n"
"def print_dist(y_sub, name):\n"
"    u, c = np.unique(y_sub, return_counts=True)\n"
"    d = {int(k): f'%{v/len(y_sub)*100:.1f}' for k, v in zip(u, c)}\n"
"    print(f'{name:10} Dağılım: {d}')\n"
"\n"
"print_dist(y_train, 'Train')\n"
"print_dist(y_val, 'Val')\n"
"print_dist(y_test, 'Test')"
        ),

        code(
"np.save('features/train_idx.npy', idx_train)\n"
"np.save('features/val_idx.npy', idx_val)\n"
"np.save('features/test_idx.npy', idx_test)\n"
"np.save('features/labels.npy', y)\n"
"print('Split indeksleri kaydedildi.')\n"
"\n"
"pd.DataFrame({\n"
"    'Set': ['Train','Validation','Test'],\n"
"    'Örnek Sayısı': [len(idx_train), len(idx_val), len(idx_test)],\n"
"    'Oran (%)': [len(idx_train)/len(y)*100, len(idx_val)/len(y)*100, len(idx_test)/len(y)*100]\n"
"})"
        ),
    ])


# ----------
#  NOTEBOOK 5 — Model Egitimi
# ----------
def nb05():
    _write("05_model_egitimi.ipynb", [

        md("# 5. Model Eğitimi\nBu notebook, TF-IDF ve Sequence vektörlerini kullanarak klasik ML ve derin öğrenme modellerini eğitir."),

        code(
"import pandas as pd\n"
"import numpy as np\n"
"import joblib\n"
"import os\n"
"import time\n"
"import json\n"
"import gc\n"
"%matplotlib inline\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"from sklearn.linear_model import LogisticRegression\n"
"from sklearn.svm import LinearSVC\n"
"from sklearn.calibration import CalibratedClassifierCV\n"
"from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score\n"
"import scipy.sparse\n"
"\n"
"TF_OK = False\n"
"try:\n"
"    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n"
"    import tensorflow as tf\n"
"    from tensorflow.keras.models import Sequential\n"
"    from tensorflow.keras.layers import (Embedding, SpatialDropout1D, LSTM,\n"
"        Dense, Dropout, Bidirectional)\n"
"    from tensorflow.keras.optimizers import Adam\n"
"    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n"
"    TF_OK = True\n"
"    print('TensorFlow yüklendi.')\n"
"except ImportError:\n"
"    print('TensorFlow yüklenemedi. LSTM/BiLSTM modelleri atlanacak.')\n"
"\n"
"print('Veriler yükleniyor...')\n"
"df = pd.read_csv('data/reviews_preprocessed.csv',\n"
"    usecols=['cleaned_text','label','review_length','word_count',\n"
"             'exclamation_count','question_count','avg_word_length',\n"
"             'uppercase_ratio','sentiment_polarity','sentiment_subjectivity'])\n"
"df = df.dropna(subset=['cleaned_text'])\n"
"\n"
"tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')\n"
"scaler = joblib.load('models/scaler.pkl')\n"
"\n"
"numeric_features = ['review_length','word_count','exclamation_count','question_count',\n"
"                    'avg_word_length','uppercase_ratio','sentiment_polarity','sentiment_subjectivity']\n"
"\n"
"print('TF-IDF + Numeric birleştiriliyor...')\n"
"X_tfidf = tfidf_vectorizer.transform(df['cleaned_text'])\n"
"X_numeric = scaler.transform(df[numeric_features])\n"
"X_combined = scipy.sparse.hstack([X_tfidf, X_numeric]).tocsr()\n"
"del X_tfidf, X_numeric; gc.collect()\n"
"\n"
"print('Sequence verileri yükleniyor...')\n"
"X_seq = np.load('features/sequences.npy')\n"
"\n"
"idx_train = np.load('features/train_idx.npy')\n"
"idx_val   = np.load('features/val_idx.npy')\n"
"idx_test  = np.load('features/test_idx.npy')\n"
"y         = np.load('features/labels.npy')\n"
"\n"
"X_train_tfidf = X_combined[idx_train]\n"
"X_val_tfidf   = X_combined[idx_val]\n"
"X_test_tfidf  = X_combined[idx_test]\n"
"X_train_seq   = X_seq[idx_train]\n"
"X_val_seq     = X_seq[idx_val]\n"
"X_test_seq    = X_seq[idx_test]\n"
"y_train = y[idx_train]\n"
"y_val   = y[idx_val]\n"
"y_test  = y[idx_test]\n"
"\n"
"print(f'X_train_tfidf: {X_train_tfidf.shape}, y_train: {y_train.shape}')"
        ),

        md("## 5.1 Model 1 — Logistic Regression"),

        code(
"print('Logistic Regression eğitimi...')\n"
"start = time.time()\n"
"best_lr = LogisticRegression(C=1, solver='lbfgs', max_iter=500,\n"
"                             class_weight='balanced', n_jobs=-1)\n"
"best_lr.fit(X_train_tfidf, y_train)\n"
"train_time_lr = time.time() - start\n"
"print(f'Eğitim süresi: {train_time_lr:.2f} s')\n"
"\n"
"os.makedirs('models', exist_ok=True)\n"
"joblib.dump(best_lr, 'models/lr_model.pkl')\n"
"\n"
"y_pred_val_lr = best_lr.predict(X_val_tfidf)\n"
"print(f'\\nVal Accuracy: {accuracy_score(y_val, y_pred_val_lr):.4f}')\n"
"print(f'Val F1 Macro: {f1_score(y_val, y_pred_val_lr, average=\"macro\"):.4f}')"
        ),

        md("## 5.2 Model 2 — SVM (LinearSVC)"),

        code(
"print('SVM eğitimi...')\n"
"start = time.time()\n"
"base_svm = LinearSVC(C=0.1, max_iter=2000, dual=False, class_weight='balanced')\n"
"calibrated_svm = CalibratedClassifierCV(base_svm, cv=3)\n"
"calibrated_svm.fit(X_train_tfidf, y_train)\n"
"train_time_svm = time.time() - start\n"
"print(f'Eğitim süresi: {train_time_svm:.2f} s')\n"
"\n"
"joblib.dump(calibrated_svm, 'models/svm_model.pkl')\n"
"\n"
"y_pred_val_svm = calibrated_svm.predict(X_val_tfidf)\n"
"print(f'\\nVal Accuracy: {accuracy_score(y_val, y_pred_val_svm):.4f}')\n"
"print(f'Val F1 Macro: {f1_score(y_val, y_pred_val_svm, average=\"macro\"):.4f}')"
        ),

        md("## 5.2.5 Derin Öğrenme İçin Alt Küme Seçimi\nDerin öğrenme GPU olmadan tüm veriyle çok uzun sürer. Eğitim için rastgele bir alt küme seçiyoruz."),

        code(
"np.random.seed(42)\n"
"DL_TRAIN = 45000\n"
"DL_VAL = 10000\n"
"\n"
"train_sub = np.random.choice(len(y_train), DL_TRAIN, replace=False)\n"
"val_sub   = np.random.choice(len(y_val), DL_VAL, replace=False)\n"
"\n"
"X_train_seq_sub = X_train_seq[train_sub]\n"
"y_train_sub     = y_train[train_sub]\n"
"X_val_seq_sub   = X_val_seq[val_sub]\n"
"y_val_sub       = y_val[val_sub]\n"
"\n"
"print(f'DL Eğitim Seti: {X_train_seq_sub.shape}')\n"
"print(f'DL Doğrulama Seti: {X_val_seq_sub.shape}')"
        ),

        md("## 5.3 Model 3 — LSTM"),

        code(
"model_lstm = None\n"
"model_bilstm = None\n"
"\n"
"if TF_OK:\n"
"    VOCAB_SIZE = 50000\n"
"    MAX_LEN = 200\n"
"    NUM_CLASSES = 3\n"
"\n"
"    print('LSTM Modeli tanımlanıyor...')\n"
"    # mask_zero: dolgu adımları yok sayılır; clipnorm + düşük öğrenme oranı\n"
"    # modelin tek sınıfa çökmesini önler. Dropout değerleri overfit'i sınırlar.\n"
"    model_lstm = Sequential([\n"
"        Embedding(VOCAB_SIZE, 64, mask_zero=True),\n"
"        SpatialDropout1D(0.3),\n"
"        LSTM(64, dropout=0.2),\n"
"        Dense(32, activation='relu'),\n"
"        Dropout(0.4),\n"
"        Dense(NUM_CLASSES, activation='softmax')\n"
"    ])\n"
"    model_lstm.compile(optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),\n"
"                       loss='sparse_categorical_crossentropy',\n"
"                       metrics=['accuracy'])\n"
"\n"
"    callbacks = [\n"
"        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),\n"
"        ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-5, verbose=1)\n"
"    ]\n"
"\n"
"    start = time.time()\n"
"    history_lstm = model_lstm.fit(\n"
"        X_train_seq_sub, y_train_sub,\n"
"        validation_data=(X_val_seq_sub, y_val_sub),\n"
"        epochs=12, batch_size=128, callbacks=callbacks)\n"
"    print(f'LSTM Eğitim süresi: {time.time()-start:.2f} s')\n"
"\n"
"    model_lstm.save('models/lstm_model.keras')\n"
"    os.makedirs('results', exist_ok=True)\n"
"    with open('results/history_lstm.json','w') as f:\n"
"        json.dump(history_lstm.history, f)\n"
"\n"
"    y_pred_val_lstm = np.argmax(model_lstm.predict(X_val_seq), axis=1)\n"
"    print(f'Val Accuracy: {accuracy_score(y_val, y_pred_val_lstm):.4f}')\n"
"    print(f'Val F1 Macro: {f1_score(y_val, y_pred_val_lstm, average=\"macro\"):.4f}')\n"
"else:\n"
"    print('TensorFlow yok, LSTM atlanıyor.')"
        ),

        md("## 5.4 Model 4 — BiLSTM\nBidirectional LSTM, metni hem ileri hem geri yönde okuyarak daha zengin bağlam çıkarır."),

        code(
"if TF_OK:\n"
"    print('BiLSTM Modeli tanımlanıyor...')\n"
"    model_bilstm = Sequential([\n"
"        Embedding(VOCAB_SIZE, 64, mask_zero=True),\n"
"        SpatialDropout1D(0.3),\n"
"        Bidirectional(LSTM(64, dropout=0.2)),\n"
"        Dense(32, activation='relu'),\n"
"        Dropout(0.4),\n"
"        Dense(NUM_CLASSES, activation='softmax')\n"
"    ])\n"
"    model_bilstm.compile(optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),\n"
"                         loss='sparse_categorical_crossentropy',\n"
"                         metrics=['accuracy'])\n"
"\n"
"    start = time.time()\n"
"    history_bilstm = model_bilstm.fit(\n"
"        X_train_seq_sub, y_train_sub,\n"
"        validation_data=(X_val_seq_sub, y_val_sub),\n"
"        epochs=12, batch_size=128, callbacks=callbacks)\n"
"    print(f'BiLSTM Eğitim süresi: {time.time()-start:.2f} s')\n"
"\n"
"    model_bilstm.save('models/bilstm_model.keras')\n"
"    with open('results/history_bilstm.json','w') as f:\n"
"        json.dump(history_bilstm.history, f)\n"
"\n"
"    y_pred_val_bilstm = np.argmax(model_bilstm.predict(X_val_seq), axis=1)\n"
"    print(f'Val Accuracy: {accuracy_score(y_val, y_pred_val_bilstm):.4f}')\n"
"    print(f'Val F1 Macro: {f1_score(y_val, y_pred_val_bilstm, average=\"macro\"):.4f}')\n"
"else:\n"
"    print('TensorFlow yok, BiLSTM atlanıyor.')"
        ),

        md("## 5.5 Model 5 — LightGBM\nGradient Boosting tabanlı güçlü bir ensemble modeli."),

        code(
"import lightgbm as lgb\n"
"\n"
"print('LightGBM eğitimi...')\n"
"model_lgbm = lgb.LGBMClassifier(\n"
"    n_estimators=500,\n"
"    num_leaves=63,\n"
"    learning_rate=0.1,\n"
"    class_weight='balanced',\n"
"    n_jobs=-1,\n"
"    random_state=42\n"
")\n"
"\n"
"start = time.time()\n"
"model_lgbm.fit(X_train_tfidf, y_train,\n"
"    eval_set=[(X_val_tfidf, y_val)],\n"
"    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=True)])\n"
"train_time_lgbm = time.time() - start\n"
"print(f'\\nLightGBM Eğitim süresi: {train_time_lgbm:.2f} s')\n"
"\n"
"joblib.dump(model_lgbm, 'models/lgbm_model.pkl')\n"
"\n"
"y_pred_val_lgbm = model_lgbm.predict(X_val_tfidf)\n"
"print(f'Val Accuracy: {accuracy_score(y_val, y_pred_val_lgbm):.4f}')\n"
"print(f'Val F1 Macro: {f1_score(y_val, y_pred_val_lgbm, average=\"macro\"):.4f}')"
        ),

        md("## 5.6 Model Karşılaştırma Özeti"),

        code(
"print('Tüm modellerin TEST seti sonuçları...')\n"
"\n"
"y_test_lr     = best_lr.predict(X_test_tfidf)\n"
"y_test_svm    = calibrated_svm.predict(X_test_tfidf)\n"
"y_test_lgbm   = model_lgbm.predict(X_test_tfidf)\n"
"\n"
"def get_metrics(y_true, y_pred):\n"
"    return {\n"
"        'Accuracy':  accuracy_score(y_true, y_pred),\n"
"        'Precision': precision_score(y_true, y_pred, average='macro'),\n"
"        'Recall':    recall_score(y_true, y_pred, average='macro'),\n"
"        'F1-Macro':  f1_score(y_true, y_pred, average='macro')\n"
"    }\n"
"\n"
"models = {\n"
"    'Logistic Regression': get_metrics(y_test, y_test_lr),\n"
"    'SVM':                 get_metrics(y_test, y_test_svm),\n"
"    'LightGBM':            get_metrics(y_test, y_test_lgbm),\n"
"}\n"
"\n"
"if model_lstm is not None:\n"
"    y_test_lstm = np.argmax(model_lstm.predict(X_test_seq), axis=1)\n"
"    models['LSTM'] = get_metrics(y_test, y_test_lstm)\n"
"if model_bilstm is not None:\n"
"    y_test_bilstm = np.argmax(model_bilstm.predict(X_test_seq), axis=1)\n"
"    models['BiLSTM'] = get_metrics(y_test, y_test_bilstm)\n"
"\n"
"comparison_df = pd.DataFrame(models).T.round(4)\n"
"comparison_df.to_csv('results/model_comparison.csv')\n"
"display(comparison_df)\n"
"\n"
"best_name = comparison_df['F1-Macro'].idxmax()\n"
"best_f1   = comparison_df['F1-Macro'].max()\n"
"print(f'\\nEn iyi model (F1-Macro={best_f1:.4f}): {best_name}')\n"
"with open('results/best_model.txt','w') as f:\n"
"    f.write(f'Best Model: {best_name}\\nF1-Macro: {best_f1}')"
        ),
    ])


# ----------
#  NOTEBOOK 6 — Model Degerlendirme
# ----------
def nb06():
    _write("06_model_degerlendirme.ipynb", [

        md("# 6. Model Değerlendirme\nBu notebook, eğitilen 5 modelin performanslarını Confusion Matrix, ROC, Precision-Recall, Radar Chart, LIME ile kapsamlı değerlendirir."),

        code(
"import pandas as pd\n"
"import numpy as np\n"
"import joblib\n"
"import os\n"
"import json\n"
"%matplotlib inline\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"from sklearn.metrics import (confusion_matrix, roc_curve, auc,\n"
"    precision_recall_curve, average_precision_score,\n"
"    accuracy_score, precision_score, recall_score, f1_score)\n"
"from sklearn.preprocessing import label_binarize\n"
"import scipy.sparse\n"
"\n"
"os.makedirs('results', exist_ok=True)\n"
"\n"
"print('Modeller yükleniyor...')\n"
"best_lr       = joblib.load('models/lr_model.pkl')\n"
"calibrated_svm = joblib.load('models/svm_model.pkl')\n"
"model_lgbm    = joblib.load('models/lgbm_model.pkl')\n"
"\n"
"model_lstm = None\n"
"model_bilstm = None\n"
"try:\n"
"    import os as _os; _os.environ['TF_CPP_MIN_LOG_LEVEL']='3'\n"
"    from tensorflow.keras.models import load_model\n"
"    if os.path.exists('models/lstm_model.keras'):\n"
"        model_lstm = load_model('models/lstm_model.keras')\n"
"    if os.path.exists('models/bilstm_model.keras'):\n"
"        model_bilstm = load_model('models/bilstm_model.keras')\n"
"except ImportError:\n"
"    print('TensorFlow yüklenemedi, DL modelleri atlanacak.')\n"
"\n"
"print('Test verileri yükleniyor...')\n"
"df = pd.read_csv('data/reviews_preprocessed.csv',\n"
"    usecols=['cleaned_text','label','review_length','word_count',\n"
"             'exclamation_count','question_count','avg_word_length',\n"
"             'uppercase_ratio','sentiment_polarity','sentiment_subjectivity'])\n"
"df = df.dropna(subset=['cleaned_text'])\n"
"\n"
"tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')\n"
"scaler_obj = joblib.load('models/scaler.pkl')\n"
"\n"
"idx_test = np.load('features/test_idx.npy')\n"
"y = np.load('features/labels.npy')\n"
"y_test = y[idx_test]\n"
"\n"
"numeric_features = ['review_length','word_count','exclamation_count','question_count',\n"
"                    'avg_word_length','uppercase_ratio','sentiment_polarity','sentiment_subjectivity']\n"
"X_tfidf = tfidf_vectorizer.transform(df['cleaned_text'])\n"
"X_num   = scaler_obj.transform(df[numeric_features])\n"
"X_combined = scipy.sparse.hstack([X_tfidf, X_num]).tocsr()\n"
"X_test_tfidf = X_combined[idx_test]\n"
"\n"
"X_seq = np.load('features/sequences.npy')\n"
"X_test_seq = X_seq[idx_test]\n"
"texts_test = df['cleaned_text'].iloc[idx_test].values\n"
"\n"
"print('Tahminler hesaplanıyor...')\n"
"y_probs_lr     = best_lr.predict_proba(X_test_tfidf)\n"
"y_probs_svm    = calibrated_svm.predict_proba(X_test_tfidf)\n"
"y_probs_lgbm   = model_lgbm.predict_proba(X_test_tfidf)\n"
"\n"
"model_probs = {\n"
"    'Logistic Regression': y_probs_lr,\n"
"    'SVM': y_probs_svm,\n"
"    'LightGBM': y_probs_lgbm\n"
"}\n"
"if model_lstm is not None:\n"
"    model_probs['LSTM'] = model_lstm.predict(X_test_seq)\n"
"if model_bilstm is not None:\n"
"    model_probs['BiLSTM'] = model_bilstm.predict(X_test_seq)\n"
"model_preds = {n: np.argmax(p, axis=1) for n, p in model_probs.items()}\n"
"class_names = ['Negatif', 'Nötr', 'Pozitif']\n"
"print('Hazır!')"
        ),

        md("## 6.1 Confusion Matrix"),

        code(
"fig, axes = plt.subplots(1, 5, figsize=(25, 5))\n"
"for ax, (name, y_pred) in zip(axes, model_preds.items()):\n"
"    cm = confusion_matrix(y_test, y_pred)\n"
"    cm_n = confusion_matrix(y_test, y_pred, normalize='true')\n"
"    annot = np.empty_like(cm, dtype=object)\n"
"    for i in range(3):\n"
"        for j in range(3):\n"
"            annot[i,j] = f'{cm[i,j]}\\n({cm_n[i,j]:.1%})'\n"
"    sns.heatmap(cm_n, annot=annot, fmt='', cmap='Blues',\n"
"                xticklabels=class_names, yticklabels=class_names,\n"
"                ax=ax, cbar=False)\n"
"    ax.set_title(f'{name}')\n"
"    ax.set_xlabel('Tahmin'); ax.set_ylabel('Gerçek')\n"
"plt.tight_layout()\n"
"plt.savefig('results/confusion_matrices_all.png', bbox_inches='tight')\n"
"plt.show()"
        ),

        md("## 6.2 ROC Curve"),

        code(
"y_test_bin = label_binarize(y_test, classes=[0,1,2])\n"
"colors = ['blue','green','red','purple','orange']\n"
"linestyles = ['-','--','-.',':','-']\n"
"\n"
"fig, axes = plt.subplots(1, 3, figsize=(21, 6))\n"
"for i, cname in enumerate(class_names):\n"
"    ax = axes[i]\n"
"    for (name, probs), col, ls in zip(model_probs.items(), colors, linestyles):\n"
"        fpr, tpr, _ = roc_curve(y_test_bin[:,i], probs[:,i])\n"
"        roc_auc = auc(fpr, tpr)\n"
"        ax.plot(fpr, tpr, color=col, linestyle=ls, lw=2,\n"
"                label=f'{name} (AUC={roc_auc:.3f})')\n"
"    ax.plot([0,1],[0,1],'k--',lw=2,label='Rastgele')\n"
"    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])\n"
"    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')\n"
"    ax.set_title(f'ROC: {cname}')\n"
"    ax.legend(loc='lower right')\n"
"plt.tight_layout()\n"
"plt.savefig('results/roc_curves_all.png', bbox_inches='tight')\n"
"plt.show()"
        ),

        md("## 6.3 Precision-Recall Curve"),

        code(
"fig, axes = plt.subplots(1, 3, figsize=(21, 6))\n"
"for i, cname in enumerate(class_names):\n"
"    ax = axes[i]\n"
"    baseline = np.sum(y_test_bin[:,i]) / len(y_test_bin[:,i])\n"
"    for (name, probs), col, ls in zip(model_probs.items(), colors, linestyles):\n"
"        prec, rec, _ = precision_recall_curve(y_test_bin[:,i], probs[:,i])\n"
"        ap = average_precision_score(y_test_bin[:,i], probs[:,i])\n"
"        ax.plot(rec, prec, color=col, linestyle=ls, lw=2,\n"
"                label=f'{name} (AP={ap:.3f})')\n"
"    ax.axhline(y=baseline, color='k', linestyle='--', label=f'Baseline ({baseline:.3f})')\n"
"    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])\n"
"    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')\n"
"    ax.set_title(f'PR: {cname}')\n"
"    ax.legend(loc='lower left')\n"
"plt.tight_layout()\n"
"plt.savefig('results/pr_curves_all.png', bbox_inches='tight')\n"
"plt.show()"
        ),

        md("## 6.4 Model Karşılaştırma ve Radar Grafiği"),

        code(
"def metrics_dict(yt, yp):\n"
"    return {\n"
"        'Accuracy':  accuracy_score(yt, yp),\n"
"        'Precision': precision_score(yt, yp, average='macro'),\n"
"        'Recall':    recall_score(yt, yp, average='macro'),\n"
"        'F1-Macro':  f1_score(yt, yp, average='macro')\n"
"    }\n"
"\n"
"metrics_data = {n: metrics_dict(y_test, p) for n, p in model_preds.items()}\n"
"df_metrics = pd.DataFrame(metrics_data).T.reset_index().rename(columns={'index':'Model'})"
        ),

        code(
"categories = ['Accuracy','Precision','Recall','F1-Macro']\n"
"N = len(categories)\n"
"angles = [n/N * 2 * np.pi for n in range(N)]\n"
"angles += angles[:1]\n"
"\n"
"plt.figure(figsize=(8, 8))\n"
"ax = plt.subplot(111, polar=True)\n"
"ax.set_theta_offset(np.pi/2)\n"
"ax.set_theta_direction(-1)\n"
"plt.xticks(angles[:-1], categories)\n"
"plt.yticks([0.2,0.4,0.6,0.8,1.0], ['0.2','0.4','0.6','0.8','1.0'], color='grey', size=10)\n"
"plt.ylim(0, 1)\n"
"\n"
"for (name, m), col, ls in zip(metrics_data.items(), colors, linestyles):\n"
"    vals = [m[c] for c in categories] + [m[categories[0]]]\n"
"    ax.plot(angles, vals, color=col, linewidth=2, linestyle=ls, label=name)\n"
"    ax.fill(angles, vals, color=col, alpha=0.1)\n"
"\n"
"plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))\n"
"plt.title('Model Performansları Radar Grafiği', size=15, y=1.1)\n"
"plt.savefig('results/model_comparison_radar.png', bbox_inches='tight')\n"
"plt.show()"
        ),

        md("## 6.5 Training History (Derin Öğrenme)"),

        code(
"dl_models = ['lstm', 'bilstm']\n"
"titles = ['LSTM', 'BiLSTM']\n"
"\n"
"fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n"
"for i, (key, title) in enumerate(zip(dl_models, titles)):\n"
"    hfile = f'results/history_{key}.json'\n"
"    if os.path.exists(hfile):\n"
"        with open(hfile) as f:\n"
"            h = json.load(f)\n"
"        axes[i,0].plot(h['loss'], label='Train Loss')\n"
"        axes[i,0].plot(h['val_loss'], label='Val Loss')\n"
"        axes[i,0].set_title(f'{title} - Loss'); axes[i,0].legend()\n"
"        axes[i,1].plot(h['accuracy'], label='Train Acc')\n"
"        axes[i,1].plot(h['val_accuracy'], label='Val Acc')\n"
"        axes[i,1].set_title(f'{title} - Accuracy'); axes[i,1].legend()\n"
"plt.tight_layout()\n"
"plt.savefig('results/training_histories.png', bbox_inches='tight')\n"
"plt.show()"
        ),

        md("## 6.6 Hata Analizi"),

        code(
"best_model_name = df_metrics.loc[df_metrics['F1-Macro'].idxmax(), 'Model']\n"
"print(f'Hata analizi modeli: {best_model_name}')\n"
"\n"
"y_pred_best = model_preds[best_model_name]\n"
"y_probs_best = model_probs[best_model_name]\n"
"\n"
"errors = y_test != y_pred_best\n"
"print(f'Toplam Test: {len(y_test):,}')\n"
"print(f'Hatalı Tahmin: {errors.sum():,} (%{errors.mean()*100:.2f})')\n"
"\n"
"err_cm = confusion_matrix(y_test[errors], y_pred_best[errors])\n"
"plt.figure(figsize=(6, 5))\n"
"sns.heatmap(err_cm, annot=True, fmt='d', cmap='Reds',\n"
"            xticklabels=class_names, yticklabels=class_names)\n"
"plt.title(f'Hata Dağılımı - {best_model_name}')\n"
"plt.xlabel('Tahmin'); plt.ylabel('Gerçek')\n"
"plt.savefig('results/error_analysis.png', bbox_inches='tight')\n"
"plt.show()"
        ),

        code(
"conf_probs = np.max(y_probs_best[errors], axis=1)\n"
"err_idx = np.where(errors)[0]\n"
"sorted_err = err_idx[np.argsort(conf_probs)[::-1]]\n"
"\n"
"print('\\n--- En Yüksek Güvenle YANLIŞ Tahmin Edilen 10 Örnek ---')\n"
"rows = []\n"
"for idx in sorted_err[:10]:\n"
"    rows.append([\n"
"        str(texts_test[idx])[:100] + '...',\n"
"        class_names[y_test[idx]],\n"
"        class_names[y_pred_best[idx]],\n"
"        f'%{np.max(y_probs_best[idx])*100:.1f}'\n"
"    ])\n"
"display(pd.DataFrame(rows, columns=['Yorum','Gerçek','Tahmin','Güven']))"
        ),

        md("## 6.7 Final Özet"),

        code(
"df_metrics.set_index('Model', inplace=True)\n"
"styled = df_metrics.style.highlight_max(axis=0, color='lightgreen').format('{:.4f}')\n"
"display(styled)\n"
"\n"
"df_metrics.to_csv('results/final_evaluation.csv')\n"
"print('Değerlendirme tamamlandı! -> results/final_evaluation.csv')"
        ),

        md("## 6.8 Örnek Tahmin Karşılaştırması"),

        code(
"np.random.seed(42)\n"
"sample_idx = np.random.choice(len(texts_test), 10, replace=False)\n"
"\n"
"rows = []\n"
"for idx in sample_idx:\n"
"    row = {\n"
"        'Yorum (150 kar.)': str(texts_test[idx])[:150] + '...',\n"
"        'Gerçek': class_names[y_test[idx]],\n"
"    }\n"
"    for mname in model_preds:\n"
"        short = mname.replace('Logistic Regression','LR')\n"
"        row[short] = class_names[model_preds[mname][idx]]\n"
"    rows.append(row)\n"
"\n"
"df_s = pd.DataFrame(rows)\n"
"def hl(val, true):\n"
"    if val == true: return 'background-color: lightgreen'\n"
"    if val in class_names: return 'background-color: lightcoral'\n"
"    return ''\n"
"\n"
"styled = df_s.style.apply(lambda x: [hl(v, x['Gerçek']) for v in x], axis=1)\n"
"display(styled)"
        ),

        md("## 6.9 Açıklanabilir Yapay Zeka (XAI) — LIME\nModelimizin bir yorumu sınıflandırırken hangi kelimelere dikkat ettiğini **LIME** ile görselleştiriyoruz."),

        code(
"from lime.lime_text import LimeTextExplainer\n"
"\n"
"np.random.seed(42)\n"
"sample_i = np.random.choice(len(texts_test))\n"
"sample_text = str(texts_test[sample_i])\n"
"true_label = class_names[y_test[sample_i]]\n"
"\n"
"orig_numeric = X_test_tfidf[sample_i, -len(numeric_features):].toarray()\n"
"\n"
"def lime_predict(texts_list):\n"
"    tfidf_m = tfidf_vectorizer.transform(texts_list)\n"
"    num_m = np.repeat(orig_numeric, len(texts_list), axis=0)\n"
"    combined = scipy.sparse.hstack([tfidf_m, scipy.sparse.csr_matrix(num_m)])\n"
"    return best_lr.predict_proba(combined)\n"
"\n"
"print('LIME açıklaması hesaplanıyor...')\n"
"explainer = LimeTextExplainer(class_names=class_names)\n"
"exp = explainer.explain_instance(sample_text, lime_predict,\n"
"                                 num_features=10, top_labels=1)\n"
"\n"
"print(f'Yorum: {sample_text[:200]}...')\n"
"print(f'Gerçek: {true_label}')\n"
"print(f'Tahmin: {class_names[np.argmax(lime_predict([sample_text])[0])]}')\n"
"\n"
"exp.save_to_file('results/lime_explanation.html')\n"
"print('LIME -> results/lime_explanation.html')\n"
"exp.show_in_notebook(text=True)"
        ),
    ])


# ----------
#  NOTEBOOK 7 — ABSA
# ----------
def nb07():
    _write("07_varlik_bazli_duygu_analizi.ipynb", [

        md("# 7. Varlık Bazlı Duygu Analizi (Aspect-Based Sentiment Analysis)\n\nBu bölümde önceden eğittiğimiz modeli kullanarak yorumları **Tat, Kalite, Fiyat, Ambalaj, Doku** gibi ürün özelliklerine göre parçalayıp duygu analizi yapıyoruz.\n\nDışarıdan hazır model (BERT vb.) kullanılmamaktadır; tamamen sıfırdan eğittiğimiz modeller ile çıkarım yapılır."),

        code(
"import pandas as pd\n"
"import numpy as np\n"
"import joblib\n"
"import nltk\n"
"import scipy.sparse\n"
"%matplotlib inline\n"
"import matplotlib.pyplot as plt\n"
"import seaborn as sns\n"
"import warnings\n"
"\n"
"warnings.filterwarnings('ignore')\n"
"nltk.download('punkt', quiet=True)\n"
"nltk.download('punkt_tab', quiet=True)\n"
"from nltk.tokenize import sent_tokenize"
        ),

        md("## 7.1 Verilerin ve Modellerin Yüklenmesi"),

        code(
"print('Modeller yükleniyor...')\n"
"tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')\n"
"scaler_obj = joblib.load('models/scaler.pkl')\n"
"lr_model = joblib.load('models/lr_model.pkl')\n"
"print('Modeller yüklendi!')\n"
"\n"
"print('\\nVeri yükleniyor (10.000 rastgele satır)...')\n"
"df = pd.read_csv('data/reviews_preprocessed.csv').dropna(subset=['cleaned_text'])\n"
"df = df.sample(n=min(10000, len(df)), random_state=42).reset_index(drop=True)\n"
"print(f'Örnek veri boyutu: {df.shape}')"
        ),

        md("## 7.2 Aspect (Varlık) Sözlüğü\nAmazon gıda ürünleri için 5 temel varlık kategorisi tanımlıyoruz."),

        code(
"aspect_keywords = {\n"
"    'Tat/Lezzet': ['taste','flavor','delicious','bland','spicy','sweet','sour',\n"
"                   'bitter','yummy','savory','tasteless','flavorful','tasty'],\n"
"    'Kalite':     ['quality','fresh','stale','expire','organic','natural',\n"
"                   'artificial','processed','ingredient','genuine','pure'],\n"
"    'Fiyat/Değer':['price','expensive','cheap','worth','value','cost',\n"
"                   'money','affordable','overpriced','bargain','deal'],\n"
"    'Ambalaj':    ['package','packaging','box','bag','seal','shipping',\n"
"                   'delivery','arrived','damaged','broken','wrapped','container'],\n"
"    'Doku/Kıvam': ['texture','crunchy','soft','hard','chewy','smooth',\n"
"                   'creamy','dry','moist','crispy','thick','thin']\n"
"}\n"
"\n"
"def get_aspects(sentence):\n"
"    s = sentence.lower()\n"
"    return [asp for asp, kws in aspect_keywords.items()\n"
"            if any(kw in s for kw in kws)]"
        ),

        md("## 7.3 Cümlelere Bölme ve Çıkarım"),

        code(
"numeric_cols = ['review_length','word_count','exclamation_count','question_count',\n"
"                'avg_word_length','uppercase_ratio','sentiment_polarity',\n"
"                'sentiment_subjectivity']\n"
"X_num_scaled = scaler_obj.transform(df[numeric_cols])\n"
"\n"
"aspect_results = []\n"
"print('Cümleler analiz ediliyor...')\n"
"for idx, row in df.iterrows():\n"
"    sentences = sent_tokenize(str(row['cleaned_text']))\n"
"    for sentence in sentences:\n"
"        aspects = get_aspects(sentence)\n"
"        if aspects:\n"
"            tfidf_m = tfidf_vectorizer.transform([sentence])\n"
"            num_m = X_num_scaled[idx].reshape(1, -1)\n"
"            combined = scipy.sparse.hstack([tfidf_m, scipy.sparse.csr_matrix(num_m)])\n"
"            pred = lr_model.predict(combined)[0]\n"
"            label_map = {0: 'Negatif', 1: 'Nötr', 2: 'Pozitif'}\n"
"            for asp in aspects:\n"
"                aspect_results.append({\n"
"                    'review_id': idx,\n"
"                    'aspect': asp,\n"
"                    'sentence': sentence,\n"
"                    'sentiment': label_map[pred]\n"
"                })\n"
"\n"
"df_asp = pd.DataFrame(aspect_results)\n"
"print(f'Toplam {len(df_asp)} adet Varlık-Cümle eşleşmesi bulundu.')\n"
"df_asp.head(10)"
        ),

        md("## 7.4 Sonuçların Görselleştirilmesi"),

        code(
"plt.figure(figsize=(12, 6))\n"
"sns.countplot(data=df_asp, x='aspect', hue='sentiment',\n"
"              palette={'Negatif':'red', 'Nötr':'gray', 'Pozitif':'green'})\n"
"plt.title('Ürün Varlıklarına (Aspects) Göre Duygu Dağılımı')\n"
"plt.xlabel('Varlık (Aspect)')\n"
"plt.ylabel('Cümle Sayısı')\n"
"plt.legend(title='Tahmin')\n"
"plt.tight_layout()\n"
"import os\n"
"os.makedirs('results', exist_ok=True)\n"
"plt.savefig('results/aspect_based_sentiment.png')\n"
"plt.show()"
        ),

        code(
"# Aspect bazında ortalama skor\n"
"score_map = {'Negatif': 0, 'Nötr': 1, 'Pozitif': 2}\n"
"df_asp['score'] = df_asp['sentiment'].map(score_map)\n"
"\n"
"aspect_avg = df_asp.groupby('aspect')['score'].mean().sort_values()\n"
"\n"
"plt.figure(figsize=(10, 6))\n"
"colors = ['red' if v < 1 else 'orange' if v < 1.5 else 'green' for v in aspect_avg.values]\n"
"aspect_avg.plot(kind='barh', color=colors)\n"
"plt.title('Ürün Varlıklarına Göre Ortalama Duygu Skoru')\n"
"plt.xlabel('Ortalama Skor (0=Negatif, 1=Nötr, 2=Pozitif)')\n"
"plt.ylabel('Varlık')\n"
"plt.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)\n"
"plt.tight_layout()\n"
"plt.savefig('results/aspect_avg_score.png')\n"
"plt.show()\n"
"\n"
"print('\\nVarlık Bazlı Duygu Özeti:')\n"
"summary = df_asp.groupby(['aspect','sentiment']).size().unstack(fill_value=0)\n"
"display(summary)"
        ),
    ])


# ----------
#  EK ANALİZLER  (tek-notebook sürümünden taşınan, 3-sınıfa uyarlanmış)
#  Mevcut notebook'lara çıktıları kaybetmeden eklenir (inject_existing).
# ----------
EXTRA_MARKER = "# === EK ANALİZ"


def extras_02():
    """NB02 EDA ekleri: korelasyon, zamansal (saat/gün), aktif kullanıcı, Zipf."""
    return [
        md("## 2.6 Ek Keşifsel Analizler\nKorelasyon matrisi, saatlik/günlük duygu trendi, en aktif kullanıcılar ve Zipf yasası."),

        code(
"""# === EK ANALİZ: Korelasyon Matrisi ===
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
dfc = pd.read_csv('data/reviews_cleaned.csv')
dfc['text'] = dfc['text'].astype(str)
dfc['review_length'] = dfc['text'].str.len()
dfc['word_count'] = dfc['text'].str.split().apply(len)
dfc['help_ratio'] = (dfc['HelpfulnessNumerator'] / dfc['HelpfulnessDenominator'].replace(0, np.nan)).fillna(0)
corr_cols = ['label','Score','review_length','word_count',
             'HelpfulnessNumerator','HelpfulnessDenominator','help_ratio']
corr = dfc[corr_cols].corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Sayısal Özellikler Korelasyon Matrisi')
plt.tight_layout(); plt.savefig('results/eda_correlation.png'); plt.show()"""
        ),

        code(
"""# === EK ANALİZ: Aylık ve Günlük Duygu Trendi ===
# Not: Time alanı yalnızca tarihi tutar (saat her zaman 00:00), bu yüzden saat
# bazlı analiz anlamlı değildir; aylık ve haftalık eğilime bakıyoruz.
dfc['timestamp'] = pd.to_datetime(dfc['Time'], unit='s')
dfc['ay'] = dfc['timestamp'].dt.month
dfc['dow'] = dfc['timestamp'].dt.dayofweek
aylar = ['Oca','Şub','Mar','Nis','May','Haz','Tem','Ağu','Eyl','Eki','Kas','Ara']
gun = ['Pzt','Sal','Çar','Per','Cum','Cmt','Paz']
ort = dfc['label'].mean()
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
aylik = dfc.groupby('ay')['label'].mean()
ax[0].plot([aylar[i-1] for i in aylik.index], aylik.values, marker='o', color='steelblue')
ax[0].axhline(ort, color='red', ls='--', alpha=0.6, label='Genel ortalama')
ax[0].set_title('Aya Göre Ortalama Duygu (0=Negatif .. 2=Pozitif)')
ax[0].set_xlabel('Ay'); ax[0].set_ylabel('Ortalama duygu'); ax[0].legend()
daily = dfc.groupby('dow')['label'].mean()
renk = ['green' if v >= ort else 'red' for v in daily.values]
ax[1].bar([gun[i] for i in daily.index], daily.values, color=renk)
ax[1].axhline(ort, color='black', ls='--', alpha=0.5)
ax[1].set_title('Haftanın Gününe Göre Ortalama Duygu'); ax[1].set_ylabel('Ortalama duygu')
plt.tight_layout(); plt.savefig('results/eda_temporal_hour_day.png'); plt.show()"""
        ),

        code(
"""# === EK ANALİZ: En Aktif Kullanıcılar (yoğun yorumcu / spam göstergesi) ===
top_users = dfc['UserId'].value_counts().head(20)
plt.figure(figsize=(12, 5))
sns.barplot(x=top_users.index, y=top_users.values, palette='rocket',
            hue=top_users.index, legend=False)
plt.xticks(rotation=75, fontsize=8)
plt.title('En Çok Yorum Yapan 20 Kullanıcı')
plt.xlabel('UserId'); plt.ylabel('Yorum sayısı')
plt.tight_layout(); plt.savefig('results/eda_top_users.png'); plt.show()
print(f'En aktif kullanıcı: {top_users.index[0]} -> {top_users.iloc[0]:,} yorum')"""
        ),

        code(
"""# === EK ANALİZ: Zipf Yasası ve Kelime Çeşitliliği ===
from collections import Counter
samp = dfc.sample(min(30000, len(dfc)), random_state=42)
freq = Counter(' '.join(samp['text'].str.lower()).split())
ranks = np.arange(1, len(freq) + 1)
freqs = np.array(sorted(freq.values(), reverse=True))
plt.figure(figsize=(8, 6))
plt.loglog(ranks, freqs, color='purple', label='Gözlenen')
plt.loglog(ranks, freqs[0] / ranks, 'r--', alpha=0.6, label='İdeal Zipf (1/n)')
plt.title('Zipf Yasası (log-log)'); plt.xlabel('Kelime sırası (rank)')
plt.ylabel('Frekans'); plt.legend()
plt.tight_layout(); plt.savefig('results/eda_zipf.png'); plt.show()
for lbl, ad in [(0, 'Negatif'), (1, 'Nötr'), (2, 'Pozitif')]:
    w = set(' '.join(samp[samp['label'] == lbl]['text'].str.lower()).split())
    print(f'{ad:5s}: {len(w):,} benzersiz kelime')
print(f'Toplam benzersiz kelime (örneklem): {len(freq):,}')"""
        ),
    ]


def extras_03():
    """NB03 ekleri: Lemmatization vs Stemming karşılaştırması."""
    return [
        md("## 3.7 Lemmatization vs Stemming\nİki kök bulma yönteminin örnek tablosu ve küçük bir örneklemde model etkisi (F1-Macro)."),

        code(
"""# === EK ANALİZ: Lemmatization vs Stemming (örnek tablo) ===
import nltk, pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer

def wordnet_hazir():
    \"\"\"WordNet korpusunu sağlar; yoksa (offline) indirmeyi dener.\"\"\"
    try:
        nltk.data.find('corpora/wordnet'); return True
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)
            nltk.data.find('corpora/wordnet'); return True
        except Exception:
            return False

stemmer = PorterStemmer(); lemmatizer = WordNetLemmatizer()
LEMMA_OK = wordnet_hazir()
ornek = ['running','better','studies','wolves','feet','flies','happily','caring','geese','tasted']
def lemma(w):
    return lemmatizer.lemmatize(w, pos='v') if LEMMA_OK else '(wordnet yok)'
tablo = pd.DataFrame({
    'Kelime': ornek,
    'Porter Stemmer': [stemmer.stem(w) for w in ornek],
    'WordNet Lemmatizer': [lemma(w) for w in ornek],
})
print('Stemming kelimeyi köke kabaca keser; Lemmatization sözlük tabanlı anlamlı köke indirger.')
if not LEMMA_OK:
    print('NOT: wordnet korpusu bulunamadı (internet gerekli). Lemmatization atlandı.')
display(tablo)"""
        ),

        code(
"""# === EK ANALİZ: Stemming vs Lemmatization model etkisi (8K örneklem) ===
import re, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
dfp = pd.read_csv('data/reviews_preprocessed.csv', usecols=['text', 'label']).dropna()
dfp = dfp.sample(min(8000, len(dfp)), random_state=42)
toks = dfp['text'].apply(lambda t: re.sub(r'[^a-z ]', ' ', str(t).lower()).split())
metinler = {'Stemming': toks.apply(lambda ws: ' '.join(stemmer.stem(w) for w in ws if len(w) > 2))}
if LEMMA_OK:
    metinler['Lemmatization'] = toks.apply(lambda ws: ' '.join(lemmatizer.lemmatize(w) for w in ws if len(w) > 2))
res = {}
for ad, metin in metinler.items():
    Xtr, Xte, ytr, yte = train_test_split(metin, dfp['label'], test_size=0.25,
                                          random_state=42, stratify=dfp['label'])
    v = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=500, class_weight='balanced').fit(v.fit_transform(Xtr), ytr)
    res[ad] = f1_score(yte, clf.predict(v.transform(Xte)), average='macro')
plt.figure(figsize=(6, 4))
plt.bar(res.keys(), res.values(), color=['#e07b39', '#3b8686'])
for i, (k, val) in enumerate(res.items()):
    plt.text(i, val, f'{val:.3f}', ha='center', va='bottom')
plt.title('Stemming vs Lemmatization (F1-Macro, 8K örneklem)'); plt.ylabel('F1-Macro')
plt.tight_layout(); plt.savefig('results/lemma_vs_stem.png'); plt.show()
if not LEMMA_OK:
    print('NOT: wordnet yok; yalnız Stemming gösterildi. İnternet olan ortamda iki yöntem de çalışır.')"""
        ),
    ]


def extras_05():
    """NB05 ekleri: GridSearchCV, 5-fold CV, eğitim süresi karşılaştırması."""
    return [
        md("## 5.7 Hiperparametre Ayarı, Cross-Validation ve Eğitim Süresi\nGridSearchCV ile sistematik arama, 5-fold CV ve model eğitim süresi / F1 ödünleşimi."),

        code(
"""# === EK ANALİZ: GridSearchCV (TF-IDF + Logistic Regression) ===
import time, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
df = pd.read_csv('data/reviews_preprocessed.csv', usecols=['cleaned_text', 'label']).dropna(subset=['cleaned_text'])
dfs = df.sample(min(20000, len(df)), random_state=42)
pipe = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True)),
                 ('clf', LogisticRegression(max_iter=500, class_weight='balanced'))])
grid = {'tfidf__max_features': [10000, 20000],
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'clf__C': [0.5, 1.0, 5.0]}
gs = GridSearchCV(pipe, grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
t0 = time.time(); gs.fit(dfs['cleaned_text'], dfs['label'])
print(f'GridSearch süresi: {time.time() - t0:.1f}s')
print('En iyi parametreler:', gs.best_params_)
print(f'En iyi CV F1-Macro : {gs.best_score_:.4f}')"""
        ),

        code(
"""# === EK ANALİZ: 5-Fold Cross-Validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cvs = cross_val_score(gs.best_estimator_, dfs['cleaned_text'], dfs['label'],
                      cv=skf, scoring='f1_macro', n_jobs=-1)
plt.figure(figsize=(7, 4))
plt.bar(range(1, 6), cvs, color='teal')
plt.axhline(cvs.mean(), color='red', ls='--', label=f'Ortalama = {cvs.mean():.3f}')
plt.title('5-Fold Cross-Validation (F1-Macro)'); plt.xlabel('Fold'); plt.ylabel('F1-Macro'); plt.legend()
plt.tight_layout(); plt.savefig('results/cross_validation.png'); plt.show()
print(f'CV F1-Macro: {cvs.mean():.4f} +/- {cvs.std():.4f}')"""
        ),

        code(
"""# === EK ANALİZ: Eğitim Süresi vs F1-Macro ===
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import lightgbm as lgb
v = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(dfs['cleaned_text'], dfs['label'],
                                              test_size=0.25, random_state=42, stratify=dfs['label'])
Xtr = v.fit_transform(Xtr_t); Xte = v.transform(Xte_t)
modeller = {
    'LogReg': LogisticRegression(max_iter=500, class_weight='balanced'),
    'LinearSVC': LinearSVC(class_weight='balanced'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=200, num_leaves=63,
                                   class_weight='balanced', n_jobs=-1, verbose=-1),
}
sure, f1l, adlar = [], [], []
for ad, m in modeller.items():
    t0 = time.time(); m.fit(Xtr, ytr_t); dt = time.time() - t0
    f1 = f1_score(yte_t, m.predict(Xte), average='macro')
    sure.append(dt); f1l.append(f1); adlar.append(ad)
    print(f'{ad:10s}  süre={dt:6.2f}s  F1-Macro={f1:.3f}')
plt.figure(figsize=(7, 5))
plt.scatter(sure, f1l, s=120, color='darkorange')
for a, x, yv in zip(adlar, sure, f1l):
    plt.annotate(a, (x, yv), xytext=(5, 5), textcoords='offset points')
plt.title('Eğitim Süresi vs F1-Macro'); plt.xlabel('Eğitim süresi (s)'); plt.ylabel('F1-Macro')
plt.tight_layout(); plt.savefig('results/training_time_comparison.png'); plt.show()"""
        ),
    ]


def extras_06():
    """NB06 ekleri: t-SNE, kural-tabanlı vs ML, özellik önemi, opsiyonel BERT."""
    return [
        md("## 6.10 Ek Değerlendirme: t-SNE, Kural-tabanlı vs ML, Özellik Önemi\nt-SNE ile sınıf kümeleri, VADER/TextBlob ile ML karşılaştırması, en etkili kelimeler ve opsiyonel DistilBERT."),

        code(
"""# === EK ANALİZ: Ortak kurulum (kayıtlı artifact'ler) ===
import pandas as pd, numpy as np, joblib, scipy.sparse, matplotlib.pyplot as plt
df = pd.read_csv('data/reviews_preprocessed.csv')
y = np.load('features/labels.npy')
test_idx = np.load('features/test_idx.npy')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')
scaler = joblib.load('models/scaler.pkl')
lr = joblib.load('models/lr_model.pkl')
NUM_COLS = ['review_length','word_count','exclamation_count','question_count',
            'avg_word_length','uppercase_ratio','sentiment_polarity','sentiment_subjectivity']
sinif_ad = {0: 'Negatif', 1: 'Nötr', 2: 'Pozitif'}
print('Kurulum tamam. Test örneği:', len(test_idx))"""
        ),

        code(
"""# === EK ANALİZ: t-SNE (TF-IDF -> SVD-50 -> 2D) ===
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
sub = np.random.RandomState(42).choice(test_idx, size=min(3000, len(test_idx)), replace=False)
X_sub = tfidf.transform(df.loc[sub, 'cleaned_text'].fillna(''))
X_red = TruncatedSVD(n_components=50, random_state=42).fit_transform(X_sub)
emb = TSNE(n_components=2, perplexity=30, random_state=42, init='pca').fit_transform(X_red)
plt.figure(figsize=(9, 7))
renk = {0: 'red', 1: 'gray', 2: 'green'}
for lbl in [0, 1, 2]:
    m = y[sub] == lbl
    plt.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.5, c=renk[lbl], label=sinif_ad[lbl])
plt.legend(); plt.title('t-SNE: TF-IDF (SVD-50) -> 2D (test örneklemi)')
plt.tight_layout(); plt.savefig('results/tsne.png'); plt.show()"""
        ),

        code(
"""# === EK ANALİZ: Kural-tabanlı (VADER / TextBlob) vs ML ===
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import f1_score
n = min(4000, len(test_idx))
sidx = np.random.RandomState(1).choice(test_idx, size=n, replace=False)
texts = df.loc[sidx, 'text'].astype(str).tolist()
y_true = y[sidx]
vader = SentimentIntensityAnalyzer()
y_vader = [2 if (c := vader.polarity_scores(t)['compound']) >= 0.05 else 0 if c <= -0.05 else 1 for t in texts]
y_tb = [2 if (p := TextBlob(t).sentiment.polarity) > 0.05 else 0 if p < -0.05 else 1 for t in texts]
Xt = tfidf.transform(df.loc[sidx, 'cleaned_text'].fillna(''))
Xn = scaler.transform(df.loc[sidx, NUM_COLS])
y_ml = lr.predict(scipy.sparse.hstack([Xt, scipy.sparse.csr_matrix(Xn)]).tocsr())
skor = {'VADER': f1_score(y_true, y_vader, average='macro'),
        'TextBlob': f1_score(y_true, y_tb, average='macro'),
        'ML (LogReg)': f1_score(y_true, y_ml, average='macro')}
plt.figure(figsize=(7, 4))
plt.bar(skor.keys(), skor.values(), color=['#999', '#5bb', '#2a6'])
for i, (k, v) in enumerate(skor.items()):
    plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
plt.title('Kural-tabanlı (VADER/TextBlob) vs ML — F1-Macro'); plt.ylabel('F1-Macro')
plt.tight_layout(); plt.savefig('results/rule_vs_ml.png'); plt.show()"""
        ),

        code(
"""# === EK ANALİZ: Özellik Önemi (LR katsayıları -> en etkili kelimeler) ===
vocab = np.array(tfidf.get_feature_names_out())
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for i, (lbl, ax) in enumerate(zip([0, 1, 2], axes)):
    coefs = lr.coef_[lbl][:len(vocab)]
    top = np.argsort(coefs)[-15:]
    ax.barh(vocab[top], coefs[top], color=['red', 'gray', 'green'][i])
    ax.set_title(f'{sinif_ad[lbl]} sınıfına en çok iten kelimeler')
plt.tight_layout(); plt.savefig('results/feature_importance.png'); plt.show()"""
        ),

        code(
"""# === EK ANALİZ (OPSİYONEL): DistilBERT — transformers + torch gerektirir ===
try:
    from transformers import pipeline as hf_pipeline
    import torch
    device = 0 if torch.cuda.is_available() else -1
    clf = hf_pipeline('sentiment-analysis',
                      model='distilbert-base-uncased-finetuned-sst-2-english', device=device)
    idx = np.random.RandomState(7).choice(test_idx, size=300, replace=False)
    txt = df.loc[idx, 'text'].astype(str).str.slice(0, 512).tolist()
    preds = clf(txt, truncation=True)
    # SST-2 ikili (POSITIVE/NEGATIVE) -> 3-sınıfa kaba eşleme: POS->Pozitif(2), NEG->Negatif(0)
    ybin = np.array([2 if p['label'] == 'POSITIVE' else 0 for p in preds])
    from sklearn.metrics import accuracy_score
    mask = np.isin(y[idx], [0, 2])
    acc = accuracy_score(np.array(y[idx])[mask], ybin[mask])
    print(f'DistilBERT (yalnız Negatif/Pozitif alt kümesinde) doğruluk: {acc:.3f}')
except ImportError:
    print('transformers/torch kurulu değil — bu hücre opsiyoneldir, atlanabilir.')
    print('Kurmak için: pip install transformers torch')"""
        ),
    ]


EXTRAS_MAP = {
    "02_kesfsel_analiz.ipynb": extras_02,
    "03_metin_on_isleme.ipynb": extras_03,
    "05_model_egitimi.ipynb": extras_05,
    "06_model_degerlendirme.ipynb": extras_06,
}


def _strip_extras(cells):
    """Daha önce eklenmiş ekstra hücreleri (sondaki blok) temizler.

    Ekstra hücreler her zaman notebook'un SONUNA, bir '## x.x ... Ek' başlığıyla
    eklenir. İlk EXTRA_MARKER'lı hücreyi bul, hemen öncesindeki markdown başlığı da
    dahil ederek o noktadan sonrasını atar. Mevcut (asıl) hücreler ve çıktıları korunur.
    """
    idx = next((i for i, c in enumerate(cells)
                if EXTRA_MARKER in "".join(c.get("source", []))), None)
    if idx is None:
        return cells
    if idx > 0 and cells[idx - 1].get("cell_type") == "markdown":
        idx -= 1
    return cells[:idx]


def inject_existing(refresh=True):
    """Mevcut (çalıştırılmış) notebook'lara ekstra hücreleri çıktıları kaybetmeden ekler.

    refresh=True ise önce eski ekstra blok temizlenip yenisi yazılır (yeniden eşitleme).
    """
    for fname, fn in EXTRAS_MAP.items():
        path = os.path.join(BASE, fname)
        if not os.path.exists(path):
            print(f"  [ATLA] {fname} bulunamadı")
            continue
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        var = any(EXTRA_MARKER in "".join(c.get("source", [])) for c in nb["cells"])
        if var and not refresh:
            print(f"  [ATLA] {fname} zaten ekstra içeriyor")
            continue
        nb["cells"] = _strip_extras(nb["cells"])
        yeni = fn()
        nb["cells"].extend(yeni)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f"  [{'YENİLE' if var else 'OK'}] {fname} +{len(yeni)} ekstra hücre")


# ----------
if __name__ == "__main__":
    print("Amazon Fine Food Reviews — Notebook Oluşturucu")
    print("=" * 50)
    nb01()
    nb02()
    nb03()
    nb04()
    nb05()
    nb06()
    nb07()
    print("--- Ek analiz hücreleri ekleniyor ---")
    inject_existing()
    print("=" * 50)
    print("Tüm notebook'lar başarıyla oluşturuldu!")
    print("config.py dosyasının da aynı dizinde olduğundan emin olun.")
