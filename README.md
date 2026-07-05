# Metin Madenciliği — Amazon Yorumları Duygu Analizi

Amazon Fine Food Reviews veri seti üzerinde uçtan uca bir **duygu analizi
(sentiment analysis)** projesi: veri hazırlamadan model eğitimi ve
değerlendirmeye, varlık bazlı (aspect-based) duygu analizine ve bir web
arayüzüne kadar. Yorumlar üç sınıfa ayrılır: **Negatif, Nötr, Pozitif**.

## Veri seti

Bu repo veri setini **içermez** (boyutu büyük). Amazon Fine Food Reviews
veri setini Kaggle'dan indir:

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

İndirdiğin `Reviews.csv` dosyasını proje kök dizinine koy
(bkz. `config.py` → `DATA_FILE`).

## Notebook akışı

| Notebook | İçerik |
|---|---|
| `01_veri_hazirlama.ipynb` | Veri yükleme ve etiketleme |
| `02_kesfsel_analiz.ipynb` | Keşifsel veri analizi (EDA) |
| `03_metin_on_isleme.ipynb` | Metin ön işleme (temizleme, normalizasyon) |
| `04_ozellik_cikarimi.ipynb` | Özellik çıkarımı (TF-IDF, gömme vektörleri) |
| `05_model_egitimi.ipynb` | Model eğitimi |
| `06_model_degerlendirme.ipynb` | Model karşılaştırma ve değerlendirme |
| `07_varlik_bazli_duygu_analizi.ipynb` | Varlık bazlı (aspect-based) duygu analizi |

## Modeller

Klasik makine öğrenmesi (Naive Bayes, SVM, Lojistik Regresyon, LightGBM),
derin öğrenme (LSTM, BiLSTM) ve **DistilBERT** ince ayarı; ayrıca bir
birleştirme (ensemble / fusion) katmanı.

Eğitilmiş model dosyaları (`models/`) ve çıkarılan özellikler (`features/`)
repoya **dahil edilmemiştir** — notebook'lar sırayla çalıştırılarak yeniden
üretilir.

## Web uygulaması

`webapp/` altında bir Flask uygulaması (`app.py`, `pipeline.py`) — eğitilen
modelle canlı tahmin arayüzü.

## Rapor

`rapor/` klasöründe proje raporu (`rapor.pdf`) ve üretilen grafikler bulunur.

## Çalıştırma

1. Gerekli kütüphaneleri kur (pandas, numpy, scikit-learn, tensorflow,
   transformers, lightgbm, flask, matplotlib, nltk ...).
2. Veri setini Kaggle'dan indir ve `Reviews.csv` olarak yerleştir.
3. Notebook'ları sırayla (01 → 07) çalıştır.
4. Web arayüzü için: `python webapp/app.py`

## Not

Büyük dosyalar (veri seti, eğitilmiş modeller, çıkarılan özellikler)
`.gitignore` ile hariç tutulmuştur; hepsi notebook'lar çalıştırılarak
yeniden üretilebilir.
