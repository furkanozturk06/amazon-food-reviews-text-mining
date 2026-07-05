# Yorum Duygu Analizi - Web Arayüzü

Projede eğitilen modelleri (Logistic Regression, SVM, LightGBM, ComplementNB,
ağırlıklı topluluk ve stacking, LSTM, BiLSTM ve ince ayarlı **DistilBERT**) tek bir
panelde toplayan Flask arayüzü. Yorumları **Negatif / Nötr / Pozitif** olarak
sınıflandırır; en yüksek başarımı DistilBERT verir (F1-Macro 0,8113).

## Sayfalar
- **Tekli Tahmin** – tek yorum, sınıf olasılıkları, belirleyici kelimeler ve tüm
  modellerin aynı metin için tahmini
- **Toplu Analiz** – çok sayıda yorum (metin kutusu veya dosya), dağılım grafiği,
  CSV indirme
- **Model Karşılaştırma** – test metrikleri tablosu ve değerlendirme grafikleri
- **Keşifsel Analiz** – sınıf dağılımı, sınıf bazlı kelimeler ve EDA görselleri
- **Hakkında** – yöntem ve sonuç özeti

## Çalıştırma
Proje kök dizininde modellerin (`models/`) ve sonuç grafiklerinin (`results/`)
hazır olması gerekir; bunlar `01`–`06` notebook'ları çalıştırıldığında üretilir.

```bash
cd webapp
python app.py
```

Tarayıcıdan `http://127.0.0.1:5000` adresini açın.

## Bağımlılıklar
flask, scikit-learn, lightgbm, tensorflow, torch, transformers, sentence-transformers,
nltk, textblob, scipy, numpy, joblib.

## Notlar
- Modeller İngilizce yorumlarla eğitildiği için en güvenilir sonucu İngilizce
  metinlerde verir.
- Görseller yalnızca `results/` altındaki bilinen dosya adları için sunulur.
