Amazon Yorumları Üç Sınıflı Duygu Analizi
Furkan Öztürk - 230229083

Bu proje, Amazon Fine Food Reviews veri seti üzerinde İngilizce ürün yorumlarını
Negatif / Nötr / Pozitif olarak sınıflandırır. Klasik modeller (Lojistik Regresyon,
SVM, LightGBM, ComplementNB), topluluk yöntemleri (ağırlıklı oylama ve stacking),
derin modeller (LSTM, BiLSTM) ve ince ayarlı DistilBERT birlikte eğitilmiştir. Bütün
modeller tek bir Flask web arayüzünde toplanmıştır; arayüzden yorum yazıp canlı
tahmin alabilirsiniz.


Ödev neden iki klasör?

Eğittiğim DistilBERT model dosyası tek başına yaklaşık 140 MB olduğu için ödevi iki
parçaya böldüm:

  230229083_proje          -> bütün kod, işlenmiş veri, notebook'lar, klasik modeller, web arayüzü,
                              
  230229083_rapor_ve_bert  -> rapor PDF'i ve büyük DistilBERT model dosyası
                              (model_int8.pt).

Web arayüzü iki klasörü birleştirmeden de çalışır; bu durumda sadece DistilBERT
seçeneği listede görünmez, diğer bütün modeller çalışır. DistilBERT'i de canlı
görmek isterseniz aşağıdaki "DistilBERT'i etkinleştirme" adımını uygulayın.


Klasör yapısı (230229083_proje/metinmadenciligi)

  01..07_*.ipynb   sırayla çalışan notebook'lar (veri hazırlama ... varlık analizi)
  config.py        ortak sabitler
  data/            işlenmiş veri (reviews_preprocessed.csv)
  features/        kaydedilmiş öznitelik dizileri (.npy)
  models/          eğitilmiş modeller (klasikler, LSTM/BiLSTM; models/v2 altında
                   topluluk modelleri ve DistilBERT)
  results/         bütün grafikler ve metrik tabloları
  webapp/          Flask arayüzü (app.py, pipeline.py)


Kurulum

1) Python 3.10 veya üstü kurulu olmalı.

2) (Önerilir) ayrı bir sanal ortam:
     python -m venv venv
     venv\Scripts\activate

3) Gerekli kütüphaneler:
     pip install flask torch transformers tensorflow lightgbm scikit-learn joblib numpy scipy nltk textblob

   Notebook'ları da baştan çalıştıracaksanız ek olarak:
     pip install pandas matplotlib seaborn wordcloud lime vaderSentiment

   NLTK'nin stopwords / wordnet / punkt verileri ilk çalıştırmada otomatik iner,
   elle bir şey indirmenize gerek yoktur.


DistilBERT'i etkinleştirme (isteğe bağlı)

İki klasörü de açtıktan sonra, büyük model dosyasını proje içine kopyalayın.

  Kaynak:
    230229083_rapor_ve_bert\metinmadenciligi\models\v2\distilbert\model_int8.pt

  Hedef:
    230229083_proje\metinmadenciligi\models\v2\distilbert\

Hedefteki distilbert klasöründe config.json, tokenizer.json gibi dosyalar zaten var;
yanlarına sadece model_int8.pt dosyasını ekliyorsunuz. Kopyaladıktan sonra web
arayüzünde DistilBERT seçeneği de gelir.


Web arayüzünü çalıştırma

  cd 230229083_proje\metinmadenciligi\webapp
  python app.py

Açıldıktan sonra tarayıcıdan şu adrese girin:

  http://127.0.0.1:5000

Sayfalar:
  - Tekli tahmin : bir yorum (ve isterseniz başlık) yazıp model seçersiniz; sonuç,
                   sınıf olasılıkları ve kelime kelime açıklama ile gösterilir.
  - Toplu analiz : birden çok yorumu aynı anda sınıflandırır.
  - Karşılaştırma: test kümesi metrikleri ve karşılaştırma grafikleri.
  - Keşif        : veri analizine ait grafikler.


Notebook'ları çalıştırma (isteğe bağlı)

Notebook'lar 01'den 07'ye doğru sırayla çalışır. İşlenmiş veri ve eğitilmiş modeller
klasörlerde hazır olduğu için sadece arayüzü çalıştırmak istiyorsanız notebook'ları
tekrar koşturmanıza gerek yoktur.

Not: 01_veri_hazirlama.ipynb, Kaggle'daki ham "Reviews.csv" dosyasını ister. Bu dosya
çok büyük olduğu için ödeve eklemedim. Ham veriyi Kaggle "Amazon Fine Food Reviews"
sayfasından indirip notebook ile aynı klasöre koyarsanız 01 baştan çalışır.
Koymazsanız da data/reviews_preprocessed.csv hazır olduğu için 02 ve sonrası sorunsuz
çalışır.


Rapor

Rapor PDF'i: 230229083_rapor_ve_bert\metinmadenciligi\rapor\ klasöründedir.
