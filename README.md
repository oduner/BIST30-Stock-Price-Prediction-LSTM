# BIST30 OpenChange Prediction Pipeline

## 📋 Proje Genel Bakışı
Bu proje, BIST30 şirketlerinin hisse senedi verilerini kullanarak açılış fiyat değişimlerini (OpenChange) ve yön tahminlerini (ODirection) yapmak için geliştirilmiş kapsamlı bir makine öğrenmesi pipeline'ıdır. Sistem, LSTM tabanlı derin öğrenme modelleri kullanarak çoklu zaman serisi tahminleri gerçekleştirir.

## 🎯 Temel Özellikler
- Çift Çıkışlı Model: Hem fiyat değişimi (regresyon) hem de yön tahmini (sınıflandırma)
- Zaman Serisi Çapraz Doğrulama: Robust model seçimi için TSCV
- Olasılık Kalibrasyonu: Isotonic regression ile güvenilir olasılık tahminleri
- Optimal Eşik Belirleme: F1 skorunu maksimize eden dinamik eşik değerleri
- Walk-Forward Validation: Gerçekçi out-of-sample tahminler
- Otomatik Veri Yönetimi: Veri indirme, güncelleme ve ön işleme

## 🏗️ Sistem Mimarisi
### Modül Yapısı
📦 BIST30-Prediction-Pipeline
├── 📊 Veri Yönetimi
│   ├── Controller.py          # Veri bütünlüğü kontrolü ve koordinasyon
│   ├── yDatas.py             # Tarihi veri indirme
│   ├── RawUpdater.py         # Güncel veri güncelleme
│   └── Preparative.py        # Veri ön işleme ve feature engineering
├── 🤖 ML Pipeline
│   ├── config_module.py      # Yapılandırma ve sabitler
│   ├── data_module.py        # Veri yükleme ve sequence oluşturma
│   ├── model_module.py       # LSTM model mimarisi
│   ├── training_module.py    # Model eğitimi ve validasyon
│   ├── calibration_module.py # Olasılık kalibrasyonu
│   └── prediction_module.py  # Tahmin ve metrik hesaplama
├── 📈 Görselleştirme & Analiz
│   ├── visualization_module.py # Eğitim geçmişi grafikleri
│   └── callbacks_module.py    # Özel training callback'leri
├── 🚀 Çalıştırıcılar
│   ├── main_module.py        # Ana giriş noktası
│   ├── runner_module.py      # Şirket bazlı pipeline koordinasyonu
│   └── pipeline.py           # Günlük tahmin pipeline'ı
└── 📁 Veri Klasörleri
    ├── yDatas/
    │   ├── Raw/              # Ham hisse verileri
    │   ├── Bist/             # Eğitim verileri
    │   └── Test/             # Test verileri (son 64 gün)
    ├── Models/OpenChange/    # Eğitilmiş modeller
    └── Results/OpenChange/   # Tahmin sonuçları ve grafikler

### Temel Yapılandırma
`config_module.py` dosyasından temel parametreleri değiştirebilirsiniz:
python
CONFIG = {
    "window_size": 64,           # Zaman serisi window boyutu
    "features": ["OpenChange", "RSI", "Volatility", "MA_20", "ODirection"],
    "lstm_units": [128, 64],     # LSTM katman boyutları
    "dense_units": [32, 16],     # Dense katman boyutları
    "epochs": 128,               # Maksimum epoch sayısı
    "batch_size": 16,            # Batch size
    "learning_rate": 0.001,      # Öğrenme oranı
    "tscv_splits": 7,            # Zaman serisi CV split sayısı
}

## 🚀 Kullanım
1. Veri Hazırlığı
# Veri bütünlüğünü kontrol et ve eksik verileri indir
python Controller.py
2. Model Eğitimi
# Tüm BIST30 şirketleri için model eğitimi
python main_module.py
# Belirli şirketler için eğitim
python -c "from runner_module import run_all_companies; run_all_companies(start_index=0, batch_size=5)"
3. Günlük Tahminler
# Fine-tuning ve günlük tahminler için
python pipeline.py

## 🔧 Teknik Detaylar
### Model Mimarisi
# Çift çıkışlı LSTM ağı:
# - Price Output: OpenChange tahmini (Linear aktivasyon)
# - Direction Output: ODirection tahmini (Sigmoid aktivasyon)

### Feature Engineering
- OpenChange: Açılış fiyatı yüzde değişimi
- RSI: Göreceli Güç Endeksi (14 gün)
- Volatility: 5 günlük volatilite
- MA_20: 20 günlük hareketli ortalama
- ODirection: Açılış yönü (0: düşüş, 1: yükseliş)

### Kalibrasyon ve Optimizasyon
- Isotonic Regression: Olasılık kalibrasyonu
- Optimal Threshold: F1 skoru maksimizasyonu
- Class Weighting: Dengesiz veri için ağırlıklandırma

## 📊 Çıktılar ve Metrikler
### Kaydedilen Çıktılar
- Modeller: `Models/OpenChange/{sembol}_model.keras`
- Scaler'lar: `Models/OpenChange/{sembol}_scaler_{X,y}.pkl`
- Sonuçlar: `Results/OpenChange/{sembol}_results.csv`
- Grafikler: `Results/OpenChange/Plotting/{sembol}/`

### Performans Metrikleri
**Regresyon Metrikleri:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)
- MSLE (Mean Squared Logarithmic Error)

**Sınıflandırma Metrikleri:**
- Accuracy, Precision, Recall, F1-Score
- Kalibre edilmiş ve ham olasılıklar
- Optimal eşik değerleri

## 🔄 İş Akışı
- Veri Kontrolü: Controller.py ile veri bütünlüğü kontrolü
- Ön İşleme: Teknik göstergeler ve feature engineering
- Model Eğitimi: TSCV ile çoklu model eğitimi ve seçimi
- Kalibrasyon: Olasılık kalibrasyonu ve eşik optimizasyonu
- Tahmin: Walk-forward validation ile test tahminleri
- Değerlendirme: Kapsamlı metrik hesaplama ve görselleştirme
- Dağıtım: Günlük fine-tuning ve tahmin pipeline'ı

## 🎯 BIST30 Şirketleri
Proje aşağıdaki BIST30 şirketlerini destekler:  
AKBNK, ARCLK, ASELS, BIMAS, EKGYO, EREGL, FROTO, GARAN, GUBRF, HEKTS  
KCHOL, KOZAA, KOZAL, KRDMD, MGROS, PGSUS, PETKM, SAHOL, SASA, SOKM  
SISE, TCELL, THYAO, TKFEN, TOASO, TUPRS, VAKBN, YKBNK, BRSAN, ALARK

## ⚠️ Önemli Notlar
- Veriler iş günlerinde güncellenir, hafta sonları güncelleme yapılmaz
- Test seti her zaman son 64 iş gününü içerir
- Model eğitimi zaman serisi sırasını korur (shuffle yok)
- Tüm işlemler reproducible olacak şekilde seed değerleri ayarlanmıştır

## 📈 Sonuçların Yorumlanması
Tahmin sonuçları CSV formatında kaydedilir ve şu sütunları içerir:
- Gerçek ve tahmini OpenChange değerleri
- Ham ve kalibre edilmiş yön olasılıkları
- Optimal ve kalibre edilmiş eşik değerleri
- Kümülatif metrikler (her adımda güncellenir)
- Hata analizi ve doğruluk ölçümleri