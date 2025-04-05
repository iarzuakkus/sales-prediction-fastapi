# 📈 Sales Prediction FastAPI

Bu proje, Northwind veritabanını kullanarak ürün bazlı satış miktarını tahmin eden bir makine öğrenmesi modelini FastAPI ile REST API olarak sunar. Model, geçmiş sipariş verilerine göre ürün satışlarını tahmin eder.

---

## 🚀 Özellikler

- FastAPI ile REST API servisi
- Ürün ve müşteri segmentasyonu
- Lineer regresyon ile tahminleme
- `has_discount` gibi türetilmiş özelliklerle geliştirilmiş veri modeli
- Swagger UI üzerinden kolay test imkânı
- API üzerinden modelin yeniden eğitilmesi (retrain)

---

## 📁 Proje Yapısı

```
sales-prediction-fastapi/
│
├── fast_api.py                # FastAPI uygulaması (endpoint tanımları)
├── main_model.py              # Model eğitim, tahmin ve feature engineering
├── database_definition.py     # Segmentleme & veri hazırlama fonksiyonları
├── database_connect.py        # PostgreSQL veri çekme fonksiyonu
├── eda_utils.py               # (Varsa) keşifsel veri analiz yardımcıları
├── model.pkl                  # Eğitilmiş model dosyası
├── scaler.pkl                 # StandardScaler nesnesi
├── README.md                  # Proje açıklaması
├── northwind_data/            # (Opsiyonel) sabit veri dosyaları
└── ...
```

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Gereken kütüphaneleri yükleyin:

```bash
pip install fastapi uvicorn pandas scikit-learn joblib psycopg2-binary
```

### 2. API’yi çalıştırın:

```bash
python -m uvicorn fast_api:app --reload
```

### 3. Tarayıcıdan erişin:

- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 API Uç Noktaları

| Endpoint         | Metot | Açıklama                                |
|------------------|--------|-----------------------------------------|
| `/products`      | GET    | Ürün listesini JSON olarak döner        |
| `/sales_summary` | GET    | Ürün bazlı toplam satışları döner       |
| `/predict`       | POST   | Satış miktarını tahmin eder             |
| `/retrain`       | POST   | Mevcut veriye göre modeli yeniden eğitir |

### 🔮 /predict örnek isteği

```json
{
  "product_id": 1,
  "customer_id": "ALFKI",
  "order_date": "15/03/1997"
}
```

---

## 🧠 Kullanılan Özellikler (Features)

Modelde kullanılan temel değişkenler şunlardır:

- `monthly_segment`: Siparişin verildiği aya göre segment
- `product_segment`: Ürün segmenti
- `product_mean_spent`: Ürünün ortalama harcama miktarı
- `stock_reorder_interaction`: Stok seviyesi × yeniden sipariş seviyesi
- `category_rank`: Ürünün kategorideki sırası
- `has_discount`: Ürün indirimli mi (1/0)

---


