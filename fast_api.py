#**FastAPI**  ile temel yapı kurulumu Aşağıdaki uç noktaların oluşturulması:

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import pandas as pd
import joblib
from database_definition import prepare_segmented_dataframe
from database_connect import get_data_from_db
from main_model import build_feature_vector, model_predict
import numpy as np
from main_model import model_predict  # model_predict fonksiyonunu kullanıyoruz
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np



model = joblib.load("model.pkl") #eğitilmiş model
scaler = joblib.load("scaler.pkl")

df = prepare_segmented_dataframe() #veri seti

app = FastAPI(title="Sales Predict Api", 
              description="Northwind DB sales quantity predict API's")



orders_df, order_details_df, products_df, customers_df, categories_df = get_data_from_db()

#/products	 GET	Ürün listesini döner

@app.get("/products")
def get_products():
    return products_df.to_dict(orient="records") #dataframei json uyumlu yapar.



#/sales_summary	GET 	Satış özet verisini döner

@app.get("/sales_summary")
def sales_summary():
    summary = df.groupby('product_id')['quantity'].sum().reset_index()
    summary = summary.rename(columns={'quantity': 'total_quantity'})
    return summary.to_dict(orient="records")



#/predict	POST	Tahmin yapılmasını sağlar

class PredictRequest(BaseModel):
    product_id: int
    customer_id: str
    order_date: str
    units_in_stock: int
    reorder_level: int

@app.post("/predict")
def predict(request: PredictRequest):
    prediction = model_predict(
        product_id=request.product_id,
        customer_id=request.customer_id,
        order_date=request.order_date,
        units_in_stock=request.units_in_stock,
        reorder_level=request.reorder_level
    )
    return {"prediction": prediction}



#/retrain	POST	Modeli tekrar eğitir

@app.post("/retrain")
def retrain():
    features = [
        'unit_price', 'discount', 'customer_segment',
        'monthly_segment', 'product_segment', 'stock_reorder_interaction'
    ]

    X = df[features]
    y = np.log1p(df['quantity'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    return {"message": "Model tekrar eğitildi."}


