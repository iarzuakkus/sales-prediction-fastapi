# Veri tabanı bağlantısı ve veri çekme işlemlerinin yapılması

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# DATABASE_URL = "postgresql+psycopg2://sevgiberk:mysecretpassword@localhost:5432/northwind"

# Engine oluştur
engine = create_engine(DATABASE_URL)

def get_data_from_db():
    orders_df = pd.read_sql("SELECT * FROM orders", engine)
    order_details_df = pd.read_sql("SELECT * FROM order_details", engine)
    products_df = pd.read_sql("SELECT * FROM products", engine)
    customers_df = pd.read_sql("SELECT * FROM customers", engine)
    categories_df = pd.read_sql("SELECT * FROM categories", engine)

    return orders_df, order_details_df, products_df, customers_df, categories_df

if __name__ == "__main__":
    print(get_data_from_db())