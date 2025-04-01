# Özellik mühendisliği analizerinin yapılması ve nihai modelin oluşturulması

import pandas as pd
from database_connect import get_data_from_db
import numpy as np

def prepare_segmented_dataframe():
    orders_df, order_details_df, products_df, customers_df, categories_df = get_data_from_db()

    # 1.Kullanilmasi on gorulen sutunlarin tablolardan cekimi
    # Amaç: Modelin hem sipariş bilgisi, hem müşteri, hem ürün bilgilerini birlikte kullanabilmesini sağlamak.
    df_final = (
        orders_df[["order_id", "customer_id", "order_date"]]
        .merge(order_details_df, on="order_id", how="left")
        .merge(products_df[["product_id", "category_id",'units_in_stock','reorder_level']],
               on="product_id", how="left")
    )
    


    # 2.Musteri bazli satis musteri segmentasyonu
    # Amaç: Müşterilerin sipariş miktarlarına göre segmentlere ayrılması

    customer_sales = df_final.groupby('customer_id')['quantity'].mean().reset_index(name='avg_quantity')
    customer_sales['customer_segment'] = pd.qcut(
        customer_sales['avg_quantity'].rank(method='first'),
        q=44, labels=range(1, 45)
    ).astype(int)

    df_final = df_final.merge(customer_sales[['customer_id', 'customer_segment']], on='customer_id', how='left')

    

    # 3.Ürün bazli satis ve urun segmentasyonu - 1
    # Amaç: Ürünlerin sipariş miktarlarına göre segmentlere ayrılması (Çok satılan ve az satılan ürünleri ayırt edebilmek.)

    product_sales = df_final.groupby('product_id')['quantity'].sum().reset_index(name='avg_quantity')
    product_sales['product_segment'] = pd.qcut(
        product_sales['avg_quantity'].rank(method='first'),
        q=77,
        labels=range(1, 78)
    ).astype(int)

    df_final = df_final.merge(product_sales[['product_id',
                                             'product_segment']], on='product_id', how='left')

   
                                             
    # 4.Ürun bazli segmentasyon - 2
    # Amaç: Ürün satış kalıplarını modelin tanımasına yardımcı olmak.
    mean_quantity = df_final.groupby('product_id')['quantity'].mean()
    df_final['product_mean_quantity'] = df_final['product_id'].map(mean_quantity)
    
  
    # 5.Aylarin numaralarinin alinmasi
    df_final['order_month_num'] = pd.to_datetime(df_final['order_date']).dt.month

    # Aylara gore satis ve segmentasyon
    # Amaç: Aylara göre satış miktarlarını ve segmentlerini modelin tanımasına yardımcı olmak.
    monthly_sales = df_final.groupby('order_month_num')['quantity'].mean().reset_index(name='avg_quantity')
    monthly_sales['monthly_segment'] = pd.qcut(
        monthly_sales['avg_quantity'].rank(method='first'),
        q=12,
        labels=range(1, 13)
    ).astype(int)

    df_final = df_final.merge(monthly_sales[['order_month_num',
                                            'monthly_segment']], on='order_month_num', how='left')


    # 6. Stok ve Yeniden Sipariş Seviyesi Etkileşimi
    # Amaç: Stok seviyeleri ile yeniden sipariş seviyeleri arasındaki etkileşimi modelin tanımasına yardımcı olmak.
    df_final['stock_reorder_interaction'] = df_final['units_in_stock'] * df_final['reorder_level']

    
    # 7. Kategorilere göre satış ve segmentasyon
    # Amaç: Ürün kategorilerine göre satış miktarlarını ve segmentlerini modelin tanımasına yardımcı olmak.
    category_sales = df_final.groupby('category_id')['quantity'].sum().reset_index()
    category_sales = category_sales.sort_values('quantity', ascending=False)
    category_sales['category_rank'] = range(1, len(category_sales) + 1)

    df_final = pd.merge(
        df_final,
        category_sales[['category_id', 'category_rank']],
        on='category_id',
        how='left'
    )
    return df_final


if __name__ == "__main__":
    print(prepare_segmented_dataframe())
    print(prepare_segmented_dataframe().columns)