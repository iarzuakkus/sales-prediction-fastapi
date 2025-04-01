 # Verideki eksik değerleri kontrol et  
 # Aykırı değerleri tespit edip veriden temizle
 # Verideki çarpıklığı azaltmak için log(1+x) dönüşümünün yapılması


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from database_definition import prepare_segmented_dataframe


import numpy as np

def check_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing)
    else:
        print("Eksik değer bulunamadı.")

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def apply_log_transform(df, column, new_column_name):
    df[new_column_name] = np.log1p(df[column])
    return df