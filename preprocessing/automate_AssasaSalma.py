import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def preprocess_data(data, target_col, identifier_col, save_path):
    df = data.copy()

    # drop kolom identifier 
    if identifier_col:
        existing_ids = [c for c in identifier_col if c in df.columns]
        df = df.drop(columns=existing_ids)

    # memisahkan fitur dan target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the DataFrame.")
    
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # identifikasi fitur numerikal dan kategorikal
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # mengatasi missing value fitur kategorikal
    for col in categorical_features:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    # mengatasi missing value fitur numerik
    for col in numeric_features:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    # scaling fitur numerik dengan RobustScaler
    scaler = RobustScaler()
    if numeric_features:
        X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # encode fitur kategorikal dengan one-hot encoding
    if categorical_features:
        X = pd.get_dummies(X, columns=categorical_features, drop_first=False)

    # gabungkan kembali fitur dan target
    preprocessed_df = pd.concat([X, y], axis=1)

    # menyimpan dataframe ke file csv
    preprocessed_df.to_csv(save_path, index=False)

    return preprocessed_df