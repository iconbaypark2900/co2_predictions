import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data(file_path, save_scaler_path=None):
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['value']])
    
    if save_scaler_path:
        joblib.dump(scaler, save_scaler_path)
        
    return data_scaled, scaler

def split_data(data, train_size=0.8):
    train_size = int(len(data) * train_size)
    train_data = data[:train_size]
    val_data = data[train_size:]
    return train_data, val_data

def inverse_transform(scaler, data_scaled):
    return scaler.inverse_transform(data_scaled)
