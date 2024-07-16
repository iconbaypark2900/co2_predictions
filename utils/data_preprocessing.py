import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

def preprocess_data(filepath, sequence_length):
    # Load the data
    data = pd.read_csv(filepath)

    # Extracting the relevant columns (datetime and CO2 value)
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')['value'].dropna()

    # Normalizing the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    return data_scaled, scaler

def split_data(data_scaled, train_size=0.8):
    split_idx = int(len(data_scaled) * train_size)
    train_data = data_scaled[:split_idx]
    val_data = data_scaled[split_idx:]
    return train_data, val_data


