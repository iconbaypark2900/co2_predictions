import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

def load_model(model_path='models/sarima_model.pkl'):
    """
    Load the SARIMA model from a file.
    Args:
        model_path (str): Path to the saved SARIMA model.

    Returns:
        model (SARIMAXResults): Loaded SARIMA model.
    """
    model = joblib.load(model_path)
    return model

def load_historical_data(data_file_path='/home/gengar/data/aiml/co2_prediction/data/raw/co2_data.csv'):
    """
    Load historical CO2 data.
    Args:
        data_file_path (str): Path to the CO2 data file.

    Returns:
        data (pd.DataFrame): Historical CO2 data.
    """
    data = pd.read_csv(data_file_path, parse_dates=['datetime'], index_col='datetime')
    return data['value']

def preprocess_data(data_file_path='/home/gengar/data/aiml/co2_prediction/data/raw/co2_data.csv'):
    """
    Preprocess the data for SARIMA model.
    Args:
        data_file_path (str): Path to the CO2 data file.

    Returns:
        data (pd.Series): Preprocessed CO2 data.
    """
    data = pd.read_csv(data_file_path, parse_dates=['datetime'], index_col='datetime')
    data = data.dropna()
    result = adfuller(data['value'])
    if result[1] > 0.05:
        data['value_diff'] = data['value'].diff().dropna()
    else:
        data['value_diff'] = data['value']
    return data

def build_and_train_model(data):
    """
    Build and train the SARIMA model.
    Args:
        data (pd.Series): Preprocessed CO2 data.

    Returns:
        model (SARIMAXResults): Trained SARIMA model.
    """
    model = SARIMAX(data['value_diff'].dropna(), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    return results

def predict(model, sequence, steps=600):
    """
    Predict future CO2 levels using the SARIMA model.
    Args:
        model (SARIMAXResults): Trained SARIMA model.
        sequence (list): List of past CO2 levels.
        steps (int): Number of steps to forecast.

    Returns:
        prediction (list): Predicted future CO2 levels.
    """
    sequence_series = pd.Series(sequence)
    prediction = model.get_forecast(steps=steps)
    return prediction.predicted_mean.tolist()

# Example usage
if __name__ == "__main__":
    data_file_path = '/home/gengar/data/aiml/co2_prediction/data/raw/co2_data.csv'
    data = preprocess_data(data_file_path)
    model = build_and_train_model(data)
    sequence = [330.91, 330.13, 330.35, 330.45, 332.69]  # Example sequence
    prediction = predict(model, sequence)
    print(f'Predicted CO2 Levels: {prediction}')