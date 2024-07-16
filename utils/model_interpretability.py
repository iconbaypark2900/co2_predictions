# import shap
# import numpy as np
# import matplotlib.pyplot as plt

# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator 
# import os
# import sys

# # Add the parent directory to the Python path for imports
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from utils.data_preprocessing import preprocess_data

# def load_trained_model(model_path):
#     return load_model(model_path)

# def explain_model_predictions(model, data_scaled, sequence_length, num_samples=100):
#     # Select a subset of the data as a background distribution for the explainer
#     # This subset should have the same shape as the test data
#     background_index = np.random.choice(data_scaled.shape[0] - sequence_length, num_samples, replace=False)
#     background = np.array([data_scaled[i:i + sequence_length] for i in background_index])
#     explainer = shap.GradientExplainer(model, background)

#     # Select a separate subset of data for explanation
#     test_index = np.random.choice(data_scaled.shape[0] - sequence_length, num_samples, replace=False)
#     test_data = np.array([data_scaled[i:i + sequence_length] for i in test_index])

#     # Compute SHAP values
#     shap_values = explainer.shap_values(test_data)

#     return shap_values



# def plot_shap_values(shap_values, output_path):
#     # Assuming shap_values is a 2D array where we've aggregated over the sequence length
#     # And we only have one feature (the univariate CO2 measurement)
#     plt.figure(figsize=(10, 5))
    
#     # We need to ensure that shap_values is 2D where the second dimension is 1 (for univariate)
#     if len(shap_values.shape) == 3:
#         shap_values = shap_values.sum(axis=1)  # This sums the SHAP values over the sequence length
    
#     shap.summary_plot(shap_values, feature_names=['CO2 Measurements'])
    
#     plt.gca().ticklabel_format(style='plain', axis='x')  # Disable scientific notation
#     plt.xticks(rotation=45, ha='right')
#     plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
#     plt.savefig(output_path)


# if __name__ == "__main__":
#     project_dir = os.path.dirname(os.path.dirname(__file__))
#     model_path = os.path.join(project_dir, 'models', 'lstm_model.h5')
#     data_file_path = os.path.join(project_dir, 'data', 'co2_data.csv')
#     plot_output_path = os.path.join(project_dir, 'data', 'shap', 'improved_shap_plot.png')

#     # Ensure the output directory exists
#     os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)

#     # Load the model and preprocessed data
#     model = load_trained_model(model_path)
#     sequence_length = 5  # This should be the same as used during model training
#     data_scaled, _ = preprocess_data(data_file_path, sequence_length)

#     # Explain model predictions using SHAP
#     shap_values = explain_model_predictions(model, data_scaled, sequence_length, num_samples=100)

#     # Reshape SHAP values if necessary (depends on the explainer's output)
#     shap_values_reshaped = np.array(shap_values).reshape(-1, sequence_length)

#     # Plot and save SHAP values
#     plot_shap_values(shap_values_reshaped, plot_output_path)


# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator 
# from sklearn.metrics import mean_squared_error
# import os
# import sys

# # Ensure correct import paths
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from utils.data_preprocessing import preprocess_data, split_data

# def load_trained_model(model_path):
#     return load_model(model_path)

# def calculate_permutation_importance(model, X, y, metric=mean_squared_error):
#     baseline_performance = metric(y, model.predict(X))
#     feature_importances = []

#     for i in range(X.shape[2]):  # Loop over all features (assuming X is 3D: samples, timesteps, features)
#         X_permuted = X.copy()
#         np.random.shuffle(X_permuted[:, :, i])  # Shuffle individual feature
#         permuted_performance = metric(y, model.predict(X_permuted))
#         importance = baseline_performance - permuted_performance
#         feature_importances.append(importance)

#     return np.array(feature_importances)

# def plot_feature_importance(feature_importances, feature_names, output_path):
#     plt.figure(figsize=(10, 6))
#     plt.bar(feature_names, feature_importances)
#     plt.xlabel('Features')
#     plt.ylabel('Importance')
#     plt.title('Feature Importance')
#     plt.savefig(output_path)

# if __name__ == "__main__":
#     project_dir = os.path.dirname(os.path.dirname(__file__))
#     model_path = os.path.join(project_dir, 'models', 'lstm_model.h5')
#     data_file_path = os.path.join(project_dir, 'data', 'co2_data.csv')
#     plot_output_path = os.path.join(project_dir, 'data', 'feature_importance.png')

#     model = load_trained_model(model_path)
#     sequence_length = 5
#     data_scaled, scaler = preprocess_data(data_file_path, sequence_length)

#     # Split the data into training and validation sets
#     train_data_scaled, val_data_scaled = split_data(data_scaled, train_size=0.8)

#     # Create TimeseriesGenerator for validation
#     val_generator = TimeseriesGenerator(val_data_scaled, val_data_scaled[:, 0], length=sequence_length, batch_size=len(val_data_scaled))

#     # Get validation data from generator
#     X_val, y_val = val_generator[0]

#     feature_importances = calculate_permutation_importance(model, X_val, y_val)

#     # Plot and save feature importance
#     plot_feature_importance(feature_importances, ['Feature 1'], plot_output_path)

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import sys

# Add the parent directory to the Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.data_preprocessing import preprocess_data, split_data

def load_trained_model(model_path):
    return load_model(model_path)

def plot_predictions_vs_actuals(predictions, actuals, output_path):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual Values')
    plt.plot(predictions, label='Predicted Values')
    plt.title('Model Predictions vs Actual Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_dir, 'models', 'lstm_model.h5')
    data_file_path = os.path.join(project_dir, 'data', 'co2_data.csv')
    plot_output_path = os.path.join(project_dir, 'data', 'predictions_vs_actuals.png')

    model = load_trained_model(model_path)
    sequence_length = 5  # Adjust as per your model's training
    data_scaled, _ = preprocess_data(data_file_path, sequence_length)

    train_data_scaled, val_data_scaled = split_data(data_scaled, train_size=0.8)

    # Preparing validation data (adjust according to your data's structure)
    X_val = np.array([val_data_scaled[i:i + sequence_length] for i in range(len(val_data_scaled) - sequence_length)])
    y_val = val_data_scaled[sequence_length:]

    predictions = model.predict(X_val)

    # Plot and save the comparison of predictions vs actual values
    plot_predictions_vs_actuals(predictions.flatten(), y_val.flatten(), plot_output_path)
