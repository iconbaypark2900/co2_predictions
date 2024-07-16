import tensorflow as tf

def create_lstm_model(input_shape, units=50, dropout=0.2):
    """
    Create an LSTM model for time series prediction.
    Args:
        input_shape (tuple): Shape of the input data (time_steps, n_features).
        units (int): Number of units in the LSTM layers.
        dropout (float): Dropout rate for regularization.

    Returns:
        model (tf.keras.Model): Compiled LSTM model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units, return_sequences=True, input_shape=input_shape, dropout=dropout),
        tf.keras.layers.LSTM(units, dropout=dropout),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32, validation_data=None):
    """
    Train the LSTM model.
    Args:
        model (tf.keras.Model): LSTM model to be trained.
        X_train (array): Training data features.
        y_train (array): Training data labels.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        validation_data (tuple): Validation data (X_val, y_val).

    Returns:
        history (History): History object containing training history.
    """
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    return history
