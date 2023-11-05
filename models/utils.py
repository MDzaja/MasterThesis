import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_X(features, window_size):
    # Normalize features to range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features.values)

    # Create the 3D input data shape [samples, time_steps, features]
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, :])

    return np.array(X)

def get_Y(labels, window_size):
    return labels[window_size:].shift(-1)[:-1].values.astype(int).reshape(-1, 1)