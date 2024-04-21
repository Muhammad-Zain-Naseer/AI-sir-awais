import numpy as np

def create_lookback_dataset(data, lookback=3):
    """
    Create dataset with lookback.

    Parameters:
        data (np.array): The original array of coordinates.
        lookback (int): The number of previous time steps to include in the features.

    Returns:
        np.array: Feature and label dataset with lookback applied.
    """
    X, Y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)].flatten())  # Flattening the window
        Y.append(data[i + lookback])
    return np.array(X), np.array(Y)

# Example Data
data = np.array([[256, 58], [264, 59], [278, 64], [290, 66], [300, 68], [310, 70]])

# Creating dataset with lookback
X, Y = create_lookback_dataset(data, lookback=1)

print("Features (X):")
print(X)
print("\nLabels (Y):")
print(Y)
