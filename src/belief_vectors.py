import numpy as np
from sklearn.preprocessing import StandardScaler

def build_belief_vectors(data):
    """
    Build richer belief representation:
    - levels (probabilities)
    - changes (momentum)
    """

    # compute changes
    changes = np.diff(data, axis=0)

    # align shapes
    levels = data[1:]

    # combine features
    combined = np.concatenate([levels, changes], axis=1)

    # normalize 
    scaler = StandardScaler()
    combined = scaler.fit_transform(combined)

    return combined


def create_sliding_windows(data, window_size=15):
    windows = []
    for i in range(len(data) - window_size):
       
        # want a 2D cloud of points (window_size x features)
        window = data[i : i + window_size]
        windows.append(window) 
    return np.array(windows)