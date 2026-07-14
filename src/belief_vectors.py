import numpy as np
from sklearn.preprocessing import StandardScaler

def build_belief_vectors(data, dim=2):
    """
    Build belief representation:
    - levels (probabilities)
    - changes (momentum)
    - acceleration (second diff) — only when dim=3

    Parameters
    ----------
    data : ndarray, shape (T, 1) or (T, F)
    dim  : int, 2 or 3

    Returns
    -------
    ndarray, shape (T-1, 2) when dim=2, (T-2, 3) when dim=3
    """
    changes = np.diff(data, axis=0)          # (T-1, F)

    if dim == 2:
        levels = data[1:]                     # (T-1, F)
        combined = np.concatenate([levels, changes], axis=1)
    elif dim == 3:
        accel = np.diff(changes, axis=0)      # (T-2, F)
        levels = data[2:]                     # (T-2, F)
        changes = changes[1:]                 # align
        combined = np.concatenate([levels, changes, accel], axis=1)
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    scaler = StandardScaler()
    combined = scaler.fit_transform(combined)
    return combined


def create_sliding_windows(data, window_size=15):
    windows = []
    for i in range(len(data) - window_size):
        window = data[i : i + window_size]
        windows.append(window)
    return np.array(windows)   # <-- "and ..." removed


# import numpy as np
# from sklearn.preprocessing import StandardScaler

# def build_belief_vectors(data):
#     """
#     Build richer belief representation:
#     - levels (probabilities)
#     - changes (momentum)
#     """

#     # compute changes
#     changes = np.diff(data, axis=0)

#     # align shapes
#     levels = data[1:]

#     # combine features
#     combined = np.concatenate([levels, changes], axis=1)

#     # normalize 
#     scaler = StandardScaler()
#     combined = scaler.fit_transform(combined)

#     return combined


# def create_sliding_windows(data, window_size=15):
#     windows = []
#     for i in range(len(data) - window_size):
       
#         # want a 2D cloud of points (window_size x features)
#         window = data[i : i + window_size]
#         windows.append(window) 
#     return np.array(windows)