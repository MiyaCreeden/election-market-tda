from ripser import ripser
import numpy as np


def compute_tda_metrics(point_cloud):
    """
    Computes persistence for H0 (clusters) and H1 (loops).
    Returns a combined scalar signal.
    """
    result = ripser(point_cloud, maxdim=1, distance_matrix=False)
    dgms   = result['dgms']

    h0_dgms = dgms[0][:-1]   # drop the infinite bar
    h0_persistence = np.sum(h0_dgms[:, 1] - h0_dgms[:, 0]) if len(h0_dgms) > 0 else 0

    h1_dgms = dgms[1]
    h1_persistence = np.sum(h1_dgms[:, 1] - h1_dgms[:, 0]) if len(h1_dgms) > 0 else 0

    return h0_persistence + (h1_persistence * 2)


def compute_tda_signal(windows, neighborhood=10):
    signal = []
    for i in range(neighborhood, len(windows)):
        block      = windows[i - neighborhood : i]
        point_cloud = block.reshape(-1, block.shape[-1])
        result     = ripser(point_cloud, maxdim=1)
        dgms       = result['dgms']
        h0         = dgms[0][:-1]
        val        = np.sum(h0[:, 1] - h0[:, 0]) if len(h0) > 0 else 0
        signal.append(val)
    return np.array(signal)   # <-- "and ..." removed


# from ripser import ripser
# import numpy as np

# def compute_tda_metrics(point_cloud):
#     """
#     Computes persistence for H0 (clusters) and H1 (loops).
#     Returns a combined signal.
#     """
#     # maxdim=1 ensures we get H0 and H1
#     result = ripser(point_cloud, maxdim=1, distance_matrix= False)
#     dgms = result['dgms']

#     # H0 Signal: how spread out are the market beliefs?
#     # ignore the point at infinity (the last cluster)
#     h0_dgms = dgms[0][:-1] 
#     h0_persistence = np.sum(h0_dgms[:, 1] - h0_dgms[:, 0]) if len(h0_dgms) > 0 else 0

#     # H1 Signal: are there cycles or inconsistencies in market pricing?
#     h1_dgms = dgms[1]
#     h1_persistence = np.sum(h1_dgms[:, 1] - h1_dgms[:, 0]) if len(h1_dgms) > 0 else 0

#     # for market shocks the total norm of the persistence diagram is 
#     # a standard indicator of volatility.
#     return h0_persistence + (h1_persistence * 2) # weighting H1 higher

# from ripser import ripser
# import numpy as np

# def compute_tda_signal(windows):
#     signal = []
#     # with a window_size of 20 a neighborhood of 10-15 is fine
#     neighborhood = 10 

#     for i in range(neighborhood, len(windows)):
#         # 1. grab the neighborhood ( neighborhood x window_size x features)
#         block = windows[i-neighborhood : i]
        
#         # 2. flatten the first two dimensions to make it 2D
#         # (neighborhood * window_size) points in feature-space
#         point_cloud = block.reshape(-1, block.shape[-1])
        
        
#         result = ripser(point_cloud, maxdim=1)
#         dgms = result['dgms']

#         # yse H0 Persistence (most stable)
#         h0 = dgms[0][:-1]
#         val = np.sum(h0[:, 1] - h0[:, 0])
#         signal.append(val)

#     return np.array(signal)