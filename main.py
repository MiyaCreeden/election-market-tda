import numpy as np
from ripser import ripser
import persim  
import matplotlib.pyplot as plt

from data.load_data import fetch_election_markets, simulate_time_series
from src.belief_vectors import build_belief_vectors, create_sliding_windows
from src.topological_analysis import compute_tda_signal

def main():
    # 1. get election markets
    df = fetch_election_markets()

    # 2. simulate time series
   
    data, shock_time = simulate_time_series(df, timesteps=300)

    # 3. build belief vectors
    beliefs = build_belief_vectors(data)

    # 4. sliding windows
    # increased window_size slightly to help Ripser find more features
    windows = create_sliding_windows(beliefs, window_size=20)

    # 5. bottleneck calculation
   
    if len(windows) > 50:
        print("Calculating Bottleneck Distance...")
        # use maxdim=1 to ensure we look for both clusters (H0) and loops (H1)
        res_stable = ripser(windows[10], maxdim=1)
        res_shock = ripser(windows[50], maxdim=1)
        
        dgm_stable = res_stable['dgms']
        dgm_shock = res_shock['dgms']
        
        # determine which dimension to compare (prefer H1 if available)
        if len(dgm_stable) > 1 and len(dgm_shock[1]) > 0 and len(dgm_stable[1]) > 0:
            h_idx = 1 
        else:
            h_idx = 0 
            
        distance = persim.bottleneck(dgm_stable[h_idx], dgm_shock[h_idx])
        print(f"Bottleneck Distance (Dimension H{h_idx}): {distance:.4f}")

    # 6. TDA signal
    print("Computing TDA signal...")
    tda_signal = compute_tda_signal(windows)

    # 7. plot
    plt.figure(figsize=(10, 5))
    
    # adjust x-axis for the sliding window offset
    # i.e.If your window_size is 15, the first signal point corresponds to t=15
    offset = len(data) - len(tda_signal)
    plt.plot(range(offset, len(data)), tda_signal, label="TDA Persistence Signal", color='blue')
    
    plt.axvline(shock_time, color='red', linestyle='--', label=f'Election Shock (t={shock_time})')
    plt.title("2024 US Election Market: Topological Persistence")
    plt.xlabel("Timestep")
    plt.ylabel("Sum of $H_0$ Persistence Lengths")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()