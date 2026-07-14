import numpy as np
from ripser import ripser
import persim
import matplotlib.pyplot as plt

from data.load_data import fetch_election_markets, fetch_extended_election_markets
from src.belief_vectors import build_belief_vectors, create_sliding_windows
from src.topological_analysis import compute_tda_signal
from src.visualization import plot_comparison_3d, pca_project


def main():
    # 1. Metadata stub (kept for interface compatibility)
    df = fetch_election_markets()

    # 2. Extended Polymarket data — candidates + House/Congress
    data, shock_time = fetch_extended_election_markets()
    n_markets = data.shape[1]

    # 3. Belief vectors: 3 features × N markets
    beliefs = build_belief_vectors(data, dim=3)
    print(f"Belief vectors: {beliefs.shape[1]}D  ({n_markets} markets × 3 features)")

    # 4. Sliding windows: shape (N, window_size, 51)
    WINDOW_SIZE = 20
    windows = create_sliding_windows(beliefs, window_size=WINDOW_SIZE)

    # 5. Bottleneck distance: stable baseline vs. election night
    shock_window = max(0, min(shock_time - WINDOW_SIZE // 2, len(windows) - 1))
    stable_window = min(100, len(windows) - 1)

    print("\nCalculating bottleneck distance (51D point clouds)...")
    print(f"  Stable window index: {stable_window}")
    print(f"  Shock window index:  {shock_window}  (data index {shock_time} → window)")
    print(f"  Total windows:       {len(windows)}")

    res_stable = ripser(windows[stable_window], maxdim=1)
    res_shock  = ripser(windows[shock_window],  maxdim=1)

    dgm_stable = res_stable['dgms']
    dgm_shock  = res_shock['dgms']

    h_idx = (
        1 if (len(dgm_stable) > 1 and len(dgm_stable[1]) > 0
              and len(dgm_shock[1]) > 0)
        else 0
    )
    distance = persim.bottleneck(dgm_stable[h_idx], dgm_shock[h_idx])
    print(f"  Bottleneck Distance (H{h_idx}): {distance:.4f}")

    # 5b. Build neighbourhood-block point clouds for VR viz
    NEIGHBORHOOD = 4
    def _block_cloud(center_idx):
        lo = max(0, center_idx - NEIGHBORHOOD)
        hi = min(len(windows), center_idx + NEIGHBORHOOD // 2)
        block = windows[lo:hi]
        return block.reshape(-1, block.shape[-1])

    cloud_stable = _block_cloud(stable_window)
    cloud_shock  = _block_cloud(shock_window)

    # Recompute diagrams on richer point clouds (still in 51D)
    dgm_stable_block = ripser(cloud_stable, maxdim=1)["dgms"]
    dgm_shock_block  = ripser(cloud_shock,  maxdim=1)["dgms"]

    # PCA-project to 3D for visualisation
    cloud_stable_3d, var_s = pca_project(cloud_stable, n_components=3)
    cloud_shock_3d,  var_e = pca_project(cloud_shock,  n_components=3)
    var_avg = (var_s + var_e) / 2
    print(f"\nPCA variance explained: PC1={var_avg[0]*100:.0f}%  PC2={var_avg[1]*100:.0f}%  PC3={var_avg[2]*100:.0f}%")

    # 5c. 3D VR complex comparison (in PCA space)
    plot_comparison_3d(
        cloud_stable_3d, cloud_shock_3d,
        dgm_stable_block, dgm_shock_block,
        stable_label="Pre-Election Baseline",
        shock_label="Election Night Spike",
        save_path="point_cloud_3d_comparison.png",
        axis_labels=("PC 1", "PC 2", "PC 3"),
        var_explained=var_avg,
    )

    # 6. Rolling TDA signal
    print("Computing TDA signal across full timeline...")
    tda_signal = compute_tda_signal(windows)

    # 7. Plot
    offset = len(data) - len(tda_signal)
    plt.figure(figsize=(12, 6))
    plt.plot(range(offset, len(data)), tda_signal,
             label="TDA Total Persistence (H₀)", color="blue")
    plt.axvline(shock_time, color="red", linestyle="--", label="Election Night (Nov 5)")
    plt.title(f"2024 US Election: {n_markets}-Market Extended TDA Signal (Candidates + Congress)")
    plt.xlabel("Timestep (hours from Oct 15)")
    plt.ylabel("Total Connected Component Persistence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



# import numpy as np
# from ripser import ripser
# import persim  
# import matplotlib.pyplot as plt

# from data.load_data import fetch_election_markets, simulate_time_series
# from src.belief_vectors import build_belief_vectors, create_sliding_windows
# from src.topological_analysis import compute_tda_signal

# def main():
#     # 1. get election markets
#     df = fetch_election_markets()

#     # 2. simulate time series
   
#     data, shock_time = simulate_time_series(df, timesteps=300)

#     # 3. build belief vectors
#     beliefs = build_belief_vectors(data)

#     # 4. sliding windows
#     # increased window_size slightly to help Ripser find more features
#     windows = create_sliding_windows(beliefs, window_size=20)

#     # 5. bottleneck calculation
   
#     if len(windows) > 50:
#         print("Calculating Bottleneck Distance...")
#         # use maxdim=1 to ensure we look for both clusters (H0) and loops (H1)
#         res_stable = ripser(windows[10], maxdim=1)
#         res_shock = ripser(windows[50], maxdim=1)
        
#         dgm_stable = res_stable['dgms']
#         dgm_shock = res_shock['dgms']
        
#         # determine which dimension to compare (prefer H1 if available)
#         if len(dgm_stable) > 1 and len(dgm_shock[1]) > 0 and len(dgm_stable[1]) > 0:
#             h_idx = 1 
#         else:
#             h_idx = 0 
            
#         distance = persim.bottleneck(dgm_stable[h_idx], dgm_shock[h_idx])
#         print(f"Bottleneck Distance (Dimension H{h_idx}): {distance:.4f}")

#     # 6. TDA signal
#     print("Computing TDA signal...")
#     tda_signal = compute_tda_signal(windows)

#     # 7. plot
#     plt.figure(figsize=(10, 5))
    
#     # adjust x-axis for the sliding window offset
#     # i.e.If your window_size is 15, the first signal point corresponds to t=15
#     offset = len(data) - len(tda_signal)
#     plt.plot(range(offset, len(data)), tda_signal, label="TDA Persistence Signal", color='blue')
    
#     plt.axvline(shock_time, color='red', linestyle='--', label=f'Election Shock (t={shock_time})')
#     plt.title("2024 US Election Market: Topological Persistence")
#     plt.xlabel("Timestep")
#     plt.ylabel("Sum of $H_0$ Persistence Lengths")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.show()

# if __name__ == "__main__":
#     main()