# CLAUDE.md

Topological Data Analysis (TDA) applied to election prediction markets. Uses persistent homology (Ripser) to detect information shocks by measuring topological change in belief vector point clouds. Case study: 2024 US Presidential Election on Polymarket.

## Commands

```bash
# Run full pipeline (fetches data, computes TDA, plots)
python3 main.py

# Headless mode (no GUI)
MPLBACKEND=Agg python3 main.py

# One-shot: fetch single-market Trump data → CSV cache
python3 -c "from data.load_data import fetch_real_historical_data; fetch_real_historical_data()"
```

No test runner, linter, or build configured.

## Dependencies

No pinned requirements file (`requirments.txt` is empty). No venv — packages installed globally under Python 3.14:
`ripser` `persim` `numpy` `pandas` `scikit-learn` `scipy` `requests` `matplotlib`

## Architecture

Five-module pipeline:

```
data/load_data.py          → Polymarket API data acquisition
src/belief_vectors.py      → Feature engineering (levels, momentum, acceleration)
src/topological_analysis.py → TDA computation (Ripser persistence)
src/visualization.py       → 3D VR complex rendering + persistence diagrams
main.py                    → Orchestration + plotting
```

### Data acquisition (`data/load_data.py`)

- **CSV cache dir**: `polymarket_data/` — 48 market CSVs (candidates + House/Congress). Paths in code reference `polymarket_data/polymarket_{id}.csv`.
- `fetch_extended_election_markets()` — **primary pipeline entry**. Loads 47 markets from `election_markets_extended.json`, fetches price history Oct 15 – Nov 30 2024, 60-min fidelity, aligns on hourly grid, filters to pre-election data only. Returns `(data_array, election_night_index)`. Typically yields 18 markets, 1105 timesteps.
- `fetch_all_election_markets()` — candidate-only variant (17 markets, no House/Congress).
- `fetch_real_historical_data()` — legacy single-market (Trump Yes, market 253591). Caches to `polymarket_data/polymarket_trump_2024.csv`.
- `_fetch_market_prices()` — chunks requests into ≤14-day intervals (API rejects longer).
- `YES_TOKEN_ID` — correct CLOB token for market 253591. Do NOT change without checking Gamma API.
- `_build_extended_registry()` — merges candidate markets + House/Congress markets into `election_markets_extended.json`.
- House event IDs: `["14033", "14268", "903514"]` (Speaker, GOP seats, House control).
- Post-resolution market data is sparse (API compresses/returns empty for resolved markets at high fidelity). Many House seat-count markets have no pre-election data — they get skipped.

### Feature engineering (`src/belief_vectors.py`)

- `build_belief_vectors(data, dim=3)` — input `(T, F)`, computes np.diff for momentum, np.diff again for acceleration. Concatenates `[levels, changes, accel]`. StandardScaler normalizes. Output `(T-2, F×3)` — for 18 markets → 54D.
- `dim=2` mode: `[levels, changes]` only, output `(T-1, F×2)`.
- `create_sliding_windows(data, window_size=20)` — sliding windows. Output `(N, window_size, F×dim)`.

### TDA computation (`src/topological_analysis.py`)

- `compute_tda_signal(windows, neighborhood=10)` — for each position, flattens a block of `neighborhood` windows into 2D point cloud (200 pts × 54D), runs Ripser `maxdim=1`, records total H0 persistence. Returns 1D signal array. **VERY SLOW** — ~1073 Ripser calls on 54D data. May need PCA pre-reduction or striding for practical runs.
- `compute_tda_metrics(point_cloud)` — computes H0 + 2×H1 persistence. **Defined but never called in main.py.**

### Visualization (`src/visualization.py`)

- `plot_comparison_3d()` — 2×2 figure: stable vs shock VR complexes (top row), persistence diagrams (bottom row). Saves to PNG.
- `pca_project(point_cloud, n_components=3)` — PCA projection to 3D. Returns `(projected, variance_ratio)`.
- `plot_vr_complex_3d()` — single VR complex: points + edges + filled triangles. Auto-picks filtration radius from persistence diagram.
- `plot_persistence_3d()` — 3D bar chart: birth × death × persistence, H0 blue, H1 red.
- `_build_vr_complex()` — manual Vietoris-Rips construction at given radius (edges + triangles).

### Pipeline (`main.py`)

1. Fetch extended multi-market data (18 markets)
2. Build belief vectors `dim=3` → `(T-2, 54)`
3. Create sliding windows `window_size=20` → `(N, 20, 54)`
4. Bottleneck distance: stable (index 100) vs election night shock. Computes on full 54D point clouds
5. Block point clouds for VR viz (neighborhood=4 blocks), PCA-project to 3D
6. 3D comparison plot → `point_cloud_3d_comparison.png`
7. Rolling TDA signal across all windows (**slow** — may hang without optimization)
8. Matplotlib time-series plot (TDA signal + red dashed election night line)

## Output files

| File | Source |
|------|--------|
| `point_cloud_3d_comparison.png` | `plot_comparison_3d()` in main.py |
| `tda_signal.png` | Rolling TDA signal plot in main.py |
| `polymarket_data/*.csv` | Per-market price caches (48 files) |
| `election_markets.json` | Candidate market registry |
| `election_markets_extended.json` | Candidate + House/Congress registry (47 markets) |
| `house_markets.json` | House-only market registry |
| `multi_market_extended.npy` | Cached aligned data array |

## Key technical detail

- Ripser `maxdim=1` computes H0 (components) and H1 (loops). Code prefers H1 for bottleneck comparison, falls back to H0. Signal uses H0 only.
- `main.py` uses `plt.show()` — blocks on GUI. Use `MPLBACKEND=Agg` for headless runs.
- Bottleneck distance: `shock_time` is raw data index → converted to window index via `shock_time - WINDOW_SIZE//2`.
- 54D Vietoris-Rips is combinatorially expensive. PC1-3 capture ~92% variance — PCA pre-reduction to ~10D viable for speed.
- Python 3.14.3, no venv (global installs). Run `python3`, not `python`.
- Commented-out legacy code in all source files references `simulate_time_series()` (now removed) — ignore.
