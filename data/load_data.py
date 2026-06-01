


import requests
import pandas as pd
import numpy as np

# -----------------------------
# 1. FETCH ELECTION MARKETS
# -----------------------------

import json # parse the string version of lists

def fetch_election_markets(min_volume=5000, debug=True):
    url = "https://gamma-api.polymarket.com/markets"
    
    # using 'closed': 'true' to get historical 2024 data
    # data rn is not historical **
    params = {
        "tag_id": 1101,
        "closed": "true", 
        "limit": 100
    }

    try:
        response = requests.get(url, params=params)
        markets = response.json()
    except Exception as e:
        print(f"API Error: {e}")
        markets = []

    data = []
    for m in markets:
        # get the price field
        raw_prices = m.get("outcomePrices")
        volume = float(m.get("volumeNum", 0))
        
        if volume > min_volume and raw_prices:
            try:
                #convert the stringified list into a Python list
                
                parsed_prices = json.loads(raw_prices)
                
                # take the first price and convert to float
                price = float(parsed_prices[0])

                data.append({
                    "market": m.get("question"),
                    "price": price,
                    "volume": volume
                })
            except (json.JSONDecodeError, ValueError, IndexError):
                continue

    df = pd.DataFrame(data)
    
    if df.empty:
        print("\nWARNING: No real data found. Using high-dim fallback for TDA.")
        # created 20 fake markets so the TDA has actual dimensions to work with
        df = pd.DataFrame({
            "market": [f"Market_{i}" for i in range(20)],
            "price": np.random.uniform(0.4, 0.6, 20),
            "volume": [10000] * 20
        })

    if debug:
        print(f"\nLoaded {len(df)} markets.")
    
    return df
# -----------------------------
# 2. SIMULATE TIME SERIES
# -----------------------------

def simulate_time_series(df, timesteps=100):
    # check valid base for simulation
    base = df["price"].values
    n_markets = len(base)

    data = []

    for t in range(timesteps):
        # generate random noise
        noise = np.random.normal(0, 0.005, n_markets)
        if t == 0:
            current_vals = base + noise
        else:
            current_vals = data[-1] + noise
            
        current_vals = np.clip(current_vals, 0.01, 0.99)
        data.append(current_vals)

    data = np.array(data)

    # inject "election shock" at timestep 50
    # creates a localized cluster shift for TDA to detect
    shock_time = 50
    data[shock_time:shock_time+5] += np.random.normal(0.08, 0.02, (5, n_markets))
    data = np.clip(data, 0.01, 0.99)

    return data, shock_time

