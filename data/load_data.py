import json
import os
import numpy as np
import pandas as pd
import requests

# Polymarket API rejects startTs->endTs intervals longer than ~14 days
MAX_FETCH_DAYS = 14

# Single-market backward-compat token (Trump Yes, market 253591)
YES_TOKEN_ID = "21742633143463906290569050155826241533067272736897614950488156847949938836455"
CACHE_FILE = "polymarket_data/polymarket_trump_2024.csv"

# Event-level discovery
EVENT_SLUG = "presidential-election-winner-2024"
MARKET_REGISTRY_FILE = "election_markets.json"
EXTENDED_REGISTRY_FILE = "election_markets_extended.json"

# House/Congress events to include in extended pipeline
HOUSE_EVENT_IDS = ["14033", "14268", "903514"]  # Speaker, GOP seats, House control


# ---- market discovery ----

def _discover_markets():
    """Return list of {id, name, yes_token, volume} for candidate markets."""
    if os.path.exists(MARKET_REGISTRY_FILE):
        with open(MARKET_REGISTRY_FILE) as f:
            markets = json.load(f)
        print(f">>> Loaded {len(markets)} markets from {MARKET_REGISTRY_FILE}")
        return markets

    print(f">>> Discovering markets for event: {EVENT_SLUG}")
    url = f"https://gamma-api.polymarket.com/events/slug/{EVENT_SLUG}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    event = resp.json()

    markets = []
    for m in event.get("markets", []):
        raw_tokens = m.get("clobTokenIds", "[]")
        tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
        yes_token = tokens[0] if len(tokens) > 0 else None
        if yes_token is None:
            continue
        name = (m["question"]
                .replace("Will ", "")
                .replace(" win the 2024 US Presidential Election?", ""))
        markets.append({
            "id": m["id"],
            "name": name,
            "yes_token": yes_token,
            "volume": float(m.get("volume", 0)),
        })
    markets.sort(key=lambda x: x["volume"], reverse=True)
    with open(MARKET_REGISTRY_FILE, "w") as f:
        json.dump(markets, f, indent=2)
    print(f">>> Cached {len(markets)} markets to {MARKET_REGISTRY_FILE}")
    return markets


def _discover_house_markets():
    """Return list of {id, question, yes_token, volume} for House/Congress markets."""
    markets = []
    for eid in HOUSE_EVENT_IDS:
        url = f"https://gamma-api.polymarket.com/events/{eid}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        ev = resp.json()
        for m in ev.get("markets", []):
            raw_tokens = m.get("clobTokenIds", "[]")
            tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
            yes_token = tokens[0] if len(tokens) > 0 else None
            if yes_token is None:
                continue
            q = m["question"]
            q = q.replace("Will ", "")
            q = q.replace(" be the first elected Speaker of the House for the 119th congress?", "")
            q = q.replace(" after the election?", "")
            q = q.replace(" after 2024 election?", "")
            markets.append({
                "id": m["id"],
                "question": q,
                "yes_token": yes_token,
                "volume": float(m.get("volume", 0)),
            })
    print(f">>> Discovered {len(markets)} House/Congress markets")
    return markets


def _build_extended_registry():
    """Combine candidate markets + house markets into single registry."""
    if os.path.exists(EXTENDED_REGISTRY_FILE):
        with open(EXTENDED_REGISTRY_FILE) as f:
            markets = json.load(f)
        print(f">>> Loaded {len(markets)} extended markets from {EXTENDED_REGISTRY_FILE}")
        return markets
    candidates = _discover_markets()
    house = _discover_house_markets()
    all_markets = candidates + house
    with open(EXTENDED_REGISTRY_FILE, "w") as f:
        json.dump(all_markets, f, indent=2)
    print(f">>> Cached {len(all_markets)} extended markets to {EXTENDED_REGISTRY_FILE}")
    return all_markets


# ---- per-market fetching ----

def _fetch_market_prices(token_id, start_ts, end_ts, label=""):
    """Fetch price history for a single CLOB token, chunking by MAX_FETCH_DAYS."""
    chunks = []
    chunk_start = start_ts
    max_interval = MAX_FETCH_DAYS * 24 * 3600
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + max_interval, end_ts)
        chunk_df = _fetch_chunk(token_id, chunk_start, chunk_end)
        if len(chunk_df) > 0:
            chunks.append(chunk_df)
        chunk_start = chunk_end
    if not chunks:
        return pd.DataFrame(columns=["timestamp", "price"])
    df = pd.concat(chunks, ignore_index=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _fetch_chunk(token_id, start_ts, end_ts):
    """Single API call - returns DataFrame (may be empty)."""
    base = "https://clob.polymarket.com/prices-history"
    resp = requests.get(
        base,
        params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 60},
        timeout=15,
    )
    resp.raise_for_status()
    records = resp.json().get("history", [])
    if not records:
        return pd.DataFrame(columns=["timestamp", "price"])
    df = pd.DataFrame(records).rename(columns={"t": "timestamp", "p": "price"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


# ---- public API ----

def fetch_election_markets():
    """Legacy stub - kept for interface compatibility."""
    return pd.DataFrame({"market_id": range(100)})


def fetch_real_historical_data():
    """Legacy single-market fetch (Trump Yes). Kept for backward compat."""
    print(">>> Fetching single-market (Trump) election data...")
    start_ts = int(pd.Timestamp("2024-10-15", tz="UTC").timestamp())
    end_ts = int(pd.Timestamp("2024-11-30", tz="UTC").timestamp())
    if os.path.exists(CACHE_FILE):
        print(f">>> Loading from cache: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, parse_dates=["timestamp"])
    else:
        df = _fetch_market_prices(YES_TOKEN_ID, start_ts, end_ts, label="Trump")
        df.to_csv(CACHE_FILE, index=False)
        print(f">>> Cached {len(df)} records to {CACHE_FILE}")
    election_night_utc = pd.Timestamp("2024-11-05 23:00:00", tz="UTC")
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    election_night_index = int(np.abs((df["timestamp"] - election_night_utc).dt.total_seconds()).argmin())
    data = df["price"].to_numpy(dtype=float).reshape(-1, 1)
    print(f">>> Data shape: {data.shape}, election night index: {election_night_index}")
    return data, election_night_index


def _fetch_and_align(markets, start_ts, end_ts, election_night_utc,
                     require_pre_election=False):
    """Shared pipeline: fetch each market, align to hourly grid, return (data, idx)."""
    price_series = []
    columns = []

    for i, m in enumerate(markets):
        cache_file = f"polymarket_data/polymarket_{m['id']}.csv"
        col_name = m.get("question", m.get("name", m["id"]))[:60]

        if os.path.exists(cache_file):
            print(f">>> [{i+1}/{len(markets)}] Loading {col_name} from cache")
            df = pd.read_csv(cache_file, parse_dates=["timestamp"])
        else:
            vol_str = "${:,.0f}".format(m["volume"])
            print(f">>> [{i+1}/{len(markets)}] Fetching {col_name}  (vol={vol_str})")
            df = _fetch_market_prices(m["yes_token"], start_ts, end_ts, label=col_name)
            df.to_csv(cache_file, index=False)
            print(f">>>   Cached {len(df)} records to {cache_file}")

        if len(df) == 0:
            continue

        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        if require_pre_election and df["timestamp"].min() > election_night_utc:
            print(f">>>   Skipping {col_name} - no pre-election data (starts {df['timestamp'].min().strftime('%b %d')})")
            continue

        df = df.set_index("timestamp")
        df = df[~df.index.duplicated()]
        series = df["price"].rename(col_name)
        price_series.append(series)
        columns.append(col_name)

    full_index = pd.date_range(
        start=pd.Timestamp(start_ts, unit="s", tz="UTC"),
        end=pd.Timestamp(end_ts, unit="s", tz="UTC"),
        freq="h",
    )
    aligned = pd.DataFrame(index=full_index)
    for col, s in zip(columns, price_series):
        floored = s.copy()
        floored.index = floored.index.floor("h")
        floored = floored[~floored.index.duplicated(keep="last")]
        aligned[col] = floored
    aligned.ffill(inplace=True)
    aligned.dropna(inplace=True)

    data = aligned.to_numpy(dtype=float)
    idx = int(np.abs((aligned.index - election_night_utc).total_seconds()).argmin())

    print(f">>> Aligned data: {data.shape[0]} timesteps x {data.shape[1]} markets")
    print(f">>> Columns ({len(columns)}): {', '.join(columns[:5])}... +{len(columns)-5} more")
    print(f">>> Election night index: {idx}")
    return data, idx


def fetch_all_election_markets():
    """Fetch candidate markets only (17 markets)."""
    markets = _discover_markets()
    start_ts = int(pd.Timestamp("2024-10-15", tz="UTC").timestamp())
    end_ts = int(pd.Timestamp("2024-11-30", tz="UTC").timestamp())
    election_night_utc = pd.Timestamp("2024-11-05 23:00:00", tz="UTC")
    return _fetch_and_align(markets, start_ts, end_ts, election_night_utc)


def fetch_extended_election_markets():
    """Fetch candidate markets + House/Congress markets (those with pre-election data)."""
    markets = _build_extended_registry()
    start_ts = int(pd.Timestamp("2024-10-15", tz="UTC").timestamp())
    end_ts = int(pd.Timestamp("2024-11-30", tz="UTC").timestamp())
    election_night_utc = pd.Timestamp("2024-11-05 23:00:00", tz="UTC")
    return _fetch_and_align(markets, start_ts, end_ts, election_night_utc,
                            require_pre_election=True)
