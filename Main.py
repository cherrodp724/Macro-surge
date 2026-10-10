import datetime
import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Macro Collateral & Surge Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# API Configuration (Streamlit Secrets, Env Vars, or Sidebar)
# -------------------------------------------------------------
default_key = ""
if "POLYGON_API_KEY" in st.secrets:
    default_key = st.secrets["POLYGON_API_KEY"]
elif "MASSIVE_API_KEY" in st.secrets:
    default_key = st.secrets["MASSIVE_API_KEY"]
else:
    default_key = os.environ.get("POLYGON_API_KEY", "") or os.environ.get("MASSIVE_API_KEY", "")

with st.sidebar:
    st.header("1. API Configuration")
    api_key_input = st.text_input(
        "Massive / Polygon API Key",
        value=default_key,
        type="password",
        help="Paste your key here or add it to Streamlit App Secrets as POLYGON_API_KEY."
    )
    
    st.header("2. Scanner Parameters")
    min_tranche = st.slider("Min Maturity Tranche ($B)", 50, 250, 100, step=25)
    min_rvol_input = st.slider("Min RVOL Threshold", 1.5, 10.0, 2.0, step=0.5)
    
    # Lookback selector to bypass 429 limits or force deep scans
    baseline_lookback = st.slider(
        "Baseline Lookback Days", 
        3, 10, 5, 
        help="Free tiers allow 5 req/min. Use 3-5 days for fast scans. Use 10 days for deeper baseline (will trigger the 65s pause)."
    )
    
    today = datetime.date.today()
    start_date = st.date_input("Start Maturity Window", today - datetime.timedelta(days=30))
    end_date = st.date_input("End Maturity Window", today)

# -------------------------------------------------------------
# 1. Fetch Treasury Maturing Tranches ($100B+)
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_treasury_maturities(start_date, end_date, min_size_billions=100):
    min_offering_amt = min_size_billions * 1_000_000_000
    url = (
        f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
        f"?filter=maturity_date:gte:{start_date},maturity_date:lte:{end_date}"
        f"&page[size]=1000"
    )
    try:
        res = requests.get(url, timeout=15).json()
    except Exception as e:
        st.error(f"Error fetching Treasury Fiscal Data: {e}")
        return pd.DataFrame()

    records = []
    for item in res.get("data", []):
        offering = float(item.get("offering_amt") or 0)
        accepted = float(item.get("total_accepted") or 0)
        size = offering if offering > 0 else accepted

        if size >= min_offering_amt:
            records.append({
                "cusip": item.get("cusip"),
                "security_type": item.get("security_type"),
                "security_term": item.get("security_term"),
                "offering_amt": size,
                "offering_b": round(size / 1_000_000_000, 2),
                "maturity_date": item.get("maturity_date"),
                "auction_date": item.get("auction_date"),
                "issue_date": item.get("issue_date")
            })
    return pd.DataFrame(records)

# -------------------------------------------------------------
# 2. Market Data Helpers & Resilient Rate-Limit Management
# -------------------------------------------------------------
def get_historical_trading_days(target_date, lookback=10):
    """Generates prior business days, skipping weekends."""
    days = []
    curr = target_date - datetime.timedelta(days=1)
    while len(days) < lookback:
        if curr.weekday() < 5:
            days.append(curr.strftime("%Y-%m-%d"))
        curr -= datetime.timedelta(days=1)
    return days

@st.cache_data(ttl=86400)
def fetch_raw_day_data(d_str, api_key):
    """Pulls grouped daily bars with progressive rate-limit backoff and custom countdown."""
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{d_str}?adjusted=true&apiKey={api_key}"
    retries = 2
    
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=15)
            
            # Catch 429 Rate Limit - Trigger 'Paytience' Countdown
            if res.status_code == 429:
                status_box = st.empty()
                for sec in range(65, 0, -1):
                    status_box.warning(
                        f"⏳ **Paytience...** API limit reached. Countdown to next batch of results: **{sec}s**"
                    )
                    time.sleep(1)
                status_box.empty()
                continue  # Retry after the 65s wait is over
                
            # Catch unauthorized keys explicitly
            if res.status_code in [401, 403]:
                st.error("❌ Invalid API Key. Please verify in the sidebar.")
                return []
                
            if res.status_code != 200:
                return []
                
            data = res.json()
            results = data.get("results", [])
            if not results:
                return []  # Return empty for weekends/holidays
                
            rows = []
            for r in results:
                vol = r.get("v", 0)
                close = r.get("c", 0)
                ticker = r.get("T", "")
                # Institutional liquidity filter
                if vol >= 500_000 and (vol * close) >= 20_000_000 and len(ticker) <= 5 and "." not in ticker:
                    rows.append({
                        "ticker": ticker,
                        "close": close,
                        "volume": vol,
                        "dollar_volume": vol * close
                    })
            return rows
            
        except Exception:
            time.sleep(1)
            
    return []

def build_smoothed_baseline(trading_days, api_key):
    """Averages volume per ticker across the dynamic lookback sessions."""
    dfs = []
    pbar = st.progress(0, text="Building Baseline...")
    
    for idx, d in enumerate(trading_days):
        day_records = fetch_raw_day_data(d, api_key)
        if day_records and isinstance(day_records, list):
            dfs.append(pd.DataFrame(day_records))
        pbar.progress((idx + 1) / len(trading_days), text=f"Processing Session {d}...")
        time.sleep(0.5)  
        
    pbar.empty()
    if not dfs:
        return pd.DataFrame()

    full_history = pd.concat(dfs, ignore_index=True)
    return full_history.groupby("ticker")["volume"].mean().reset_index(name="avg_volume_10d")

# -------------------------------------------------------------
# 3. Main Interface & Execution
# -------------------------------------------------------------
st.title("Macro Collateral & Volume Surge Scanner")
st.caption("Detecting institutional accumulation around $100B+ Treasury maturities using smoothed multi-day RVOL math.")

if not api_key_input:
    st.warning("Enter your Massive/Polygon API Key in the sidebar to begin.")
    st.stop()

# Step A: Table of Maturing Tranches
st.subheader(f"1. Maturing Debt Schedules (>= ${min_tranche}B)")
df_treasury = fetch_treasury_maturities(str(start_date), str(end_date), min_size_billions=min_tranche)

if df_treasury.empty:
    st.info("No maturity records matching that tranche threshold in this date range.")
    st.stop()

df_display = df_treasury[[
    "cusip", "security_type", "security_term", "offering_b", "auction_date", "issue_date", "maturity_date"
]].copy()

# Format and Sort chronologically
df_display.columns = ["CUSIP", "Security Type", "Term", "Amount ($B)", "Auction Date", "Issue Date", "Maturity Date"]
df_display = df_display.sort_values(by="Maturity Date", ascending=False).reset_index(drop=True)

st.dataframe(df_display, use_container_width=True)

# Step B: Select maturity date to anchor scan
st.subheader("2. Run Auction Scan via Maturity Anchor")
target_maturity_selected = st.selectbox(
    "Select Maturing Date to Search & Trigger Auction-Day Scan",
    options=sorted(df_treasury["maturity_date"].unique(), reverse=True)
)

matching_tranches = df_treasury[df_treasury["maturity_date"] == target_maturity_selected].copy()

if not matching_tranches.empty:
    target_auction_date = matching_tranches["auction_date"].iloc[0]
    
    st.info(
        f"⚓ Maturing Date chosen: **{target_maturity_selected}**.\n"
        f"🎯 Target Auction Date: **{target_auction_date}** ($T+0$). Searching window using Fallback checking to prevent market holidays."
    )
    
