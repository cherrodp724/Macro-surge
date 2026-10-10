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
    min_tranche = st.slider("Min Cumulative Maturity ($B)", 50, 250, 100, step=25)
    min_rvol_input = st.slider("Min RVOL Threshold", 1.5, 10.0, 2.0, step=0.5)
    
    baseline_lookback = st.slider(
        "Baseline Lookback Days", 
        3, 10, 5, 
        help="Free tiers allow 5 req/min. Use 3-5 days for fast scans. Use 10 days for deeper baseline (will trigger the 65s pause)."
    )
    
    today = datetime.date.today()
    start_date = st.date_input("Start Maturity Window", today - datetime.timedelta(days=30))
    end_date = st.date_input("End Maturity Window", today)

# -------------------------------------------------------------
# 1. Fetch Treasury Maturing Tranches (Aggregated Openings + Reopenings)
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_treasury_maturities(start_date, end_date, min_size_billions=100):
    url = (
        f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
        f"?filter=maturity_date:gte:{start_date},maturity_date:lte:{end_date}"
        f"&page[size]=5000"  # Increased page size to capture all micro-auctions
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
        
        if size > 0:
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
            
    df = pd.DataFrame(records)
    if df.empty:
        return df
        
    # Group by maturity date to calculate the TRUE combined tranche size (Openings + Reopenings)
    daily_totals = df.groupby("maturity_date")["offering_b"].sum().reset_index()
    
    # Find which maturity dates meet your minimum threshold
    valid_dates = daily_totals[daily_totals["offering_b"] >= min_size_billions]["maturity_date"]
    
    # Keep all individual auction events that belong to those massive maturity dates
    filtered_df = df[df["maturity_date"].isin(valid_dates)].copy()
    return filtered_df

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
                continue
                
            if res.status_code in [401, 403]:
                st.error("❌ Invalid API Key. Please verify in the sidebar.")
                return []
                
            if res.status_code != 200:
                return []
                
            data = res.json()
            results = data.get("results", [])
            if not results:
                return []
                
            rows = []
            for r in results:
                vol = r.get("v", 0)
                close = r.get("c", 0)
                ticker = r.get("T", "")
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
st.subheader(f"1. Maturing Debt Schedules (Cumulative >= ${min_tranche}B)")
df_treasury = fetch_treasury_maturities(str(start_date), str(end_date), min_size_billions=min_tranche)

if df_treasury.empty:
    st.info("No maturity records matching that tranche threshold in this date range.")
    st.stop()

df_display = df_treasury[[
    "cusip", "security_type", "security_term", "offering_b", "auction_date", "issue_date", "maturity_date"
]].copy()

# Format and Sort chronologically
df_display.columns = ["CUSIP", "Security Type", "Term", "Amount ($B)", "Auction Date", "Issue Date", "Maturity Date"]
df_display = df_display.sort_values(by=["Maturity Date", "Auction Date"], ascending=[False, True]).reset_index(drop=True)

st.dataframe(df_display, use_container_width=True)

# Step B: Select maturity date to anchor scan
st.subheader("2. Run Auction Scan via Maturity Anchor")
target_maturity_selected = st.selectbox(
    "Select Maturing Date to Search & Trigger Auction-Day Scans",
    options=sorted(df_treasury["maturity_date"].unique(), reverse=True)
)

matching_tranches = df_treasury[df_treasury["maturity_date"] == target_maturity_selected].copy()

if not matching_tranches.empty:
    # Identify ALL unique auction dates (Opening and Reopenings) associated with this maturity date
    unique_auction_dates = sorted(matching_tranches["auction_date"].dropna().unique())
    total_liquidity = matching_tranches['offering_b'].sum()
    
    st.info(
        f"⚓ **Maturing Date:** {target_maturity_selected} | **Cumulative Size:** ${total_liquidity:,.1f}B\n\n"
        f"🎯 Found **{len(unique_auction_dates)}** associated Auction Date(s): {', '.join(unique_auction_dates)}."
    )
    
    if st.button("Run T-1 & T+0 Settlement Scan for All Auctions"):
        # Iterate over every unique auction date found for this tranche
        for target_auction_date in unique_auction_dates:
            st.markdown("---")
            st.subheader(f"🏦 Scanning Auction Event: {target_auction_date}")
            
            # Determine target date datetime
            t_0_dt = datetime.datetime.strptime(target_auction_date, "%Y-%m-%d").date()
            
            # Check if auction is in the future
            if t_0_dt > datetime.date.today():
                st.warning(f"Auction date {target_auction_date} is in the future. Skipping market data pull.")
                continue

            while t_0_dt.weekday() >= 5:
                t_0_dt -= datetime.timedelta(days=1)
                
            t_0_str = str(t_0_dt)
            
            # ACTIVE BACKTRACKING FOR T+0
            df_t0 = pd.DataFrame()
            with st.spinner(f"[{target_auction_date}] Verifying active session on Auction Date..."):
                for _ in range(5): 
                    t0_data = fetch_raw_day_data(t_0_str, api_key_input)
                    if t0_data:
                        df_t0 = pd.DataFrame(t0_data)
                        break
                    else:
                        st.caption(f"⚠️ {t_0_str} was closed (possible holiday). Searching previous session...")
                        t_0_dt -= datetime.timedelta(days=1)
                        while t_0_dt.weekday() >= 5:
                            t_0_dt -= datetime.timedelta(days=1)
                        t_0_str = str(t_0_dt)
                        time.sleep(0.5)

            # ACTIVE BACKTRACKING FOR T-1
            df_t1 = pd.DataFrame()
            if not df_t0.empty:
                t_minus_1_dt = t_0_dt - datetime.timedelta(days=1)
                while t_minus_1_dt.weekday() >= 5:
                    t_minus_1_dt -= datetime.timedelta(days=1)
                    
                t_minus_1_str = str(t_minus_1_dt)
                
                with st.spinner(f"[{target_auction_date}] Verifying active trading day prior to {t_0_str}..."):
                    for _ in range(5):
                        t1_data = fetch_raw_day_data(t_minus_1_str, api_key_input)
                        if t1_data:
                            df_t1 = pd.DataFrame(t1_data)
                            break
                        else:
                            st.caption(f"⚠️ {t_minus_1_str} was closed. Searching previous session...")
                            t_minus_1_dt -= datetime.timedelta(days=1)
                            while t_minus_1_dt.weekday() >= 5:
                                t_minus_1_dt -= datetime.timedelta(days=1)
                            t_minus_1_str = str(t_minus_1_dt)
                            time.sleep(0.5)
                            
            if df_t0.empty or df_t1.empty:
                st.error(f"No active trading sessions found within a 5-day window of {target_auction_date}. Skipping.")
                continue
                
            # Lookback baseline
            lookback_days = get_historical_trading_days(t_minus_1_dt, lookback=baseline_lookback)
            
            with st.spinner(f"[{target_auction_date}] Evaluating {baseline_lookback}-day lookback session baseline..."):
                df_base = build_smoothed_baseline(lookback_days, api_key_input)
                
            if df_base.empty:
                st.error(f"Historical baseline could not be generated for {target_auction_date}.")
                continue
                
            # Merge and Calculate
            df_t1 = df_t1[['ticker', 'volume', 'close']].rename(columns={'volume': 'vol_T1', 'close': 'close_T1'})
            df_t0 = df_t0[['ticker', 'volume', 'close']].rename(columns={'volume': 'vol_T0', 'close': 'close_T0'})
            
            merged = pd.merge(df_t1, df_t0, on="ticker", how="inner")
            merged = pd.merge(merged, df_base, on="ticker", how="inner")
            
            merged = merged[merged["avg_volume_10d"] >= 500_000].copy()
            
            merged["rvol_T1"] = merged["vol_T1"] / merged["avg_volume_10d"]
            merged["rvol_T0"] = merged["vol_T0"] / merged["avg_volume_10d"]
            
            merged = merged[(merged["rvol_T1"] >= min_rvol_input) | (merged["rvol_T0"] >= min_rvol_input)].copy()
            
            merged["double_surge"] = (merged["rvol_T1"] >= min_rvol_input) & (merged["rvol_T0"] >= min_rvol_input)
            
            merged["base_score"] = np.log10(merged["vol_T1"] + merged["vol_T0"]) * (merged["rvol_T1"] + merged["rvol_T0"])
            merged["surge_score"] = np.where(merged["double_surge"], merged["base_score"] * 2.0, merged["base_score"])
            
            results = merged.sort_values(by="surge_score", ascending=False).reset_index(drop=True)
            
            st.write(f"**Results for Verified Windows: {t_minus_1_str} ($T-1$) & {t_0_str} ($T+0$)**")
            if not results.empty:
                out = results.head(25).copy()
                
                out["Type"] = np.where(out["double_surge"], "🔥 DOUBLE SURGE", "Single Spike")
                out["vol_T1"] = out["vol_T1"].apply(lambda x: f"{x:,.0f}")
                out["vol_T0"] = out["vol_T0"].apply(lambda x: f"{x:,.0f}")
                out["avg_vol"] = out["avg_volume_10d"].apply(lambda x: f"{x:,.0f}")
                out["rvol_T1"] = out["rvol_T1"].apply(lambda x: f"{x:.2f}x")
                out["rvol_T0"] = out["rvol_T0"].apply(lambda x: f"{x:.2f}x")
                out["surge_score"] = out["surge_score"].apply(lambda x: f"{x:.2f}")
                out["close_T0"] = out["close_T0"].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(
                    out[["ticker", "Type", "close_T0", "avg_vol", "vol_T1", "rvol_T1", "vol_T0", "rvol_T0", "surge_score"]],
                    use_container_width=True
                )
            else:
                st.info(f"No tickers exceeded the surge criteria across the {target_auction_date} window.")
