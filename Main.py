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
# 2. Market Data Helpers & Rate-Limit Management
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
    """Pulls grouped daily bars and waits 60s if the 5 req/min quota is tripped."""
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{d_str}?adjusted=true&apiKey={api_key}"
    try:
        res = requests.get(url, timeout=15)
        
        # Handle 429 Too Many Requests
        if res.status_code == 429:
            status_box = st.empty()
            for sec in range(60, 0, -1):
                status_box.warning(f"API rate limit (5/min) reached. Cooling down: {sec}s remaining...")
                time.sleep(1)
            status_box.empty()
            res = requests.get(url, timeout=15)

        data = res.json()
        results = data.get("results", [])
        if results:
            rows = []
            for r in results:
                vol = r.get("v", 0)
                close = r.get("c", 0)
                ticker = r.get("T", "")
                # Institutional liquidity gate: eliminates warrants, units, and sub-penny noise
                if vol >= 500_000 and (vol * close) >= 20_000_000 and len(ticker) <= 5 and "." not in ticker:
                    rows.append({
                        "ticker": ticker,
                        "close": close,
                        "volume": vol,
                        "dollar_volume": vol * close
                    })
            return rows
    except Exception:
        pass
    return []

def build_smoothed_baseline(trading_days, api_key):
    """Averages volume per ticker across the full 10-session lookback window."""
    dfs = []
    pbar = st.progress(0, text="Building 10-Day Volume Baseline...")
    
    for idx, d in enumerate(trading_days):
        day_records = fetch_raw_day_data(d, api_key)
        if day_records:
            dfs.append(pd.DataFrame(day_records))
        pbar.progress((idx + 1) / len(trading_days), text=f"Processing session {d}...")
        
    pbar.empty()
    if not dfs:
        return pd.DataFrame()

    full_history = pd.concat(dfs, ignore_index=True)
    return full_history.groupby("ticker")["volume"].mean().reset_index(name="avg_volume_10d")

# -------------------------------------------------------------
# 3. Main Interface & Execution
# -------------------------------------------------------------
st.title("Macro Collateral & Volume Surge Scanner")
st.caption("Detecting institutional accumulation around $100B+ Treasury maturities using 10-day smoothed 48-hour RVOL math.")

if not api_key_input:
    st.warning("Enter your Massive/Polygon API Key in the sidebar to begin.")
    st.stop()

# Step A: Table of Maturing Tranches (CUSIP, dates and amounts detailed)
st.subheader(f"1. Maturing Debt Schedules (>= ${min_tranche}B)")
df_treasury = fetch_treasury_maturities(str(start_date), str(end_date), min_size_billions=min_tranche)

if df_treasury.empty:
    st.info("No maturity records matching that tranche threshold in this date range.")
    st.stop()

# Order chronological by maturity date for visibility
df_display = df_treasury[[
    "cusip", "security_type", "security_term", "offering_b", "auction_date", "issue_date", "maturity_date"
]].copy()

# Format Display table
df_display.columns = ["CUSIP", "Security Type", "Term", "Amount ($B)", "Auction Date", "Issue Date", "Maturity Date"]
df_display = df_display.sort_values(by="Maturity Date", ascending=False).reset_index(drop=True)

st.dataframe(df_display, use_container_width=True)

# Step B: Select maturity date to anchor search
st.subheader("2. Run Auction Scan via Maturity Anchor")
target_maturity_selected = st.selectbox(
    "Select Maturing Date to Search & Trigger Auction-Day Scan",
    options=sorted(df_treasury["maturity_date"].unique(), reverse=True)
)

# Extract specific info tied to the chosen maturity anchor
matching_tranches = df_treasury[df_treasury["maturity_date"] == target_maturity_selected].copy()

# Show details of selection & calculate corresponding auction window (T+0)
if not matching_tranches.empty:
    # Identify the companion auction_date and check for single date
    target_auction_date = matching_tranches["auction_date"].iloc[0]
    
    st.info(
        f"⚓ Maturing Date chosen: **{target_maturity_selected}**.\n"
        f"🎯 Scan target: **Auction Date {target_auction_date}** ($T+0$) and prior session ($T-1$) using CUSIP **{matching_tranches['cusip'].iloc[0]}** as anchor."
    )
    
    # Check issue date as well
    issue_dates_found = ", ".join(matching_tranches["issue_date"].unique())
    st.caption(f"Tranche Amount: ${matching_tranches['offering_b'].sum():,.1f}B | Terms: {', '.join(matching_tranches['security_term'].unique())} | Issue Date(s): {issue_dates_found}")

    if st.button("Run T-1 & T+0 Settlement Scan"):
        # Resolve dates using Auction Date (as T+0) instead of maturity date
        t_0_dt = datetime.datetime.strptime(target_auction_date, "%Y-%m-%d").date()
        while t_0_dt.weekday() >= 5:
            t_0_dt -= datetime.timedelta(days=1)
            
        t_minus_1_str = get_historical_trading_days(t_0_dt, lookback=1)[0]
        t_minus_1_dt = datetime.datetime.strptime(t_minus_1_str, "%Y-%m-%d").date()
        t_0_str = str(t_0_dt)
        
        # Lookback baseline days based on day prior to start of position
        lookback_days = get_historical_trading_days(t_minus_1_dt, lookback=10)
        
        with st.spinner(f"Pulling trading sessions {t_minus_1_str} ($T-1$) & {t_0_str} ($T+0$ Auction) against 10-day baseline..."):
            df_base = build_smoothed_baseline(lookback_days, api_key_input)
            t0_data = fetch_raw_day_data(t_0_str, api_key_input)
            t1_data = fetch_raw_day_data(t_minus_1_str, api_key_input)

            df_t0 = pd.DataFrame(t0_data) if t0_data else pd.DataFrame()
            df_t1 = pd.DataFrame(t1_data) if t1_data else pd.DataFrame()
            
        if df_t0.empty or df_t1.empty or df_base.empty:
            st.warning("Incomplete market bars returned. Verify that both dates were active trading sessions.")
        else:
            # Standardize and merge
            df_t1 = df_t1[['ticker', 'volume', 'close']].rename(columns={'volume': 'vol_T1', 'close': 'close_T1'})
            df_t0 = df_t0[['ticker', 'volume', 'close']].rename(columns={'volume': 'vol_T0', 'close': 'close_T0'})
            
            merged = pd.merge(df_t1, df_t0, on="ticker", how="inner")
            merged = pd.merge(merged, df_base, on="ticker", how="inner")
            
            # Use institutional liquidity floor
            merged = merged[merged["avg_volume_10d"] >= 500_000].copy()
            
            # Calculate Independent RVOL for T-1 (Pre-Auction) and T+0 (Auction Day)
            merged["rvol_T1"] = merged["vol_T1"] / merged["avg_volume_10d"]
            merged["rvol_T0"] = merged["vol_T0"] / merged["avg_volume_10d"]
            
            # Filter: At least one day must meet the target rvol multiple
            merged = merged[(merged["rvol_T1"] >= min_rvol_input) | (merged["rvol_T0"] >= min_rvol_input)].copy()
            
            # Double Surge Qualifier
            merged["double_surge"] = (merged["rvol_T1"] >= min_rvol_input) & (merged["rvol_T0"] >= min_rvol_input)
            
            # Compute score (Aggregated Rel Vol * Size Log)
            merged["base_score"] = np.log10(merged["vol_T1"] + merged["vol_T0"]) * (merged["rvol_T1"] + merged["rvol_T0"])
            
            # 2.0x Double Surge Multiplier
            merged["surge_score"] = np.where(merged["double_surge"], merged["base_score"] * 2.0, merged["base_score"])
            
            results = merged.sort_values(by="surge_score", ascending=False).reset_index(drop=True)
            
            st.subheader(f"Results for Pre-Auction & Auction Windows: {t_minus_1_str} & {t_0_str}")
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
                st.info("No tickers exceeded the surge threshold across this window.")
