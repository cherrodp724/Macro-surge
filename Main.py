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
# API Configuration
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
    min_rvol_input = st.slider("Min RVOL Threshold", 1.0, 10.0, 1.5, step=0.1)
    
    baseline_lookback = st.slider(
        "Volume SMA Lookback (Days)", 
        3, 14, 8, 
        help="Calculates the baseline prior to T+0. Default is 8 days."
    )
    
    today = datetime.date.today()
    start_date = st.date_input("Start Maturity Window", today - datetime.timedelta(days=30))
    end_date = st.date_input("End Maturity Window", today)

# -------------------------------------------------------------
# 1. Fetch Treasury Maturing Tranches (Aggregated)
# -------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_treasury_maturities(start_date, end_date, min_size_billions=100):
    url = (
        f"https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
        f"?filter=maturity_date:gte:{start_date},maturity_date:lte:{end_date}"
        f"&page[size]=5000"
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
        
    daily_totals = df.groupby("maturity_date")["offering_b"].sum().reset_index()
    valid_dates = daily_totals[daily_totals["offering_b"] >= min_size_billions]["maturity_date"]
    
    filtered_df = df[df["maturity_date"].isin(valid_dates)].copy()
    return filtered_df

# -------------------------------------------------------------
# 2. Market Data Helpers & ETF Filters 
# -------------------------------------------------------------
@st.cache_data(ttl=604800)
def get_common_stocks():
    """Fetches the official consolidated tape to strip ETFs, ETNs, and Warrants."""
    try:
        url = "http://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
        df = pd.read_csv(url, sep='|')
        df = df[:-1] 
        valid_df = df[(df['ETF'] == 'N') & (df['Test Issue'] == 'N')]
        return set(valid_df['Symbol'].dropna().tolist())
    except Exception:
        return set()

def get_historical_trading_days(target_date, lookback=8):
    """Returns business days strictly prior to target_date."""
    days = []
    curr = target_date - datetime.timedelta(days=1)
    while len(days) < lookback:
        if curr.weekday() < 5:
            days.append(curr.strftime("%Y-%m-%d"))
        curr -= datetime.timedelta(days=1)
    return days

@st.cache_data(ttl=86400)
def fetch_raw_day_data(d_str, api_key):
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{d_str}?adjusted=true&apiKey={api_key}"
    retries = 2
    valid_stocks = get_common_stocks()
    
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=15)
            
            if res.status_code == 429:
                status_box = st.empty()
                for sec in range(65, 0, -1):
                    status_box.warning(f"⏳ **Paytience...** API limit reached. Countdown to next batch of results: **{sec}s**")
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
                
                if valid_stocks and ticker not in valid_stocks:
                    continue
                    
                # Strict floor filtering for options availability & structural viability
                if vol >= 500_000 and close >= 5.0 and (vol * close) >= 20_000_000 and len(ticker) <= 5 and "." not in ticker:
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
    dfs = []
    pbar = st.progress(0, text="Building Baseline...")
    
    for idx, d in enumerate(trading_days):
        day_records = fetch_raw_day_data(d, api_key)
        if day_records and isinstance(day_records, list):
            dfs.append(pd.DataFrame(day_records))
        pbar.progress((idx + 1) / len(trading_days), text=f"Processing Baseline Session {d}...")
        time.sleep(0.5)  
        
    pbar.empty()
    if not dfs:
        return pd.DataFrame()

    full_history = pd.concat(dfs, ignore_index=True)
    return full_history.groupby("ticker")["volume"].mean().reset_index(name="avg_volume_sma")

# -------------------------------------------------------------
# 3. Main Interface & Execution
# -------------------------------------------------------------
st.title("Macro Collateral & Volume Surge Scanner")
st.caption("Targeting the Top 25 Market Volume Leaders on T+0 (Auction Date) and T+1 (When-Issued / Post-Auction Settlement), filtered for institutional RVOL against an 8-day SMA.")

if not api_key_input:
    st.warning("Enter your Massive/Polygon API Key in the sidebar to begin.")
    st.stop()

st.subheader(f"1. Maturing Debt Schedules (Cumulative >= ${min_tranche}B)")
df_treasury = fetch_treasury_maturities(str(start_date), str(end_date), min_size_billions=min_tranche)

if df_treasury.empty:
    st.info("No maturity records matching that tranche threshold in this date range.")
    st.stop()

df_display = df_treasury[[
    "cusip", "security_type", "security_term", "offering_b", "auction_date", "issue_date", "maturity_date"
]].copy()

df_display.columns = ["CUSIP", "Security Type", "Term", "Amount ($B)", "Auction Date", "Issue Date", "Maturity Date"]
df_display = df_display.sort_values(by=["Maturity Date", "Auction Date"], ascending=[False, True]).reset_index(drop=True)

st.dataframe(df_display, use_container_width=True)

st.subheader("2. Run Auction Scan via Maturity Anchor")
target_maturity_selected = st.selectbox(
    "Select Maturing Date to Search & Trigger Auction-Day Scans",
    options=sorted(df_treasury["maturity_date"].unique(), reverse=True)
)

matching_tranches = df_treasury[df_treasury["maturity_date"] == target_maturity_selected].copy()

if not matching_tranches.empty:
    unique_auction_dates = sorted(matching_tranches["auction_date"].dropna().unique())
    total_liquidity = matching_tranches['offering_b'].sum()
    
    st.info(
        f"⚓ **Maturing Date:** {target_maturity_selected} | **Cumulative Size:** ${total_liquidity:,.1f}B\n\n"
        f"🎯 Found **{len(unique_auction_dates)}** associated Auction Date(s): {', '.join(unique_auction_dates)}."
    )
    
    if st.button("Run T0 & T+1 Settlement Scan for All Auctions"):
        
        master_auction_results = []
        
        for target_auction_date in unique_auction_dates:
            st.markdown("---")
            st.subheader(f"🏦 Scanning Auction Event: {target_auction_date}")
            
            t_0_dt = datetime.datetime.strptime(target_auction_date, "%Y-%m-%d").date()
            if t_0_dt > datetime.date.today():
                st.warning(f"Auction date {target_auction_date} is in the future. Skipping market data pull.")
                continue

            # Ensure T0 is a weekday (or backtrack to prior active session)
            while t_0_dt.weekday() >= 5:
                t_0_dt -= datetime.timedelta(days=1)
                
            t_0_str = str(t_0_dt)
            
            # --- 1. VERIFY T+0 SESSION ---
            df_t0 = pd.DataFrame()
            with st.spinner(f"[{target_auction_date}] Verifying active session on T0 Auction Date..."):
                for _ in range(5): 
                    t0_data = fetch_raw_day_data(t_0_str, api_key_input)
                    if t0_data:
                        df_t0 = pd.DataFrame(t0_data)
                        break
                    else:
                        st.caption(f"⚠️ {t_0_str} was closed. Searching previous session...")
                        t_0_dt -= datetime.timedelta(days=1)
                        while t_0_dt.weekday() >= 5:
                            t_0_dt -= datetime.timedelta(days=1)
                        t_0_str = str(t_0_dt)
                        time.sleep(0.5)

            # --- 2. VERIFY T+1 SESSION ---
            df_t1 = pd.DataFrame()
            if not df_t0.empty:
                t_plus_1_dt = t_0_dt + datetime.timedelta(days=1)
                while t_plus_1_dt.weekday() >= 5:
                    t_plus_1_dt += datetime.timedelta(days=1)
                    
                t_plus_1_str = str(t_plus_1_dt)
                
                if t_plus_1_dt > datetime.date.today():
                    st.warning(f"Post-Auction session ($T+1$: {t_plus_1_str}) has not closed yet. Skipping.")
                    continue
                
                with st.spinner(f"[{target_auction_date}] Verifying active trading on post-auction session ($T+1$: {t_plus_1_str})..."):
                    for _ in range(5):
                        t1_data = fetch_raw_day_data(t_plus_1_str, api_key_input)
                        if t1_data:
                            df_t1 = pd.DataFrame(t1_data)
                            break
                        else:
                            st.caption(f"⚠️ {t_plus_1_str} was closed. Forward-checking next business day...")
                            t_plus_1_dt += datetime.timedelta(days=1)
                            while t_plus_1_dt.weekday() >= 5:
                                t_plus_1_dt += datetime.timedelta(days=1)
                            t_plus_1_str = str(t_plus_1_dt)
                            time.sleep(0.5)
                            
            if df_t0.empty or df_t1.empty:
                st.error(f"Incomplete trading sessions found for {target_auction_date} ($T0/$T+1$). Skipping.")
                continue
                
            df_t0 = df_t0[['ticker', 'volume', 'close']].rename(columns={'volume': 'vol_T0', 'close': 'close_T0'})
            df_t1 = df_t1[['ticker', 'volume', 'close']].rename(columns={'volume': 'vol_T1', 'close': 'close_T1'})
            
            merged = pd.merge(df_t0, df_t1, on="ticker", how="inner")
            
            # --- 3. IDENTIFY TOP 25 VOLUME LEADERS ACROSS T0 AND T+1 ---
            merged["total_event_vol"] = merged["vol_T0"] + merged["vol_T1"]
            top_25_leaders = merged.sort_values(by="total_event_vol", ascending=False).head(25).copy()
            
            if top_25_leaders.empty:
                st.info(f"No optionable volume leaders found for {target_auction_date}.")
                continue
                
            # --- 4. FETCH THE SMA BASELINE PRIOR TO T0 ---
            lookback_days = get_historical_trading_days(t_0_dt, lookback=baseline_lookback)
            
            with st.spinner(f"[{target_auction_date}] Evaluating {baseline_lookback}-day SMA baseline prior to Auction Date..."):
                df_base = build_smoothed_baseline(lookback_days, api_key_input)
                
            if df_base.empty:
                st.error(f"Historical baseline could not be generated for {target_auction_date}.")
                continue
                
            # --- 5. CALCULATE RVOL ON THE TOP 25 ---
            merged_final = pd.merge(top_25_leaders, df_base, on="ticker", how="inner")
            
            merged_final["rvol_T0"] = merged_final["vol_T0"] / merged_final["avg_volume_sma"]
            merged_final["rvol_T1"] = merged_final["vol_T1"] / merged_final["avg_volume_sma"]
            
            # --- 6. APPLY RVOL GATE & DOUBLE SURGE LOGIC ---
            merged_final = merged_final[(merged_final["rvol_T0"] >= min_rvol_input) | (merged_final["rvol_T1"] >= min_rvol_input)].copy()
            merged_final["double_surge"] = (merged_final["rvol_T0"] >= min_rvol_input) & (merged_final["rvol_T1"] >= min_rvol_input)
            
            merged_final["base_score"] = np.log10(merged_final["vol_T0"] + merged_final["vol_T1"]) * (merged_final["rvol_T0"] + merged_final["rvol_T1"])
            merged_final["surge_score"] = np.where(merged_final["double_surge"], merged_final["base_score"] * 2.0, merged_final["base_score"])
            
            results = merged_final.sort_values(by="surge_score", ascending=False).reset_index(drop=True)
            
            if not results.empty:
                store_df = results.copy()
                store_df["auction_date"] = target_auction_date
                master_auction_results.append(store_df)
            
            st.write(f"**Top Volume Leaders Passing RVOL Surge: {t_0_str} ($T0$ Auction) & {t_plus_1_str} ($T+1$ Post-Auction)**")
            if not results.empty:
                out = results.head(25).copy()
                
                out["Type"] = np.where(out["double_surge"], "🔥 DOUBLE SURGE", "Single Spike")
                out["vol_T0"] = out["vol_T0"].apply(lambda x: f"{x:,.0f}")
                out["vol_T1"] = out["vol_T1"].apply(lambda x: f"{x:,.0f}")
                out["avg_vol"] = out["avg_volume_sma"].apply(lambda x: f"{x:,.0f}")
                out["rvol_T0"] = out["rvol_T0"].apply(lambda x: f"{x:.2f}x")
                out["rvol_T1"] = out["rvol_T1"].apply(lambda x: f"{x:.2f}x")
                out["surge_score"] = out["surge_score"].apply(lambda x: f"{x:.2f}")
                out["close_T1"] = out["close_T1"].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(
                    out[["ticker", "Type", "close_T1", "avg_vol", "vol_T0", "rvol_T0", "vol_T1", "rvol_T1", "surge_score"]],
                    use_container_width=True
                )
            else:
                st.info(f"None of the Top 25 Volume Leaders exceeded the RVOL surge criteria during the {target_auction_date} window.")
                
        # ---------------------------------------------------------
        # AGGREGATE RECURRING LEADERS
        # ---------------------------------------------------------
        if len(unique_auction_dates) > 1 and master_auction_results:
            st.markdown("---")
            st.header("🏆 Multi-Auction Recurring Volume Leaders")
            st.caption("Stocks exhibiting consistent elevated RVOL across multiple distinct auction openings/reopenings for this maturity.")
            
            agg_df = pd.concat(master_auction_results, ignore_index=True)
            agg_df["total_event_vol"] = agg_df["vol_T0"] + agg_df["vol_T1"]
            
            leaderboard = agg_df.groupby("ticker").agg(
                Appearances=("ticker", "count"),
                Avg_RVOL_T0=("rvol_T0", "mean"),
                Avg_RVOL_T1=("rvol_T1", "mean"),
                Avg_Score=("surge_score", "mean"),
                Total_Surge_Vol=("total_event_vol", "sum")
            ).reset_index()
            
            recurring_leaders = leaderboard[leaderboard["Appearances"] > 1].copy()
            
            if not recurring_leaders.empty:
                recurring_leaders = recurring_leaders.sort_values(by=["Appearances", "Avg_Score"], ascending=[False, False])
                
                recurring_leaders["Avg_RVOL_T0"] = recurring_leaders["Avg_RVOL_T0"].apply(lambda x: f"{x:.2f}x")
                recurring_leaders["Avg_RVOL_T1"] = recurring_leaders["Avg_RVOL_T1"].apply(lambda x: f"{x:.2f}x")
                recurring_leaders["Avg_Score"] = recurring_leaders["Avg_Score"].apply(lambda x: f"{x:.2f}")
                recurring_leaders["Total_Surge_Vol"] = recurring_leaders["Total_Surge_Vol"].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(recurring_leaders, use_container_width=True)
            else:
                st.info("No single ticker managed to clear the minimum accumulation threshold across multiple different auction dates.")
