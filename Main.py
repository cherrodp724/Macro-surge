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
    initial_sidebar_state="expanded",
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
    default_key = os.environ.get("POLYGON_API_KEY", "") or os.environ.get(
        "MASSIVE_API_KEY", ""
    )

with st.sidebar:
    st.header("1. API Configuration")
    api_key_input = st.text_input(
        "Massive / Polygon API Key",
        value=default_key,
        type="password",
        help="Paste your key here or add it to Streamlit App Secrets as POLYGON_API_KEY.",
    )

    st.header("2. Scanner Parameters")
    min_tranche = st.slider("Min Maturity Tranche ($B)", 50, 250, 100, step=25)
    min_rvol_input = st.slider("Min RVOL Threshold", 1.5, 10.0, 2.0, step=0.5)

    today = datetime.date.today()
    start_date = st.date_input(
        "Start Maturity Window", today - datetime.timedelta(days=30)
    )
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
            records.append(
                {
                    "cusip": item.get("cusip"),
                    "security_type": item.get("security_type"),
                    "security_term": item.get("security_term"),
                    "offering_amt": size,
                    "offering_b": round(size / 1_000_000_000, 2),
                    "maturity_date": item.get("maturity_date"),
                    "auction_date": item.get("auction_date"),
                    "issue_date": item.get("issue_date"),
                }
            )
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
                status_box.warning(
                    f"API rate limit (5/min) reached. Cooling down: {sec}s remaining..."
                )
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
                if (
                    vol >= 500_000
                    and (vol * close) >= 20_000_000
                    and len(ticker) <= 5
                    and "." not in ticker
                ):
                    rows.append(
                        {
                            "ticker": ticker,
                            "close": close,
                            "volume": vol,
                            "dollar_volume": vol * close,
                        }
                    )
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
        pbar.progress(
            (idx + 1) / len(trading_days), text=f"Processing session {d}..."
        )

    pbar.empty()
    if not dfs:
        return pd.DataFrame()

    full_history = pd.concat(dfs, ignore_index=True)
    return (
        full_history.groupby("ticker")["volume"]
        .mean()
        .reset_index(name="avg_volume_10d")
    )


# -------------------------------------------------------------
# 3. Main Interface & Execution
# -------------------------------------------------------------
st.title("Macro Collateral & Volume Surge Scanner")
st.caption(
    "Detecting institutional accumulation around $100B+ Treasury maturities using 10-day smoothed 48-hour RVOL math."
)

if not api_key_input:
    st.warning("Enter your Massive/Polygon API Key in the sidebar to begin.")
    st.stop()

# Step A: Table of Maturing Tranches
st.subheader(f"1. Maturing Debt Schedules (>= ${min_tranche}B)")
df_treasury = fetch_treasury_maturities(
    str(start_date), str(end_date), min_size_billions=min_tranche
)

if df_treasury.empty:
    st.info("No maturity records matching that tranche threshold in this date range.")
    st.stop()

tranche_summary = (
    df_treasury.groupby("maturity_date")
    .agg(
        total_offering_b=("offering_b", "sum"),
        tranche_count=("cusip", "count"),
        terms=("security_term", lambda x: ", ".join(x.unique())),
    )
    .reset_index()
    .sort_values(by="maturity_date", ascending=False)
)

st.dataframe(tranche_summary, use_container_width=True)

# Step B: Pick Date & Run 48-Hour Macro Scan
target_date_selected = st.selectbox(
    "Select Maturing Date (T+0) to Scan Institutional Flow",
    options=tranche_summary["maturity_date"].tolist(),
)

if st.button("Run T-1 & T+0 Settlement Scan"):
    # Normalize T+0 to a weekday
    t_0_dt = datetime.datetime.strptime(target_date_selected, "%Y-%m-%d").date()
    while t_0_dt.weekday() >= 5:
        t_0_dt -= datetime.timedelta(days=1)

    t_minus_1_str = get_historical_trading_days(t_0_dt, lookback=1)[0]
    t_minus_1_dt = datetime.datetime.strptime(t_minus_1_str, "%Y-%m-%d").date()
    t_0_str = str(t_0_dt)

    # 10-day lookback before T-1 positioning starts
    lookback_days = get_historical_trading_days(t_minus_1_dt, lookback=10)

    with st.spinner(
        f"Scanning 48-hour window ({t_minus_1_str} to {t_0_str}) against 10-day baseline..."
    ):
        df_base = build_smoothed_baseline(lookback_days, api_key_input)
        t0_data = fetch_raw_day_data(t_0_str, api_key_input)
        t1_data = fetch_raw_day_data(t_minus_1_str, api_key_input)

        df_t0 = pd.DataFrame(t0_data) if t0_data else pd.DataFrame()
        df_t1 = pd.DataFrame(t1_data) if t1_data else pd.DataFrame()

    if df_t0.empty or df_t1.empty or df_base.empty:
        st.warning(
            "Incomplete market bars returned. Verify that both dates were active trading sessions."
        )
    else:
        # Standardize and merge
        df_t1 = df_t1[["ticker", "volume", "close"]].rename(
            columns={"volume": "vol_T1", "close": "close_T1"}
        )
        df_t0 = df_t0[["ticker", "volume", "close"]].rename(
            columns={"volume": "vol_T0", "close": "close_T0"}
        )

        merged = pd.merge(df_t1, df_t0, on="ticker", how="inner")
        merged = pd.merge(merged, df_base, on="ticker", how="inner")

        # Hard baseline liquidity floor
        merged = merged[merged["avg_volume_10d"] >= 500_000].copy()

        # Calculate Independent RVOL for T-1 and T+0
        merged["rvol_T1"] = merged["vol_T1"] / merged["avg_volume_10d"]
        merged["rvol_T0"] = merged["vol_T0"] / merged["avg_volume_10d"]

        # Filter: At least one day must clear the RVOL threshold
        merged = merged[
            (merged["rvol_T1"] >= min_rvol_input)
            | (merged["rvol_T0"] >= min_rvol_input)
        ].copy()

        # Double Surge Qualifier
        merged["double_surge"] = (merged["rvol_T1"] >= min_rvol_input) & (
            merged["rvol_T0"] >= min_rvol_input
        )

        # Composite Surge Math
        merged["base_score"] = np.log10(
            merged["vol_T1"] + merged["vol_T0"]
        ) * (merged["rvol_T1"] + merged["rvol_T0"])

        # 2.0x Double Surge Multiplier
        merged["surge_score"] = np.where(
            merged["double_surge"],
            merged["base_score"] * 2.0,
            merged["base_score"],
        )

        results = merged.sort_values(
            by="surge_score", ascending=False
        ).reset_index(drop=True)

        st.subheader(f"Macro Accumulation Window: {t_minus_1_str} & {t_0_str}")
        if not results.empty:
            out = results.head(25).copy()

            out["Type"] = np.where(
                out["double_surge"], "🔥 DOUBLE SURGE", "Single Spike"
            )
            out["vol_T1"] = out["vol_T1"].apply(lambda x: f"{x:,.0f}")
            out["vol_T0"] = out["vol_T0"].apply(lambda x: f"{x:,.0f}")
            out["avg_vol"] = out["avg_volume_10d"].apply(lambda x: f"{x:,.0f}")
            out["rvol_T1"] = out["rvol_T1"].apply(lambda x: f"{x:.2f}x")
            out["rvol_T0"] = out["rvol_T0"].apply(lambda x: f"{x:.2f}x")
            out["surge_score"] = out["surge_score"].apply(lambda x: f"{x:.2f}")
            out["close_T0"] = out["close_T0"].apply(lambda x: f"${x:.2f}")

            st.dataframe(
                out[
                    [
                        "ticker",
                        "Type",
                        "close_T0",
                        "avg_vol",
                        "vol_T1",
                        "rvol_T1",
                        "vol_T0",
                        "rvol_T0",
                        "surge_score",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.info("No tickers exceeded the surge threshold across this window.")
