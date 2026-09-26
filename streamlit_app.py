import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
from itertools import zip_longest

try:
    from kiteconnect import KiteConnect
except ImportError:
    st.error("KiteConnect library not found. Please install it using `pip install kiteconnect`.")
    st.stop()

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Invsion Connect - Compliant vs Non-Compliant Returns", layout="wide")
st.title("Invsion Connect")
st.markdown("Compare 3-year returns of **Compliant** vs **Non-Compliant** firms using live Kite Connect data.")

DEFAULT_EXCHANGE = "NSE"
YEARS_LOOKBACK = 3
BENCHMARK_SYMBOL = "NIFTY 50"
TRADING_DAYS_PER_YEAR = 252

# --- Session State ---
if "kite_access_token" not in st.session_state: st.session_state["kite_access_token"] = None
if "analysis_done" not in st.session_state: st.session_state["analysis_done"] = False
if "output_csv" not in st.session_state: st.session_state["output_csv"] = None
if "enforcement_csv" not in st.session_state: st.session_state["enforcement_csv"] = None


# --- Load Credentials from Streamlit Secrets ---
def load_secrets():
    kite_conf = st.secrets.get("kite", {})
    if not kite_conf.get("api_key") or not kite_conf.get("api_secret") or not kite_conf.get("redirect_uri"):
        st.error("Missing Kite credentials (api_key, api_secret, redirect_uri) in `.streamlit/secrets.toml`.")
        st.stop()
    return kite_conf

KITE_CREDENTIALS = load_secrets()


# --- KiteConnect Client Initialization ---
@st.cache_resource(ttl=3600)
def init_kite_unauth_client(api_key: str) -> KiteConnect:
    return KiteConnect(api_key=api_key)

kite_unauth_client = init_kite_unauth_client(KITE_CREDENTIALS["api_key"])
login_url = kite_unauth_client.login_url()


def get_authenticated_kite_client(api_key: str | None, access_token: str | None) -> KiteConnect | None:
    if api_key and access_token:
        k_instance = KiteConnect(api_key=api_key)
        k_instance.set_access_token(access_token)
        return k_instance
    return None


@st.cache_data(ttl=86400, show_spinner="Loading instruments...")
def load_instruments_cached(api_key: str, access_token: str, exchange: str = None) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame()
    try:
        instruments = kite_instance.instruments(exchange) if exchange else kite_instance.instruments()
        df = pd.DataFrame(instruments)
        if "instrument_token" in df.columns:
            df["instrument_token"] = df["instrument_token"].astype("int64")
        return df
    except Exception:
        return pd.DataFrame()


def find_instrument_token(df: pd.DataFrame, tradingsymbol: str, exchange: str = DEFAULT_EXCHANGE) -> int | None:
    if df.empty:
        return None
    mask = (df.get("exchange", pd.Series(dtype=str)).astype(str).str.upper() == exchange.upper()) & \
           (df.get("tradingsymbol", pd.Series(dtype=str)).astype(str).str.upper() == tradingsymbol.upper())
    hits = df[mask]
    return int(hits.iloc[0]["instrument_token"]) if not hits.empty else None


@st.cache_data(ttl=3600)
def get_historical_data_cached(api_key: str, access_token: str, symbol: str, from_date, to_date,
                                exchange: str = DEFAULT_EXCHANGE) -> pd.DataFrame:
    kite_instance = get_authenticated_kite_client(api_key, access_token)
    if not kite_instance:
        return pd.DataFrame({"_error": ["Kite not authenticated."]})
    instruments_df = load_instruments_cached(api_key, access_token, exchange)
    token = find_instrument_token(instruments_df, symbol, exchange)
    if not token and symbol.upper() in ["NIFTY 50", "NIFTY50", "NIFTY BANK", "BANKNIFTY"]:
        # Indices sit on NSE regardless of the exchange passed in
        instruments_df = load_instruments_cached(api_key, access_token, "NSE")
        token = find_instrument_token(instruments_df, symbol, "NSE")
    if not token:
        return pd.DataFrame({"_error": [f"Instrument token not found for {symbol}."]})
    try:
        data = kite_instance.historical_data(
            token,
            from_date=datetime.combine(from_date, datetime.min.time()),
            to_date=datetime.combine(to_date, datetime.max.time()),
            interval="day"
        )
        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(
                pd.to_numeric, errors="coerce")
            df.dropna(subset=["close"], inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame({"_error": [str(e)]})


def extract_firm_info(df: pd.DataFrame) -> list[dict]:
    """Normalize column names and pull out Symbol / Company per firm."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    symbol_col = next((c for c in ["symbol", "tradingsymbol", "ticker"] if c in df.columns), None)
    if symbol_col is None:
        return []
    name_col = next((c for c in ["company", "company_name", "name", "firm", "firm_name"] if c in df.columns), None)

    df[symbol_col] = df[symbol_col].astype(str).str.strip().str.upper()
    df = df.dropna(subset=[symbol_col]).drop_duplicates(subset=[symbol_col])

    firms = []
    for _, row in df.iterrows():
        firms.append({
            "Symbol": row[symbol_col],
            "Company": str(row[name_col]).strip() if name_col and pd.notna(row.get(name_col)) else row[symbol_col],
        })
    return firms


def compute_security_metrics(hist_df: pd.DataFrame, benchmark_returns: pd.DataFrame | None) -> dict | None:
    """Total/annualized return, annualized volatility, and beta vs the benchmark."""
    if hist_df.empty or "_error" in hist_df.columns or "close" not in hist_df.columns or len(hist_df) < 2:
        return None
    hist = hist_df.copy()
    hist["daily_return"] = hist["close"].pct_change()

    start_row, end_row = hist.iloc[0], hist.iloc[-1]
    start_price, end_price = start_row["close"], end_row["close"]
    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return None

    total_return = ((end_price - start_price) / start_price) * 100
    days_held = (end_row["date"] - start_row["date"]).days
    years_held = days_held / 365.25 if days_held > 0 else np.nan
    annualized_return = (((end_price / start_price) ** (1 / years_held)) - 1) * 100 if years_held and years_held > 0 else np.nan

    daily_returns = hist["daily_return"].dropna()
    volatility = daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) * 100 if len(daily_returns) > 1 else np.nan

    beta = np.nan
    if benchmark_returns is not None and not benchmark_returns.empty:
        merged = pd.merge(
            hist[["date", "daily_return"]], benchmark_returns, on="date", how="inner"
        ).dropna()
        if len(merged) > 2:
            covariance = merged["daily_return"].cov(merged["bm_return"])
            bm_variance = merged["bm_return"].var()
            beta = covariance / bm_variance if bm_variance > 0 else np.nan

    return {
        "Start Date": start_row["date"].date(),
        "Start Price": round(start_price, 2),
        "End Date": end_row["date"].date(),
        "End Price": round(end_price, 2),
        "Total Return (%)": round(total_return, 2),
        "Annualized Return (%)": round(annualized_return, 2) if not pd.isna(annualized_return) else None,
        "Beta": round(beta, 3) if not pd.isna(beta) else None,
        "Volatility (Annualized %)": round(volatility, 2) if not pd.isna(volatility) else None,
        "Data Points": len(hist_df),
    }


# --- Sidebar: Kite Login ---
with st.sidebar:
    st.markdown("### Login to Kite Connect")
    if not st.session_state["kite_access_token"]:
        st.link_button("🔗 Open Kite login", login_url, use_container_width=True)
    request_token_param = st.query_params.get("request_token")
    if request_token_param and not st.session_state["kite_access_token"]:
        with st.spinner("Authenticating..."):
            try:
                data = kite_unauth_client.generate_session(request_token_param, api_secret=KITE_CREDENTIALS["api_secret"])
                st.session_state["kite_access_token"] = data.get("access_token")
                st.success("Kite authentication successful.")
                st.query_params.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Authentication failed: {e}")
    if st.session_state["kite_access_token"]:
        st.success("Kite Authenticated ✅")
        if st.button("Logout from Kite", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    else:
        st.info("Not authenticated with Kite yet.")

k = get_authenticated_kite_client(KITE_CREDENTIALS["api_key"], st.session_state["kite_access_token"])
api_key = KITE_CREDENTIALS["api_key"]
access_token = st.session_state["kite_access_token"]


# --- Main UI ---
st.subheader("1. Upload Firm Lists")
col1, col2 = st.columns(2)
with col1:
    compliant_file = st.file_uploader("Compliant Firms CSV", type="csv", key="compliant_upload")
with col2:
    noncompliant_file = st.file_uploader("Non-Compliant Firms CSV", type="csv", key="noncompliant_upload")

st.caption(
    "Each CSV needs at least a `Symbol` column with NSE trading symbols. "
    "An optional `Company` column will populate company names in the Enforcement Index."
)

if not k:
    st.info("Please login to Kite Connect above before running the analysis.")

run_disabled = not (k and compliant_file and noncompliant_file)

if st.button("🚀 Fetch 3-Year Data & Compare Returns", type="primary", disabled=run_disabled, use_container_width=True):
    compliant_firms = extract_firm_info(pd.read_csv(compliant_file))
    noncompliant_firms = extract_firm_info(pd.read_csv(noncompliant_file))

    if not compliant_firms:
        st.error("No 'Symbol' column found in the Compliant Firms file.")
        st.stop()
    if not noncompliant_firms:
        st.error("No 'Symbol' column found in the Non-Compliant Firms file.")
        st.stop()

    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=365 * YEARS_LOOKBACK)

    # Benchmark data, used to compute beta for every security
    benchmark_hist = get_historical_data_cached(api_key, access_token, BENCHMARK_SYMBOL, from_date, to_date, "NSE")
    benchmark_returns = None
    if not benchmark_hist.empty and "_error" not in benchmark_hist.columns:
        benchmark_returns = benchmark_hist[["date"]].copy()
        benchmark_returns["bm_return"] = benchmark_hist["close"].pct_change()
    else:
        st.warning(f"Could not fetch benchmark data for {BENCHMARK_SYMBOL} — Beta will be left blank.")

    all_rows, consolidated_rows, failed_symbols = [], [], []
    per_group_records = {"Compliant": [], "Non-Compliant": []}
    groups = [("Compliant", compliant_firms), ("Non-Compliant", noncompliant_firms)]
    total_symbols = len(compliant_firms) + len(noncompliant_firms)
    progress = st.progress(0, text="Fetching historical data...")
    processed = 0

    for group_name, firms in groups:
        for firm in firms:
            symbol = firm["Symbol"]
            hist_df = get_historical_data_cached(api_key, access_token, symbol, from_date, to_date, DEFAULT_EXCHANGE)
            metrics = compute_security_metrics(hist_df, benchmark_returns)
            if metrics is None:
                failed_symbols.append(f"{symbol} ({group_name})")
            else:
                record = {"Symbol": symbol, "Company": firm["Company"], "Group": group_name}
                record.update(metrics)
                all_rows.append(record)
                per_group_records[group_name].append(record)
                for _, r in hist_df.iterrows():
                    consolidated_rows.append({
                        "Date": r["date"].date(), "Symbol": symbol, "Group": group_name,
                        "Open": r["open"], "High": r["high"], "Low": r["low"],
                        "Close": r["close"], "Volume": r["volume"],
                    })
            processed += 1
            progress.progress(processed / total_symbols, text=f"Fetched {symbol} ({group_name})")

    progress.empty()

    if failed_symbols:
        st.warning(f"Could not fetch data for: {', '.join(failed_symbols)}")

    if not all_rows:
        st.error("No return data could be computed for any security.")
        st.stop()

    returns_df = pd.DataFrame(all_rows)
    consolidated_df = pd.DataFrame(consolidated_rows)

    compliant_returns = returns_df.loc[returns_df["Group"] == "Compliant", "Total Return (%)"]
    noncompliant_returns = returns_df.loc[returns_df["Group"] == "Non-Compliant", "Total Return (%)"]

    def safe(fn, series):
        series = series.dropna()
        return round(fn(series), 2) if not series.empty else None

    summary_rows = [
        {"Metric": "Number of Firms", "Compliant": len(compliant_returns), "Non-Compliant": len(noncompliant_returns)},
        {"Metric": "Average Total Return (%)", "Compliant": safe(pd.Series.mean, compliant_returns), "Non-Compliant": safe(pd.Series.mean, noncompliant_returns)},
        {"Metric": "Median Total Return (%)", "Compliant": safe(pd.Series.median, compliant_returns), "Non-Compliant": safe(pd.Series.median, noncompliant_returns)},
        {"Metric": "Best Total Return (%)", "Compliant": safe(pd.Series.max, compliant_returns), "Non-Compliant": safe(pd.Series.max, noncompliant_returns)},
        {"Metric": "Worst Total Return (%)", "Compliant": safe(pd.Series.min, compliant_returns), "Non-Compliant": safe(pd.Series.min, noncompliant_returns)},
        {"Metric": "Std Dev of Return (%)", "Compliant": safe(pd.Series.std, compliant_returns), "Non-Compliant": safe(pd.Series.std, noncompliant_returns)},
    ]
    summary_df = pd.DataFrame(summary_rows)

    if not compliant_returns.empty and not noncompliant_returns.empty:
        outperformance = round(compliant_returns.mean() - noncompliant_returns.mean(), 2)
        summary_df.loc[len(summary_df)] = {
            "Metric": "Compliant Outperformance vs Non-Compliant (percentage points)",
            "Compliant": outperformance,
            "Non-Compliant": "",
        }

    st.session_state["returns_df"] = returns_df
    st.session_state["summary_df"] = summary_df
    st.session_state["consolidated_df"] = consolidated_df
    st.session_state["analysis_done"] = True

    # --- Build one combined CSV: security returns + summary + consolidated data ---
    buf = StringIO()
    buf.write("SECURITY RETURNS (3-YEAR)\n")
    returns_df.to_csv(buf, index=False)
    buf.write("\n\nSUMMARY: COMPLIANT vs NON-COMPLIANT\n")
    summary_df.to_csv(buf, index=False)
    buf.write("\n\nCONSOLIDATED HISTORICAL DATA (DAILY)\n")
    consolidated_df.to_csv(buf, index=False)
    st.session_state["output_csv"] = buf.getvalue()

    # --- Build the Enforcement Index: one row per compliant/non-compliant pair, matched by upload order ---
    beta_compliant_avg = safe(pd.Series.mean, pd.Series([r["Beta"] for r in per_group_records["Compliant"] if r["Beta"] is not None]))
    beta_noncompliant_avg = safe(pd.Series.mean, pd.Series([r["Beta"] for r in per_group_records["Non-Compliant"] if r["Beta"] is not None]))
    vol_compliant_avg = safe(pd.Series.mean, pd.Series([r["Volatility (Annualized %)"] for r in per_group_records["Compliant"] if r["Volatility (Annualized %)"] is not None]))
    vol_noncompliant_avg = safe(pd.Series.mean, pd.Series([r["Volatility (Annualized %)"] for r in per_group_records["Non-Compliant"] if r["Volatility (Annualized %)"] is not None]))

    enforcement_rows = []
    for c, nc in zip_longest(per_group_records["Compliant"], per_group_records["Non-Compliant"]):
        c_return = c["Total Return (%)"] if c else None
        nc_return = nc["Total Return (%)"] if nc else None
        return_gap_pct_pts = round(c_return - nc_return, 2) if c_return is not None and nc_return is not None else None

        enforcement_rows.append({
            "Compliant Company": c["Company"] if c else None,
            "compliant_ticker": c["Symbol"] if c else None,
            "return_3years_compliant": c_return,
            "Non-Compliant Company": nc["Company"] if nc else None,
            "non_compliant_ticker": nc["Symbol"] if nc else None,
            "return_3years_non_compliant": nc_return,
            "return_gap_pct_pts": return_gap_pct_pts,
            "beta_compliant_firms": beta_compliant_avg,
            "beta_non_compliant_firms": beta_noncompliant_avg,
            "standard_deviation_compliant": vol_compliant_avg,
            "standard_deviation_non_compliant": vol_noncompliant_avg,
        })

    enforcement_df = pd.DataFrame(enforcement_rows)
    st.session_state["enforcement_df"] = enforcement_df
    st.session_state["enforcement_csv"] = enforcement_df.to_csv(index=False)

    st.success("✅ Analysis complete!")

# --- Display Results ---
if st.session_state.get("analysis_done"):
    st.markdown("---")
    st.subheader("2. Security-Level Returns")
    st.dataframe(st.session_state["returns_df"], use_container_width=True)

    st.subheader("3. Summary: Compliant vs Non-Compliant")
    st.dataframe(st.session_state["summary_df"], use_container_width=True)

    st.subheader("4. Consolidated Historical Data")
    st.dataframe(st.session_state["consolidated_df"], use_container_width=True, height=400)

    st.subheader("5. Enforcement Index")
    st.caption("Compliant firms paired with non-compliant firms in upload order.")
    st.dataframe(st.session_state["enforcement_df"], use_container_width=True)

    st.markdown("---")
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "📥 Download Combined CSV Report",
            st.session_state["output_csv"],
            f"compliant_vs_noncompliant_returns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            "📥 Download Enforcement Index CSV",
            st.session_state["enforcement_csv"],
            f"enforcement_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True,
        )

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Invsion Connect</strong> - Compliant vs Non-Compliant Returns</p>
    <p style='font-size: 0.9em;'>⚠️ For informational purposes only. Always consult with qualified professionals for investment decisions.</p>
    <p style='font-size: 0.8em;'>Powered by KiteConnect API</p>
</div>
""", unsafe_allow_html=True)
