# app.py
"""
Streamlit dashboard (updated):
 - removes allocation output
 - prints VIX once
 - shows per-stock Implied Volatility (IV) computed from nearest option expiry (best-effort)
 - displays per-stock parameters in a structured Parameter|Value|Comment table
 - supports ETFs (technical analysis applies; analyst/IV may be missing)
"""

import io
import math
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- CONFIG ----------------
SMA_SHORT = 50
SMA_LONG = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_SPIKE_FACTOR = 1.0
VIX_THRESHOLD = 25.0
DEFAULT_PERIOD = "2y"
BATCH_SIZE = 8

st.set_page_config(layout="wide", page_title="Multi-Stock Dashboard (IV & Table)", initial_sidebar_state="expanded")

# ---------------- Utility ----------------
def safe_float_last(x):
    try:
        if x is None:
            return None
        if hasattr(x, "iloc"):
            v = x.iloc[-1]
        else:
            v = x
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None

# ---------------- Indicators ----------------
def compute_indicators(df):
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(window=SMA_SHORT, min_periods=1).mean()
    df["SMA200"] = df["Close"].rolling(window=SMA_LONG, min_periods=1).mean()
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)
    # MACD
    ema_fast = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["MACD_HIST"] = macd - macd_sig
    # Vol MA
    df["Vol_MA20"] = df["Volume"].rolling(window=20, min_periods=1).mean()
    return df

# ---------------- Data fetch & caching ----------------
@st.cache_data(ttl=300)
def batch_download(tickers, period=DEFAULT_PERIOD):
    out = {}
    if not tickers:
        return out
    try:
        batch_df = yf.download(" ".join(tickers), period=period, group_by="ticker", auto_adjust=False, progress=False)
    except Exception:
        batch_df = None

    for tk in tickers:
        df = None
        if batch_df is not None:
            try:
                if isinstance(batch_df.columns, pd.MultiIndex):
                    df = batch_df[tk].copy()
                else:
                    if len(tickers) == 1:
                        df = batch_df.copy()
            except Exception:
                df = None
        if df is None:
            try:
                df = yf.download(tk, period=period, auto_adjust=False, progress=False)
            except Exception:
                df = None
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index).tz_localize(None)
            out[tk] = df
    return out

@st.cache_data(ttl=600)
def fetch_vix_value():
    try:
        vix_df = yf.download("^VIX", period="7d", interval="1d", progress=False, auto_adjust=False)
        if vix_df is None or vix_df.empty:
            vix_df = yf.download("VIXY", period="7d", interval="1d", progress=False, auto_adjust=False)
        if vix_df is None or vix_df.empty:
            return None
        return safe_float_last(vix_df["Close"])
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_analyst_data(ticker):
    rec_mean = None
    analyst_count = None
    target_mean = None
    try:
        t = yf.Ticker(ticker)
        # try earnings_trend
        try:
            trend = getattr(t, "earnings_trend", None)
            if trend and isinstance(trend, dict) and "trend" in trend and len(trend["trend"])>0:
                curr = trend["trend"][0]
                rec_mean = curr.get("ratingMean") or curr.get("ratingmean")
                analyst_count = curr.get("numberOfAnalysts") or curr.get("numberofanalysts")
                target_mean = curr.get("targetMean") or curr.get("targetmean")
        except Exception:
            pass
        # fallback get_earnings_trend
        if rec_mean is None or analyst_count is None:
            try:
                if hasattr(t, "get_earnings_trend"):
                    trend2 = t.get_earnings_trend()
                    if trend2 and "trend" in trend2 and len(trend2["trend"])>0:
                        curr = trend2["trend"][0]
                        rec_mean = rec_mean or curr.get("ratingMean")
                        analyst_count = analyst_count or curr.get("numberOfAnalysts")
                        target_mean = target_mean or curr.get("targetMean")
            except Exception:
                pass
        # fallback to info
        try:
            info = t.get_info() if hasattr(t, "get_info") else getattr(t, "info", {}) or {}
            if rec_mean is None:
                rec_mean = info.get("recommendationMean") or info.get("recommendationMean")
            if target_mean is None:
                target_mean = info.get("targetMeanPrice") or info.get("targetMean")
        except Exception:
            pass
        # recommendations table count fallback
        try:
            recs = t.recommendations
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                analyst_count = analyst_count or recs.shape[0]
        except Exception:
            pass
    except Exception:
        pass
    # sanitize
    try:
        rec_mean = float(rec_mean) if rec_mean is not None and not math.isnan(rec_mean) else None
    except Exception:
        rec_mean = None
    try:
        analyst_count = int(analyst_count) if analyst_count is not None else None
    except Exception:
        analyst_count = None
    try:
        target_mean = float(target_mean) if target_mean is not None and not math.isnan(target_mean) else None
    except Exception:
        target_mean = None
    return rec_mean, analyst_count, target_mean

# ---------------- Implied Volatility (IV) ----------------
@st.cache_data(ttl=300)
def fetch_implied_volatility(ticker):
    """
    Best-effort IV: take nearest option expiry (>=7 days away), compute mean impliedVolatility
    across calls+puts. Returns IV as percentage (e.g., 25.4) or None.
    """
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return None
        # select nearest expiry >= 7 days out (to avoid weekly noise); fallback to first
        today = datetime.utcnow().date()
        chosen = None
        for e in exps:
            try:
                ed = datetime.strptime(e, "%Y-%m-%d").date()
                if (ed - today).days >= 7:
                    chosen = e
                    break
            except Exception:
                continue
        if chosen is None:
            chosen = exps[0]
        try:
            opt = t.option_chain(chosen)
            calls = opt.calls
            puts = opt.puts
            ivs = []
            if isinstance(calls, pd.DataFrame) and not calls.empty and "impliedVolatility" in calls.columns:
                ivs += list(calls["impliedVolatility"].dropna().astype(float))
            if isinstance(puts, pd.DataFrame) and not puts.empty and "impliedVolatility" in puts.columns:
                ivs += list(puts["impliedVolatility"].dropna().astype(float))
            if not ivs:
                return None
            # yfinance impliedVolatility is between 0-1; convert to percent
            mean_iv = float(np.mean(ivs)) * 100.0
            return mean_iv
        except Exception:
            return None
    except Exception:
        return None

# ---------------- Factors & scoring (same logic) ----------------
def compute_factors(summary, vix_value, vix_threshold=VIX_THRESHOLD):
    factors = {}
    factors["Trend_OK"] = (summary.get("sma50") is not None and summary.get("sma200") is not None and summary["sma50"] > summary["sma200"])
    factors["RSI_OK"] = (summary.get("rsi") is not None) and (45 <= summary["rsi"] <= 65)
    factors["MACD_OK"] = (summary.get("macd_hist") is not None) and (summary["macd_hist"] > 0)
    factors["Vol_OK"] = (summary.get("vol20") is not None) and (summary.get("vol_today") is not None) and (summary["vol_today"] > summary["vol20"] * VOLUME_SPIKE_FACTOR)
    factors["Analyst_OK"] = (summary.get("rec_mean") is not None and summary.get("analyst_count") is not None and summary["analyst_count"] >= 10 and summary["rec_mean"] <= 2.5)
    factors["Sentiment_OK"] = (vix_value is not None) and (vix_value < vix_threshold)
    return factors

def score_positives(summary, factors):
    score = 0
    score += int(bool(factors.get("Trend_OK")))
    if summary.get("rsi") is not None:
        score += int(bool(factors.get("RSI_OK")))
    if summary.get("macd_hist") is not None:
        score += int(bool(factors.get("MACD_OK")))
    if summary.get("vol20") is not None and summary.get("vol_today") is not None:
        score += int(bool(factors.get("Vol_OK")))
    if summary.get("rec_mean") is not None and summary.get("analyst_count") is not None:
        score += int(bool(factors.get("Analyst_OK")))
    if summary.get("vix") is not None:
        score += int(bool(factors.get("Sentiment_OK")))
    return score

def final_recommendation(summary, factors):
    sma50 = summary.get("sma50"); sma200 = summary.get("sma200")
    rsi = summary.get("rsi"); macd = summary.get("macd_hist")
    if (sma50 is not None and sma200 is not None and sma50 < sma200) or (macd is not None and macd < 0) or (rsi is not None and rsi > 75):
        return "SELL", "Bearish structure or weak momentum / overbought"
    if (factors.get("Trend_OK") and factors.get("MACD_OK") and factors.get("RSI_OK") and factors.get("Analyst_OK") and factors.get("Sentiment_OK")):
        return "BUY", "Multiple strict confirmations aligned"
    if factors.get("Trend_OK") and factors.get("MACD_OK"):
        return "BUY", "Trend positive and momentum supportive (conditional)"
    return "WAIT", "Mixed or insufficient confirmations"

# ---------------- Plot & Table rendering ----------------
def render_plot(df, ticker):
    fig, ax = plt.subplots(figsize=(10,4.5))
    ax.plot(df.index, df["Close"], label="Close", linewidth=1.2)
    ax.plot(df.index, df["SMA50"], label=f"SMA{SMA_SHORT}", linewidth=0.9)
    ax.plot(df.index, df["SMA200"], label=f"SMA{SMA_LONG}", linewidth=0.9)
    ax.set_title(f"{ticker} — Close + SMA{SMA_SHORT}/{SMA_LONG}")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left")
    ax.grid(True)
    ax2 = ax.twinx()
    vol = df["Volume"].fillna(0)
    ax2.bar(df.index, vol, alpha=0.12, width=1)
    ax2.set_yticks([])
    fig.tight_layout()
    return fig

def build_parameter_table(ticker, summary, factors, iv_value, is_etf):
    # rows: Parameter, Value, Comment/Verdict
    rows = []
    cur = summary.get("current")
    rows.append(("Price", f"${cur:.2f}" if cur is not None else "N/A", "Latest close"))
    s50 = summary.get("sma50"); s200 = summary.get("sma200")
    if s50 is not None and s200 is not None:
        rows.append((f"SMA{SMA_SHORT}/{SMA_LONG}", f"{s50:.2f} / {s200:.2f}", "Trend structure"))
        rows.append(("Trend", "UP" if factors.get("Trend_OK") else "DOWN", "SMA50 > SMA200?" ))
    else:
        rows.append(("SMA", "N/A", "Insufficient data"))
    rsi = summary.get("rsi")
    rows.append(("RSI(14)", f"{rsi:.1f}" if rsi is not None else "N/A", "45-65 = healthy momentum" if rsi is not None else "N/A"))
    macd = summary.get("macd_hist")
    rows.append(("MACD hist", f"{macd:.4f}" if macd is not None else "N/A", "Positive = bullish momentum"))
    vol_today = summary.get("vol_today"); vol20 = summary.get("vol20")
    rows.append(("Volume (today)", f"{int(vol_today):,}" if vol_today is not None else "N/A", "Daily volume"))
    rows.append(("Volume 20d avg", f"{int(vol20):,}" if vol20 is not None else "N/A", "20-day average"))
    rows.append(("Volume verdict", "Above avg" if factors.get("Vol_OK") else "Not above avg", "Participation check"))
    rec_mean = summary.get("rec_mean"); a_count = summary.get("analyst_count")
    rows.append(("Analyst rec (mean)", f"{rec_mean:.2f}" if rec_mean is not None else "N/A", f"{a_count if a_count is not None else 'N/A'} analysts"))
    rows.append(("Target mean", f"${summary['target_mean']:.2f}" if summary.get("target_mean") is not None else "N/A", "Mean analyst target"))
    rows.append(("Implied Vol (IV)", f"{iv_value:.1f}% " if iv_value is not None else "N/A", "Avg IV from nearest expiry (calls+puts)"))
    rows.append(("VIX (market)", f"{summary.get('vix'):.2f}" if summary.get("vix") is not None else "N/A", f"Global market volatility; used for sentiment threshold {VIX_THRESHOLD}"))
    rows.append(("ETF?", "Yes" if is_etf else "No", "Detected from yfinance quoteType (if available)"))
    rows.append(("Positive factors", f"{summary.get('positives')} / 6", "Count of satisfied checks"))
    rows.append(("Recommendation", summary.get("recommendation"), summary.get("reason")))
    df_table = pd.DataFrame(rows, columns=["Parameter", "Value", "Comment/Verdict"])
    return df_table

# ---------------- App UI ----------------
st.title("Multi-Stock Dashboard — IV, Indicators & Table")
st.write("Enter tickers (comma-separated). The dashboard prints VIX once, shows IV per symbol (best-effort), and a structured parameter table.")

with st.sidebar:
    st.header("Settings")
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, NVDA")
    period = st.selectbox("Data period", options=["1y","2y","5y","max"], index=1)
    run_button = st.button("Analyze")

if not tickers_input:
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if run_button:
    with st.spinner("Running analysis..."):
        # Fetch VIX once and display at top
        vix_val = fetch_vix_value()
        if vix_val is not None:
            st.metric("VIX (latest)", f"{vix_val:.2f}")
        else:
            st.write("VIX not available")

        dfs = batch_download(tickers, period=period)

        for tk in tickers:
            if tk not in dfs:
                st.warning(f"{tk}: no data found or failed download. Skipping.")
                continue
            df = dfs[tk]
            df = compute_indicators(df)
            last = df.iloc[-1]
            summary = {
                "current": safe_float_last(last["Close"]),
                "sma50": safe_float_last(df["SMA50"]),
                "sma200": safe_float_last(df["SMA200"]),
                "rsi": safe_float_last(df["RSI"]),
                "macd_hist": safe_float_last(df["MACD_HIST"]),
                "vol_today": safe_float_last(last["Volume"]),
                "vol20": safe_float_last(df["Vol_MA20"]),
                "rec_mean": None,
                "analyst_count": None,
                "target_mean": None,
                "vix": vix_val
            }

            # analyst
            rec_mean, a_count, tgt = fetch_analyst_data(tk)
            summary["rec_mean"] = rec_mean
            summary["analyst_count"] = a_count
            summary["target_mean"] = tgt

            # detect if ETF (best-effort)
            is_etf = False
            try:
                info = yf.Ticker(tk).get_info()
                if isinstance(info, dict):
                    qt = info.get("quoteType") or info.get("quote_type")
                    if qt and str(qt).upper() == "ETF":
                        is_etf = True
            except Exception:
                is_etf = False

            # implied vol
            iv_val = fetch_implied_volatility(tk)

            # compute factors and score
            factors = compute_factors(summary, vix_val)
            positives = score_positives(summary, factors)
            rec, reason = final_recommendation(summary, factors)
            summary["recommendation"] = rec
            summary["reason"] = reason
            summary["positives"] = positives

            # plot
            fig = render_plot(df, tk)
            st.pyplot(fig)

            # table and text
            table = build_parameter_table(tk, summary, factors, iv_val, is_etf)
            st.dataframe(table, use_container_width=True)

else:
    st.info("Enter tickers in the sidebar and press Analyze.")
