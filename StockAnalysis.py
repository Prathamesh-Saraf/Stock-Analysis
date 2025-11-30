# app.py
"""
Streamlit dashboard: enter tickers -> run analysis -> get per-stock plot + decision summary.
Designed for deployment to Streamlit Cloud (or run locally with `streamlit run app.py`).

Requirements:
    - Python 3.9+
    - See requirements.txt below
"""

import io
from datetime import datetime
import math

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------- CONFIG (tweakable) ----------------
SMA_SHORT = 50
SMA_LONG = 200
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_SPIKE_FACTOR = 1.0
VIX_THRESHOLD = 25.0
DEFAULT_PERIOD = "2y"
DEFAULT_CAPITAL = 6000.0
BATCH_SIZE = 8  # used only for internal batching when downloading many tickers

st.set_page_config(layout="wide", page_title="Multi-Stock Dashboard", initial_sidebar_state="expanded")

# ---------------- Utility helpers ----------------
def safe_float_last(x):
    try:
        if x is None:
            return None
        if hasattr(x, "iloc"):
            val = x.iloc[-1]
        else:
            val = x
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)
    except Exception:
        return None

def safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None

# ---------------- Indicators ----------------
def compute_indicators(df,
                       sma_short=SMA_SHORT, sma_long=SMA_LONG,
                       rsi_period=RSI_PERIOD,
                       macd_fast=MACD_FAST, macd_slow=MACD_SLOW, macd_signal=MACD_SIGNAL):
    df = df.copy()
    df["SMA50"] = df["Close"].rolling(window=sma_short, min_periods=1).mean()
    df["SMA200"] = df["Close"].rolling(window=sma_long, min_periods=1).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=macd_signal, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_SIGNAL"] = macd_sig
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

    df["Vol_MA20"] = df["Volume"].rolling(window=20, min_periods=1).mean()

    return df

# ---------------- Data fetch (cached) ----------------
@st.cache_data(ttl=60)  # cache for 60s; adjust as desired
def batch_download(tickers, period=DEFAULT_PERIOD):
    """
    Downloads multiple tickers in one yf.download call if possible.
    Returns dict ticker -> DataFrame
    """
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
                    # if only one ticker in batch, yf returns single-frame
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

@st.cache_data(ttl=300)
def fetch_vix_value():
    """Fetch VIX close; fallback to VIXY if necessary."""
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
    """
    Best-effort: tries earnings_trend, get_earnings_trend, info, recommendations.
    Returns (rec_mean, analyst_count, target_mean)
    """
    rec_mean = None
    analyst_count = None
    target_mean = None
    try:
        t = yf.Ticker(ticker)
        # try attribute earnings_trend
        try:
            trend = getattr(t, "earnings_trend", None)
            if trend and isinstance(trend, dict) and "trend" in trend and len(trend["trend"]) > 0:
                curr = trend["trend"][0]
                rec_mean = curr.get("ratingMean") or curr.get("ratingmean")
                analyst_count = curr.get("numberOfAnalysts") or curr.get("numberofanalysts")
                target_mean = curr.get("targetMean") or curr.get("targetmean")
        except Exception:
            pass

        # fallback to method get_earnings_trend
        if rec_mean is None or analyst_count is None:
            try:
                if hasattr(t, "get_earnings_trend"):
                    trend2 = t.get_earnings_trend()
                    if trend2 and "trend" in trend2 and len(trend2["trend"]) > 0:
                        curr = trend2["trend"][0]
                        rec_mean = rec_mean or curr.get("ratingMean")
                        analyst_count = analyst_count or curr.get("numberOfAnalysts")
                        target_mean = target_mean or curr.get("targetMean")
            except Exception:
                pass

        # final fallback to info dict
        try:
            info = t.get_info() if hasattr(t, "get_info") else getattr(t, "info", {}) or {}
            if rec_mean is None:
                rec_mean = info.get("recommendationMean") or info.get("recommendationMean")
            if target_mean is None:
                target_mean = info.get("targetMeanPrice") or info.get("targetMean")
        except Exception:
            pass

        # attempt to count recommendations table if present
        try:
            recs = t.recommendations
            if isinstance(recs, pd.DataFrame) and not recs.empty:
                analyst_count = analyst_count or recs.shape[0]
        except Exception:
            pass

    except Exception:
        pass

    # sanitize numeric conversions
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

# ---------------- Factor computation & scoring ----------------
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

# ---------------- Plotting / display ----------------
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

def print_text_dashboard(ticker, summary, factors, positives, allocation):
    st.markdown("---")
    st.subheader(f"{ticker} — Recommendation: **{summary['recommendation']}**")
    col1, col2 = st.columns([2,1])
    with col1:
        if summary.get("current") is not None:
            st.write(f"**Price:** ${summary['current']:.2f}")
        else:
            st.write("**Price:** N/A")
        if summary.get("sma50") is not None and summary.get("sma200") is not None:
            st.write(f"**Trend (SMA{SMA_SHORT}/{SMA_LONG}):** {summary['sma50']:.2f} vs {summary['sma200']:.2f} → {'UP' if factors.get('Trend_OK') else 'DOWN'}")
        else:
            st.write("**Trend (SMA):** N/A")
        if summary.get("rsi") is not None:
            st.write(f"**RSI(14):** {summary['rsi']:.1f} → {'OK' if factors.get('RSI_OK') else ('Overbought' if summary['rsi']>70 else 'Weak/Neutral')}")
        else:
            st.write("**RSI(14):** N/A")
        if summary.get("macd_hist") is not None:
            st.write(f"**MACD hist:** {summary['macd_hist']:.4f} → {'Bullish' if factors.get('MACD_OK') else 'Weak/Bearish'}")
        else:
            st.write("**MACD hist:** N/A")
        vol20_str = f"{int(summary['vol20']):,}" if summary.get("vol20") is not None else "N/A"
        voltoday_str = f"{int(summary['vol_today']):,}" if summary.get("vol_today") is not None else "N/A"
        st.write(f"**Volume:** today={voltoday_str} | 20d_avg={vol20_str} → {'Above avg' if factors.get('Vol_OK') else 'Not above avg'}")
    with col2:
        rec_mean = summary.get("rec_mean")
        a_count = summary.get("analyst_count")
        st.write(f"**Analyst rec (mean):** {rec_mean if rec_mean is not None else 'N/A'}")
        st.write(f"**Analyst count:** {a_count if a_count is not None else 'N/A'}")
        if summary.get("target_mean") is not None and summary.get("current") is not None:
            upside = (summary['target_mean']/summary['current'] - 1) * 100.0
            st.write(f"**Target mean:** ${summary['target_mean']:.2f} ({upside:.1f}%)")
        else:
            st.write("**Target mean:** N/A")
        vix = summary.get("vix")
        st.write(f"**VIX:** {vix if vix is not None else 'N/A'} → {'Stable' if (vix is not None and vix < VIX_THRESHOLD) else 'Risky/Unknown'}")
        st.write(f"**Positive factors:** {positives} / 6")
        st.write(f"**Reason:** {summary.get('reason')}")
        st.write(f"**Suggested allocation:** ${allocation:.2f}")

# ---------------- App UI ----------------
st.title("Multi-Stock Technical + Sentiment Dashboard")
st.write("Enter comma-separated tickers and press Analyze. Works best for batches of 5–20 tickers.")

with st.sidebar:
    st.header("Settings")
    tickers_input = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, NVDA")
    period = st.selectbox("Data period", options=["1y","2y","5y","max"], index=1)
    capital = st.number_input("Total capital (USD)", value=float(DEFAULT_CAPITAL), step=100.0)
    run_button = st.button("Analyze")

if not tickers_input:
    st.stop()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
allocation_per_ticker = capital / len(tickers) if tickers else 0.0

if run_button:
    with st.spinner("Downloading data and running analysis..."):
        vix_value = fetch_vix_value()
        if vix_value is not None:
            st.write(f"VIX (latest): {vix_value:.2f}")
        else:
            st.write("VIX not available")

        dfs = batch_download(tickers, period=period)
        for tk in tickers:
            if tk not in dfs:
                st.warning(f"{tk}: no data found or failed download. Skipping.")
                continue
            df = dfs[tk]
            df = compute_indicators(df)

            last_row = df.iloc[-1]
            summary = {
                "current": safe_float_last(last_row["Close"]),
                "sma50": safe_float_last(df["SMA50"]),
                "sma200": safe_float_last(df["SMA200"]),
                "rsi": safe_float_last(df["RSI"]),
                "macd_hist": safe_float_last(df["MACD_HIST"]),
                "vol_today": safe_float_last(last_row["Volume"]),
                "vol20": safe_float_last(df["Vol_MA20"]),
                "rec_mean": None,
                "analyst_count": None,
                "target_mean": None,
                "vix": vix_value
            }

            rec_mean, a_count, tgt = fetch_analyst_data(tk)
            summary["rec_mean"] = rec_mean
            summary["analyst_count"] = a_count
            summary["target_mean"] = tgt

            factors = compute_factors(summary, vix_value)
            positives = score_positives(summary, factors)
            rec, reason = final_recommendation(summary, factors)
            summary["recommendation"] = rec
            summary["reason"] = reason

            # plot
            fig = render_plot(df, tk)
            st.pyplot(fig)

            # dashboard
            print_text_dashboard(tk, summary, factors, positives, allocation_per_ticker)
else:
    st.info("Enter tickers and press Analyze to run the dashboard.")
