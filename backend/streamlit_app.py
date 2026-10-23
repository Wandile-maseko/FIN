"""FinLearn — investor learning dashboard (Streamlit).

Run from the ``backend/`` directory:

    pip install -r requirements.txt          # streamlit + plotly included
    streamlit run streamlit_app.py

Features
--------
* Sidebar ticker selector (any Yahoo Finance symbol; data cached to SQLite via
  the same schema as ``scripts/fetch_aapl.py`` when that DB exists).
* Plotly **candlestick** chart with volume and a 30-day rolling-beta overlay.
* **Risk panel** computed locally by :class:`app.learning.risk_calculator.RiskCalculator`
  (Sharpe, max drawdown, volatility, annualized growth).
* News **sentiment gauge**: average score in [-1, 1] from headlines you paste in
  (uses scripts/sentiment_pipeline.py keyword scorer so the app runs even
  without Ollama; if the LLM stack is installed it upgrades automatically).
* Insider Form-4 table when the AAPL database has been populated.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make `app.learning.risk_calculator` importable regardless of CWD.
BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(BACKEND_ROOT / "scripts"))

from app.learning.risk_calculator import RiskCalculator  # noqa: E402

DB_PATH = BACKEND_ROOT.parent / "data" / "finlearn.db"
BENCHMARK = "^GSPC"  # S&P 500 total-price proxy for beta


st.set_page_config(page_title="FinLearn — Investor Dashboard", layout="wide",
                   page_icon="📈")


# --------------------------------------------------------------------------- #
# Data access (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=900, show_spinner="Fetching daily prices…")
def get_prices(ticker: str) -> pd.DataFrame:
    """Last year of OHLCV bars for *ticker* (yfinance), falling back to the
    local SQLite DB if the network is unavailable."""
    try:
        import yfinance as yf
        df = yf.download(ticker.upper(), period="1y", interval="1d",
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            return df
        raise RuntimeError("empty frame")
    except Exception:
        if DB_PATH.exists():
            d = pd.read_sql(
                "SELECT trade_date, open, high, low, close, volume FROM daily_prices "
                "WHERE ticker=? ORDER BY trade_date",
                __import__("sqlite3").connect(DB_PATH), params=(ticker.upper(),),
                parse_dates=["trade_date"],
            ).set_index("trade_date")
            if not d.empty:
                st.sidebar.caption(f"⚠ Offline mode — served {len(d)} rows from SQLite.")
                return d
        raise


@st.cache_data(ttl=3600, show_spinner="Fetching benchmark (^GSPC)…")
def get_benchmark_returns() -> pd.Series:
    try:
        b = get_prices(BENCHMARK)
        return b["Close"].pct_change().dropna()
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600)
def get_insider_filings(ticker: str) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    con = __import__("sqlite3").connect(DB_PATH)
    try:
        return pd.read_sql(
            "SELECT transaction_date, insider_name, insider_title, transaction_code, "
            "shares, price_per_share, acquired_disposed FROM form4_filings "
            "ORDER BY transaction_date DESC LIMIT 200", con)
    finally:
        con.close()


def returns_frame(prices: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    asset = prices["Close"].pct_change().rename("asset")
    df = asset.to_frame()
    if not bench.empty:
        df["benchmark"] = bench.reindex(df.index)
    return df


# --------------------------------------------------------------------------- #
# Sentiment scoring (LLM if available, keyword fallback otherwise)
# --------------------------------------------------------------------------- #
def headline_scores(headlines: list[str]) -> list[dict]:
    try:
        from sentiment_pipeline import run_pipeline  # type: ignore
        return run_pipeline(headlines, use_embeddings=False)
    except Exception:
        from math import tanh
        bull = {"beat": .6, "beats": .6, "surges": .7, "record": .6, "rally": .5,
                "upgrade": .6, "growth": .4, "profit": .4, "high": .3, "soars": .8}
        bear = {"miss": -.6, "falls": -.4, "fall": -.4, "lawsuit": -.5, "recall": -.6,
                "probe": -.4, "downgrade": -.6, "loss": -.5, "plunge": -.8, "cut": -.3}
        out = []
        import re
        for h in headlines:
            s = sum(bull.get(w, 0) + bear.get(w, 0) for w in re.findall(r"[a-z]+", h.lower()))
            v = round(tanh(s / 1.5), 3)
            out.append({"headline": h, "score": v,
                        "label": "bullish" if v > .15 else "bearish" if v < -.15 else "neutral",
                        "method": "keyword"})
        return out


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("📈 FinLearn")
st.sidebar.subheader("Investor learning tool")

PRESET = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM"]
choice = st.sidebar.selectbox("Popular tickers", PRESET)
custom = st.sidebar.text_input("…or any ticker", "").strip().upper()
TICKER = custom or choice

months = st.sidebar.slider("History window (months)", 3, 12, 12, step=3)
rf = st.sidebar.number_input("Risk-free rate (annual %)", 0.0, 10.0, 4.5, 0.5) / 100.0
st.sidebar.markdown("---")
st.sidebar.write("Sentiment engine:", "LangChain+Ollama" if
                 Path(BACKEND_ROOT / "scripts" / "sentiment_pipeline.py").exists() else "keyword")

# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title(f"{TICKER} — Company Study")

try:
    prices_full = get_prices(TICKER)
except Exception as exc:
    st.error(f"Could not load price data for {TICKER}: {exc}")
    st.stop()

prices = prices_full.tail(int(months * 21))
bench = get_benchmark_returns()
rets = returns_frame(prices, bench)

# ---------------- risk metrics (our own math) ----------------
rc = RiskCalculator(rets, risk_free_rate=rf)
summary = rc.summary()
beta_series = rc.rolling_beta()
drawdown = rc.drawdown_series()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("30-day Beta", f"{summary['beta_30d']:.2f}" if np.isfinite(summary["beta_30d"]) else "n/a")
c2.metric("Sharpe Ratio", f"{summary['sharpe_ratio']:.2f}" if np.isfinite(summary["sharpe_ratio"]) else "n/a")
c3.metric("Max Drawdown", f"{summary['maximum_drawdown']*100:.1f}%")
c4.metric("Ann. Volatility", f"{summary['annualized_volatility']*100:.1f}%")
c5.metric("Ann. Return", f"{summary['annualized_return']*100:.1f}%")

tab_px, tab_risk, tab_news, tab_ins = st.tabs(["🕯 Price", "🧮 Risk & Growth", "📰 News Sentiment", "👤 Insider Filings"])

with tab_px:
    fig = go.Figure(data=[
        go.Candlestick(
            x=prices.index, open=prices["Open"], high=prices["High"],
            low=prices["Low"], close=prices["Close"], name=TICKER,
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        )
    ])
    fig.add_trace(go.Scatter(x=prices.index,
                             y=prices["Close"].rolling(20).mean(),
                             name="SMA 20", line=dict(color="#ffa726", width=1.5)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10),
                      template="plotly_white")
    st.plotly_chart(fig, width="stretch")

    vol = go.Figure(go.Bar(x=prices.index, y=prices["Volume"],
                           marker_color="#90a4ae", name="Volume"))
    vol.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white")
    st.plotly_chart(vol, width="stretch")

with tab_risk:
    left, right = st.columns(2)
    with left:
        st.subheader("30-day rolling beta vs S&P 500")
        fb = beta_series.dropna()
        if not fb.empty:
            st.plotly_chart(px.line(fb, labels={"value": "beta", "index": ""},
                                    template="plotly_white", height=300),
                            width="stretch")
        else:
            st.info("Benchmark data unavailable — beta needs ^GSPC returns.")
    with right:
        st.subheader("Drawdown underwater curve")
        st.plotly_chart(px.area(drawdown, labels={"value": "drawdown", "index": ""},
                                template="plotly_white", height=300),
                        width="stretch")
    pk, tr = rc.max_drawdown_dates()
    if pk is not None:
        st.caption(f"Worst drawdown: peak {pk.date()} → trough {tr.date()} "
                   f"({summary['maximum_drawdown']*100:.1f}%).")

with tab_news:
    st.subheader("Score your own news feed")
    default = ("Apple hits record high on AI iPhone demand\n"
               "Analysts downgrade Apple on China weakness\n"
               "Apple buyback expands by $110 billion")
    raw = st.text_area("One headline per line", value=default, height=140)
    headlines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if st.button("Analyze sentiment", type="primary") and headlines:
        results = headline_scores(headlines)
        avg = float(np.mean([r["score"] for r in results]))

        # --- gauge chart ---
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg,
            number={"valueformat": ".2f", "font": {"size": 42}},
            title={"text": "Aggregate Sentiment (-1 … +1)"},
            gauge={
                "axis": {"range": [-1, 1], "tickwidth": 1},
                "bar": {"color": "#26a69a" if avg >= 0 else "#ef5350"},
                "steps": [
                    {"range": [-1, -0.55], "color": "#b71c1c"},
                    {"range": [-0.55, -0.15], "color": "#ef9a9a"},
                    {"range": [-0.15, 0.15], "color": "#e0e0e0"},
                    {"range": [0.15, 0.55], "color": "#a5d6a7"},
                    {"range": [0.55, 1], "color": "#1b5e20"},
                ],
                "threshold": {"line": {"color": "black", "width": 3},
                              "thickness": 0.8, "value": avg},
            },
        ))
        gauge.update_layout(height=380, margin=dict(l=40, r=40, t=60, b=10))
        st.plotly_chart(gauge, width="stretch")

        st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)

with tab_ins:
    ins = get_insider_filings(TICKER)
    if ins.empty:
        st.info("No insider filings in the local database yet. Run "
                "`python scripts/fetch_aapl.py` to populate Form 4 data for AAPL.")
    else:
        st.subheader("SEC Form 4 transactions (latest)")
        CODES = {"P": "Purchase", "S": "Sale", "A": "Grant", "F": "Tax", "M": "Option Ex."}
        ins["code_meaning"] = ins["transaction_code"].map(CODES).fillna(ins["transaction_code"])
        buys = ins[ins["acquired_disposed"] == "A"]["shares"].fillna(0).sum()
        sells = ins[ins["acquired_disposed"] == "D"]["shares"].fillna(0).sum()
        b1, b2 = st.columns(2)
        b1.metric("Insider shares acquired", f"{buys:,.0f}")
        b2.metric("Insider shares disposed", f"{sells:,.0f}")
        st.dataframe(ins, width="stretch", hide_index=True)

st.caption("Educational tool only — not investment advice. Risk metrics are computed "
           "locally from stored daily returns by RiskCalculator.")
