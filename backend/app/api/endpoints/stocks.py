"""Stock price / risk endpoints backed by SQLite + RiskCalculator."""

from __future__ import annotations

import math
import sqlite3

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.learning.risk_calculator import RiskCalculator

router = APIRouter()


def _conn() -> sqlite3.Connection:
    if not settings.db_path.exists():
        raise HTTPException(404, "Database not populated. Run scripts/fetch_aapl.py first.")
    return sqlite3.connect(settings.db_path)


@router.get("/stocks/{ticker}/prices")
def prices(ticker: str, months: int = Query(12, ge=1, le=24)):
    """OHLCV bars for the last *months* months."""
    with _conn() as con:
        df = pd.read_sql(
            "SELECT trade_date, open, high, low, close, adj_close, volume "
            "FROM daily_prices WHERE ticker=? ORDER BY trade_date",
            con, params=(ticker.upper(),),
        )
    if df.empty:
        raise HTTPException(404, f"No stored prices for {ticker}")
    df = df.tail(months * 21)
    return {"ticker": ticker.upper(), "rows": json_safe(df.to_dict("records"))}


@router.get("/stocks/{ticker}/risk")
def risk(ticker: str):
    """Risk & growth metrics computed locally from stored returns."""
    with _conn() as con:
        px = pd.read_sql(
            "SELECT trade_date, close FROM daily_prices WHERE ticker=? ORDER BY trade_date",
            con, params=(ticker.upper(),), parse_dates=["trade_date"],
        ).set_index("trade_date")
        bench = pd.read_sql(
            "SELECT trade_date, close FROM daily_prices WHERE ticker='^GSPC' ORDER BY trade_date",
            con, parse_dates=["trade_date"],
        ).set_index("trade_date")
    if px.empty:
        raise HTTPException(404, f"No stored prices for {ticker}")
    rets = px["close"].pct_change().rename("asset").to_frame()
    if not bench.empty:
        rets["benchmark"] = bench["close"].pct_change().reindex(rets.index)
    rc = RiskCalculator(rets, risk_free_rate=settings.risk_free_rate)
    s = rc.summary()
    beta = rc.rolling_beta().dropna()
    return {
        "ticker": ticker.upper(),
        "summary": {k: (None if isinstance(v, float) and not math.isfinite(v) else round(v, 5))
                    for k, v in s.items()},
        "beta_series": [{"date": d.strftime("%Y-%m-%d"), "beta": round(float(b), 4)}
                        for d, b in beta.tail(90).items()],
    }


@router.get("/stocks")
def list_tickers():
    with _conn() as con:
        rows = con.execute("SELECT DISTINCT ticker FROM daily_prices ORDER BY ticker").fetchall()
    return {"tickers": [r[0] for r in rows]}


def json_safe(records: list[dict]) -> list[dict]:
    out = []
    for rec in records:
        out.append({k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                        and not (isinstance(v, float) and math.isnan(v)) else v)
                    for k, v in rec.items()})
    return out
