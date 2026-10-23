"""Insider (Form 4) endpoints."""

from __future__ import annotations

import sqlite3

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings

router = APIRouter()


@router.get("/insiders")
def recent_filings(limit: int = Query(100, ge=1, le=500)):
    with sqlite3.connect(settings.db_path) as con:
        df = pd.read_sql(
            "SELECT transaction_date, insider_name, insider_title, transaction_code, "
            "shares, price_per_share, acquired_disposed FROM form4_filings "
            "ORDER BY transaction_date DESC LIMIT ?", con, params=(limit,))
    return {"rows": df.to_dict("records")}


@router.get("/insiders/netflow")
def net_flow(days: int = Query(90, ge=7, le=365)):
    """Aggregate acquired vs disposed shares over trailing *days* — a feature the
    KnowledgeEngine consumes as `insider_net_norm`."""
    with sqlite3.connect(settings.db_path) as con:
        row = con.execute(
            "SELECT COALESCE(SUM(CASE WHEN acquired_disposed='A' THEN shares END),0), "
            "COALESCE(SUM(CASE WHEN acquired_disposed='D' THEN shares END),0) "
            "FROM form4_filings WHERE date(transaction_date) >= date('now', ?)",
            (f"-{days} days",),
        ).fetchone()
    acquired, disposed = float(row[0]), float(row[1])
    denom = acquired + disposed
    net_norm = (acquired - disposed) / denom if denom else 0.0
    return {"days": days, "acquired": acquired, "disposed": disposed,
            "net_normalized": round(net_norm, 4)}
