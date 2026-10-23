#!/usr/bin/env python3
"""Fetch Apple (AAPL) market + insider data and persist it to SQLite.

What this script does
---------------------
1. **Prices** — pulls the last calendar year of *daily* OHLCV bars for AAPL via
   ``yfinance`` and writes them into the ``daily_prices`` table.
2. **Insider trades** — downloads every Form 4 (statement of changes in
   beneficial ownership) filed by Apple Inc. (CIK 0000320193) using the
   ``sec-edgar-downloader`` library, then parses each filing's XML ownership
   tables into normalized rows in the ``form4_filings`` table.

Everything is stored in a local SQLite database (default: ``data/finlearn.db``,
override with ``--db``). Inserts use ``INSERT OR REPLACE`` on natural keys so the
script is fully idempotent — run it as often as you like to keep data fresh.

Usage
-----
    python scripts/fetch_aapl.py                 # default DB at ../../data/finlearn.db
    python scripts/fetch_aapl.py --db my.db
    python scripts/fetch_aapl.py --ticker AAPL   # generalize later; AAPL is the seed

Notes
-----
* SEC EDGAR requires a User-Agent of the form "Name email@example.com".
  Set ``SEC_USER_AGENT`` env var or pass ``--user-agent``.
* Form 4 download can take a couple minutes for ~1 year of filings; the raw
  documents are also kept under ``data/form4/`` for auditing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_aapl")

TICKER = "AAPL"
APPLE_CIK = "0000320193"

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker      TEXT NOT NULL,
    trade_date  TEXT NOT NULL,           -- ISO YYYY-MM-DD
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    adj_close   REAL,
    volume      INTEGER,
    PRIMARY KEY (ticker, trade_date)
);

CREATE TABLE IF NOT EXISTS form4_filings (
    accession_no     TEXT NOT NULL,      -- e.g. 0000320193-24-000123
    transaction_date TEXT NOT NULL,      -- ISO YYYY-MM-DD
    insider_name     TEXT,
    insider_title    TEXT,
    transaction_code TEXT,               -- P=purchase, S=sale, A=grant, F=tax, M=option ex.
    shares           REAL,
    price_per_share  REAL,
    acquired_disposed TEXT,              -- A or D
    equity_classification TEXT,          -- Common Stock / Option...
    filing_date      TEXT,
    source_file      TEXT,
    PRIMARY KEY (accession_no, transaction_date, insider_name,
                 transaction_code, shares, price_per_share)
);

CREATE INDEX IF NOT EXISTS ix_form4_ticker_date ON form4_filings (transaction_date);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open (and if needed create) the SQLite database with our schema."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    log.info("SQLite ready at %s", db_path)
    return conn


# --------------------------------------------------------------------------- #
# 1) Prices via yfinance
# --------------------------------------------------------------------------- #
def fetch_daily_prices(ticker: str) -> list[tuple[Any, ...]]:
    """Return the last year of daily OHLCV rows for *ticker* as tuples.

    Each tuple is ``(ticker, trade_date_iso, open, high, low, close,
    adj_close, volume)`` with ``None`` where the source had gaps.
    """
    import pandas as pd
    import yfinance as yf  # deferred so --help works without deps

    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=365)
    log.info("Fetching %s daily prices %s -> %s", ticker, start, end)

    df = yf.download(
        tickers=ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    # yfinance >=0.2.28 returns MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rows: list[tuple[Any, ...]] = []
    for ts, r in df.iterrows():
        def _f(name: str) -> float | None:
            if name not in r.index:
                return None
            v = r[name]
            if v is None or pd.isna(v):
                return None
            return float(v)

        trade_date = ts.date().isoformat() if isinstance(ts, pd.Timestamp) else str(ts)[:10]
        vol = _f("Volume")
        rows.append(
            (
                ticker.upper(),
                trade_date,
                _f("Open"), _f("High"), _f("Low"), _f("Close"), _f("Adj Close"),
                int(vol) if vol is not None else None,
            )
        )
    log.info("Fetched %d daily price rows", len(rows))
    return rows


def save_prices(conn: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> int:
    conn.executemany(
        """INSERT OR REPLACE INTO daily_prices
           (ticker, trade_date, open, high, low, close, adj_close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    log.info("Saved %d price rows to daily_prices", len(rows))
    return len(rows)


# --------------------------------------------------------------------------- #
# 2) Form 4 insider filings via sec-edgar-downloader
# --------------------------------------------------------------------------- #
def download_form4_files(download_dir: Path, user_agent: str) -> list[Path]:
    """Download all Form 4 filings for Apple CIK. Returns list of file paths.

    Uses ``sec-edgar-downloader`` v5 API:
        Downloader(company_name, email_address, download_folder)
        .get(form="4", ticker_or_cik="aapl", after=date)

    *user_agent* must look like ``"Name someone@example.com"`` per SEC rules;
    the name/email are split out and handed to the library.
    """
    from sec_edgar_downloader import Downloader  # deferred import

    parts = user_agent.split()
    name = " ".join(parts[:-1]) or "FinLearn Research"
    email = parts[-1] if "@" in parts[-1] else "research@example.com"

    dl = Downloader(
        company_name=name,
        email_address=email,
        download_folder=str(download_dir),
    )
    log.info("Downloading Form 4 filings for AAPL (CIK %s) ...", APPLE_CIK)
    # `after` keeps the pull bounded but generous (covers > 1 full year)
    after = date.today() - timedelta(days=400)
    n = dl.get("4", "aapl", after=after.isoformat(), include_amends=True)
    log.info("sec-edgar-downloader reported %d filings fetched", n)

    files = [p for p in download_dir.rglob("*") if p.is_file()]
    log.info("Found %d raw files under %s", len(files), download_dir)
    return files


def parse_form4_file(path: Path) -> list[tuple[Any, ...]]:
    """Parse one downloaded Form 4 document into flat transaction rows.

    Handles both standalone ``ownsecform4.xml`` primary docs and the legacy
    full-submission wrapper (``full-submission.txt`` / ``.txt``) in which the
    XML lives between ``<XML>`` and ``</XML>`` markers.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    # Extract embedded XML from a full-submission wrapper if present.
    if "<ownershipDocument>" in raw:
        start = raw.index("<ownershipDocument>")
        end = raw.find("</ownershipDocument>", start)
        xml_str = raw[start:end + len("</ownershipDocument>")]
    elif path.suffix.lower() == ".xml":
        xml_str = raw
    else:
        return []

    rows: list[tuple[Any, ...]] = []
    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError:
        return rows

    def find(el: str, node: ET.Element) -> ET.Element | None:
        """Find first descendant whose tag ends with *el* (supports 'a/b' paths)."""
        if node is None:
            return None
        parts = el.split("/")
        cur: ET.Element | None = node
        for part in parts:
            cur = next((c for c in cur.iter() if c.tag.endswith(part)), None) \
                if cur is not None else None
            if cur is None:
                return None
        return cur

    def text(el: str, node: ET.Element | None, default: str = "") -> str:
        found = find(el, node)
        return (found.text or default).strip() if found is not None and found.text else default

    # Accession number: prefer the XML element, then a path component that
    # looks like an accession (NNNNNNNN-NN-NNNNNN), then the file stem.
    accession = text("accessionNumber", root)
    if not accession:
        for comp in reversed(path.parts):
            if len(comp) >= 16 and comp.count("-") == 2 and comp[:9].isdigit():
                accession = comp
                break
    accession = accession or path.stem

    filing_date_raw = ""
    # The XML doc may sit beside full-submission.txt (same dir) or be the
    # primary doc inside the accession folder itself.
    for cand in (path.parent / "full-submission.txt",
                 path.parent.parent / "full-submission.txt"):
        if cand.exists():
            try:
                for line in cand.read_text(errors="ignore").splitlines()[:20]:
                    if line.startswith("ACCEPTANCE-DATETIME"):
                        filing_date_raw = line.split(":", 1)[1].strip()
                        break
            except OSError:
                pass
            if filing_date_raw:
                break
    if not filing_date_raw:
        filing_date_raw = text("filingDate", root) or text("dateOfChange", root)

    owner = find("reportingOwner", root)
    if owner is None:
        owner = find("ownerInfo", root)
    name = text("rptOwnerName", owner) if owner is not None else ""
    title = text("officerTitle", owner) if owner is not None else ""
    if not title and owner is not None:
        rel = find("reportingOwnerRelationship", owner)
        if rel is not None and text("isDirector", rel).lower() in ("true", "1"):
            title = "Director"

    txn_tags = ("nonDerivativeTransaction", "derivativeTransaction")
    for txn in [t for t in root.iter() if any(t.tag.endswith(x) for x in txn_tags)]:
        code = text("transactionCode", txn)
        shares = text("transactionShares/value", txn)
        price = (text("transactionPricePerShare/value", txn)
                 or text("conversionOrExercisePrice/value", txn)
                 or text("exercisePrice/value", txn))
        acq_disp = text("transactionAcquiredDisposedCode/value", txn)
        cls = text("securityTitle/value", txn) or text("underlyingSecurityTitle/value", txn)
        date_raw = text("transactionDate/value", txn) or text("transactionDate", txn)
        if not date_raw:
            date_raw = text("periodOfReport", root)
        rows.append((
            accession,
            _norm_date(date_raw),
            name,
            title,
            code,
            _to_float(shares),
            _to_float(price),
            acq_disp[:1].upper() if acq_disp else None,
            cls,
            _norm_date(filing_date_raw),
            str(path),
        ))
    return rows


def _to_float(s: str) -> float | None:
    try:
        return float(str(s).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return None


def _norm_date(s: str) -> str:
    """Normalize EDGAR dates (YYYYMMDD or YYYY-MM-DD) to ISO."""
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s[:10], fmt).date().isoformat()
        except ValueError:
            continue
    return s


def save_form4(conn: sqlite3.Connection, rows: list[tuple[Any, ...]]) -> int:
    conn.executemany(
        """INSERT OR REPLACE INTO form4_filings
           (accession_no, transaction_date, insider_name, insider_title,
            transaction_code, shares, price_per_share, acquired_disposed,
            equity_classification, filing_date, source_file)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    log.info("Saved %d Form 4 transaction rows", len(rows))
    return len(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    here = Path(__file__).resolve()
    default_db = here.parents[2] / "data" / "finlearn.db"

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=default_db, help="SQLite database path")
    ap.add_argument("--ticker", default=TICKER, help="Ticker (default AAPL)")
    ap.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT", "FinLearn Research research@example.com"),
        help='SEC required UA: "Name email@example.com"',
    )
    args = ap.parse_args()

    conn = init_db(args.db)
    download_dir = args.db.parent / "form4"

    # --- prices -------------------------------------------------------------
    try:
        price_rows = fetch_daily_prices(args.ticker)
        save_prices(conn, price_rows)
    except Exception as exc:  # keep going even if one feed fails
        log.error("Price feed failed: %s", exc)

    # --- insider filings ----------------------------------------------------
    try:
        files = download_form4_files(download_dir, args.user_agent)
        all_rows: list[tuple[Any, ...]] = []
        for f in files:
            all_rows.extend(parse_form4_file(f))
        if all_rows:
            save_form4(conn, all_rows)
        else:
            log.warning("Parsed 0 Form 4 transactions (EDGAR format may vary); "
                        "raw files retained in %s", download_dir)
    except Exception as exc:
        log.error("Form 4 feed failed: %s", exc)

    # --- summary ------------------------------------------------------------
    n_prices = conn.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0]
    n_f4 = conn.execute("SELECT COUNT(*) FROM form4_filings").fetchone()[0]
    log.info("DONE. daily_prices=%d rows, form4_filings=%d rows -> %s",
             n_prices, n_f4, args.db)
    conn.close()


if __name__ == "__main__":
    main()
