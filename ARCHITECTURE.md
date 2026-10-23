# FinLearn — Investor Learning Dashboard: Architecture

A self-learning financial knowledge tool that ingests market feeds (yfinance), SEC
filings (Form 4 insider activity), and news sentiment (LangChain + local LLM via
Ollama), computes risk/growth statistics itself, and surfaces everything through a
clean dashboard.

## High-Level Data Flow

```
             +------------------+     +---------------------+
 Feeds ----> | Ingestion Layer  | --> | SQLite (data/*.db)  |
 (yfinance,  | backend/scripts  |     | prices / insiders / |
  SEC EDGAR, | + services       |     | sentiment / signals |
  news)      +------------------+     +----------+----------+
                                                 |
                                                 v
                                   +---------------------------+
                                   | Self-Learning Engine      |
                                   | backend/app/learning      |
                                   | - online logistic         |
                                   |   regressor (incremental  |
                                   |   knowledge building)     |
                                   | - feature store           |
                                   | - pattern memory          |
                                   +-----------+---------------+
                                               |
              +--------------------+           v
 Analytics -> | RiskCalculator     |   +-------------------+   +----------------+
 (returns)    | beta/sharpe/mdd    |   | FastAPI backend   |-> | React frontend |
              +--------------------+   | /api/* endpoints  |   | Vite + Recharts|
                                       +-------------------+   +----------------+
                                               ^
                                       +-------+--------+
                                       | Streamlit app  |  (analyst/learning UI:
                                       | streamlit_app  |   candlesticks + gauges)
                                       +----------------+
```

## Directory Structure

```
/workspace
├── ARCHITECTURE.md                  # this file
├── README.md                        # setup & run instructions
├── data/                            # SQLite databases (gitignored content)
│
├── backend/
│   ├── requirements.txt             # Python dependencies
│   ├── .env.example                 # configuration template
│   ├── app/
│   │   ├── main.py                  # FastAPI application factory + CORS
│   │   ├── core/
│   │   │   └── config.py            # pydantic-settings config (DB path, Ollama URL)
│   │   ├── db/
│   │   │   └── database.py          # SQLite engine/session helpers
│   │   ├── models/
│   │   │   ├── price.py             # ORM: daily_prices
│   │   │   ├── insider.py           # ORM: form4_filings
│   │   │   └── sentiment.py         # ORM: news_sentiment
│   │   ├── schemas/
│   │   │   └── responses.py         # Pydantic response models
│   │   ├── services/
│   │   │   ├── market_data.py       # yfinance fetch/caching service
│   │   │   ├── sec_edgar.py         # Form 4 downloader service
│   │   │   └── news_sentiment.py    # LangChain/Ollama sentiment pipeline
│   │   ├── learning/
│   │   │   ├── risk_calculator.py   # RiskCalculator (beta, Sharpe, max drawdown)
│   │   │   └── knowledge_engine.py  # self-learning signal/outcome loop
│   │   └── api/
│   │       ├── router.py            # aggregates sub-routers
│   │       └── endpoints/
│   │           ├── stocks.py        # GET /stocks/{ticker}/prices|risk|stats
│   │           ├── insiders.py      # GET /insiders/{ticker}
│   │           ├── sentiment.py     # POST /sentiment, GET /sentiment/{ticker}
│   │           └── learn.py         # GET/POST /learn/status, /learn/observe
│   ├── scripts/
│   │   ├── fetch_aapl.py            # yfinance + sec-edgar-downloader -> SQLite
│   │   └── seed_demo_data.py        # offline demo data generator
│   ├── streamlit_app.py             # investor learning GUI (plotly candles/gauge)
│   └── tests/
│       └── test_risk_calculator.py
│
└── frontend/
    ├── package.json                 # Node dependencies
    ├── vite.config.ts               # dev proxy to :8000
    ├── index.html
    └── src/
        ├── main.tsx
        ├── App.tsx                  # layout + routing
        ├── api/client.ts            # typed fetch wrapper for backend
        ├── hooks/useStockData.ts    # polling/react-query style hook
        ├── components/
        │   ├── CandlestickChart.tsx
        │   ├── RiskPanel.tsx        # beta/sharpe/drawdown cards
        │   ├── InsiderTable.tsx
        │   └── SentimentGauge.tsx
        └── pages/
            ├── Dashboard.tsx
            └── CompanyDetail.tsx
```

## Key Design Decisions

1. **SQLite as the system of record** — single-file, zero-ops storage; every feed
   (prices, Form 4s, sentiment scores) lands in a table with an upsert key so
   ingestion is idempotent and re-runnable offline.
2. **Self-learning = incremental, not batch.** `KnowledgeEngine` uses
   `SGDClassifier(log_loss)` with `partial_fit`, so each new observation
   (features → next-day return direction) updates weights online. It persists
   coefficients back to SQLite ("knowledge") and can be reloaded on boot.
3. **Risk math is done locally, never trusted from vendors.** `RiskCalculator`
   derives rolling beta, annualized Sharpe, and max drawdown directly from the
   stored daily-return series.
4. **Local-first AI.** Sentiment runs through LangChain → Ollama (`llama3.2` by
   default) with sentence-transformer embeddings for similarity dedup/clustering;
   no external API keys required. Graceful keyword-based fallback if Ollama is down.
5. **Two UIs, one core.** Streamlit for quick analyst exploration (candles + gauge);
   React/FastAPI for the production dashboard. Both consume the same services.
