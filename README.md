# FinLearn 📈

**A self-learning, local-first financial knowledge and investor learning dashboard.**

FinLearn ingests live market feeds, SEC Form 4 insider filings, and financial news sentiment. It processes this data locally to compute proprietary risk/growth statistics and uses an incremental machine learning engine to build knowledge about market patterns. All of this is surfaced through a clean, modern dual-UI dashboard (React + Streamlit).

> ⚠️ **Disclaimer:** This tool is designed for **educational and learning purposes only**. It is not a trading bot, and nothing in this repository constitutes financial advice. Always do your own research or consult a licensed financial advisor.

---

## ✨ Key Features

- **🧠 Incremental Self-Learning Engine:** The `KnowledgeEngine` uses `SGDClassifier` with `partial_fit` to continuously update its understanding of market patterns. Each new observation (features → next-day return direction) updates weights online, persisting "knowledge" directly to SQLite.
- **🔒 Local-First AI:** Runs sentiment analysis and NLP pipelines locally using LangChain and Ollama (`llama3.2`), with `sentence-transformers` for embedding and deduplication. **No external API keys required.** Includes a graceful keyword-based fallback if Ollama is unavailable.
- **📊 Proprietary Risk Math:** Calculates rolling Beta, Annualized Sharpe Ratio, and Maximum Drawdown directly from stored daily-return series via the `RiskCalculator`. Vendor-provided risk metrics are never blindly trusted.
- **🏛️ Insider Tracking:** Automatically fetches and parses SEC Form 4 filings to track legitimate insider buying/selling activity.
- **🖥️ Dual UI Architecture:** 
  - **React + Vite + Tailwind:** A production-ready, responsive frontend with interactive candlestick charts (TradingView Lightweight Charts) and risk gauges (Recharts).
  - **Streamlit:** A rapid-iteration analyst exploration UI for quick data visualization, debugging, and learning.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Market Feeds: yfinance, SEC EDGAR, News] --> B(Ingestion Layer)
    B --> C[(SQLite Database<br/>prices / insiders / sentiment)]
    C --> D[Self-Learning Engine<br/>SGDClassifier + Feature Store]
    C --> E[Risk Calculator<br/>Beta, Sharpe, Max Drawdown]
    D --> F[FastAPI Backend<br/>/api/* endpoints]
    E --> F
    F --> G[React + Vite Frontend<br/>Dashboard & Charts]
    F --> H[Streamlit Analyst App<br/>Quick Exploration]
 ```
---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend API** | Python, FastAPI, Uvicorn, Pydantic, SQLAlchemy |
| **Data & Feeds** | SQLite, Pandas, NumPy, `yfinance`, `sec-edgar-downloader` |
| **AI & Machine Learning** | Scikit-Learn (Online Learning), Joblib, LangChain, Ollama, `sentence-transformers`, PyTorch |
| **Frontend** | React, TypeScript, Vite, Tailwind CSS, TanStack React Query |
| **Visualization** | Recharts, TradingView Lightweight Charts, Plotly (Streamlit) |
| **Dev & Testing** | Ruff, ESLint, Pytest, HTTPX |

---

## 📂 Project Structure

```text
/workspace
 ├── ARCHITECTURE.md                  # Detailed system design and data flow
 ├── README.md                        # This file
 ├── data/                            # SQLite databases (gitignored)
 │
 ├── backend/
 │   ├── requirements.txt             # Python dependencies
 │   ├── .env.example                 # Configuration template
 │   ├── app/
 │   │   ├── main.py                  # FastAPI application factory + CORS
 │   │   ├── core/config.py           # Pydantic-settings config (DB path, Ollama URL)
 │   │   ├── db/database.py           # SQLite engine/session helpers
 │   │   ├── models/                  # SQLAlchemy ORM models (price, insider, sentiment)
 │   │   ├── schemas/responses.py     # Pydantic response models
 │   │   ├── services/                # Data fetchers (yfinance, SEC EDGAR, LangChain/Ollama)
 │   │   ├── learning/                # RiskCalculator & KnowledgeEngine (partial_fit)
 │   │   └── api/endpoints/           # FastAPI routers (stocks, insiders, sentiment, learn)
 │   ├── scripts/                     # Data ingestion and seeding scripts
 │   ├── streamlit_app.py             # Investor learning GUI (Plotly candles/gauge)
 │   └── tests/                       # Pytest suite
 │
 └── frontend/
     ├── package.json                 # Node dependencies
     ├── vite.config.ts               # Dev proxy to backend :8000
     ├── index.html
     └── src/
         ├── api/client.ts            # Typed fetch wrapper for backend
         ├── hooks/useStockData.ts    # React Query polling hook
         ├── components/              # CandlestickChart, RiskPanel, InsiderTable, SentimentGauge
         └── pages/                   # Dashboard, CompanyDetail

---

## 🚀 Installation & Setup

| Step | Phase | Action | Command / Details |
| :--- | :--- | :--- | :--- |
| **1** | **Clone** | Clone the repository | `git clone https://github.com/Wandile-maseko/FIN.git` <br> `cd FIN` |
| **2** | **Backend** | Set up Python Virtual Environment | `python -m venv venv` <br> `source venv/bin/activate` *(Windows: `venv\Scripts\activate`)* |
| **3** | **Backend** | Install Python Dependencies | `pip install -r backend/requirements.txt` |
| **4** | **Backend** | Configure Environment | `cp backend/.env.example backend/.env` <br> *(Optional: Defaults are already set)* |
| **5** | **Frontend** | Install Node Dependencies | `cd frontend` <br> `npm install` |

---

## 🏃‍♂️ How to Run

| Service | Terminal Setup | Run Command | Access URL |
| :--- | :--- | :--- | :--- |
| **Backend API** | `cd backend` | `uvicorn app.main:app --reload --port 8000` | API: `http://localhost:8000` <br> Docs: `http://localhost:8000/docs` |
| **React Frontend** | `cd frontend` | `npm run dev` | Dashboard: `http://localhost:5173` |
| **Streamlit App** | `cd backend` | `streamlit run streamlit_app.py` | Analyst UI: `http://localhost:8501` |

---

## 🧪 Testing & Linting

| Environment | Setup | Test Command | Lint Command |
| :--- | :--- | :--- | :--- |
| **Backend (Python)** | `cd backend` | `pytest tests/ -v` | `ruff check .` |
| **Frontend (React)** | `cd frontend` | *(No test script provided)* | `npm run lint` |

---

## 🤝 Contributing

| Step | Action | Command / Details |
| :--- | :--- | :--- |
| **1** | Fork the repository | *(Use the GitHub UI)* |
| **2** | Create your feature branch | `git checkout -b feature/AmazingFeature` |
| **3** | Commit your changes | `git commit -m 'Add some AmazingFeature'` |
| **4** | Push to the branch | `git push origin feature/AmazingFeature` |
| **5** | Open a Pull Request | *(Use the GitHub UI)* |
| **Issues** | Report bugs or request features | [Open an Issue](https://github.com/Wandile-maseko/FIN/issues) |
         
