# FIN
Financial Learning Dashboard Architecture

# FinLearn 📈

**A self-learning, local-first financial knowledge and investor learning dashboard.**

FinLearn ingests live market feeds, SEC Form 4 insider filings, and financial news sentiment. It processes this data locally to compute risk/growth statistics and uses an incremental machine learning engine to build knowledge about market patterns. All of this is surfaced through a clean, modern dual-UI dashboard (React + Streamlit).

> ⚠️ **Disclaimer:** This tool is designed for **educational and learning purposes only**. It is not a trading bot, and nothing in this repository constitutes financial advice. Always do your own research or consult a licensed financial advisor.

---

## ✨ Key Features

- **🧠 Self-Learning Engine:** Uses incremental online learning (`SGDClassifier`) to continuously update its understanding of market patterns based on new feature observations and outcomes.
- **🔒 Local-First AI:** Runs sentiment analysis and NLP pipelines locally using LangChain and Ollama (`llama3.2`), with `sentence-transformers` for embedding. No external API keys required.
- **📊 Proprietary Risk Math:** Calculates rolling Beta, Annualized Sharpe Ratio, and Maximum Drawdown directly from stored daily-return series. Never trusts vendor-provided risk metrics.
- **🏛️ Insider Tracking:** Automatically fetches and parses SEC Form 4 filings to track legitimate insider buying/selling activity.
- **🖥️ Dual UI Architecture:** 
  - **React + Vite + Tailwind:** A production-ready, responsive frontend with interactive candlestick charts (Lightweight Charts) and risk gauges (Recharts).
  - **Streamlit:** A rapid-iteration analyst exploration UI for quick data visualization and debugging.

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Market Feeds: yfinance, SEC EDGAR, News] --> B(Ingestion Layer)
    B --> C[(SQLite Database)]
    C --> D[Self-Learning Engine<br/>SGDClassifier + Feature Store]
    C --> E[Risk Calculator<br/>Beta, Sharpe, Max Drawdown]
    D --> F[FastAPI Backend<br/>/api/* endpoints]
    E --> F
    F --> G[React + Vite Frontend<br/>Dashboard & Charts]
    F --> H[Streamlit Analyst App<br/>Quick Exploration]
