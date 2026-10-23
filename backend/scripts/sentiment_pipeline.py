#!/usr/bin/env python3
"""News-headline sentiment pipeline: embeddings + local LLM scoring.

Pipeline
--------
1. **Embed** every headline with a sentence-transformers model
   (``all-MiniLM-L6-v2`` by default) so we can deduplicate near-identical
   stories via cosine similarity before paying the LLM for them.
2. **Score** each unique headline with a local LLM running under Ollama,
   driven through LangChain's ``ChatOllama`` interface. The prompt forces a
   single float in [-1, 1] (-1 = very bearish, 0 = neutral, +1 = very bullish).
3. **Validate & clamp** the parsed score, retry once on malformed output, and
   fall back to a lightweight keyword heuristic if Ollama is unreachable —
   so the script never crashes mid-feed.

Usage
-----
    # start a local model first:  ollama pull llama3.2 && ollama serve
    python scripts/sentiment_pipeline.py --headlines "Apple beats EPS estimates" "Fed holds rates"
    python scripts/sentiment_pipeline.py --file headlines.txt          # one per line
    echo "Nvidia stalls after rally" | python scripts/sentiment_pipeline.py -
    python scripts/sentiment_pipeline.py --demo                        # built-in sample set
    python scripts/sentiment_pipeline.py --demo --out scores.json      # persist results

Output: JSON list of ``{"headline", "score", "label", "method"}`` printed to
stdout (and optionally written with ``--out``).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentiment")

DEFAULT_MODEL = "llama3.2"           # any instruct model available in Ollama
DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"
OLLAMA_BASE_URL = "http://localhost:11434"

# --------------------------------------------------------------------------- #
# 1) Embeddings (dedup near-identical headlines)
# --------------------------------------------------------------------------- #
def embed_headlines(headlines: list[str], model_name: str = DEFAULT_EMBEDDER) -> Any:
    """Return a (n, d) embedding matrix for *headlines* using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    log.info("Loading sentence-transformer '%s' ...", model_name)
    model = SentenceTransformer(model_name)
    vecs = model.encode(headlines, normalize_embeddings=True, show_progress_bar=False)
    log.info("Embedded %d headlines -> dim %d", len(headlines), vecs.shape[1])
    return vecs


def dedupe_by_similarity(headlines: list[str], threshold: float = 0.92) -> list[int]:
    """Indices of headlines to keep, dropping ones >= *threshold* cosine-similar
    to an earlier kept headline (embeddings are normalized, so dot == cosine)."""
    if not headlines:
        return []
    vecs = embed_headlines(headlines)
    keep: list[int] = []
    for i in range(len(headlines)):
        dup = False
        for j in keep:
            sim = float(vecs[i] @ vecs[j])
            if sim >= threshold:
                dup = True
                log.info("Duplicate skipped: %r ~ %r (%.3f)", headlines[i], headlines[j], sim)
                break
        if not dup:
            keep.append(i)
    return keep


# --------------------------------------------------------------------------- #
# 2) LLM scoring via LangChain + Ollama
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a precise financial sentiment analyst. For each headline you are "
    "given, output ONLY a single floating point number between -1 and 1 that "
    "captures market sentiment toward the mentioned company/market:\n"
    "  -1.0 = extremely bearish, -0.5 = mildly bearish, 0.0 = neutral,\n"
    "  +0.5 = mildly bullish, +1.0 = extremely bullish.\n"
    "No words, no units, no explanation. Just the number."
)

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def build_chat_model(model: str = DEFAULT_MODEL, base_url: str = OLLAMA_BASE_URL):
    """Instantiate LangChain's ChatOllama bound to the local model."""
    from langchain_ollama import ChatOllama  # provided by langchain-community

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0.0,     # deterministic scoring
        num_predict=8,       # we only need a number
        format="text",
    )


def parse_score(raw: str) -> float | None:
    """Extract the first float in [-1, 1] from an LLM response; clamp it."""
    m = _NUM_RE.search(raw or "")
    if not m:
        return None
    try:
        v = float(m.group())
    except ValueError:
        return None
    return max(-1.0, min(1.0, v))


def score_with_llm(llm: Any, headline: str) -> float | None:
    """Ask the local LLM for one headline's score; retry once on garbage output."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=headline)]
    for attempt in (1, 2):
        try:
            resp = llm.invoke(messages)
            score = parse_score(resp.content)
            if score is not None:
                return score
            log.warning("Unparseable LLM output (attempt %d): %r", attempt, resp.content[:80])
        except Exception as exc:  # connection refused, model missing, etc.
            log.warning("Ollama call failed (attempt %d): %s", attempt, exc)
            break
    return None


# --------------------------------------------------------------------------- #
# 3) Fallback heuristic (used only when the LLM is unavailable)
# --------------------------------------------------------------------------- #
_BULL = {
    "beat": 0.6, "beats": 0.6, "soars": 0.8, "surges": 0.7, "rally": 0.5, "rallies": 0.5,
    "record": 0.6, "upgrade": 0.6, "buy": 0.4, "growth": 0.4, "profit": 0.4, "profits": 0.4,
    "gain": 0.4, "gains": 0.4, "high": 0.3, "wins": 0.4, "approval": 0.5, "expands": 0.4,
    "launches": 0.3, "breakthrough": 0.7, "outperforms": 0.6, "dividend": 0.2,
}
_BEAR = {
    "miss": -0.6, "misses": -0.6, "plunge": -0.8, "plunges": -0.8, "drop": -0.4, "drops": -0.4,
    "fall": -0.4, "falls": -0.4, "lawsuit": -0.5, "probe": -0.4, "recall": -0.6, "layoff": -0.6,
    "layoffs": -0.6, "downgrade": -0.6, "sell": -0.4, "loss": -0.5, "losses": -0.5, "debt": -0.3,
    "crash": -0.9, "slump": -0.6, "stalls": -0.4, "cut": -0.3, "cuts": -0.3, "warning": -0.5,
    "investigation": -0.5, "bearish": -0.6,
}


def keyword_score(headline: str) -> float:
    """Sum bullish/bearish keyword weights, squash to (-1, 1) with tanh."""
    import math

    words = re.findall(r"[a-z]+", headline.lower())
    s = sum(_BULL.get(w, 0.0) + _BEAR.get(w, 0.0) for w in words)
    return round(math.tanh(s / 1.5), 3)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def label_for(score: float) -> str:
    if score >= 0.55:
        return "strong_bullish"
    if score >= 0.15:
        return "bullish"
    if score > -0.15:
        return "neutral"
    if score > -0.55:
        return "bearish"
    return "strong_bearish"


def run_pipeline(
    headlines: list[str],
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    use_embeddings: bool = True,
) -> list[dict[str, Any]]:
    """Embed, dedup, LLM-score (with fallback), and return per-headline results."""
    results: dict[str, dict[str, Any]] = {}

    keep_idx = dedupe_by_similarity(headlines) if use_embeddings else list(range(len(headlines)))
    unique = [headlines[i] for i in keep_idx]
    log.info("%d headlines -> %d unique after embedding dedup", len(headlines), len(unique))

    llm = None
    try:
        from langchain_core.messages import HumanMessage

        llm = build_chat_model(model=model, base_url=base_url)
        # cheap liveness probe
        llm.invoke([HumanMessage(content="ping")])
    except Exception as exc:
        log.error("Local LLM unavailable (%s) -> falling back to keyword heuristic.", exc)
        llm = None

    for h in unique:
        score: float | None = None
        method = "llm"
        if llm is not None:
            score = score_with_llm(llm, h)
        if score is None:
            score, method = keyword_score(h), "keyword_fallback"
        results[h] = {
            "headline": h,
            "score": round(float(score), 3),
            "label": label_for(score),
            "method": method,
        }

    # map duplicates back onto their representative's score, preserving input order
    out: list[dict[str, Any]] = []
    rep_of = {}
    for slot, orig_i in enumerate(keep_idx):
        rep_of[orig_i] = unique[slot]
    for i, h in enumerate(headlines):
        rep = rep_of.get(i)
        rec = dict(results[rep]) if rep in results else {"headline": h, "score": 0.0,
                                                         "label": "neutral", "method": "dropped"}
        rec["headline"] = h
        out.append(rec)
    return out


DEMO_HEADLINES = [
    "Apple smashes Q3 earnings estimates as iPhone supercycle accelerates",
    "Apple stock falls on weaker-than-expected China demand guidance",
    "Fed holds interest rates steady, signals patience on cuts",
    "Tesla recalls 200,000 vehicles after NHTSA investigation",
    "Nvidia unveils next-gen AI chip, analysts issue price target hikes",
    "Meta faces new EU antitrust lawsuit over ad practices",
    "Microsoft Azure growth beats consensus on enterprise AI adoption",
    "Boeing slips as union strike threatens 737 MAX output",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--headlines", nargs="+", help="Headlines given inline")
    src.add_argument("--file", type=Path, help="File with one headline per line")
    src.add_argument("--stdin", action="store_true", help="Read headlines from stdin")
    src.add_argument("--demo", action="store_true", help="Use built-in demo headlines")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model tag")
    ap.add_argument("--base-url", default=OLLAMA_BASE_URL, help="Ollama server URL")
    ap.add_argument("--no-embed", action="store_true", help="Skip embedding-based dedup")
    ap.add_argument("--out", type=Path, help="Also write JSON results to this file")
    args = ap.parse_args()

    if args.demo:
        headlines = DEMO_HEADLINES
    elif args.headlines:
        headlines = args.headlines
    elif args.file:
        headlines = [ln.strip() for ln in args.file.read_text().splitlines() if ln.strip()]
    else:  # --stdin
        headlines = [ln.strip() for ln in sys.stdin.read().splitlines() if ln.strip()]

    results = run_pipeline(
        headlines,
        model=args.model,
        base_url=args.base_url,
        use_embeddings=not args.no_embed,
    )

    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)
        log.info("Wrote %d scored headlines to %s", len(results), args.out)

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    log.info("Aggregate sentiment: %.3f (%s)", avg, label_for(avg))


if __name__ == "__main__":
    main()
