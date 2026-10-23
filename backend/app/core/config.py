"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="FINLEARN_", extra="ignore")

    db_path: Path = BACKEND_ROOT.parent / "data" / "finlearn.db"
    risk_free_rate: float = 0.045
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    embedder_model: str = "all-MiniLM-L6-v2"


settings = Settings()
