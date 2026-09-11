from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def normalize_model_id(value: str) -> str:
    """Remove obsolete structured-output suffixes from configured model IDs."""
    model = (value or "").strip()
    while model.endswith(":structured"):
        model = model[: -len(":structured")].rstrip(":")
    return model


def is_free_model_id(value: str) -> bool:
    model = normalize_model_id(value)
    return model == "openrouter/free" or model.endswith(":free")


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    paper_trading: bool = env_bool("PAPER_TRADING", True)
    alpaca_data_feed: str = os.getenv("ALPACA_DATA_FEED", "iex").lower()

    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model_default: str = os.getenv("MODEL_DEFAULT", "nvidia/nemotron-3-ultra-550b-a55b:free")
    # Single explicit fallback model, tried only after the primary (per-agent
    # override, or else MODEL_DEFAULT) has exhausted its own retry
    # attempts. If MODEL_FALLBACK is left unset it mirrors
    # MODEL_DEFAULT, so agents without an override have no distinct
    # fallback (same as before), while agents WITH an override automatically
    # gain the global default as a safety net. Set it explicitly to any other
    # free OpenRouter model ID to enable real model-level fallback everywhere.
    model_fallback: str = os.getenv("MODEL_FALLBACK", os.getenv("MODEL_DEFAULT", "nvidia/nemotron-3-ultra-550b-a55b:free"))
    llm_daily_request_budget: int = env_int("LLM_DAILY_REQUEST_BUDGET", 50)
    llm_min_request_interval_seconds: float = env_float("LLM_MIN_REQUEST_INTERVAL_SECONDS", 3.0)
    site_url: str = os.getenv("SITE_URL", "http://localhost:8000")
    site_name: str = os.getenv("SITE_NAME", "StockPaperSim")

    model_scout: str = os.getenv("MODEL_SCOUT", "")
    model_quant: str = os.getenv("MODEL_QUANT", "")
    model_macro: str = os.getenv("MODEL_MACRO", "")
    model_news: str = os.getenv("MODEL_NEWS", "")
    model_finance: str = os.getenv("MODEL_FINANCE", "")
    model_risk: str = os.getenv("MODEL_RISK", "")
    model_pm: str = os.getenv("MODEL_PM", "")

    trading_enabled: bool = env_bool("TRADING_ENABLED", True)
    start_autopilot: bool = env_bool("START_AUTOPILOT", False)
    reset_runtime_controls_on_start: bool = env_bool("RESET_RUNTIME_CONTROLS_ON_START", True)
    benchmark_symbol: str = os.getenv("BENCHMARK_SYMBOL", "^GSPC").upper()
    market_hours_only: bool = env_bool("MARKET_HOURS_ONLY", True)

    scanner_interval_seconds: int = env_int("SCANNER_INTERVAL_SECONDS", 60)
    analysis_cooldown_seconds: int = env_int("ANALYSIS_COOLDOWN_SECONDS", 0)
    analysis_candidates: int = env_int("ANALYSIS_CANDIDATES", 2)
    ai_cycles_per_session: int = env_int("AI_CYCLES_PER_SESSION", 5)
    activity_target_entries: int = env_int("ACTIVITY_TARGET_ENTRIES", 4)
    scanner_top: int = env_int("SCANNER_TOP", 80)
    sp500_source_url: str = os.getenv("SP500_SOURCE_URL", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp500_refresh_hours: float = env_float("SP500_REFRESH_HOURS", 24.0)
    stream_symbol_limit: int = env_int("STREAM_SYMBOL_LIMIT", 30)
    event_trigger_pct: float = env_float("EVENT_TRIGGER_PCT", 1.25)
    event_trigger_volume_ratio: float = env_float("EVENT_TRIGGER_VOLUME_RATIO", 3.0)
    min_scan_score: float = env_float("MIN_SCAN_SCORE", 4.0)

    max_position_weight: float = env_float("MAX_POSITION_WEIGHT", 0.10)
    max_order_notional: float = env_float("MAX_ORDER_NOTIONAL", 5000.0)
    max_daily_loss_pct: float = env_float("MAX_DAILY_LOSS_PCT", 0.03)
    max_total_open_orders: int = env_int("MAX_TOTAL_OPEN_ORDERS", 10)
    max_symbol_orders_per_hour: int = env_int("MAX_SYMBOL_ORDERS_PER_HOUR", 3)
    min_confidence: float = env_float("MIN_CONFIDENCE", 0.65)
    min_liquidity_dollars: float = env_float("MIN_LIQUIDITY_DOLLARS", 250_000.0)
    min_order_notional: float = env_float("MIN_ORDER_NOTIONAL", 25.0)
    min_order_qty: float = env_float("MIN_ORDER_QTY", 0.01)

    prompt_dir: Path = Path(os.getenv("PROMPT_DIR", str(ROOT / "config" / "agents")))
    database_url: str = os.getenv("DATABASE_URL", "postgresql+psycopg://trading:trading@localhost:5432/trading_floor")
    history_bars: int = env_int("HISTORY_BARS", 180)
    chart_bars: int = env_int("CHART_BARS", 120)

    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = env_int("PORT", 8000)
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    frontend_dist: Path = ROOT / "frontend" / "dist"

    def validate(self) -> None:
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required.")
        if not self.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required.")
        if not self.paper_trading:
            raise RuntimeError("This build is paper-only. Set PAPER_TRADING=true.")
        if self.alpaca_data_feed not in {"iex", "sip"}:
            raise RuntimeError("ALPACA_DATA_FEED must be iex or sip.")
        if not self.model_default:
            raise RuntimeError("MODEL_DEFAULT must not be empty.")
        if not is_free_model_id(self.model_default):
            raise RuntimeError(
                "MODEL_DEFAULT must be a free OpenRouter model ID (use :free or openrouter/free)."
            )
        if self.model_fallback and not is_free_model_id(self.model_fallback):
            raise RuntimeError(
                "MODEL_FALLBACK must be a free OpenRouter model ID (use :free or openrouter/free)."
            )
        overrides = {
            "SCOUT": self.model_scout, "QUANT": self.model_quant, "MACRO": self.model_macro,
            "NEWS": self.model_news, "FINANCE": self.model_finance,
            "RISK": self.model_risk, "PM": self.model_pm,
        }
        invalid_overrides = {name: model for name, model in overrides.items() if model and not is_free_model_id(model)}
        if invalid_overrides:
            raise RuntimeError(
                "Per-agent model overrides must be free OpenRouter model IDs: "
                f"{invalid_overrides}"
            )
        if self.analysis_candidates < 1:
            raise RuntimeError("ANALYSIS_CANDIDATES must be at least 1.")
        if self.ai_cycles_per_session < 1:
            raise RuntimeError("AI_CYCLES_PER_SESSION must be at least 1.")
        if self.sp500_refresh_hours <= 0:
            raise RuntimeError("SP500_REFRESH_HOURS must be greater than 0.")
        if self.activity_target_entries < 0:
            raise RuntimeError("ACTIVITY_TARGET_ENTRIES must be non-negative.")
        if not (0 < self.max_position_weight <= 1):
            raise RuntimeError("MAX_POSITION_WEIGHT must be between 0 and 1.")
        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite"):
            (self.root_dir / "data").mkdir(parents=True, exist_ok=True)


settings = Settings()
