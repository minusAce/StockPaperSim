from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Candidate(StrictModel):
    symbol: str
    score: float
    reason: str
    last_price: float | None = None
    change_pct: float | None = None
    volume: float | None = None
    daily_dollar_volume: float | None = None
    dollar_volume: float | None = None
    source: str


class QuantReport(StrictModel):
    symbol: str
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0, le=1)
    momentum_score: float = Field(ge=-1, le=1)
    volatility_score: float = Field(ge=-1, le=1)
    liquidity_score: float = Field(ge=0, le=1)
    summary: str
    risks: list[str]
    data_quality: Literal["GOOD", "LIMITED", "INSUFFICIENT"]


class MacroReport(StrictModel):
    regime: Literal["RISK_ON", "NEUTRAL", "RISK_OFF"]
    score: float = Field(ge=-1, le=1)
    summary: str
    key_drivers: list[str]


class NewsReport(StrictModel):
    symbol: str
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "INSUFFICIENT"]
    catalyst_score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    summary: str
    catalysts: list[str]
    sources: list[str]
    data_quality: Literal["GOOD", "LIMITED", "INSUFFICIENT"]


class FinanceReport(StrictModel):
    symbol: str
    stance: Literal["ATTRACTIVE", "NEUTRAL", "UNATTRACTIVE", "INSUFFICIENT"]
    score: float = Field(ge=-1, le=1)
    confidence: float = Field(ge=0, le=1)
    summary: str
    positives: list[str]
    concerns: list[str]
    data_quality: Literal["GOOD", "LIMITED", "INSUFFICIENT"]


class ScoutChoice(StrictModel):
    symbol: str
    score: float = Field(ge=0, le=100)
    reason: str


class ScoutResponse(StrictModel):
    selected: list[ScoutChoice]
    summary: str


class PMProposal(StrictModel):
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0, le=1)
    target_weight: float = Field(ge=0, le=1)
    urgency: Literal["LOW", "MEDIUM", "HIGH"]
    thesis: str
    invalidation: str
    max_price_slippage_pct: float = Field(ge=0, le=0.05)


class PMDecision(StrictModel):
    environment: Literal["TRADE", "NO_TRADE"]
    summary: str
    proposals: list[PMProposal]


class RiskReview(StrictModel):
    symbol: str
    action: Literal["BUY", "SELL", "HOLD"]
    llm_risk: Literal["LOW", "MEDIUM", "HIGH", "REJECT"]
    confidence: float = Field(ge=0, le=1)
    reason: str


class RiskBatch(StrictModel):
    reviews: list[RiskReview]


class ExecutionResult(StrictModel):
    ok: bool
    symbol: str
    action: str
    qty: float
    order_id: str | None
    status: str | None
    error: str | None
    timestamp: datetime = Field(default_factory=now_utc)


class ControlPayload(StrictModel):
    enabled: bool | None = None
    active: bool | None = None
    paused: bool | None = None


class BacktestRequest(StrictModel):
    symbol: str = Field(min_length=1, max_length=12)
    days: int = Field(default=30, ge=1, le=365)
    initial_cash: float = Field(default=100_000, ge=1000)
    fast_window: int = Field(default=20, ge=5, le=100)
    slow_window: int = Field(default=50, ge=10, le=200)

    @model_validator(mode="after")
    def _check_window_order(self) -> "BacktestRequest":
        # Each is individually valid per its own ge/le bounds, but the crossover
        # strategy assumes fast_window < slow_window. Without this check a
        # request like fast_window=100, slow_window=10 slips through and the
        # backtest loop's closes[i-fast_window:i] slice goes negative for the
        # first several bars, silently producing NaN moving averages (and thus
        # a degenerate, trade-free backtest) instead of a clear error.
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        return self


class BacktestResult(StrictModel):
    symbol: str
    days: int
    initial_cash: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: int
    win_rate_pct: float
    strategy: str
    notes: list[str]
