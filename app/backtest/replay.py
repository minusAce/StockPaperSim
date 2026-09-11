from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from app.config import Settings
from app.market.alpaca import AlpacaService
from app.schemas import BacktestRequest, BacktestResult


async def run_moving_average_backtest(alpaca: AlpacaService, settings: Settings,
                                      request: BacktestRequest) -> BacktestResult:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=request.days)
    bars_req = StockBarsRequest(symbol_or_symbols=request.symbol.upper(), start=start, end=end,
                                timeframe=TimeFrame(5, TimeFrameUnit.Minute), feed=alpaca.feed)
    result = await asyncio.to_thread(alpaca.market.get_stock_bars, bars_req)
    bars = list(result[request.symbol.upper()]) if request.symbol.upper() in result else []
    closes = np.array([float(b.close) for b in bars], dtype=float)
    if len(closes) < max(request.fast_window, request.slow_window) + 5:
        return BacktestResult(symbol=request.symbol.upper(), days=request.days, initial_cash=request.initial_cash,
                              final_equity=request.initial_cash, total_return_pct=0, max_drawdown_pct=0, trades=0,
                              win_rate_pct=0, strategy="SMA crossover",
                              notes=["Insufficient historical bars for the requested window."])
    cash = float(request.initial_cash);
    shares = 0.0;
    equity_curve = [];
    trade_pnls = [];
    entry = 0.0;
    trades = 0
    for i in range(request.slow_window, len(closes)):
        fast = closes[i - request.fast_window:i].mean();
        slow = closes[i - request.slow_window:i].mean();
        price = closes[i]
        if shares == 0 and fast > slow:
            shares = cash / price;
            cash = 0.0;
            entry = price;
            trades += 1
        elif shares > 0 and fast < slow:
            cash = shares * price;
            trade_pnls.append(price - entry);
            shares = 0.0
        equity_curve.append(cash + shares * price)
    if shares > 0:
        final = shares * closes[-1];
        trade_pnls.append(closes[-1] - entry);
        cash = final;
        shares = 0
    final_equity = float(cash)
    curve = np.array(equity_curve + [final_equity], dtype=float)
    peak = np.maximum.accumulate(curve);
    drawdown = (curve / peak - 1) if len(curve) else np.array([0])
    wins = sum(1 for x in trade_pnls if x > 0)
    return BacktestResult(symbol=request.symbol.upper(), days=request.days, initial_cash=request.initial_cash,
                          final_equity=final_equity, total_return_pct=(final_equity / request.initial_cash - 1) * 100,
                          max_drawdown_pct=float(drawdown.min() * 100), trades=trades,
                          win_rate_pct=(wins / len(trade_pnls) * 100 if trade_pnls else 0), strategy="SMA crossover",
                          notes=["This is a deterministic research baseline, not an LLM backtest.",
                                 "Use it to sanity-check market behavior before trusting autonomous paper decisions."])
