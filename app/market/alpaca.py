from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any, Awaitable, Callable

import httpx
import pandas as pd
from alpaca.data.enums import DataFeed, MostActivesBy
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import MarketMoversRequest, MostActivesRequest, StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus, OrderSide, TimeInForce
from alpaca.trading.requests import GetAssetsRequest, GetOrdersRequest, MarketOrderRequest

from app.config import Settings
from app.market.features import build_features

logger = logging.getLogger(__name__)


class AlpacaService:
    BASE_DATA = "https://data.alpaca.markets"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.headers = {"APCA-API-KEY-ID": settings.alpaca_api_key, "APCA-API-SECRET-KEY": settings.alpaca_secret_key}
        self.trading = TradingClient(settings.alpaca_api_key, settings.alpaca_secret_key, paper=True)
        self.market = StockHistoricalDataClient(settings.alpaca_api_key, settings.alpaca_secret_key)
        self.screener = ScreenerClient(settings.alpaca_api_key, settings.alpaca_secret_key)
        self.feed = DataFeed.IEX if settings.alpaca_data_feed == "iex" else DataFeed.SIP
        self.stream = StockDataStream(settings.alpaca_api_key, settings.alpaca_secret_key, feed=self.feed,
                                      data_timeout=180)
        self.latest_quotes: dict[str, dict[str, float | None]] = {}
        self.latest_bars: dict[str, dict[str, Any]] = {}
        self.previous_bar: dict[str, dict[str, Any]] = {}
        self.bar_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=500))
        self.assets: dict[str, dict[str, Any]] = {}
        self._stream_task: asyncio.Task | None = None
        self._trade_stream_task: asyncio.Task | None = None
        self._stream_running = False
        self._trade_stream_running = False
        self._on_event: Callable[[str, dict], Awaitable[None]] | None = None
        self._on_trade_event: Callable[[dict], Awaitable[None]] | None = None
        self._subscribed: set[str] = set()
        self.trade_stream = None
        self.http = httpx.AsyncClient(timeout=15, headers={"User-Agent": "StockPaperSim/1.0"})
        self._sp500_symbols: list[str] = []
        self._sp500_loaded_at: float = 0.0
        self._sp500_lock = asyncio.Lock()

    async def close(self):
        await self.stop_stream()
        await self.stop_trade_stream()
        await self.http.aclose()

    async def clock(self) -> dict[str, Any]:
        """Return Alpaca market clock information for session-aware scheduling."""
        clock = await asyncio.to_thread(self.trading.get_clock)
        return clock.model_dump() if hasattr(clock, "model_dump") else dict(clock)

    async def account(self) -> dict[str, Any]:
        account = await asyncio.to_thread(self.trading.get_account)
        return account.model_dump() if hasattr(account, "model_dump") else dict(account)

    async def positions(self) -> list[dict[str, Any]]:
        positions = await asyncio.to_thread(self.trading.get_all_positions)
        return [p.model_dump() if hasattr(p, "model_dump") else dict(p) for p in positions]

    async def open_orders(self) -> list[dict[str, Any]]:
        orders = await asyncio.to_thread(self.trading.get_orders, GetOrdersRequest(status="open", limit=500))
        return [o.model_dump() if hasattr(o, "model_dump") else dict(o) for o in orders]

    async def load_assets(self) -> dict[str, dict[str, Any]]:
        request = GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
        assets = await asyncio.to_thread(self.trading.get_all_assets, request)
        self.assets = {}
        for asset in assets:
            data = asset.model_dump() if hasattr(asset, "model_dump") else dict(asset)
            symbol = str(data.get("symbol", "")).upper()
            if not symbol or not data.get("tradable"):
                continue
            self.assets[symbol] = data
        return self.assets

    async def screeners(self, top: int) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
        """Return Alpaca's dynamic opportunity overlay.

        The S&P 500 core universe is loaded separately from a live external
        constituent source so daily movers can add stocks outside the index.
        """
        top = min(max(int(top), 1), 100)
        # Two independent screener endpoints; run their blocking SDK calls on
        # separate threads concurrently instead of one after the other.
        most, movers = await asyncio.gather(
            asyncio.to_thread(self.screener.get_most_actives, MostActivesRequest(top=top, by=MostActivesBy.VOLUME)),
            asyncio.to_thread(self.screener.get_market_movers, MarketMoversRequest(top=min(top, 50))),
        )
        most_data = most.model_dump() if hasattr(most, "model_dump") else dict(most)
        mover_data = movers.model_dump() if hasattr(movers, "model_dump") else dict(movers)
        return list(most_data.get("most_actives", [])), {"gainers": mover_data.get("gainers", []),
                                                         "losers": mover_data.get("losers", [])}

    async def sp500_symbols(self) -> list[str]:
        """Fetch the current S&P 500 security list; never hard-code constituents.

        Membership is refreshed on a configurable interval. A previously successful
        in-memory result is retained only if a later refresh temporarily fails.
        """
        now = time.time()
        refresh_seconds = self.settings.sp500_refresh_hours * 3600.0
        if self._sp500_symbols and now - self._sp500_loaded_at < refresh_seconds:
            return list(self._sp500_symbols)
        async with self._sp500_lock:
            now = time.time()
            if self._sp500_symbols and now - self._sp500_loaded_at < refresh_seconds:
                return list(self._sp500_symbols)
            response = await self.http.get(self.settings.sp500_source_url)
            response.raise_for_status()
            tables = await asyncio.to_thread(pd.read_html, StringIO(response.text))
            frame = next((t for t in tables if "Symbol" in t.columns), None)
            if frame is None:
                raise RuntimeError("S&P 500 source did not contain a Symbol column")
            symbols = []
            for raw in frame["Symbol"].tolist():
                symbol = str(raw).strip().upper()
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
            if len(symbols) < 450:
                raise RuntimeError(
                    f"S&P 500 source returned only {len(symbols)} symbols; refusing to replace the previous universe")
            tradable = [symbol for symbol in symbols if not self.assets or symbol in self.assets]
            if len(tradable) < 400:
                raise RuntimeError(
                    f"Only {len(tradable)} fetched S&P symbols are currently tradable in Alpaca; refusing to replace the previous universe")
            self._sp500_symbols = tradable
            self._sp500_loaded_at = time.time()
            logger.info("Loaded dynamic S&P 500 universe: %s symbols from %s", len(tradable),
                        self.settings.sp500_source_url)
            return list(self._sp500_symbols)

    async def snapshots(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch multi-symbol Alpaca snapshots in one request per batch."""
        cleaned = []
        for symbol in symbols:
            value = str(symbol).upper().strip()
            if value and value not in cleaned:
                cleaned.append(value)
        if not cleaned:
            return {}
        result: dict[str, dict[str, Any]] = {}
        # Keep a generous per-request batch while allowing room for future API limits.
        for start in range(0, len(cleaned), 500):
            batch = cleaned[start:start + 500]
            req = StockSnapshotRequest(symbol_or_symbols=batch, feed=self.feed)
            payload = await asyncio.to_thread(self.market.get_stock_snapshot, req)
            if hasattr(payload, "model_dump"):
                mapping = payload.model_dump()
            elif isinstance(payload, dict):
                mapping = payload
            else:
                mapping = getattr(payload, "data", {}) or {}
            for symbol, snap in (mapping or {}).items():
                data = snap.model_dump() if hasattr(snap, "model_dump") else dict(snap)
                result[str(symbol).upper()] = data
        return result

    async def bars(self, symbol: str, limit: int | None = None, days: int = 10) -> list[Any]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        req = StockBarsRequest(
            symbol_or_symbols=symbol.upper(), start=start, end=end,
            limit=limit or self.settings.history_bars, timeframe=TimeFrame(1, TimeFrameUnit.Minute), feed=self.feed,
        )
        result = await asyncio.to_thread(self.market.get_stock_bars, req)
        try:
            return list(result[symbol.upper()])
        except Exception:
            try:
                return list(result[symbol])
            except Exception:
                return []

    async def features(self, symbol: str, daily_dollar_volume: float | None = None) -> dict[str, Any]:
        symbol = symbol.upper()
        bars = await self.bars(symbol)
        if daily_dollar_volume is None:
            snap = await self.snapshot(symbol)
            daily_bar = snap.get("daily_bar") or snap.get("dailyBar") or {}
            trade = snap.get("latest_trade") or snap.get("latestTrade") or {}
            try:
                price = float(trade.get("p", trade.get("price")))
                volume = float(daily_bar.get("v", daily_bar.get("volume")))
                daily_dollar_volume = price * volume
            except (TypeError, ValueError):
                daily_dollar_volume = 0.0
        data = build_features(bars, daily_dollar_volume=daily_dollar_volume)
        quote = self.latest_quotes.get(symbol, {})
        if quote:
            data.update({"bid": quote.get("bid"), "ask": quote.get("ask"), "spread_pct": quote.get("spread_pct")})
        return data

    async def snapshot(self, symbol: str) -> dict[str, Any]:
        req = StockSnapshotRequest(symbol_or_symbols=symbol.upper(), feed=self.feed)
        result = await asyncio.to_thread(self.market.get_stock_snapshot, req)
        try:
            snap = result[symbol.upper()]
        except Exception:
            return {}
        return snap.model_dump() if hasattr(snap, "model_dump") else dict(snap)

    async def benchmark(self, symbol: str | None = None) -> dict[str, Any]:
        """Return a live S&P 500 benchmark view; stock symbols still use Alpaca snapshots."""
        symbol = (symbol or self.settings.benchmark_symbol).upper()

        def val(obj: dict[str, Any], *keys: str):
            for key in keys:
                if key in obj and obj[key] is not None:
                    try:
                        return float(obj[key])
                    except (TypeError, ValueError):
                        pass
            return None

        try:
            if symbol in {"^GSPC", "GSPC", "SP500", "S&P500", "S&P 500"}:
                url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
                response = await self.http.get(url, params={"range": "1d", "interval": "1m"},
                                               headers={"User-Agent": "StockPaperSim/1.0"})
                response.raise_for_status()
                payload = response.json().get("chart", {}).get("result", [{}])[0] or {}
                meta = payload.get("meta", {}) or {}
                quote = (payload.get("indicators", {}).get("quote", [{}]) or [{}])[0] or {}
                closes = [float(x) for x in (quote.get("close") or []) if x is not None]
                opens = [float(x) for x in (quote.get("open") or []) if x is not None]
                price = val(meta, "regularMarketPrice") or (closes[-1] if closes else None)
                prev_close = val(meta, "previousClose", "chartPreviousClose")
                session_open = opens[0] if opens else None
                change_pct = ((price - prev_close) / prev_close * 100) if price is not None and prev_close else None
                change = (price - prev_close) if price is not None and prev_close else None
                return {
                    "symbol": "S&P 500",
                    "underlying_symbol": "^GSPC",
                    "price": price,
                    "day_change": change,
                    "day_change_pct": change_pct,
                    "previous_close": prev_close,
                    "session_open": session_open,
                    "direction": "UP" if (
                            change_pct is not None and change_pct >= 0) else "DOWN" if change_pct is not None else "UNKNOWN",
                }

            snap = await self.snapshot(symbol)
            daily = snap.get("daily_bar") or snap.get("dailyBar") or {}
            previous = snap.get("prev_daily_bar") or snap.get("previous_daily_bar") or snap.get("prevDailyBar") or {}
            close = val(daily, "c", "close")
            open_price = val(daily, "o", "open")
            prev_close = val(previous, "c", "close")
            latest_trade = snap.get("latest_trade") or snap.get("latestTrade") or {}
            latest_price = val(latest_trade, "p", "price")
            price = latest_price or close
            if prev_close and price:
                change = price - prev_close
                change_pct = (change / prev_close) * 100
            elif open_price and price:
                change = price - open_price
                change_pct = (change / open_price) * 100
            else:
                change = None
                change_pct = None
            return {
                "symbol": symbol,
                "price": price,
                "day_change": change,
                "day_change_pct": change_pct,
                "previous_close": prev_close,
                "session_open": open_price,
                "direction": "UP" if (
                        change_pct is not None and change_pct >= 0) else "DOWN" if change_pct is not None else "UNKNOWN",
            }
        except Exception as exc:
            logger.warning("Benchmark lookup failed for %s: %s", symbol, exc)
            return {"symbol": "S&P 500" if symbol in {"^GSPC", "GSPC", "SP500", "S&P500", "S&P 500"} else symbol,
                    "underlying_symbol": symbol, "price": None, "day_change": None, "day_change_pct": None,
                    "direction": "UNKNOWN", "error": str(exc)}

    async def news(self, symbol: str, limit: int = 5) -> list[dict[str, Any]]:
        response = await self.http.get(f"{self.BASE_DATA}/v1beta1/news", headers=self.headers,
                                       params={"symbols": symbol.upper(), "limit": limit, "sort": "desc",
                                               "include_content": "false"})
        response.raise_for_status()
        return response.json().get("news", [])

    async def history(self, symbol: str, limit: int = 120) -> list[dict[str, Any]]:
        symbol = symbol.upper()
        if len(self.bar_history[symbol]) >= min(limit, 20):
            return list(self.bar_history[symbol])[-limit:]
        bars = await self.bars(symbol, limit=limit)
        result = []
        for b in bars:
            result.append(
                {"timestamp": str(b.timestamp), "open": float(b.open), "high": float(b.high), "low": float(b.low),
                 "close": float(b.close), "volume": float(b.volume)})
        return result[-limit:]

    async def submit_market_order(self, symbol: str, side: str, qty: float, client_order_id: str):
        order = MarketOrderRequest(
            symbol=symbol.upper(), qty=qty, side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY, client_order_id=client_order_id,
        )
        return await asyncio.to_thread(self.trading.submit_order, order)

    async def start_stream(self, on_event: Callable[[str, dict], Awaitable[None]]) -> None:
        self._on_event = on_event
        if self._stream_running: return
        self._stream_running = True
        self._stream_task = asyncio.create_task(asyncio.to_thread(self._run_stream))

    def _run_stream(self):
        try:
            self.stream.run()
        except Exception:
            self._stream_running = False
            logger.exception("Alpaca market stream stopped")

    async def start_trade_stream(self, on_event: Callable[[dict], Awaitable[None]]) -> None:
        if self._trade_stream_running: return
        from alpaca.trading.stream import TradingStream
        self._on_trade_event = on_event
        self.trade_stream = TradingStream(self.settings.alpaca_api_key, self.settings.alpaca_secret_key, paper=True)
        self.trade_stream.subscribe_trade_updates(self._trade_update_handler)
        self._trade_stream_running = True
        self._trade_stream_task = asyncio.create_task(asyncio.to_thread(self.trade_stream.run))

    async def _trade_update_handler(self, update) -> None:
        data = update.model_dump() if hasattr(update, "model_dump") else getattr(update, "__dict__", {})
        if self._on_trade_event: await self._on_trade_event(data)

    async def stop_trade_stream(self):
        if not self._trade_stream_running: return
        self._trade_stream_running = False
        try:
            await self.trade_stream.stop_ws()
        except Exception:
            logger.exception("Failed stopping trading stream")
        if self._trade_stream_task:
            try:
                await asyncio.wait_for(self._trade_stream_task, timeout=10)
            except Exception:
                self._trade_stream_task.cancel()

    async def _quote_handler(self, quote) -> None:
        symbol = quote.symbol.upper()
        bid = float(quote.bid_price or 0);
        ask = float(quote.ask_price or 0)
        mid = (bid + ask) / 2 if bid and ask else None
        spread_pct = ((ask - bid) / mid * 100) if mid else None
        self.latest_quotes[symbol] = {"bid": bid, "ask": ask, "mid": mid, "spread_pct": spread_pct}
        if self._on_event:
            await self._on_event("quote",
                                 {"symbol": symbol, "bid": bid, "ask": ask, "mid": mid, "spread_pct": spread_pct})

    async def _bar_handler(self, bar) -> None:
        symbol = bar.symbol.upper()
        payload = {"symbol": symbol, "open": float(bar.open), "high": float(bar.high), "low": float(bar.low),
                   "close": float(bar.close), "volume": float(bar.volume), "timestamp": str(bar.timestamp)}
        self.previous_bar[symbol] = self.latest_bars.get(symbol, {})
        self.latest_bars[symbol] = payload
        self.bar_history[symbol].append(payload)
        if self._on_event: await self._on_event("bar", payload)

    async def update_subscriptions(self, symbols: list[str]) -> None:
        target = {s.upper() for s in symbols if s}
        add = sorted(target - self._subscribed);
        remove = sorted(self._subscribed - target)
        if add:
            try:
                self.stream.subscribe_quotes(self._quote_handler, *add)
                self.stream.subscribe_bars(self._bar_handler, *add)
            except Exception:
                logger.exception("Failed to subscribe %s", add)
        if remove:
            try:
                self.stream.unsubscribe_quotes(*remove);
                self.stream.unsubscribe_bars(*remove)
            except Exception:
                logger.exception("Failed to unsubscribe %s", remove)
        self._subscribed = target

    async def stop_stream(self):
        if not self._stream_running: return
        self._stream_running = False
        try:
            await self.stream.stop_ws()
        except Exception:
            logger.exception("Failed stopping market stream")
        if self._stream_task:
            try:
                await asyncio.wait_for(self._stream_task, timeout=10)
            except Exception:
                self._stream_task.cancel()
