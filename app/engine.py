from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from app.agents.llm import OpenRouterQuotaExceeded
from app.agents.team import AgentTeam
from app.config import Settings
from app.event_bus import EventBus
from app.market.alpaca import AlpacaService
from app.storage.db import Database
from app.trading.execution import ExecutionEngine
from app.trading.risk import RiskEngine

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, settings: Settings, alpaca: AlpacaService, team: AgentTeam, risk: RiskEngine,
                 execution: ExecutionEngine, db: Database, bus: EventBus):
        self.settings, self.alpaca, self.team, self.risk, self.execution, self.db, self.bus = settings, alpaca, team, risk, execution, db, bus
        self.running = False
        self.autopilot = db.get_control("autopilot", settings.start_autopilot)
        self._task = None;
        self._cycle_lock = asyncio.Lock();
        self._event_trigger_task = None;
        self._analysis_task = None
        self.last_scan = 0.0;
        self.last_cycle = 0.0
        self._last_portfolio_refresh = 0.0;
        self._last_benchmark_refresh = 0.0;
        self._last_market_monitor_refresh = 0.0
        self._market_close_shutdown_task = None
        self._display_refresh_task = None
        self.candidates: list[dict[str, Any]] = []
        self.recent_symbols: dict[str, float] = {}
        self.portfolio: dict[str, Any] = {}
        self.market_event_count = 0
        self.last_market_event: dict[str, Any] = {}
        self.cycle_count = 0
        self._ai_slot_index = int(self.db.get_control_value("ai_schedule_slot", "0") or 0)
        self._ai_schedule_day = self.db.get_control_value("ai_schedule_day", "")
        self._pending_ai_events: dict[str, dict[str, Any]] = {}

    async def start(self):
        self.running = True
        self.team.set_enabled(self.autopilot)
        await self._refresh_ai_schedule_state()
        await self._refresh_portfolio()
        try:
            await self.alpaca.load_assets()
        except Exception as exc:
            logger.warning("Asset universe load failed: %s", exc)
        # Read-only market data initializes independently of autopilot so Market Monitor
        # and the live tape are populated immediately after launch, even while trading is paused.
        await self.scan()
        await self._refresh_benchmark_display()
        await self.alpaca.start_stream(self._on_market_event)
        if self.autopilot:
            await self.alpaca.start_trade_stream(self._on_trade_update)
        self._task = asyncio.create_task(self._loop())
        self._display_refresh_task = asyncio.create_task(self._display_refresh_loop())
        await self._publish("system", {"level": "INFO",
                                       "message": f"Trading floor online — paper mode — autopilot={'ON' if self.autopilot else 'OFF'}"})

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._event_trigger_task: self._event_trigger_task.cancel()
        if self._display_refresh_task:
            self._display_refresh_task.cancel()
            try:
                await self._display_refresh_task
            except asyncio.CancelledError:
                pass
        if self._analysis_task and not self._analysis_task.done(): self._analysis_task.cancel()
        self.team.set_enabled(False)
        await self.alpaca.close()

    async def _loop(self):
        while self.running:
            try:
                now = time.time()
                await self._check_market_close()
                if now - self._last_portfolio_refresh >= 10.0:
                    await self._refresh_portfolio()
                    self._last_portfolio_refresh = now
                if not self.autopilot:
                    await asyncio.sleep(2)
                    continue
                if now - self.last_scan >= self.settings.scanner_interval_seconds:
                    await self.scan();
                    self.last_scan = now
                if (
                        self.autopilot
                        and self.settings.trading_enabled
                        and not self.risk.manual_pause
                        and self.candidates
                        and await self._ai_slot_due()
                ):
                    self._analysis_task = asyncio.create_task(self.analysis_cycle("scheduled"))
                    try:
                        await self._analysis_task
                    except asyncio.CancelledError:
                        if self.running:
                            pass
                    finally:
                        self._analysis_task = None
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Engine loop failed")
                await self._publish("system", {"level": "ERROR", "message": str(exc)})
                await asyncio.sleep(10)

    async def _display_refresh_loop(self) -> None:
        """Refresh all visible market-data widgets together every 60 seconds.

        This loop is deliberately independent of scanning and AI analysis. The Market
        Monitor/ticker candidate data and the S&P 500 benchmark share the same refresh
        cycle so the UI presents a synchronized market snapshot.
        """
        refresh_interval = 60.0
        while self.running:
            try:
                await asyncio.sleep(2)
                if self.risk.kill_switch:
                    continue
                now = time.time()
                if now - self._last_market_monitor_refresh >= refresh_interval:
                    # Keep candidate/ticker data and benchmark on one independent cycle.
                    await asyncio.gather(
                        self._refresh_market_monitor(),
                        self._refresh_benchmark_display(),
                        return_exceptions=True,
                    )
                    refreshed_at = time.time()
                    self._last_market_monitor_refresh = refreshed_at
                    self._last_benchmark_refresh = refreshed_at
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Display refresh loop failed: %s", exc)

    async def _refresh_benchmark_display(self) -> None:
        try:
            benchmark = await self.alpaca.benchmark(self.settings.benchmark_symbol)
            await self._publish("benchmark", benchmark)
        except Exception as exc:
            logger.debug("Benchmark realtime update failed: %s", exc)

    async def _check_market_close(self) -> None:
        """Activate the kill switch and stop the Docker stack after regular close.

        Only the regular U.S. equity session is considered. Before the open, the
        simulator remains available; once the session has ended on a weekday, the
        same kill-switch path used by the UI is invoked.

        Gated by settings.market_hours_only. Set MARKET_HOURS_ONLY=false (e.g. for
        local development) to disable this automatic shutdown entirely, regardless
        of the current time of day.
        """
        if not self.settings.market_hours_only:
            return
        if self.risk.kill_switch or self._market_close_shutdown_task:
            return
        try:
            from zoneinfo import ZoneInfo
            from datetime import time as dt_time
            now_et = datetime.now(ZoneInfo("America/New_York"))
            if now_et.weekday() >= 5 or now_et.time() < dt_time(16, 0):
                return
            self.risk.activate_kill_switch()
            await self.set_autopilot(False)
            await self._publish("system", {
                "level": "WARN",
                "message": "REGULAR MARKET CLOSED — KILL SWITCH ACTIVE — shutting down StockPaperSim stack",
            })
            from app.shutdown import shutdown_docker_stack
            async def shutdown():
                await asyncio.sleep(0.35)
                await self.stop()
                await shutdown_docker_stack()

            self._market_close_shutdown_task = asyncio.create_task(shutdown())
        except Exception as exc:
            logger.exception("Automatic market-close shutdown failed: %s", exc)

    async def set_autopilot(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self.autopilot:
            return
        self.autopilot = enabled
        self.db.set_control("autopilot", enabled)
        self.team.set_enabled(enabled)
        if not enabled:
            if self._event_trigger_task and not self._event_trigger_task.done():
                self._event_trigger_task.cancel()
            if self._analysis_task and not self._analysis_task.done():
                self._analysis_task.cancel()
            # Keep the read-only market stream alive while autopilot is paused;
            # only broker trade updates stop with automated trading.
            await self.alpaca.stop_trade_stream()
        else:
            if not self.alpaca._stream_running:
                await self.alpaca.start_stream(self._on_market_event)
            await self.alpaca.start_trade_stream(self._on_trade_update)
            await self.scan()
        state = "ON" if enabled else "OFF"
        activity = "resumed" if enabled else "stopped"
        await self._publish("system",
                            {"level": "INFO", "message": f"Autopilot {state} — automated API/AI activity {activity}"})

    @staticmethod
    def _snapshot_metrics(snapshot: dict[str, Any]) -> dict[str, float | None]:
        def val(obj: Any, *keys: str):
            if not isinstance(obj, dict):
                return None
            for key in keys:
                value = obj.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        pass
            return None

        daily = snapshot.get("daily_bar") or snapshot.get("dailyBar") or {}
        previous = snapshot.get("prev_daily_bar") or snapshot.get("previous_daily_bar") or snapshot.get(
            "prevDailyBar") or {}
        latest_trade = snapshot.get("latest_trade") or snapshot.get("latestTrade") or {}
        close = val(daily, "c", "close")
        prev_close = val(previous, "c", "close")
        price = val(latest_trade, "p", "price") or close
        volume = val(daily, "v", "volume")
        change_pct = ((price - prev_close) / prev_close * 100) if price is not None and prev_close else None
        daily_dollar_volume = (price * volume) if price is not None and volume is not None else None
        return {
            "last_price": price,
            "change_pct": change_pct,
            "volume": volume,
            "daily_dollar_volume": daily_dollar_volume,
            "dollar_volume": daily_dollar_volume,
        }

    async def scan(self):
        try:
            sp500 = await self.alpaca.sp500_symbols()
        except Exception as exc:
            # A temporary constituent-source outage must not disable the dynamic
            # Alpaca discovery overlay or current-position monitoring.
            logger.warning("S&P 500 universe refresh unavailable; continuing with Alpaca overlay: %s", exc)
            await self._publish("system", {"level": "WARN",
                                           "message": f"S&P 500 universe refresh unavailable; using Alpaca discovery overlay: {exc}"})
            sp500 = []
        try:
            (active, movers), positions = await asyncio.gather(
                self.alpaca.screeners(self.settings.scanner_top), self.alpaca.positions()
            )
        except Exception as exc:
            await self._publish("system", {"level": "ERROR", "message": f"Scanner failed: {exc}"});
            return

        # Stable core: dynamically fetched S&P 500 constituents. Discovery overlay:
        # Alpaca's live most-active/gainer/loser lists can introduce stocks outside the index.
        core = {symbol.upper() for symbol in sp500 if symbol}
        overlay: dict[str, dict[str, Any]] = {}

        for row in active:
            symbol = str(row.get("symbol", "")).upper()
            if not symbol:
                continue
            overlay[symbol] = {
                "source": "alpaca_most_active",
                "most_active": True,
                "most_active_volume": _num(row.get("volume")),
                "most_active_trades": _num(row.get("trade_count")),
            }

        for side in ("gainers", "losers"):
            for row in movers.get(side, []):
                symbol = str(row.get("symbol", "")).upper()
                if not symbol:
                    continue
                entry = overlay.setdefault(symbol, {})
                change = _num(row.get("percent_change"))
                entry.update({
                    "source": entry.get("source") or "alpaca_mover",
                    "mover_side": side[:-1],
                    "mover_change_pct": change,
                    "mover_price": _num(row.get("price")),
                    "mover_volume": _num(row.get("volume")),
                })

        candidates: dict[str, dict[str, Any]] = {}
        for symbol in core:
            candidates[symbol] = {
                "symbol": symbol,
                "source": "sp500",
                "universe": "S&P 500",
                "score": 5.0,
                "reason": "Current S&P 500 constituent",
            }

        for symbol, extra in overlay.items():
            candidate = candidates.setdefault(symbol, {
                "symbol": symbol,
                "source": "alpaca_overlay",
                "universe": "Alpaca opportunity overlay",
                "score": 0.0,
                "reason": "Alpaca dynamic opportunity",
            })
            sources = []
            if symbol in core:
                candidate["source"] = "sp500+alpaca_overlay"
            if extra.get("most_active"):
                volume = _num(extra.get("most_active_volume")) or 0.0
                active_score = min((volume / 10_000_000.0) * 2.5, 7.5)
                candidate["score"] += active_score
                sources.append("most active")
            mover_change = _num(extra.get("mover_change_pct"))
            if mover_change is not None:
                candidate["change_pct"] = mover_change
                candidate["score"] += min(abs(mover_change) * 1.8, 24.0)
                sources.append(f"{extra.get('mover_side', 'mover')} {mover_change:.2f}%")
            if sources:
                candidate["reason"] += "; " + ", ".join(sources)

        for p in positions:
            symbol = str(p.get("symbol", "")).upper()
            if not symbol:
                continue
            candidate = candidates.setdefault(symbol, {
                "symbol": symbol,
                "source": "position",
                "universe": "Current portfolio",
                "score": 8.0,
                "reason": "Current portfolio position",
            })
            candidate["portfolio_position"] = True
            candidate["score"] += 3.0

        # One multi-symbol snapshot pass gives the scanner current price, daily volume,
        # and dollar volume for the entire core + overlay universe. This is the point
        # where the liquidity gate happens, before expensive historical feature calls.
        snapshots = await self.alpaca.snapshots(list(candidates))
        ranked: list[dict[str, Any]] = []
        for candidate in candidates.values():
            symbol = candidate["symbol"]
            metrics = self._snapshot_metrics(snapshots.get(symbol, {}))
            candidate.update({k: v for k, v in metrics.items() if v is not None})
            candidate["change_pct"] = _num(candidate.get("change_pct"))
            candidate["daily_dollar_volume"] = _num(candidate.get("daily_dollar_volume"))
            candidate["dollar_volume"] = candidate["daily_dollar_volume"]
            candidate["volume"] = _num(candidate.get("volume"))
            candidate["last_price"] = _num(candidate.get("last_price"))

            daily_dollar_volume = float(candidate.get("daily_dollar_volume") or 0.0)
            if daily_dollar_volume > 0:
                liquidity_multiple = daily_dollar_volume / max(self.settings.min_liquidity_dollars, 1.0)
                candidate["score"] += min(max(liquidity_multiple, 0.0) ** 0.25 * 4.0, 10.0)
            candidate["researchable"] = bool(
                candidate.get("score", 0) >= self.settings.min_scan_score
                and daily_dollar_volume >= self.settings.min_liquidity_dollars
                and candidate.get("last_price") is not None
            )
            if candidate["researchable"]:
                ranked.append(candidate)

        ranked.sort(key=lambda x: (float(x.get("score") or 0), abs(float(x.get("change_pct") or 0)),
                                   float(x.get("daily_dollar_volume") or 0)), reverse=True)

        # Only the deterministic shortlist gets 180-minute feature history work.
        feature_limit = max(self.settings.stream_symbol_limit, self.settings.analysis_candidates * 8, 30)
        enriched: list[dict[str, Any]] = []
        for candidate in ranked[:feature_limit]:
            try:
                f = await self.alpaca.features(candidate["symbol"], candidate.get("daily_dollar_volume"))
                # Keep the live Alpaca snapshot price already captured above.
                # Feature bars may be delayed/historical and must not overwrite the
                # price shown in the market monitor or ticker.
                candidate["features"] = f
            except Exception as exc:
                logger.debug("Feature enrichment failed for %s: %s", candidate["symbol"], exc)
            enriched.append(candidate)

        self.candidates = enriched
        stream_limit = self.settings.stream_symbol_limit
        if self.settings.alpaca_data_feed == "iex":
            # Alpaca Basic currently permits 30 equity websocket symbols. Clamp
            # regardless of .env so a stale setting cannot break the market stream.
            stream_limit = min(stream_limit, 30)
            if self.settings.stream_symbol_limit > 30:
                logger.warning(
                    "STREAM_SYMBOL_LIMIT=%s exceeds Alpaca Basic IEX limit; clamping live subscriptions to 30",
                    self.settings.stream_symbol_limit)
        await self.alpaca.update_subscriptions([x["symbol"] for x in enriched[:stream_limit]])
        self.last_scan = time.time()
        await self._publish("candidates", self.candidates[:50])

    async def _refresh_market_monitor(self) -> None:
        """Refresh displayed candidate market metrics without rerunning the scanner.

        The scanner determines the ranked candidate set. This lightweight pass keeps
        the visible Market Monitor/ticker prices and daily metrics live every 60 seconds
        without waiting for feature enrichment or a new scanner ranking.
        """
        if not self.candidates:
            return
        symbols = [str(x.get("symbol", "")).upper() for x in self.candidates[:50] if x.get("symbol")]
        try:
            snapshots = await self.alpaca.snapshots(symbols)
            changed = False
            for candidate in self.candidates:
                symbol = str(candidate.get("symbol", "")).upper()
                snapshot = snapshots.get(symbol, {})
                metrics = self._snapshot_metrics(snapshot)
                for key in ("last_price", "change_pct", "volume", "dollar_volume"):
                    value = metrics.get(key)
                    if value is not None:
                        candidate[key] = value
                        changed = True
            if changed:
                await self._publish("candidates", self.candidates[:50])
        except Exception as exc:
            logger.debug("Market monitor refresh failed: %s", exc)

    async def _refresh_ai_schedule_state(self) -> None:
        day = datetime.now(timezone.utc).date().isoformat()
        if self._ai_schedule_day != day:
            self._ai_schedule_day = day
            self._ai_slot_index = 0
            self.db.set_control_value("ai_schedule_day", day)
            self.db.set_control_value("ai_schedule_slot", "0")

    async def _ai_slot_times(self) -> list[float]:
        """Return evenly distributed AI capacity windows across today's regular session."""
        await self._refresh_ai_schedule_state()
        try:
            from zoneinfo import ZoneInfo

            clock = await self.alpaca.clock()
            if not clock.get("next_close"):
                return []

            def ts(value: Any) -> float:
                if isinstance(value, (int, float)):
                    return float(value)
                text = str(value).replace("Z", "+00:00")
                return datetime.fromisoformat(text).timestamp()

            close_ts = ts(clock["next_close"])
            if clock.get("is_open"):
                # Alpaca's live clock timestamp is not the session open. On an open
                # regular session, derive 09:30 ET from the current trading date.
                now_et = datetime.now(ZoneInfo("America/New_York"))
                open_et = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
                start_ts = open_et.timestamp()
            else:
                start_ts = ts(clock.get("next_open")) if clock.get("next_open") else 0.0
                if not start_ts:
                    return []

            if start_ts >= close_ts:
                return []

            span = close_ts - start_ts
            # Spread the configured AI waves across the session with 10-minute edge buffers.
            # With the default of five waves, each wave is a ten-request AI capacity window.
            edge = min(600.0, span * 0.04)
            first = start_ts + edge
            last = close_ts - edge
            cycles = max(1, self.settings.ai_cycles_per_session)
            step = (last - first) / (cycles - 1) if cycles > 1 and last > first else 0.0
            return [first + i * step for i in range(cycles) if start_ts <= first + i * step < close_ts]
        except Exception as exc:
            logger.warning("AI schedule clock lookup failed: %s", exc)
            return []

    async def _ai_slot_due(self) -> bool:
        if self._ai_slot_index >= self.settings.ai_cycles_per_session:
            return False
        slots = await self._ai_slot_times()
        if self._ai_slot_index >= len(slots):
            return False
        return time.time() >= slots[self._ai_slot_index]

    def _consume_ai_slot(self) -> None:
        self._ai_slot_index += 1
        self.db.set_control_value("ai_schedule_slot", str(self._ai_slot_index))

    async def analysis_cycle(self, trigger: str = "scheduled", execute_trades: bool = True):
        if self._cycle_lock.locked(): return
        async with self._cycle_lock:
            if (not self.autopilot and not trigger.startswith(
                    "manual-test")) or self.risk.manual_pause or not self.candidates: return
            if trigger.startswith("scheduled") and not await self._ai_slot_due():
                return
            # Each AI wave is deliberately sized to 10 requests: SCOUT + MACRO +
            # (QUANT/NEWS/FINANCE x 2) + PM + RISK. Five waves therefore provide
            # exactly 50 request opportunities across the regular session.
            # OpenRouter remains authoritative for actual quota exhaustion.
            quota = self.team.quota_status()
            cycle_candidates = min(2, max(1, int(self.settings.analysis_candidates)))
            estimated_calls = self.team.estimated_cycle_calls(cycle_candidates)
            self._consume_ai_slot()
            started = time.time();
            self.cycle_count += 1
            self.db.journal(None, "CYCLE_START",
                            f"Analysis cycle #{self.cycle_count} started with {cycle_candidates} research slots",
                            {"trigger": trigger, "candidate_slots": cycle_candidates, "quota": quota})
            await self._publish(
                "system",
                {
                    "level": "INFO",
                    "message": (
                        f"Analysis cycle #{self.cycle_count} started ({trigger}) — "
                        f"{estimated_calls} AI calls planned; {quota['remaining']} quota requests remaining"
                    ),
                },
            )
            now = time.time()
            eligible_all = [
                c for c in self.candidates
                if c.get("researchable")
                   and now - self.recent_symbols.get(c["symbol"], 0) >= self.settings.analysis_cooldown_seconds
            ]
            eligible_symbols = {c["symbol"].upper() for c in eligible_all}
            priority_symbols = [symbol for symbol in self._pending_ai_events if symbol in eligible_symbols]
            ranked_eligible = eligible_all[: max(cycle_candidates * 6, 18)]
            priority_candidates = [c for c in eligible_all if c["symbol"].upper() in priority_symbols]
            seen_eligible = {c["symbol"].upper() for c in priority_candidates}
            eligible = priority_candidates + [c for c in ranked_eligible if c["symbol"].upper() not in seen_eligible]
            if not eligible:
                await self._publish("system", {"level": "WARN",
                                               "message": "Analysis cycle skipped — no liquidity-qualified candidates available."})
                self.last_cycle = time.time()
                return
            stale_events = [symbol for symbol in self._pending_ai_events if symbol not in eligible_symbols]
            for symbol in stale_events:
                if not any(c["symbol"].upper() == symbol for c in self.candidates):
                    self._pending_ai_events.pop(symbol, None)

            scout_payload = []
            for c in eligible:
                f = c.get("features") or await self.alpaca.features(c["symbol"], c.get("daily_dollar_volume"))
                scout_payload.append({**c, "features": f})
            try:
                scout = await self.team.scout(scout_payload)
                await self._publish("agent", {"agent": "SCOUT", "data": scout.model_dump()})
                by_symbol = {c["symbol"].upper(): c for c in scout_payload}
                # Event-driven candidates are promoted into the next available AI wave.
                # This does not add requests; it changes which symbols receive the
                # already-budgeted research slots.
                selected = [by_symbol[symbol] for symbol in priority_symbols if symbol in by_symbol][:cycle_candidates]
                selected_symbols = {c["symbol"].upper() for c in selected}
                for picked in scout.selected:
                    symbol = picked.symbol.upper()
                    if len(selected) >= cycle_candidates:
                        break
                    if symbol in by_symbol and symbol not in selected_symbols:
                        selected.append(by_symbol[symbol])
                        selected_symbols.add(symbol)
                # Fill missing Scout choices from the deterministic scanner ranking.
                if len(selected) < cycle_candidates:
                    for fallback in sorted(scout_payload, key=lambda x: float(x.get("score") or 0), reverse=True):
                        if fallback["symbol"] not in selected_symbols:
                            selected.append(fallback)
                            selected_symbols.add(fallback["symbol"])
                        if len(selected) >= cycle_candidates:
                            break
            except OpenRouterQuotaExceeded as exc:
                await self._publish("system", {"level": "ERROR",
                                               "message": f"OpenRouter daily AI quota reached; stopping cycle: {exc}"})
                self.last_cycle = time.time()
                return
            except Exception as exc:
                # Fail closed: fall back to the deterministic scanner ordering.
                # This keeps the cycle alive without inventing AI output.
                await self._publish("system", {"level": "WARN",
                                               "message": f"SCOUT unavailable; using deterministic scanner fallback: {exc}"})
                selected = []
            if not selected:
                selected = sorted(scout_payload, key=lambda x: float(x.get("score") or 0), reverse=True)[
                    :cycle_candidates]
            try:
                macro = await self.team.macro(await self._macro_context())
                macro_data = macro.model_dump()
                await self._publish("agent", {"agent": "MACRO", "data": macro_data})
            except OpenRouterQuotaExceeded as exc:
                await self._publish("system", {"level": "ERROR",
                                               "message": f"OpenRouter daily AI quota reached; stopping cycle: {exc}"})
                self.last_cycle = time.time()
                return
            except Exception as exc:
                # Never interpret a missing MACRO review as a neutral market regime.
                # Neutral is valid AI output; unavailable is a data-integrity failure.
                await self._publish("system", {"level": "WARN",
                                               "message": f"MACRO unavailable; rejecting cycle for safety: {exc}"})
                self.last_cycle = time.time()
                return
            # Refresh portfolio state before research so a position review is built
            # from the same live holdings that will be shown to the PM.
            portfolio = await self._portfolio_payload()
            results = []
            for c in selected:
                symbol = c["symbol"]
                try:
                    features = c.get("features") or await self.alpaca.features(symbol, c.get("daily_dollar_volume"))
                    news = await self.alpaca.news(symbol, 5)
                    event = self._pending_ai_events.get(symbol.upper())
                    position_review = self._position_review_context(
                        symbol, c, features, portfolio, event,
                    )
                    result = await self.team.research_symbol(symbol, features, news)
                    if position_review:
                        result["position_review"] = position_review
                    results.append(result)
                    self.recent_symbols[symbol] = now
                    self._pending_ai_events.pop(symbol.upper(), None)
                    for agent_name, report in result["reports"].items(): await self._publish("agent",
                                                                                             {"agent": agent_name,
                                                                                              "data": report})
                except OpenRouterQuotaExceeded as exc:
                    await self._publish("system", {"level": "ERROR",
                                                   "message": f"OpenRouter daily AI quota reached; stopping cycle: {exc}"})
                    self.last_cycle = time.time()
                    return
                except Exception as exc:
                    await self._publish("system", {"level": "ERROR",
                                                   "message": f"Research failed for {symbol}; candidate rejected: {exc}"})
            if not results:
                await self._publish("system", {"level": "WARN",
                                               "message": "No complete research packets available; cycle rejected before PM."})
                self.last_cycle = time.time()
                return
            activity = self.db.session_entry_status()
            activity.update({"target_entries": self.settings.activity_target_entries, "remaining_target": max(0,
                                                                                                              self.settings.activity_target_entries - int(
                                                                                                                  activity.get(
                                                                                                                      "entries_today",
                                                                                                                      0) or 0))})
            position_reviews = [r["position_review"] for r in results if r.get("position_review")]
            pm_payload = {"portfolio": portfolio, "macro": macro_data, "research": results,
                          "position_reviews": position_reviews,
                          "activity_objective": activity,
                          "risk_limits": {"max_position_weight": self.settings.max_position_weight,
                                          "max_order_notional": self.settings.max_order_notional,
                                          "min_confidence": self.settings.min_confidence,
                                          "min_liquidity_dollars": self.settings.min_liquidity_dollars}}
            try:
                pm = await self.team.pm(pm_payload)
                await self._publish("agent", {"agent": "PM", "data": pm.model_dump()})
            except OpenRouterQuotaExceeded as exc:
                await self._publish("system", {"level": "ERROR",
                                               "message": f"OpenRouter daily AI quota reached; stopping cycle: {exc}"})
                self.last_cycle = time.time()
                return
            except Exception as exc:
                # PM is the final AI trade proposal stage. Never trade on a PM failure.
                await self._publish("system",
                                    {"level": "WARN", "message": f"PM unavailable; fail-safe NO_TRADE: {exc}"})
                self.last_cycle = time.time()
                return
            proposals = [p.model_dump() for p in pm.proposals if p.action != "HOLD"]
            if pm.environment != "TRADE" or not proposals:
                self.last_cycle = time.time();
                return
            proposal_symbols = [str(p["symbol"]).upper() for p in proposals]
            research_symbols = {str(r["symbol"]).upper() for r in results}
            duplicates = sorted({s for s in proposal_symbols if proposal_symbols.count(s) > 1})
            outside_research = sorted(set(proposal_symbols) - research_symbols)
            if duplicates or outside_research:
                reasons = []
                if duplicates:
                    reasons.append(f"duplicate symbols: {', '.join(duplicates)}")
                if outside_research:
                    reasons.append(f"symbols without current research: {', '.join(outside_research)}")
                await self._publish("system", {"level": "WARN",
                                               "message": f"PM output rejected for safety — {'; '.join(reasons)}"})
                self.last_cycle = time.time()
                return
            try:
                risk_batch = await self.team.risk(proposals, portfolio)
                await self._publish("agent", {"agent": "RISK", "data": risk_batch.model_dump()})
                review_symbols = [r.symbol.upper() for r in risk_batch.reviews]
                expected_symbols = [p["symbol"].upper() for p in proposals]
                if len(review_symbols) != len(set(review_symbols)) or set(review_symbols) != set(expected_symbols):
                    raise RuntimeError(
                        f"RISK coverage invalid: expected {expected_symbols}, received {review_symbols}"
                    )
                reviews = {r.symbol.upper(): r.model_dump() for r in risk_batch.reviews}
            except OpenRouterQuotaExceeded as exc:
                await self._publish("system", {"level": "ERROR",
                                               "message": f"OpenRouter daily AI quota reached; stopping cycle: {exc}"})
                self.last_cycle = time.time()
                return
            except Exception as exc:
                # Missing AI risk review is fail-closed: no autonomous orders.
                await self._publish("system", {"level": "WARN",
                                               "message": f"LLM RISK unavailable; rejecting cycle for safety: {exc}"})
                self.last_cycle = time.time()
                return
            open_orders = await self.alpaca.open_orders()
            for proposal in proposals:
                symbol = proposal["symbol"].upper();
                research = next((r for r in results if r["symbol"] == symbol), None)
                if not research: continue
                f = research["features"];
                price = _num(f.get("last_price")) or 0.0;
                daily_dollar_volume = _num(f.get("daily_dollar_volume", f.get("dollar_volume"))) or 0.0
                position = portfolio.get("positions_by_symbol", {}).get(symbol, {});
                equity = float(portfolio.get("equity", 0) or 0);
                current_weight = float(position.get("weight", 0) or 0)
                target_weight = max(0.0, min(1.0, float(proposal.get("target_weight", 0) or 0)))
                if proposal["action"] == "BUY":
                    desired = max(0.0, target_weight - current_weight) * equity
                    qty = desired / price if price else 0.0
                else:
                    # SELL is AI-sized in portfolio-weight terms. There is no fixed
                    # percentage trim: 0% is a legitimate full exit when the thesis
                    # is invalidated. The deterministic risk engine still prevents
                    # invalid quantities such as selling more shares than held.
                    if not position or float(position.get("qty", 0) or 0) <= 0:
                        continue
                    if target_weight > current_weight + 1e-9:
                        decision = {
                            "id": str(uuid.uuid4()), "symbol": symbol, "action": proposal["action"],
                            "confidence": proposal["confidence"], "thesis": proposal["thesis"],
                            "sell_rationale": proposal.get("sell_rationale", ""),
                            "approved": False, "rejection_reason": "SELL_TARGET_ABOVE_CURRENT_WEIGHT",
                            "target_weight": target_weight, "current_weight": current_weight,
                            "price": price, "trigger": trigger,
                        }
                        self.db.decision(decision);
                        self.db.journal(symbol, "DECISION", "SELL REJECTED — target weight is above current weight",
                                        decision)
                        await self._publish("decision", decision)
                        continue
                    desired_reduction = max(0.0, current_weight - target_weight) * equity
                    qty = desired_reduction / price if price else 0.0
                    if not str(proposal.get("sell_rationale", "")).strip():
                        decision = {
                            "id": str(uuid.uuid4()), "symbol": symbol, "action": proposal["action"],
                            "confidence": proposal["confidence"], "thesis": proposal["thesis"],
                            "sell_rationale": "", "approved": False, "rejection_reason": "SELL_JUSTIFICATION_REQUIRED",
                            "target_weight": target_weight, "current_weight": current_weight,
                            "price": price, "trigger": trigger,
                        }
                        self.db.decision(decision);
                        self.db.journal(symbol, "DECISION", "SELL REJECTED — explicit sell rationale required",
                                        decision)
                        await self._publish("decision", decision)
                        continue
                enriched = {**proposal, "symbol": symbol, "price": price, "quantity": qty,
                            "target_weight": target_weight, "current_weight": current_weight,
                            "daily_dollar_volume": daily_dollar_volume, "dollar_volume": daily_dollar_volume}
                llm_risk = reviews.get(symbol, {})
                if llm_risk.get("llm_risk") == "REJECT":
                    approved, reason, meta = False, "LLM_RISK_REJECT", {}
                else:
                    approved, reason, meta = self.risk.validate(enriched, portfolio, len(open_orders))
                qty = float(meta.get("quantity", qty)) if approved else qty
                decision = {
                    "id": str(uuid.uuid4()),
                    "symbol": symbol,
                    "action": proposal["action"],
                    "confidence": proposal["confidence"],
                    "thesis": proposal["thesis"],
                    "sell_rationale": proposal.get("sell_rationale", ""),
                    "approved": approved,
                    "rejection_reason": None if approved else reason,
                    "quantity": qty,
                    "price": price,
                    "target_weight": target_weight,
                    "current_weight": current_weight,
                    "trigger": trigger,
                }
                self.db.decision(decision);
                self.db.journal(symbol, "DECISION",
                                f"{proposal['action']} {'APPROVED' if approved else 'REJECTED'} — {reason}", decision)
                await self._publish("decision", {**decision, "reason": reason})
                if not execute_trades or not approved or not self.settings.trading_enabled or self.risk.manual_pause: continue
                result = await self.execution.execute(symbol, proposal["action"], qty)
                if result.ok:
                    self.risk.record_order(symbol)
                    open_orders.append({"symbol": symbol})
                    if proposal["action"] == "BUY":
                        self.db.record_session_entry()
                self.db.journal(symbol, "ORDER", f"{result.action} {result.qty:.4f} submitted: {result.status}",
                                result.model_dump())
                await self._publish("trade", result.model_dump())
            self.last_cycle = time.time()
            duration = time.time() - started
            self.db.journal(None, "CYCLE_COMPLETE", f"Analysis cycle #{self.cycle_count} complete in {duration:.1f}s",
                            {"trigger": trigger, "duration_seconds": duration, "quota": self.team.quota_status(),
                             "activity": self.db.session_entry_status()})
            await self._publish("system", {"level": "INFO", "message": f"Cycle complete in {duration:.1f}s"})

    async def _macro_context(self):
        # Three independent Alpaca reads with no ordering dependency between
        # them; fetch concurrently instead of one benchmark at a time.
        symbols = ("SPY", "QQQ", "IWM")
        features = await asyncio.gather(*(self.alpaca.features(s) for s in symbols))
        return {"benchmarks": dict(zip(symbols, features))}

    async def _refresh_portfolio(self):
        account, positions = await asyncio.gather(self.alpaca.account(), self.alpaca.positions())
        equity = float(account.get("equity") or 0);
        cash = float(account.get("cash") or 0);
        buying_power = float(account.get("buying_power") or 0);
        last_equity = float(account.get("last_equity") or equity)
        daily_pnl = equity - last_equity;
        daily_pnl_pct = daily_pnl / last_equity if last_equity else 0

        # Keep a persistent weekly anchor so the PM can distinguish a bad day from
        # broader portfolio pressure over the current trading week. The anchor is
        # established from the first portfolio refresh of each UTC week and survives
        # process restarts through control_state. This is contextual information only;
        # it is not itself a trade veto.
        week_start = (datetime.now(timezone.utc).date()).isoformat()
        monday = datetime.now(timezone.utc).date()
        week_start = (monday.fromordinal(monday.toordinal() - monday.weekday())).isoformat()
        stored_week = self.db.get_control_value("weekly_pnl_anchor_week", "")
        stored_anchor = self.db.get_control_value("weekly_pnl_anchor_equity", "")
        try:
            weekly_anchor_equity = float(stored_anchor)
        except (TypeError, ValueError):
            weekly_anchor_equity = 0.0
        if stored_week != week_start or weekly_anchor_equity <= 0:
            weekly_anchor_equity = equity
            self.db.set_control_value("weekly_pnl_anchor_week", week_start)
            self.db.set_control_value("weekly_pnl_anchor_equity", f"{equity:.12f}")
        weekly_pnl = equity - weekly_anchor_equity
        weekly_pnl_pct = weekly_pnl / weekly_anchor_equity if weekly_anchor_equity else 0
        by_symbol = {};
        clean = []
        for p in positions:
            symbol = str(p.get("symbol", "")).upper();
            mv = float(p.get("market_value") or 0);
            qty = float(p.get("qty") or 0);
            pl = float(p.get("unrealized_pl") or 0)
            entry_decision = self.db.latest_approved_buy_decision(symbol)
            entry_payload = (entry_decision or {}).get("payload_json") or {}
            item = {
                "symbol": symbol,
                "qty": qty,
                "market_value": mv,
                "weight": abs(mv) / equity if equity else 0,
                "unrealized_pl": pl,
                "current_price": float(p.get("current_price") or 0),
                "avg_entry_price": float(p.get("avg_entry_price") or 0),
                "unrealized_plpc": float(p.get("unrealized_plpc") or 0),
                "change_today": float(p.get("change_today") or 0),
                "entry_thesis": entry_decision.get("thesis") if entry_decision else None,
                "entry_invalidation": entry_payload.get("invalidation") if isinstance(entry_payload, dict) else None,
                "entry_confidence": entry_decision.get("confidence") if entry_decision else None,
                "entry_timestamp": entry_decision.get("timestamp") if entry_decision else None,
            }
            by_symbol[symbol] = item;
            clean.append(item)
        self.portfolio = {"equity": equity, "cash": cash, "buying_power": buying_power, "daily_pnl": daily_pnl,
                          "daily_pnl_pct": daily_pnl_pct, "weekly_pnl": weekly_pnl,
                          "weekly_pnl_pct": weekly_pnl_pct, "weekly_pnl_anchor_week": week_start,
                          "positions": clean, "positions_by_symbol": by_symbol, "updated_at": time.time()}
        self.db.portfolio_snapshot(self.portfolio);
        await self._publish("portfolio", self.portfolio)

    async def _portfolio_payload(self):
        await self._refresh_portfolio();
        return self.portfolio

    def _position_review_context(
            self,
            symbol: str,
            candidate: dict[str, Any],
            features: dict[str, Any],
            portfolio: dict[str, Any],
            event: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Build an explicit PM review packet for a currently held position.

        A market move is only the trigger. The PM receives the current exposure,
        original entry thesis/invalidation, and latest market evidence so it can
        decide whether the position should be held, added, reduced, or exited.
        """
        symbol = str(symbol or "").upper().strip()
        position = (portfolio.get("positions_by_symbol") or {}).get(symbol)
        if not position or float(position.get("qty", 0) or 0) <= 0:
            return None

        current_price = _num(candidate.get("last_price"))
        if current_price is None:
            current_price = _num(features.get("last_price"))
        current_weight = float(position.get("weight", 0) or 0)
        return {
            "review_type": "POSITION_REVIEW",
            "symbol": symbol,
            "trigger": {
                "reason": (event or {}).get("reason", "portfolio_position"),
                "review_type": (event or {}).get("review_type", "POSITION_REVIEW"),
                "move_pct": (event or {}).get("move_pct"),
                "queued_at": (event or {}).get("queued_at"),
                "last_triggered_at": (event or {}).get("last_triggered_at"),
            },
            "position": {
                "qty": float(position.get("qty", 0) or 0),
                "market_value": float(position.get("market_value", 0) or 0),
                "current_weight": current_weight,
                "current_price": current_price,
                "avg_entry_price": float(position.get("avg_entry_price", 0) or 0),
                "unrealized_pl": float(position.get("unrealized_pl", 0) or 0),
                "unrealized_plpc": float(position.get("unrealized_plpc", 0) or 0),
                "change_today": float(position.get("change_today", 0) or 0),
            },
            "original_entry": {
                "thesis": position.get("entry_thesis"),
                "invalidation": position.get("entry_invalidation"),
                "confidence": position.get("entry_confidence"),
                "timestamp": position.get("entry_timestamp"),
            },
            "current_market": {
                "last_price": current_price,
                "change_pct": _num(candidate.get("change_pct")),
                "daily_dollar_volume": _num(
                    candidate.get("daily_dollar_volume", features.get("daily_dollar_volume"))
                ),
                "quant_features": features,
            },
            "portfolio_context": {
                "daily_pnl_pct": portfolio.get("daily_pnl_pct"),
                "weekly_pnl_pct": portfolio.get("weekly_pnl_pct"),
                "cash": portfolio.get("cash"),
                "buying_power": portfolio.get("buying_power"),
            },
            "pm_question": (
                "Does the original thesis remain valid given current evidence? "
                "Choose HOLD, ADD, REDUCE, or EXIT via the corresponding target_weight. "
                "A price decline is a review trigger, not an automatic sell signal."
            ),
        }

    async def _on_market_event(self, kind: str, payload: dict[str, Any]):
        self.market_event_count += 1;
        self.last_market_event = payload
        if kind == "bar":
            previous = self.alpaca.previous_bar.get(payload["symbol"], {})
            move = ((payload["close"] / previous["close"] - 1) * 100) if previous.get("close") else 0.0
            # Market Monitor/ticker remain live regardless of the trading mode.
            await self._publish("market", {**payload, "move_pct": move})
            if self.autopilot and abs(move) >= self.settings.event_trigger_pct and not self.risk.manual_pause:
                symbol = str(payload["symbol"]).upper()
                held = symbol in (self.portfolio.get("positions_by_symbol") or {})
                reason = "position_price_move" if held else "price_move"
                await self._trigger_event_cycle(symbol, reason, move_pct=move)

    async def _trigger_event_cycle(self, symbol: str, reason: str, move_pct: float | None = None):
        symbol = str(symbol).upper().strip()
        if not symbol:
            return
        # Event triggers do not create an extra AI cycle. They enqueue a priority
        # opportunity so the next available scheduled AI wave researches it first.
        existing = self._pending_ai_events.get(symbol)
        self._pending_ai_events[symbol] = {
            "reason": reason,
            "review_type": "POSITION_REVIEW" if reason.startswith("position_") else "OPPORTUNITY_REVIEW",
            "move_pct": move_pct,
            "queued_at": existing.get("queued_at", time.time()) if existing else time.time(),
            "last_triggered_at": time.time(),
        }
        queue = list(self._pending_ai_events)
        await self._publish(
            "system",
            {
                "level": "INFO",
                "message": (
                    f"AI opportunity queued: {symbol} after {reason} — "
                    f"priority for next available AI slot ({len(queue)} queued)"
                ),
                "symbol": symbol,
                "reason": reason,
                "review_type": self._pending_ai_events[symbol]["review_type"],
                "move_pct": move_pct,
                "queue_size": len(queue),
            },
        )

    async def _on_trade_update(self, data: dict[str, Any]):
        order = data.get("order", data) if isinstance(data, dict) else {}
        order_id = str(order.get("id") or order.get("order_id") or "")
        status = str(order.get("status") or data.get("event") or "unknown")
        if order_id: self.db.update_order_by_alpaca_id(order_id, status, data)
        symbol = str(order.get("symbol") or "").upper() or None
        event = str(data.get("event") or "TRADE_UPDATE")
        self.db.event("trade_update", data);
        self.db.journal(symbol, event, f"Broker update: {status}", data)
        await self._publish("trade_update", data)
        if status in {"filled", "partially_filled", "canceled", "rejected", "expired"}: await self._refresh_portfolio()

    async def _publish(self, kind: str, data):
        await self.bus.publish(kind, data);
        self.db.event(kind, data)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
