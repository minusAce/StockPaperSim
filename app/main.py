from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agents.team import AgentTeam
from app.backtest import run_moving_average_backtest
from app.config import settings
from app.engine import TradingEngine
from app.event_bus import EventBus
from app.market.alpaca import AlpacaService
from app.schemas import BacktestRequest
from app.shutdown import shutdown_docker_stack
from app.storage.db import Database
from app.trading.execution import ExecutionEngine
from app.trading.risk import RiskEngine

settings.validate()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

bus = EventBus(); db = Database(settings); alpaca = AlpacaService(settings); team = AgentTeam(settings, db); risk = RiskEngine(settings, db); execution = ExecutionEngine(alpaca, db); engine = TradingEngine(settings, alpaca, team, risk, execution, db, bus)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start(); yield; await engine.stop()


app = FastAPI(title="StockPaperSim", version="2.0.0", lifespan=lifespan)

if settings.frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=settings.frontend_dist / "assets"), name="assets")


@app.get("/")
async def index():
    built = settings.frontend_dist / "index.html"
    if built.exists(): return FileResponse(built)
    return JSONResponse({"message": "React frontend not built yet. Run 'cd frontend && npm install && npm run build'.", "docs": "/docs"})


@app.get("/api/status")
async def status():
    return {"running": engine.running, "autopilot": engine.autopilot, "trading_enabled": settings.trading_enabled, "paper_trading": settings.paper_trading, "data_feed": settings.alpaca_data_feed, "model": settings.model_default, "model_default": settings.model_default, "model_fallback": settings.model_fallback or None, "ai_quota": team.quota_status(), "activity": db.session_entry_status() | {"target_entries": settings.activity_target_entries}, "kill_switch": risk.kill_switch, "paused": risk.manual_pause, "stream_running": alpaca._stream_running, "trade_stream_running": alpaca._trade_stream_running, "market_event_count": engine.market_event_count, "last_scan": engine.last_scan, "last_cycle": engine.last_cycle, "cycle_count": engine.cycle_count, "asset_universe": len(alpaca.assets), "db_counts": db.counts()}


@app.get("/api/agents")
async def agents(): return team.status()


@app.get("/api/portfolio")
async def portfolio():
    if not engine.autopilot and engine.portfolio:
        return engine.portfolio
    await engine._refresh_portfolio()
    return engine.portfolio


@app.get("/api/candidates")
async def candidates(): return engine.candidates[:50]


@app.get("/api/decisions")
async def decisions(): return db.latest("decisions", 100)


@app.get("/api/orders")
async def orders(): return db.latest("orders", 100)


@app.get("/api/journal")
async def journal(): return db.latest("trade_journal", 100)


@app.get("/api/events")
async def events(): return db.latest("events", 100)



@app.get("/api/performance")
async def performance(): return db.performance()


@app.get("/api/market/{symbol}/history")
async def market_history(symbol: str, limit: int = 120): return await alpaca.history(symbol, min(max(limit, 20), 300))


@app.get("/api/benchmark")
async def benchmark():
    return await alpaca.benchmark(settings.benchmark_symbol)


@app.post("/api/control/pause")
async def pause(payload: dict):
    risk.set_pause(bool(payload.get("paused", True))); await bus.publish("system", {"level": "WARN", "message": f"Trading {'paused' if risk.manual_pause else 'resumed'}"}); return {"paused": risk.manual_pause}


@app.post("/api/control/kill-switch")
async def kill_switch(payload: dict):
    if not bool(payload.get("active", True)):
        risk.reset_kill_switch()
        return {"kill_switch": risk.kill_switch, "paused": risk.manual_pause}
    risk.activate_kill_switch()
    await engine.set_autopilot(False)
    await bus.publish("system", {"level": "WARN", "message": "KILL SWITCH ACTIVE — shutting down StockPaperSim stack"})
    asyncio.create_task(_delayed_stack_shutdown())
    return {"kill_switch": True, "shutdown_scheduled": True}


async def _delayed_stack_shutdown():
    await asyncio.sleep(0.35)
    await engine.stop()
    await shutdown_docker_stack()


@app.post("/api/control/autopilot")
async def autopilot(payload: dict):
    enabled = bool(payload.get("enabled", True))
    await engine.set_autopilot(enabled)
    return {"autopilot": engine.autopilot, "api_active": engine.autopilot}


@app.post("/api/control/run-cycle")
async def run_cycle():
    if not engine.autopilot:
        return JSONResponse(status_code=409, content={"ok": False, "message": "Autopilot is OFF."})
    await engine.analysis_cycle("manual-now")
    return {"ok": True}


@app.post("/api/control/run-ai-test")
async def run_ai_test():
    if risk.kill_switch:
        return JSONResponse(status_code=409, content={"ok": False, "message": "Kill switch is active."})
    was_enabled = engine.autopilot
    team.set_enabled(True)
    try:
        if not engine.candidates:
            await engine.scan()
        if not engine.candidates:
            return JSONResponse(status_code=409, content={"ok": False, "message": "No market candidates are available for the AI test."})
        await engine.analysis_cycle("manual-test", execute_trades=False)
    finally:
        team.set_enabled(was_enabled)
    return {"ok": True, "message": "Manual AI test cycle completed. No orders were executed."}


@app.post("/api/control/rescan")
async def rescan():
    if not engine.autopilot:
        return JSONResponse(status_code=409, content={"ok": False, "message": "Autopilot is OFF."})
    await engine.scan()
    return {"ok": True, "candidates": len(engine.candidates)}


@app.post("/api/backtest")
async def backtest(request: BacktestRequest): return (await run_moving_average_backtest(alpaca, settings, request)).model_dump()


@app.get("/health")
async def health(): return {"ok": True, "paper": settings.paper_trading, "running": engine.running}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await bus.connect(websocket)
    await websocket.send_json({"type": "connected", "data": {"timestamp": time.time()}})
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=20)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "data": {"timestamp": time.time(), "ai_quota": team.quota_status()}})
    except WebSocketDisconnect:
        await bus.disconnect(websocket)
    except Exception:
        await bus.disconnect(websocket)
