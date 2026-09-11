from __future__ import annotations

import uuid

from app.market.alpaca import AlpacaService
from app.schemas import ExecutionResult
from app.storage.db import Database


class ExecutionEngine:
    def __init__(self, alpaca: AlpacaService, db: Database):
        self.alpaca, self.db = alpaca, db

    async def execute(self, symbol: str, action: str, qty: float) -> ExecutionResult:
        execution_id = str(uuid.uuid4()); client_order_id = f"aitf-{execution_id[:20]}"
        try:
            order = await self.alpaca.submit_market_order(symbol, action, qty, client_order_id)
            data = order.model_dump() if hasattr(order, "model_dump") else dict(order)
            result = ExecutionResult(ok=True, symbol=symbol, action=action, qty=qty, order_id=str(data.get("id")) if data.get("id") else None, status=str(data.get("status")) if data.get("status") else None, error=None)
            self.db.order({"id": execution_id, "symbol": symbol, "action": action, "qty": qty, "status": result.status, "alpaca_order_id": result.order_id, "payload": data})
            return result
        except Exception as exc:
            result = ExecutionResult(ok=False, symbol=symbol, action=action, qty=qty, order_id=None, status="ERROR", error=str(exc))
            self.db.order({"id": execution_id, "symbol": symbol, "action": action, "qty": qty, "status": "ERROR", "error": str(exc), "payload": {}})
            return result
