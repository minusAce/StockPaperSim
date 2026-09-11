from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()
        self.lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self.lock:
            self.connections.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self.lock:
            self.connections.discard(ws)

    async def publish(self, kind: str, data) -> None:
        message = {"type": kind, "data": data}
        # Serialize once, up front, with default=str so payload fields that
        # aren't natively JSON-safe (e.g. a raw datetime from
        # ExecutionResult.model_dump()) still send instead of raising inside
        # send_json. Without this, json.dumps's TypeError was caught by the
        # per-socket except below and every currently-connected client got
        # dropped as "dead" on every single message that carried a datetime,
        # even though the sockets themselves were perfectly healthy.
        try:
            text = json.dumps(message, default=str)
        except Exception:
            logger.exception("EventBus payload for %r is not JSON-serializable; dropping message", kind)
            return
        async with self.lock:
            sockets = list(self.connections)
        dead = []
        for ws in sockets:
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        if dead:
            async with self.lock:
                for ws in dead:
                    self.connections.discard(ws)
