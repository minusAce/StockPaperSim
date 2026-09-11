from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, MetaData, String, Table, Text, case, create_engine, \
    func, select
from sqlalchemy.engine import Engine

from app.config import Settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(settings.database_url, future=True, pool_pre_ping=True,
                                            connect_args=connect_args)
        self.metadata = MetaData()
        self._define_tables()
        self.metadata.create_all(self.engine)
        self._lock = threading.RLock()
        self._seed_control_state()

    def _define_tables(self) -> None:
        self.agent_runs = Table(
            "agent_runs", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("agent", String(32), nullable=False),
            Column("symbol", String(32)),
            Column("model", String(200)),
            Column("ok", Boolean, nullable=False),
            Column("input_json", Text),
            Column("output_json", Text),
            Column("error", Text),
        )
        self.decisions = Table(
            "decisions", self.metadata,
            Column("id", String(64), primary_key=True),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("symbol", String(32), nullable=False),
            Column("action", String(16), nullable=False),
            Column("confidence", Float, nullable=False),
            Column("thesis", Text, nullable=False),
            Column("approved", Boolean),
            Column("rejection_reason", Text),
            Column("payload_json", Text),
        )
        self.orders = Table(
            "orders", self.metadata,
            Column("id", String(64), primary_key=True),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("symbol", String(32), nullable=False),
            Column("action", String(16), nullable=False),
            Column("qty", Float, nullable=False),
            Column("status", String(32)),
            Column("alpaca_order_id", String(64)),
            Column("error", Text),
            Column("payload_json", Text),
        )
        self.events = Table(
            "events", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("event_type", String(64), nullable=False),
            Column("payload_json", Text, nullable=False),
        )
        self.portfolio_snapshots = Table(
            "portfolio_snapshots", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("payload_json", Text, nullable=False),
        )
        self.trade_journal = Table(
            "trade_journal", self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("symbol", String(32)),
            Column("event", String(64), nullable=False),
            Column("message", Text, nullable=False),
            Column("payload_json", Text),
        )
        self.control_state = Table(
            "control_state", self.metadata,
            Column("key", String(64), primary_key=True),
            Column("value", String(256), nullable=False),
        )

    def _seed_control_state(self) -> None:
        """Create runtime controls and reset them to a safe boot state when configured.

        The trading database may live on a persistent Docker volume, so these values
        must not inherit a previous kill/autopilot state after a rebuild/restart.
        This resets control state only; historical decisions/orders remain intact.
        """
        with self._lock, self.engine.begin() as conn:
            rows = set(conn.execute(select(self.control_state.c.key)).scalars().all())
            defaults = {"autopilot": False, "pause": False, "kill_switch": False}
            missing = [{"key": key, "value": "1" if value else "0"} for key, value in defaults.items() if
                       key not in rows]
            if missing:
                conn.execute(self.control_state.insert(), missing)
            if self.settings.reset_runtime_controls_on_start:
                for key, value in defaults.items():
                    conn.execute(self.control_state.update().where(self.control_state.c.key == key).values(
                        value="1" if value else "0"))

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, default=str, ensure_ascii=False)

    @staticmethod
    def _row(row) -> dict[str, Any]:
        result = dict(row._mapping)
        for key in ("timestamp",):
            if isinstance(result.get(key), datetime):
                result[key] = result[key].isoformat()
        return result

    def agent_run(self, agent: str, symbol: str | None, model: str, ok: bool, input_data: Any, output_data: Any,
                  error: str | None = None) -> None:
        with self._lock, self.engine.begin() as conn:
            conn.execute(
                self.agent_runs.insert().values(timestamp=utc_now(), agent=agent, symbol=symbol, model=model, ok=ok,
                                                input_json=self._json(input_data), output_json=self._json(output_data),
                                                error=error))

    def decision(self, data: dict[str, Any]) -> None:
        values = self._decision_values(data)
        with self._lock, self.engine.begin() as conn:
            existing = conn.execute(select(self.decisions.c.id).where(self.decisions.c.id == values["id"])).first()
            if existing:
                conn.execute(self.decisions.update().where(self.decisions.c.id == values["id"]).values(**values))
            else:
                conn.execute(self.decisions.insert().values(**values))

    def latest_approved_buy_decision(self, symbol: str) -> dict[str, Any] | None:
        """Return the most recent approved BUY decision for a symbol.

        This preserves the investment thesis that opened the position so later
        portfolio reviews can compare the current evidence against the original
        reason for entry.
        """
        symbol = str(symbol or "").upper()
        if not symbol:
            return None
        with self._lock, self.engine.connect() as conn:
            row = conn.execute(
                select(self.decisions)
                .where(
                    self.decisions.c.symbol == symbol,
                    self.decisions.c.action == "BUY",
                    self.decisions.c.approved.is_(True),
                )
                .order_by(self.decisions.c.timestamp.desc())
                .limit(1)
            ).first()
        if not row:
            return None
        item = self._row(row)
        if item.get("payload_json"):
            try:
                item["payload_json"] = json.loads(item["payload_json"])
            except Exception:
                pass
        return item

    def _decision_values(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": data["id"], "timestamp": _parse_dt(data.get("timestamp")), "symbol": data["symbol"],
            "action": data["action"],
            "confidence": float(data["confidence"]), "thesis": data["thesis"], "approved": data.get("approved"),
            "rejection_reason": data.get("rejection_reason"), "payload_json": self._json(data),
        }

    def order(self, data: dict[str, Any]) -> None:
        values = {
            "id": data["id"], "timestamp": _parse_dt(data.get("timestamp")), "symbol": data["symbol"],
            "action": data["action"],
            "qty": float(data.get("qty", 0)), "status": data.get("status"),
            "alpaca_order_id": data.get("alpaca_order_id"),
            "error": data.get("error"), "payload_json": self._json(data),
        }
        with self._lock, self.engine.begin() as conn:
            existing = conn.execute(select(self.orders.c.id).where(self.orders.c.id == values["id"])).first()
            if existing:
                conn.execute(self.orders.update().where(self.orders.c.id == values["id"]).values(**values))
            else:
                conn.execute(self.orders.insert().values(**values))

    def update_order_by_alpaca_id(self, alpaca_order_id: str, status: str, payload: dict[str, Any]) -> None:
        with self._lock, self.engine.begin() as conn:
            conn.execute(
                self.orders.update().where(self.orders.c.alpaca_order_id == alpaca_order_id).values(status=status,
                                                                                                    payload_json=self._json(
                                                                                                        payload)))

    def event(self, event_type: str, payload: Any) -> None:
        with self._lock, self.engine.begin() as conn:
            conn.execute(self.events.insert().values(timestamp=utc_now(), event_type=event_type,
                                                     payload_json=self._json(payload)))

    def portfolio_snapshot(self, payload: Any) -> None:
        with self._lock, self.engine.begin() as conn:
            conn.execute(
                self.portfolio_snapshots.insert().values(timestamp=utc_now(), payload_json=self._json(payload)))

    def journal(self, symbol: str | None, event: str, message: str, payload: Any = None) -> None:
        with self._lock, self.engine.begin() as conn:
            conn.execute(
                self.trade_journal.insert().values(timestamp=utc_now(), symbol=symbol, event=event, message=message,
                                                   payload_json=self._json(payload or {})))

    def record_session_entry(self) -> int:
        """Increment today's successful BUY count used only as a soft PM activity objective."""
        today = utc_now().date().isoformat()
        with self._lock, self.engine.begin() as conn:
            day_row = conn.execute(
                select(self.control_state.c.value).where(self.control_state.c.key == "session_entry_day")).first()
            count_row = conn.execute(
                select(self.control_state.c.value).where(self.control_state.c.key == "session_entry_count")).first()
            day = str(day_row[0]) if day_row else ""
            count = int(count_row[0]) if count_row and str(count_row[0]).isdigit() else 0
            if day != today:
                count = 0
                self._set_control_value_conn(conn, "session_entry_day", today)
            count += 1
            self._set_control_value_conn(conn, "session_entry_count", str(count))
            return count

    def session_entry_status(self) -> dict[str, Any]:
        today = utc_now().date().isoformat()
        with self._lock, self.engine.connect() as conn:
            day_row = conn.execute(
                select(self.control_state.c.value).where(self.control_state.c.key == "session_entry_day")).first()
            count_row = conn.execute(
                select(self.control_state.c.value).where(self.control_state.c.key == "session_entry_count")).first()
        day = str(day_row[0]) if day_row else today
        count = int(count_row[0]) if count_row and str(count_row[0]).isdigit() and day == today else 0
        return {"day_utc": today, "entries_today": count}

    def _set_control_value_conn(self, conn, key: str, value: str) -> None:
        existing = conn.execute(select(self.control_state.c.key).where(self.control_state.c.key == key)).first()
        if existing:
            conn.execute(self.control_state.update().where(self.control_state.c.key == key).values(value=str(value)))
        else:
            conn.execute(self.control_state.insert().values(key=key, value=str(value)))

    def set_control_value(self, key: str, value: str) -> None:
        """Persist an arbitrary small control value for cross-restart state."""
        with self._lock, self.engine.begin() as conn:
            existing = conn.execute(select(self.control_state.c.key).where(self.control_state.c.key == key)).first()
            if existing:
                conn.execute(
                    self.control_state.update().where(self.control_state.c.key == key).values(value=str(value)))
            else:
                conn.execute(self.control_state.insert().values(key=key, value=str(value)))

    def get_control_value(self, key: str, default: str = "") -> str:
        with self._lock, self.engine.connect() as conn:
            row = conn.execute(select(self.control_state.c.value).where(self.control_state.c.key == key)).first()
        return default if not row else str(row[0])

    def set_control(self, key: str, value: bool) -> None:
        encoded = "1" if value else "0"
        with self._lock, self.engine.begin() as conn:
            existing = conn.execute(select(self.control_state.c.key).where(self.control_state.c.key == key)).first()
            if existing:
                conn.execute(self.control_state.update().where(self.control_state.c.key == key).values(value=encoded))
            else:
                conn.execute(self.control_state.insert().values(key=key, value=encoded))

    def get_control(self, key: str, default: bool = False) -> bool:
        with self._lock, self.engine.connect() as conn:
            row = conn.execute(select(self.control_state.c.value).where(self.control_state.c.key == key)).first()
        return default if not row else row[0] == "1"

    def latest(self, table: str, limit: int = 50) -> list[dict[str, Any]]:
        mapping = {"agent_runs": self.agent_runs, "decisions": self.decisions, "orders": self.orders,
                   "events": self.events, "portfolio_snapshots": self.portfolio_snapshots,
                   "trade_journal": self.trade_journal}
        if table not in mapping:
            raise ValueError("Invalid table")
        t = mapping[table]
        order_col = t.c.id
        with self._lock, self.engine.connect() as conn:
            rows = conn.execute(select(t).order_by(order_col.desc()).limit(limit)).fetchall()
        result = [self._row(r) for r in rows]
        if table in {"agent_runs", "decisions", "orders", "events", "portfolio_snapshots", "trade_journal"}:
            for item in result:
                for k in ("input_json", "output_json", "payload_json"):
                    if k in item and item[k]:
                        try:
                            item[k] = json.loads(item[k])
                        except Exception:
                            pass
        return result

    def counts(self) -> dict[str, int]:
        names = {
            "agent_runs": self.agent_runs, "decisions": self.decisions, "orders": self.orders,
            "events": self.events, "portfolio_snapshots": self.portfolio_snapshots, "trade_journal": self.trade_journal,
        }
        with self._lock, self.engine.connect() as conn:
            return {name: int(conn.execute(select(func.count()).select_from(table)).scalar_one()) for name, table in
                    names.items()}

    def performance(self) -> dict[str, Any]:
        agent_stats: list[dict[str, Any]] = []
        with self._lock, self.engine.connect() as conn:
            rows = conn.execute(select(self.agent_runs.c.agent, func.count(self.agent_runs.c.id).label("runs"),
                                       func.sum(case((self.agent_runs.c.ok.is_(True), 1), else_=0)).label(
                                           "ok_runs")).group_by(self.agent_runs.c.agent)).fetchall()
            for r in rows:
                agent_stats.append({"agent": r.agent, "runs": int(r.runs or 0), "successful": int(r.ok_runs or 0),
                                    "success_rate": (float(r.ok_runs or 0) / float(r.runs or 1))})
            decision_count = int(conn.execute(select(func.count()).select_from(self.decisions)).scalar_one())
            approved = int(conn.execute(select(func.count()).select_from(self.decisions).where(
                self.decisions.c.approved.is_(True))).scalar_one())
            orders = int(conn.execute(select(func.count()).select_from(self.orders)).scalar_one())
            filled = int(conn.execute(select(func.count()).select_from(self.orders).where(
                self.orders.c.status.in_(["filled", "partially_filled"]))).scalar_one())
        return {"agents": agent_stats, "decisions": decision_count, "approved_decisions": approved, "orders": orders,
                "filled_orders": filled}


def _parse_dt(value: Any) -> datetime:
    if not value:
        return utc_now()
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return utc_now()
