from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Any

from app.config import Settings


class RiskEngine:
    def __init__(self, settings: Settings, db=None):
        self.settings = settings
        self.db = db
        self.kill_switch = bool(db.get_control("kill_switch", False)) if db else False
        self.manual_pause = bool(db.get_control("pause", False)) if db else False
        self.order_times: dict[str, deque[float]] = defaultdict(deque)

    def activate_kill_switch(self):
        self.kill_switch = True
        if self.db: self.db.set_control("kill_switch", True)

    def reset_kill_switch(self):
        self.kill_switch = False
        if self.db: self.db.set_control("kill_switch", False)

    def set_pause(self, paused: bool):
        self.manual_pause = paused
        if self.db: self.db.set_control("pause", paused)

    def validate(self, proposal: dict[str, Any], portfolio: dict[str, Any], open_order_count: int) -> tuple[
        bool, str, dict[str, Any]]:
        symbol = str(proposal.get("symbol", "")).upper()
        action = str(proposal.get("action", "")).upper()
        try:
            confidence = float(proposal.get("confidence", 0) or 0)
            price = float(proposal.get("price", 0) or 0)
            qty = float(proposal.get("quantity", 0) or 0)
            equity = float(portfolio.get("equity", 0) or 0)
            cash = float(portfolio.get("cash", 0) or 0)
            daily_loss_pct = float(portfolio.get("daily_pnl_pct", 0) or 0)
            liquidity = float(proposal.get("daily_dollar_volume", proposal.get("dollar_volume", 0)) or 0)
        except (TypeError, ValueError):
            return False, "NON_NUMERIC_RISK_INPUT", {}
        if not all(math.isfinite(v) for v in (confidence, price, qty, equity, cash, daily_loss_pct, liquidity)):
            return False, "NON_FINITE_RISK_INPUT", {}
        position = portfolio.get("positions_by_symbol", {}).get(symbol, {})
        current_weight = float(position.get("weight", 0) or 0)
        current_qty = float(position.get("qty", 0) or 0)

        if self.kill_switch: return False, "KILL_SWITCH_ACTIVE", {}
        if self.manual_pause: return False, "MANUAL_PAUSE", {}
        if not self.settings.trading_enabled: return False, "TRADING_DISABLED", {}
        if action not in {"BUY", "SELL"}: return False, "INVALID_ACTION", {}
        if confidence < self.settings.min_confidence: return False, "CONFIDENCE_BELOW_MINIMUM", {}
        if equity <= 0 or price <= 0: return False, "INVALID_PORTFOLIO_OR_PRICE", {}
        if liquidity < self.settings.min_liquidity_dollars: return False, "INSUFFICIENT_LIQUIDITY", {}
        if open_order_count >= self.settings.max_total_open_orders: return False, "TOO_MANY_OPEN_ORDERS", {}

        recent = self.order_times[symbol]
        cutoff = time.time() - 3600
        while recent and recent[0] < cutoff: recent.popleft()
        if len(recent) >= self.settings.max_symbol_orders_per_hour:
            return False, "SYMBOL_ORDER_RATE_LIMIT", {}

        daily_loss_limit = abs(self.settings.max_daily_loss_pct)
        emergency_loss_limit = abs(self.settings.emergency_daily_loss_pct)
        daily_loss_buy_scale = 1.0

        if action == "BUY":
            desired_notional = max(0.0, float(proposal.get("target_weight", 0)) - current_weight) * equity
            desired_notional = min(desired_notional, self.settings.max_order_notional)
            qty = math.floor(min(qty, desired_notional / price if price else 0.0))

            # Daily loss is a graduated new-risk control. It never overrides the PM
            # proposal by itself until the emergency threshold is reached. Between
            # the caution and emergency thresholds, scale only the NEW BUY quantity.
            loss = max(0.0, -daily_loss_pct)
            if loss >= emergency_loss_limit:
                return False, "EMERGENCY_DAILY_LOSS_LIMIT", {}
            if loss > daily_loss_limit:
                daily_loss_buy_scale = (emergency_loss_limit - loss) / (emergency_loss_limit - daily_loss_limit)
                daily_loss_buy_scale = max(0.0, min(1.0, daily_loss_buy_scale))
                qty = math.floor(qty * daily_loss_buy_scale)

            if qty < self.settings.min_order_qty:
                return False, ("DAILY_LOSS_NEW_RISK_TOO_LARGE" if loss > daily_loss_limit
                               else "QUANTITY_TOO_SMALL"), {
                    "quantity": qty,
                    "daily_loss_buy_scale": daily_loss_buy_scale,
                }
            if qty * price < self.settings.min_order_notional:
                return False, ("DAILY_LOSS_ORDER_TOO_SMALL" if loss > daily_loss_limit
                               else "ORDER_TOO_SMALL"), {
                    "quantity": qty,
                    "daily_loss_buy_scale": daily_loss_buy_scale,
                }
            if qty * price > cash:
                return False, "INSUFFICIENT_CASH", {"daily_loss_buy_scale": daily_loss_buy_scale}
            projected_weight = current_weight + (qty * price / equity)
            if projected_weight > self.settings.max_position_weight + 1e-9:
                return False, "POSITION_WEIGHT_LIMIT", {"daily_loss_buy_scale": daily_loss_buy_scale}
        else:
            target_weight = float(proposal.get("target_weight", current_weight) or 0)
            if target_weight < -1e-9 or target_weight > current_weight + 1e-9:
                return False, "INVALID_SELL_TARGET_WEIGHT", {}
            # The PM may request any sell size, including a complete exit. The only
            # quantity guard here is mechanical: never submit more shares than held.
            qty = math.floor(min(qty, max(current_qty, 0.0)))
            if qty < self.settings.min_order_qty: return False, "NOTHING_TO_SELL", {}
            if qty * price < self.settings.min_order_notional: return False, "ORDER_TOO_SMALL", {}

        return True, "APPROVED", {
            "quantity": qty,
            "daily_loss_buy_scale": daily_loss_buy_scale if action == "BUY" else None,
        }

    def record_order(self, symbol: str):
        self.order_times[symbol.upper()].append(time.time())
