from __future__ import annotations

import asyncio
from typing import Any

from app.agents.llm import Agent, LLMRequestGate, OpenRouterQuotaExceeded
from app.config import Settings
from app.schemas import FinanceReport, MacroReport, NewsReport, PMDecision, QuantReport, RiskBatch, ScoutResponse
from app.storage.db import Database


class AgentTeam:
    def __init__(self, settings: Settings, db: Database):
        p = settings.prompt_dir
        self.quota = LLMRequestGate(settings, db)
        self.agents = {
            "SCOUT": Agent("SCOUT", "Opportunity Hunter", p / "scout.yaml", settings.model_scout, settings, db,
                           self.quota),
            "MACRO": Agent("MACRO", "Market Regime", p / "macro.yaml", settings.model_macro, settings, db, self.quota),
            "QUANT": Agent("QUANT", "Technical Analysis", p / "quant.yaml", settings.model_quant, settings, db,
                           self.quota),
            "NEWS": Agent("NEWS", "Catalysts", p / "news.yaml", settings.model_news, settings, db, self.quota),
            "FINANCE": Agent("FINANCE", "Company Context", p / "finance.yaml", settings.model_finance, settings, db,
                             self.quota),
            "PM": Agent("PM", "Portfolio Manager", p / "pm.yaml", settings.model_pm, settings, db, self.quota),
            "RISK": Agent("RISK", "Risk Review", p / "risk.yaml", settings.model_risk, settings, db, self.quota),
        }

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                "name": a.name,
                "role": a.role,
                "model": a.model,
                "requested_model": a.last_requested_model,
                "provider": a.last_served_provider,
                "status": a.status,
                "last_run": a.last_run,
                "last_summary": a.last_summary,
                "last_error": a.last_error,
            }
            for a in self.agents.values()
        ]

    def quota_status(self) -> dict[str, Any]:
        return self.quota.snapshot()

    def set_enabled(self, enabled: bool) -> None:
        for agent in self.agents.values():
            agent.set_enabled(enabled)

    def estimated_cycle_calls(self, candidates: int) -> int:
        # One request per agent: SCOUT + MACRO + (QUANT/NEWS/FINANCE x N) + PM + RISK.
        # A wave is capped at two fully researched symbols so the default ten-request
        # wave fits the 50-request daily budget exactly across five session waves.
        candidate_count = min(2, max(1, int(candidates)))
        return 4 + (3 * candidate_count)

    async def scout(self, candidates):
        return await self.agents["SCOUT"].run({"candidates": candidates}, ScoutResponse)

    async def quant(self, symbol, features):
        return await self.agents["QUANT"].run({"symbol": symbol, "features": features}, QuantReport)

    async def macro(self, context):
        return await self.agents["MACRO"].run(context, MacroReport)

    async def news(self, symbol, news):
        return await self.agents["NEWS"].run({"symbol": symbol, "news": news}, NewsReport)

    async def finance(self, symbol, context):
        return await self.agents["FINANCE"].run({"symbol": symbol, "context": context}, FinanceReport)

    async def research_symbol(self, symbol, features, news):
        tasks = {
            "QUANT": asyncio.create_task(self.quant(symbol, features)),
            "NEWS": asyncio.create_task(self.news(symbol, news)),
            "FINANCE": asyncio.create_task(self.finance(symbol, {"features": features, "news": news})),
        }
        reports: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}

        # Research remains concurrent for throughput, but OpenRouter daily-quota
        # exhaustion is authoritative. As soon as one task receives that 429, cancel
        # every sibling task so they cannot take later turns and generate a cascade of
        # identical quota errors.
        task_to_name = {task: name for name, task in tasks.items()}
        done_list: list[asyncio.Task] = []
        pending = set(tasks.values())
        quota_error = None

        # Watch the concurrent research tasks incrementally. FIRST_EXCEPTION lets us
        # notice a provider-level quota failure as soon as that task fails instead of
        # waiting for the other research calls to finish. Non-quota errors do not stop
        # the remaining research; only the authoritative OpenRouter quota error does.
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_EXCEPTION)
            done_list.extend(done)
            for task in done:
                if task.cancelled():
                    continue
                try:
                    exc = task.exception()
                except asyncio.CancelledError:
                    continue
                if isinstance(exc, OpenRouterQuotaExceeded):
                    quota_error = exc
                    break
            if quota_error is not None:
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                pending.clear()
                # Consume exceptions from already-completed siblings so asyncio does not
                # report them as unhandled when the cycle exits immediately.
                for task in done_list:
                    if not task.cancelled():
                        try:
                            task.exception()
                        except asyncio.CancelledError:
                            pass
                raise quota_error

        # Build a stable name -> task result map regardless of which task completed first.
        for task in done_list:
            name = task_to_name[task]
            result = None
            if not task.cancelled():
                try:
                    result = task.result()
                except Exception as exc:
                    result = exc
            if isinstance(result, Exception):
                errors[name] = str(result)
                continue
            reports[name] = result.model_dump()

        # A trade candidate is only as good as its complete research packet.
        # Missing any research agent is a hard failure for that symbol; an explicit
        # INSUFFICIENT report is fine because that is valid agent output.
        if errors:
            detail = "; ".join(f"{name}: {message}" for name, message in sorted(errors.items()))
            raise RuntimeError(f"Research incomplete for {symbol}: {detail}")
        return {"symbol": symbol, "features": features, "news": news[:5], "reports": reports}

    async def pm(self, payload):
        return await self.agents["PM"].run(payload, PMDecision)

    async def risk(self, proposals, portfolio):
        return await self.agents["RISK"].run({"proposals": proposals, "portfolio": portfolio}, RiskBatch)
