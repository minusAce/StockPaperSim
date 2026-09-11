# StockPaperSim

## Current runtime architecture

- React performs one REST bootstrap on page load.
- Live dashboard state is delivered through `/ws` via the EventBus.
- WebSocket sends a `connected` message immediately and a heartbeat every ~20 seconds when idle.
- Market bar events, trade updates, decisions, agent events, candidates, portfolio snapshots, and benchmark snapshots
  are pushed over WebSocket.
- Browser REST polling is intentionally disabled.
- Server-side portfolio refresh is paced at 10 seconds while autopilot is active.
- Server-side benchmark refresh is paced at 30 seconds while autopilot is active.
- Alpaca IEX live subscriptions are capped at 30 symbols.

## Temporary AI test control

`POST /api/control/run-ai-test` runs one manual AI analysis pass using the configured AI pipeline, without enabling
autopilot and without executing orders. The dashboard exposes a temporary `RUN AI NOW` button for testing. Remove this
endpoint/button when manual validation is complete.

## Model configuration

Every agent's model comes from configuration, never from a hardcoded model ID in the code. Each agent resolves its
primary model as its own `MODEL_<AGENT>` override (e.g. `MODEL_SCOUT`) if set, else `MODEL_DEFAULT`. If that primary
model fails after its own retry attempts, the agent tries `MODEL_FALLBACK` once, but only when the fallback is a
different model than the primary — so an agent with no override and a fallback equal to the default tries exactly one
model, while an agent with its own override automatically gets the shared default as a safety net. All three settings
must be free OpenRouter model IDs (a `:free` suffix, or `openrouter/free`).

## Runtime database and AI session behavior

Docker Compose runs the application against the PostgreSQL service (`DATABASE_URL=postgresql+psycopg://...`). The
application uses PostgreSQL in Docker, persisted in the `trading_floor_pg` Docker volume. There is no checked-in SQLite
trading database.

The scanner uses a dynamically fetched S&P 500 constituent universe as its stable core. Membership is refreshed every 24
hours from the configured external source; no S&P ticker list is hardcoded. Alpaca most-active/gainer/loser screeners
remain a dynamic discovery overlay, so new stocks outside the index can still enter the funnel. The scanner uses
multi-symbol Alpaca snapshots to apply the liquidity gate (default $50K dollar volume) before historical feature work or
AI research, then enriches only the deterministic shortlist. The default AI session is five scheduled waves of two fully
researched candidates (10 AI requests per wave), and market events queue priority candidates for the next available
wave. The PM receives a soft activity target of four successful paper BUY entries for the session; this never overrides
confidence, liquidity, cash, position, order-count, daily-loss, or other hard risk controls.
