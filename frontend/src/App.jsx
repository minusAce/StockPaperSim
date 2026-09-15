import {useEffect, useRef, useState} from 'react'

const money = n => `$${Number(n || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`
const pct = n => `${Number(n || 0).toFixed(2)}%`
const escText = v => String(v ?? '')
const confPct = n => `${Math.round(Number(n || 0) * 100)}%`
const parseTs = ts => {
    if (!ts) return null;
    const dt = new Date(ts);
    return Number.isNaN(dt.getTime()) ? null : dt
}
const fmtDate = dt => dt.toLocaleDateString('en-US', {month: 'short', day: 'numeric'})
const fmtDateFull = dt => dt.toLocaleDateString('en-US', {month: 'short', day: 'numeric', year: 'numeric'})
const fmtTime = dt => dt.toLocaleTimeString('en-US', {hour: 'numeric', minute: '2-digit', hour12: true})
const fmtTimeFull = dt => dt.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    hour12: true
})
const prettyReason = s => String(s || '').split('_').filter(Boolean).map(w => w.charAt(0) + w.slice(1).toLowerCase()).join(' ') || '—'
// Strips any enum-style prefix (e.g. "OrderStatus.FILLED" -> "FILLED") before
// humanizing, so both legacy and clean status values render the same way.
const prettyStatus = s => {
    const raw = String(s || '').split('.').pop()
    return raw.split('_').filter(Boolean).map(w => w.charAt(0) + w.slice(1).toLowerCase()).join(' ') || '—'
}


function StaffScene({agents}) {
    return <div
        className="staff-stage"
        data-agent-count={agents.length}
        aria-label={`AI Staff visual stage ready for the PixiJS scene (${agents.length} agents connected)`}
    >
        <div className="staff-stage-status">
            <span>AI STAFF DATA LINK</span>
            <b>{agents.length} AGENTS CONNECTED</b>
            <small>PixiJS scene layer reserved</small>
        </div>
    </div>
}

export default function App() {
    const [status, setStatus] = useState({});
    const [now, setNow] = useState(new Date());
    const [wsState, setWsState] = useState('DISCONNECTED');
    const [lastUpdateAt, setLastUpdateAt] = useState(null);
    const [manualAi, setManualAi] = useState(false);
    const [agents, setAgents] = useState([]);
    const [portfolio, setPortfolio] = useState({positions: []});
    const [candidates, setCandidates] = useState([]);
    const [benchmark, setBenchmark] = useState({symbol: 'S&P 500'});
    const [decisions, setDecisions] = useState([]);
    const [orders, setOrders] = useState([]);
    const [performance, setPerformance] = useState({agents: []});
    const [logs, setLogs] = useState([]);
    const [killSwitchLatched, setKillSwitchLatched] = useState(false);
    const [activeDecision, setActiveDecision] = useState(null);
    const wsRef = useRef(null);
    const reconnectTimerRef = useRef(null);
    const wsPausedRef = useRef(false)

    const refresh = async () => {
        try {
            const urls = ['/api/status', '/api/agents', '/api/portfolio', '/api/candidates', '/api/benchmark', '/api/decisions', '/api/orders', '/api/performance']
            const [s, a, p, c, b, d, o, perf] = await Promise.all(urls.map(u => fetch(u).then(r => r.json())))
            setStatus(s);
            if (s?.kill_switch) setKillSwitchLatched(true)
            setAgents(a);
            setPortfolio(p);
            setCandidates(c);
            setBenchmark(b);
            setDecisions(d);
            setOrders(o);
            setPerformance(perf)
            return s
        } catch {
            return null
        }
    }

    const pushLog = (agent, msg, kind = '') => setLogs(x => [{
        time: new Date().toLocaleTimeString(),
        agent,
        msg,
        kind
    }, ...x].slice(0, 150))

    const connect = () => {
        wsPausedRef.current = false
        if (wsRef.current && [WebSocket.OPEN, WebSocket.CONNECTING].includes(wsRef.current.readyState)) return
        const proto = location.protocol === 'https:' ? 'wss' : 'ws'
        const ws = new WebSocket(`${proto}://${location.host}/ws`);
        wsRef.current = ws
        setWsState('CONNECTING')
        ws.onopen = () => setWsState('CONNECTED')
        ws.onmessage = e => {
            const m = JSON.parse(e.data), d = m.data
            if (m.type === 'heartbeat') {
                setLastUpdateAt(new Date(d?.timestamp ? Number(d.timestamp) * 1000 : Date.now()))
                if (d?.ai_quota) setStatus(prev => ({...prev, ai_quota: d.ai_quota}))
                return
            }
            if (m.type === 'connected') return
            if (['portfolio', 'candidates', 'benchmark', 'market'].includes(m.type)) setLastUpdateAt(new Date())
            if (m.type === 'portfolio') setPortfolio(d)
            if (m.type === 'candidates') setCandidates(d)
            if (m.type === 'benchmark') setBenchmark(d)
            if (m.type === 'decision') {
                setDecisions(x => [d, ...x].slice(0, 80));
                pushLog('RISK', `${d.symbol} ${d.action} ${d.approved ? 'APPROVED' : `REJECTED — ${prettyReason(d.rejection_reason || d.reason)}`}`, 'risk')
            }
            if (m.type === 'trade') {
                setOrders(x => [d, ...x].slice(0, 80));
                pushLog('EXEC', `${d.action} ${d.qty} ${d.symbol} ${d.status || ''}`, 'trade')
            }
            if (m.type === 'trade_update') pushLog('BROKER', `${escText(d.event || 'update')} ${escText(d.order?.symbol || '')}`, 'trade')
            if (m.type === 'agent') {
                const a = d?.agent, body = d?.data || {}
                if (a === 'RISK' && Array.isArray(body.reviews)) {
                    // RiskBatch has no top-level summary — build one readable line per
                    // reviewed symbol instead of dumping (and truncating) the raw JSON.
                    if (body.reviews.length) {
                        body.reviews.forEach(r => pushLog('RISK', `${r?.symbol || '—'}: ${r?.llm_risk || '—'} risk (${confPct(r?.confidence)} conf) — ${r?.reason || 'No reasoning provided.'}`, 'risk'))
                    } else {
                        pushLog('RISK', 'Risk review returned no symbols.', 'risk')
                    }
                } else {
                    const msg = body.summary || body.thesis || body.reason || JSON.stringify(body).slice(0, 260)
                    pushLog(a, msg, 'agent')
                }
            }
            if (m.type === 'system') pushLog('SYS', d.message || JSON.stringify(d), d.level === 'ERROR' ? 'error' : 'system')
        }
        ws.onclose = () => {
            setWsState('DISCONNECTED');
            if (!wsPausedRef.current) reconnectTimerRef.current = window.setTimeout(connect, 2000)
        }
        ws.onerror = () => setWsState('ERROR')
    }

    const post = async (url, body) => {
        try {
            await fetch(url, {
                method: 'POST',
                headers: {'content-type': 'application/json'},
                body: JSON.stringify(body || {})
            })
        } finally {
            await refresh();
            connect()
        }
    }
    const runManualAi = async () => {
        setManualAi(true);
        try {
            const r = await fetch('/api/control/run-ai-test', {method: 'POST'});
            if (!r.ok) {
                const data = await r.json().catch(() => ({}));
                throw new Error(data.message || 'Manual AI run failed')
            }
            await refresh()
        } catch (err) {
            pushLog('SYS', err.message || 'Manual AI run failed', 'error')
        } finally {
            setManualAi(false)
        }
    }
    const navPnlClass = Number(portfolio.daily_pnl || 0) >= 0 ? 'up' : 'down'
    // portfolio.pnl is never sent by the API (only daily_pnl is) — that's why this
    // was frozen at $0.00. Total open P/L is the sum of each position's unrealized P/L.
    const totalPnl = (portfolio.positions || []).reduce((sum, p) => sum + Number(p.unrealized_pl || 0), 0)
    const quota = status.ai_quota || {used: 0, budget: 0, remaining: 0}
    const autopilotOn = Boolean(status.autopilot)

    useEffect(() => {
        let alive = true
        const boot = async () => {
            const s = await refresh()
            if (!alive) return
            connect()
        }
        boot()
        const c = setInterval(() => setNow(new Date()), 1000)
        // Control metrics stay synchronized through the same websocket heartbeat that
        // drives LAST UPDATE; do not run a separate 1-second status poll.
        return () => {
            alive = false;
            clearInterval(c);
            if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
            wsPausedRef.current = true;
            wsRef.current?.close()
        }
    }, [])

    useEffect(() => {
        if (autopilotOn || !wsPausedRef.current) {
            connect()
        } else {
            wsPausedRef.current = true
            if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
            if (wsRef.current && wsRef.current.readyState !== WebSocket.CLOSED) wsRef.current.close()
            wsRef.current = null
        }
        return () => {
        }
    }, [autopilotOn])

    const benchmarkTicker = benchmark?.price ? [{
        symbol: benchmark.symbol,
        last_price: benchmark.price,
        change_pct: benchmark.day_change_pct,
        isBenchmark: true
    }] : []
    const tickerItems = [...benchmarkTicker, ...candidates.slice(0, 15)]
    const tickerLoop = tickerItems.length ? [...tickerItems, ...tickerItems] : []
    const benchmarkClass = benchmark?.direction === 'UP' ? 'up' : benchmark?.direction === 'DOWN' ? 'down' : ''
    const movementClass = value => value == null ? '' : Number(value) > 0 ? 'up' : Number(value) < 0 ? 'down' : ''
    const engineState = status.kill_switch ? 'killed' : (status.running && autopilotOn ? 'live' : 'paused')
    const marketDataKilled = Boolean(status.kill_switch || killSwitchLatched)
    const marketDataClass = marketDataKilled ? 'killed' : wsState === 'CONNECTED' ? 'connected' : wsState === 'CONNECTING' ? 'connecting' : 'offline'

    return <div className="app-shell">
        <header className="topbar">
            <div className="brand">StockPaperSim</div>
            <div className="topbar-center">
                <Metric label="NAV" value={money(portfolio.equity)} className="nav-value"/>
                <Metric label="DAY P/L" value={money(portfolio.daily_pnl)} className={`pnl-value ${navPnlClass}`}/>
                <Metric label="P/L" value={money(totalPnl)} className={`pnl-value ${totalPnl >= 0 ? 'up' : 'down'}`}/>
                <Metric label="ORDERS" value={status.db_counts?.orders ?? 0} className="orders-value"/>
                <Metric label="FILLED" value={performance.filled_orders ?? 0} className="filled-value"/>
            </div>
            <div className="topbar-right">
                <div className={`live-engine ${engineState}`} aria-label={`Engine status: ${engineState}`}>
                    <i/> <span>{engineState === 'killed' ? 'KILLED' : engineState === 'live' ? 'LIVE' : 'PAUSED'}</span>
                </div>
                <div className={`metric topbar-right-metric realtime-status ${marketDataClass}`}
                     title={lastUpdateAt ? `Last update ${lastUpdateAt.toLocaleTimeString()}` : 'Waiting for market data'}
                     aria-label={`Market data: ${wsState}`}>
                    <div className="realtime-main"><i/> <span>MARKET DATA</span></div>
                    <div className="realtime-last">LAST
                        UPDATE <b>{lastUpdateAt ? lastUpdateAt.toLocaleTimeString() : '—'}</b></div>
                </div>
                <div className="metric topbar-right-metric clock"><span>EAST COAST</span>
                    <time dateTime={now.toISOString()}>{now.toLocaleTimeString('en-US', {
                        timeZone: 'America/New_York',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: true
                    })}</time>
                </div>
            </div>
        </header>

        <div className="ticker-bar">
            <div className="ticker-viewport">
                <div className="ticker-track">
                    {tickerLoop.map((c, i) => {
                        const direction = movementClass(c.change_pct);
                        return <div key={`${c.symbol}-${i}`}
                                    className={`ticker ${direction} ${c.isBenchmark ? 'benchmark-ticker' : ''}`}>
                            <b>{c.symbol}</b><small>{c.last_price ? money(c.last_price) : '—'}</small><span
                            className={direction}>{c.change_pct == null ? '—' : `${c.change_pct >= 0 ? '+' : ''}${pct(c.change_pct)}`}</span>
                        </div>
                    })}
                </div>
            </div>
        </div>

        <main className="floor-grid">
            <section className="panel staff">
                <PanelTitle>AI STAFF</PanelTitle>
                <div className="staff-grid">
                    <StaffScene agents={agents}/>
                </div>
            </section>

            <section className="panel market">
                <PanelTitle aside="TOP MOVERS (TODAY)">MARKET MONITOR</PanelTitle>
                <div className="benchmark-banner">
                    <div>
                        <span className="benchmark-label">BENCHMARK · {benchmark.symbol || 'S&P 500'}</span>
                        <strong>{benchmark.price ? money(benchmark.price) : '—'}</strong>
                    </div>
                    <div className={`benchmark-change ${benchmarkClass}`}>
                        <b>{benchmark.day_change_pct == null ? '—' : `${benchmark.day_change_pct >= 0 ? '+' : ''}${pct(benchmark.day_change_pct)}`}</b>

                    </div>
                </div>
                <div className="panel-scroll monitor-table">
                    <div className="table-head monitor-head">
                        <span>SYMBOL</span><span>PRICE</span><span>TODAY</span><span>VOLUME</span><span>SCORE</span>
                    </div>
                    {candidates.slice(0, 18).map(c => {
                        const direction = movementClass(c.change_pct);
                        return <div className={`monitor-row ${direction}`} key={c.symbol}><b>{c.symbol}</b><span
                            className="monitor-price">{c.last_price ? money(c.last_price) : '—'}</span><span
                            className={direction}>{c.change_pct == null ? '—' : `${c.change_pct >= 0 ? '+' : ''}${pct(c.change_pct)}`}</span><span>{c.volume ? Number(c.volume).toLocaleString() : '—'}</span><span>{Number(c.score || 0).toFixed(1)}</span>
                        </div>
                    })}
                </div>
            </section>

            <div className="portfolio-controls-stack">
                <section className="panel portfolio">
                    <PanelTitle>PORTFOLIO</PanelTitle>
                    <div className="stat-grid"><Stat label="CASH" v={money(portfolio.cash)}/><Stat label="BUYING POWER"
                                                                                                   v={money(portfolio.buying_power)}/>
                    </div>
                    <div className="subhead">OPEN POSITIONS</div>
                    <div className="panel-scroll positions-list">
                        <div className="table-head position-head" aria-hidden="true">
                            <span>SYMBOL</span>
                            <span>SHARES</span>
                            <span>PORTFOLIO</span>
                            <span>COST / SHARE</span>
                            <span>G/L / SHARE</span>
                            <span>G/L</span>
                        </div>
                        {(portfolio.positions || []).map(p => {
                            const qty = Number(p.qty || 0)
                            const currentPrice = Number(p.current_price || 0)
                            const costPerShare = Number(p.avg_entry_price || 0)
                            const glPerShare = qty < 0 ? costPerShare - currentPrice : currentPrice - costPerShare
                            const glClass = glPerShare >= 0 ? 'up' : 'down'
                            const totalGlClass = Number(p.unrealized_pl || 0) >= 0 ? 'up' : 'down'
                            return <div className="pos-row" key={p.symbol}>
                                <b>{p.symbol}</b>
                                <span>{Math.abs(qty).toLocaleString()}</span>
                                <span>{money(p.market_value)}</span>
                                <span>{money(costPerShare)}</span>
                                <span className={glClass}>{money(glPerShare)}</span>
                                <span className={totalGlClass}>{money(p.unrealized_pl)}</span>
                            </div>
                        })}
                        {!(portfolio.positions || []).length && <div className="empty">NO OPEN POSITIONS</div>}
                    </div>
                </section>

                <section className="panel controls">
                    <PanelTitle>DESK CONTROLS</PanelTitle>
                    <div className="budget-card">
                        <div><span>AI BUDGET</span><b>{quota.remaining}/{quota.budget}</b></div>
                        <div className="budget-track"><i
                            style={{width: `${quota.budget ? Math.max(0, Math.min(100, (quota.remaining / quota.budget) * 100)) : 0}%`}}/>
                        </div>
                    </div>
                    <div className="control-stack">
                        <button className={autopilotOn ? 'primary' : ''}
                                onClick={() => post('/api/control/autopilot', {enabled: !autopilotOn})}>{autopilotOn ? 'AUTOPILOT ON' : 'AUTOPILOT OFF'}</button>
                        <button onClick={runManualAi}
                                disabled={manualAi || Boolean(status.kill_switch)}>{manualAi ? 'RUNNING AI…' : 'RUN AI NOW'}</button>
                        <button className="danger" onClick={() => {
                            setKillSwitchLatched(true);
                            post('/api/control/kill-switch', {active: true})
                        }}
                                disabled={Boolean(status.kill_switch)}>{status.kill_switch ? 'KILL SWITCH ACTIVE' : 'KILL SWITCH'}</button>
                    </div>
                </section>
            </div>

            <section className="panel squawk">
                <PanelTitle>DESK CHATTER</PanelTitle>
                <div className="panel-scroll squawk-log">
                    {logs.map((l, i) => <div className={`squawk-line ${l.kind}`} key={i}>
                        <time>{l.time}</time>
                        <b>{l.agent}</b><span>{l.msg}</span></div>)}
                    {!logs.length && <div className="empty">NO CHATTER YET</div>}
                </div>
            </section>

            <section className="panel decisions">
                <PanelTitle>DECISION BOOK</PanelTitle>
                <div className="panel-scroll scroll-pane standalone-table">
                    <div className="table-head decision-head">
                        <span>TIME</span><span>DECISION</span><span>SYMBOL</span><span>THESIS</span></div>
                    {decisions.slice(0, 18).map(d =>
                        <div className="decision-row" key={d.id}>
                            <TimeCell ts={d.timestamp}/>
                            <span
                                className={`decision-action ${(d.action || '').toLowerCase()}`}>{d.action || '—'}</span>
                            <span>{d.symbol || '—'}</span>
                            <button type="button" className="thesis-btn" onClick={() => setActiveDecision(d)}
                                    aria-haspopup="dialog"
                                    aria-label={`View thesis for ${d.symbol || 'this decision'}`}>
                                VIEW<span aria-hidden="true">›</span>
                            </button>
                        </div>
                    )}
                    {!decisions.length && <div className="empty">NO DECISIONS YET</div>}
                </div>
            </section>

            <section className="panel broker-orders">
                <PanelTitle>BROKER ORDERS</PanelTitle>
                <div className="panel-scroll scroll-pane standalone-table">
                    <div className="table-head order-head">
                        <span>TIME</span><span>SYMBOL</span><span>SIDE</span><span>QTY</span><span>STATUS</span></div>
                    {orders.slice(0, 18).map((o, i) => <div className="order-row" key={o.id || i}><TimeCell
                        ts={o.timestamp}/><b>{o.symbol || '—'}</b><span
                        className={`order-side ${(o.action || '').toLowerCase()}`}>{o.action || '—'}</span><span>{Number(o.qty || 0).toLocaleString()}</span><span
                        className="order-status">{prettyStatus(o.status)}</span>
                    </div>)}
                    {!orders.length && <div className="empty">NO ORDERS YET</div>}
                </div>
            </section>

        </main>

        {activeDecision && <DecisionModal decision={activeDecision} onClose={() => setActiveDecision(null)}/>}
    </div>
}

function Metric({label, value, className = ''}) {
    return <div className="metric"><span>{label}</span><b className={className}>{value}</b></div>
}

function Stat({label, v, cls = ''}) {
    return <div className="stat"><span>{label}</span><b className={cls}>{v}</b></div>
}

function PanelTitle({children, aside = ''}) {
    return <div className="panel-title"><span>{children}</span>
        <div className="panel-title-actions">{aside && <em>{aside}</em>}<i/></div>
    </div>
}

function TimeCell({ts}) {
    const dt = parseTs(ts)
    if (!dt) return <span className="cell-time"><b>—</b></span>
    return <span className="cell-time"><b>{fmtDate(dt)}</b><small>{fmtTime(dt)}</small></span>
}

function DecisionModal({decision, onClose}) {
    useEffect(() => {
        const onKey = e => {
            if (e.key === 'Escape') onClose()
        }
        window.addEventListener('keydown', onKey)
        return () => window.removeEventListener('keydown', onKey)
    }, [onClose])

    const d = decision
    const dt = parseTs(d.timestamp)
    const action = (d.action || '—').toUpperCase()
    const thesisText = d.thesis || d.reason || d.rejection_reason || 'No thesis was recorded for this decision.'

    return <div className="modal-overlay" onClick={onClose}>
        <div className="modal-card decision-modal" role="dialog" aria-modal="true"
             aria-labelledby="decision-modal-symbol" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={onClose} aria-label="Close">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     strokeWidth="2.5" strokeLinecap="round">
                    <line x1="4" y1="4" x2="20" y2="20"/>
                    <line x1="20" y1="4" x2="4" y2="20"/>
                </svg>
            </button>
            <div className="decision-modal-header">
                <span className={`decision-action-badge ${action.toLowerCase()}`}>{action}</span>
                <h3 id="decision-modal-symbol">{d.symbol || '—'}</h3>
            </div>
            <div className="decision-modal-meta">
                <div><span>WHEN</span>{dt ? <b>{fmtDateFull(dt)} {fmtTimeFull(dt)}</b> : <b>—</b>}
                </div>
                <div><span>CONFIDENCE</span><b>{confPct(d.confidence)}</b></div>
            </div>
            <div className="decision-modal-thesis">
                <span className="decision-modal-label">THESIS</span>
                <p>{thesisText}</p>
            </div>
        </div>
    </div>
}