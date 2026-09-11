from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_features(bars: list[Any], daily_dollar_volume: float | None = None) -> dict[str, Any]:
    if not bars:
        return {"data_quality": "INSUFFICIENT", "last_price": None, "daily_dollar_volume": 0.0, "dollar_volume": 0.0,
                "minute_dollar_volume": 0.0}
    rows = []
    for b in bars:
        rows.append(
            {"ts": getattr(b, "timestamp", None), "open": float(b.open), "high": float(b.high), "low": float(b.low),
             "close": float(b.close), "volume": float(b.volume)})
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    close = df["close"];
    volume = df["volume"]
    ret_1 = float(close.pct_change(1).iloc[-1]) if len(df) > 1 else 0.0
    ret_5 = float(close.pct_change(5).iloc[-1]) if len(df) > 5 else ret_1
    ret_20 = float(close.pct_change(20).iloc[-1]) if len(df) > 20 else ret_5
    sma20 = float(close.rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(close.mean())
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(df) >= 50 else sma20
    vol_avg20 = float(volume.rolling(20).mean().iloc[-1]) if len(df) >= 20 else float(volume.mean())
    volume_ratio = float(volume.iloc[-1] / vol_avg20) if vol_avg20 else 0.0
    tr = pd.concat([(df.high - df.low), (df.high - df.close.shift()).abs(), (df.low - df.close.shift()).abs()],
                   axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(df) >= 14 else float(tr.mean())
    ret_series = close.pct_change().dropna()
    vol_annualized = float(ret_series.std() * np.sqrt(390)) if len(ret_series) > 2 else 0.0
    delta = close.diff();
    gain = delta.clip(lower=0).rolling(14).mean();
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    # pd.isna is the correct NaN check here: `x in (0, np.nan)` never matches a
    # real NaN (NaN != NaN under ==), so that comparison was a silent no-op.
    # It happened not to change the final result because np.isfinite(rs) below
    # already routes NaN to the same rsi=100.0 fallback, but spelling out the
    # actual check avoids relying on that coincidence.
    last_loss = loss.iloc[-1]
    rs = gain.iloc[-1] / last_loss if not pd.isna(last_loss) and last_loss != 0 else np.inf
    rsi = float(100 - (100 / (1 + rs))) if np.isfinite(rs) else 100.0
    last = float(close.iloc[-1])
    minute_dollar_volume = float(last * volume.iloc[-1])
    return {
        "data_quality": "GOOD" if len(df) >= 50 else "LIMITED",
        "bars": len(df), "last_price": last, "return_1m": ret_1 * 100, "return_5m": ret_5 * 100,
        "return_20m": ret_20 * 100,
        "sma20": sma20, "sma50": sma50, "above_sma20": last > sma20, "above_sma50": last > sma50,
        "volume": float(volume.iloc[-1]), "volume_ratio": volume_ratio,
        "daily_dollar_volume": float(daily_dollar_volume or 0.0),
        "dollar_volume": float(daily_dollar_volume or 0.0),
        "minute_dollar_volume": minute_dollar_volume,
        "atr14": atr14, "atr_pct": (atr14 / last * 100) if last else 0.0, "annualized_volatility": vol_annualized,
        "rsi14": rsi,
        "trend_score": float(np.clip(((last / sma20 - 1) * 4 + (last / sma50 - 1) * 3 + ret_5 * 2), -1, 1)),
    }
