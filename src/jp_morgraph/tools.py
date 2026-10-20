# src/trading_floor/tools.py
import zlib
import numpy as np
import pandas as pd
from langchain_core.tools import tool


def _series(ticker: str, as_of: str) -> pd.Series:
    """Cours hebdomadaires factices, tronqués à as_of (aucune donnée du futur)."""
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    prices = 100 * np.cumprod(1 + rng.normal(0.002, 0.03, 156))
    idx = pd.date_range(end="2026-10-09", periods=156, freq="W-FRI")
    s = pd.Series(prices, index=idx)
    return s[s.index <= as_of]


@tool
def get_price_history(ticker: str, as_of: str, weeks: int = 12) -> list[dict]:
    """Renvoie les derniers cours de clôture hebdomadaires d'un titre,
    jusqu'à la date as_of (AAAA-MM-JJ)."""
    s = _series(ticker, as_of).tail(weeks)
    return [{"date": d.strftime("%Y-%m-%d"), "close": round(float(p), 2)}
            for d, p in s.items()]


@tool
def compute_indicators(ticker: str, as_of: str) -> dict:
    """Calcule les indicateurs techniques d'un titre à la date as_of (AAAA-MM-JJ) :
    rendements 4 et 12 semaines, moyennes mobiles 10 et 40 semaines, RSI, volatilité, drawdown."""
    s = _series(ticker, as_of)
    delta = s.diff()
    gain = delta.clip(lower=0).tail(14).mean()
    loss = (-delta.clip(upper=0)).tail(14).mean()
    return {
        "ret_4w": round(float(s.iloc[-1] / s.iloc[-5] - 1), 4),
        "ret_12w": round(float(s.iloc[-1] / s.iloc[-13] - 1), 4),
        "ma_10w": round(float(s.tail(10).mean()), 2),
        "ma_40w": round(float(s.tail(40).mean()), 2),
        "rsi_14": round(float(100 - 100 / (1 + gain / loss)), 1),
        "vol_12w": round(float(s.pct_change().tail(12).std()), 4),
        "drawdown": round(float(s.iloc[-1] / s.cummax().iloc[-1] - 1), 4),
    }