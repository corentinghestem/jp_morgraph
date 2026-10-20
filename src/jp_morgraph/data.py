# src/jp_morgraph/data.py
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")
BENCH = "SPY"

def download_daily(tickers: list[str], force: bool = False) -> pd.DataFrame:
    """Un seul appel groupé pour tous les titres, puis cache Parquet (évite les 429)."""
    path = DATA_DIR / "daily.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    DATA_DIR.mkdir(exist_ok=True)
    df = yf.download(tickers + [BENCH], period="3y", interval="1d",
                     auto_adjust=True, progress=False)  # colonnes : (Open/Close/..., ticker)
    df.to_parquet(path)
    return df

def weekly_close(daily: pd.DataFrame) -> pd.DataFrame:
    # dernier cours de chaque semaine, daté du vendredi
    return daily["Close"].resample("W-FRI").last().dropna(how="all")

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)

def indicators(close: pd.Series) -> pd.DataFrame:
    ret = close.pct_change()
    return pd.DataFrame({
        "ret_4w": close.pct_change(4),
        "ret_12w": close.pct_change(12),
        "sma_10": close.rolling(10).mean(),
        "sma_40": close.rolling(40).mean(),
        "rsi_14": rsi(close),
        "vol_12w": ret.rolling(12).std() * np.sqrt(52),
        "drawdown": close / close.cummax() - 1,
    })