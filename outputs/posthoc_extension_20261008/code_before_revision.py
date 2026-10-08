"""Initial post-hoc exploration factor. Kept immutable as the pre-revision snapshot."""

import numpy as np
import pandas as pd


def turnover_shock_reversal_v1(df: pd.DataFrame) -> pd.Series:
    """Five-day reversal activated only by an unusually high recent turnover."""
    symbols = df["symbol"]
    close = df.groupby("symbol", sort=False)["close"]
    turnover = df["turnover"].where(df["turnover"] > 0)
    ret5 = close.pct_change(5, fill_method=None)
    recent = (turnover.groupby(symbols, sort=False).rolling(5, min_periods=5).mean()
              .reset_index(level=0, drop=True))
    normal = (turnover.groupby(symbols, sort=False).rolling(60, min_periods=40).mean()
              .reset_index(level=0, drop=True))
    shock = np.log(recent / normal).clip(lower=0)
    return -ret5 * shock
