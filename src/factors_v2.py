import numpy as np
import pandas as pd


def candidate_v2(df: pd.DataFrame) -> pd.Series:
    """V1 divided by trailing realized volatility; one pre-declared revision."""
    g = df.groupby("symbol", sort=False)["close"]
    mom = g.shift(5) / g.shift(60) - 1.0
    ret = g.pct_change(fill_method=None)
    vol = (ret.groupby(df["symbol"], sort=False)
              .rolling(20, min_periods=15).std()
              .reset_index(level=0, drop=True))
    return mom / vol.clip(lower=1e-4).replace([np.inf, -np.inf], np.nan)

