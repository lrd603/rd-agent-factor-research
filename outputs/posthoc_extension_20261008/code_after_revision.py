"""Sole revision after early-design diagnostics; direction is not performance-flipped."""

import numpy as np
import pandas as pd

from .factor_exploration_initial import turnover_shock_reversal_v1


def turnover_shock_reversal_v2(df: pd.DataFrame) -> pd.Series:
    """Incremental turnover-shock reversal, residualized from plain 5-day reversal."""
    initial = turnover_shock_reversal_v1(df)
    close = df.groupby("symbol", sort=False)["close"]
    reversal = -close.pct_change(5, fill_method=None)

    def residualize(index: pd.Index) -> pd.Series:
        y = initial.loc[index]
        x = reversal.loc[index]
        ok = y.notna() & x.notna()
        out = np.full(len(index), np.nan, dtype=float)
        if ok.sum() < 10 or x[ok].var() == 0:
            return pd.Series(out, index=index)
        beta = ((x[ok] - x[ok].mean()) * (y[ok] - y[ok].mean())).sum() / ((x[ok] - x[ok].mean()) ** 2).sum()
        out[ok.to_numpy()] = np.asarray(y[ok] - y[ok].mean() - beta * (x[ok] - x[ok].mean()), dtype=float)
        return pd.Series(out, index=index)

    return df.groupby("date", sort=False, group_keys=False).apply(
        lambda z: residualize(z.index), include_groups=False).sort_index()
