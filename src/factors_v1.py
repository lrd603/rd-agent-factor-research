import pandas as pd


def candidate_v1(df: pd.DataFrame) -> pd.Series:
    """Medium-term momentum excluding the latest 5 sessions."""
    g = df.groupby("symbol", sort=False)["close"]
    return g.shift(5) / g.shift(60) - 1.0

