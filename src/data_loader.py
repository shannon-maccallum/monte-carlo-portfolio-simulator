from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf


def download_adjusted_prices(
    tickers: list[str] | tuple[str, ...],
    start: str | date = "2015-01-01",
    end: str | date | None = None,
) -> pd.DataFrame:
    """Download adjusted ETF prices from Yahoo Finance."""
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        raise ValueError("Yahoo Finance returned no price data.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data.rename("Close").to_frame()
        prices.columns = list(tickers)

    prices = prices.dropna(axis=1, how="all").ffill().dropna()
    missing = sorted(set(tickers).difference(prices.columns))
    if missing:
        raise ValueError(f"Missing Yahoo Finance data for: {', '.join(missing)}")
    return prices.loc[:, list(tickers)]


def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough price history to compute returns.")
    return returns
