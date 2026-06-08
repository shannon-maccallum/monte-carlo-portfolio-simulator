from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_UNIVERSE_PATH = Path(__file__).resolve().parents[1] / "data" / "etf_universe.csv"


@dataclass(frozen=True)
class Portfolio:
    """A long-only ETF portfolio with weights summing to one."""

    tickers: tuple[str, ...]
    weights: np.ndarray
    name: str = "Custom Portfolio"

    def __post_init__(self) -> None:
        clean_tickers = tuple(ticker.strip().upper() for ticker in self.tickers)
        weights = np.asarray(self.weights, dtype=float)

        if len(clean_tickers) == 0:
            raise ValueError("Portfolio must contain at least one ETF.")
        if len(clean_tickers) != len(weights):
            raise ValueError("Tickers and weights must have the same length.")
        if len(set(clean_tickers)) != len(clean_tickers):
            raise ValueError("Duplicate tickers are not allowed.")
        if np.any(weights < 0):
            raise ValueError("Weights must be nonnegative.")
        if not np.isfinite(weights).all():
            raise ValueError("Weights must be finite numbers.")
        if weights.sum() <= 0:
            raise ValueError("At least one weight must be positive.")

        normalized = weights / weights.sum()
        object.__setattr__(self, "tickers", clean_tickers)
        object.__setattr__(self, "weights", normalized)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, name: str = "Custom Portfolio") -> "Portfolio":
        required = {"ticker", "weight"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Portfolio table missing columns: {sorted(missing)}")
        return cls(tuple(frame["ticker"]), frame["weight"].to_numpy(), name=name)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"ticker": self.tickers, "weight": self.weights})


def load_etf_universe(path: Path = DEFAULT_UNIVERSE_PATH) -> list[str]:
    universe = pd.read_csv(path)["ticker"].str.upper().tolist()
    return sorted(universe)


def validate_against_universe(portfolio: Portfolio, universe: list[str]) -> None:
    allowed = set(universe)
    invalid = [ticker for ticker in portfolio.tickers if ticker not in allowed]
    if invalid:
        raise ValueError(f"These ETFs are outside the approved universe: {', '.join(invalid)}")
