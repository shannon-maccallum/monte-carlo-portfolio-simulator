from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from src.portfolio import Portfolio


TRADING_DAYS = 252


def annualize_daily_simple_return(daily_return: np.ndarray | float, trading_days: int = TRADING_DAYS) -> np.ndarray | float:
    """Convert a daily simple return to its compounded annual equivalent."""
    return (1.0 + daily_return) ** trading_days - 1.0


class ReturnModel(Protocol):
    name: str

    def simulate_asset_returns(
        self,
        n_steps: int,
        n_simulations: int,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """Return array with shape: simulations x steps x assets."""


def validate_portfolio_model_alignment(portfolio: Portfolio, model: ReturnModel) -> None:
    if not np.isclose(portfolio.weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.0.")
    if np.any(portfolio.weights < 0) or np.any(portfolio.weights > 1):
        raise ValueError("Portfolio weights must be decimal allocations between 0 and 1.")

    model_asset_names = getattr(model, "asset_names", None)
    if model_asset_names is not None and tuple(model_asset_names) != portfolio.tickers:
        raise ValueError(
            "Portfolio tickers must be in the same order as model.asset_names. "
            f"Portfolio order: {portfolio.tickers}; model order: {tuple(model_asset_names)}"
        )


@dataclass(frozen=True)
class MultivariateNormalModel:
    """Daily multivariate normal model estimated from historical simple returns."""

    mean: np.ndarray
    covariance: np.ndarray
    asset_names: tuple[str, ...]
    name: str = "Multivariate normal returns"

    @classmethod
    def fit(cls, returns: pd.DataFrame) -> "MultivariateNormalModel":
        return cls(
            mean=returns.mean().to_numpy(),
            covariance=returns.cov().to_numpy(),
            asset_names=tuple(returns.columns),
        )

    def annualized_mean(self) -> np.ndarray:
        """Annualized drift shown for diagnostics; simulation still uses daily mean."""
        return annualize_daily_simple_return(self.mean)

    def simulate_asset_returns(
        self,
        n_steps: int,
        n_simulations: int,
        random_seed: int | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(random_seed)
        draws = rng.multivariate_normal(self.mean, self.covariance, size=(n_simulations, n_steps))
        return np.clip(draws, -0.95, None)


@dataclass(frozen=True)
class SimulationResult:
    portfolio: Portfolio
    paths: pd.DataFrame
    portfolio_returns: pd.DataFrame
    model_name: str
    time_step_days: int = 1

    @property
    def ending_values(self) -> pd.Series:
        return self.paths.iloc[-1]


def horizon_to_steps(years: float, trading_days: int = TRADING_DAYS) -> int:
    if years <= 0:
        raise ValueError("Investment horizon must be positive.")
    return max(1, int(round(years * trading_days)))


def run_monte_carlo(
    portfolio: Portfolio,
    model: ReturnModel,
    starting_value: float = 100_000,
    years: float = 5,
    n_simulations: int = 10_000,
    random_seed: int | None = None,
) -> SimulationResult:
    if starting_value <= 0:
        raise ValueError("Starting value must be positive.")
    if n_simulations <= 0:
        raise ValueError("Number of simulations must be positive.")

    validate_portfolio_model_alignment(portfolio, model)
    n_steps = horizon_to_steps(years)
    asset_returns = model.simulate_asset_returns(n_steps, n_simulations, random_seed)
    if asset_returns.shape[2] != len(portfolio.weights):
        raise ValueError("Simulated asset return dimension must match number of portfolio weights.")

    portfolio_returns = asset_returns @ portfolio.weights
    growth = np.cumprod(1.0 + portfolio_returns, axis=1)
    path_values = np.column_stack([np.full(n_simulations, starting_value), starting_value * growth])

    index = pd.RangeIndex(0, n_steps + 1, name="day")
    columns = pd.RangeIndex(1, n_simulations + 1, name="simulation")
    return_index = pd.RangeIndex(1, n_steps + 1, name="day")

    return SimulationResult(
        portfolio=portfolio,
        paths=pd.DataFrame(path_values.T, index=index, columns=columns),
        portfolio_returns=pd.DataFrame(portfolio_returns.T, index=return_index, columns=columns),
        model_name=model.name,
    )
