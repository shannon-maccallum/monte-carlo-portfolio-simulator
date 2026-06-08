from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation import SimulationResult, TRADING_DAYS
from src.time_utils import format_trading_day_years


def maximum_drawdown(paths: pd.DataFrame) -> pd.Series:
    drawdowns = paths / paths.cummax() - 1.0
    return drawdowns.min(axis=0)


def summarize_simulation(
    result: SimulationResult,
    starting_value: float,
    risk_free_rate: float = 0.0,
    var_level: float = 0.05,
) -> dict[str, float]:
    ending_values = result.ending_values
    total_returns = ending_values / starting_value - 1.0
    n_years = (len(result.paths.index) - 1) / TRADING_DAYS
    annual_returns = (1.0 + total_returns).pow(1.0 / n_years) - 1.0
    daily_portfolio_returns = result.portfolio_returns

    path_daily_std = daily_portfolio_returns.std(axis=0)
    annual_volatility = path_daily_std * np.sqrt(TRADING_DAYS)
    downside = daily_portfolio_returns.clip(upper=0).std(axis=0) * np.sqrt(TRADING_DAYS)
    mdd = maximum_drawdown(result.paths)
    var = total_returns.quantile(var_level)
    cvar = total_returns[total_returns <= var].mean()
    sharpe = (annual_returns.mean() - risk_free_rate) / annual_volatility.mean()

    return {
        "mean_ending_value": float(ending_values.mean()),
        "median_ending_value": float(ending_values.median()),
        "expected_annual_return": float(annual_returns.mean()),
        "median_annual_return": float(annual_returns.median()),
        "annual_volatility": float(annual_volatility.mean()),
        "downside_deviation": float(downside.mean()),
        "mean_max_drawdown": float(mdd.mean()),
        "probability_of_loss": float((ending_values < starting_value).mean()),
        "value_at_risk": float(var),
        "conditional_value_at_risk": float(cvar),
        "sharpe_ratio": float(sharpe),
    }


def format_summary_table(metrics: dict[str, float]) -> pd.DataFrame:
    labels = {
        "mean_ending_value": "Mean terminal value",
        "median_ending_value": "Median terminal value",
        "expected_annual_return": "Mean CAGR",
        "median_annual_return": "Median CAGR",
        "annual_volatility": "Annual volatility",
        "downside_deviation": "Downside deviation",
        "mean_max_drawdown": "Mean maximum drawdown",
        "probability_of_loss": "Probability of loss",
        "value_at_risk": "5% VaR",
        "conditional_value_at_risk": "5% CVaR",
        "sharpe_ratio": "Sharpe ratio",
        "exit_probability": "Threshold breach probability",
        "survival_probability": "Survival probability",
        "expected_exit_day": "Average exit time",
        "median_exit_day": "Median exit time",
    }
    hidden_keys = {"expected_exit_time_years", "median_exit_time_years"}
    formatted_values = {
        "expected_exit_day": format_trading_day_years(metrics.get("expected_exit_day")),
        "median_exit_day": format_trading_day_years(metrics.get("median_exit_day")),
    }
    return pd.DataFrame(
        [
            {"metric": labels.get(key, key), "value": formatted_values.get(key, value)}
            for key, value in metrics.items()
            if key not in hidden_keys
        ]
    )
