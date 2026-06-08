from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulation import TRADING_DAYS
from src.time_utils import trading_days_to_years


RISK_PROFILE_RULES = {
    "Conservative": {"max_loss": 0.10, "max_drawdown": 0.15},
    "Balanced": {"max_loss": 0.18, "max_drawdown": 0.25},
    "Growth": {"max_loss": 0.30, "max_drawdown": 0.40},
}


@dataclass(frozen=True)
class ExitTimeResult:
    exit_occurred: pd.Series
    first_exit_day: pd.Series
    rule: dict[str, float]

    def summary(self) -> dict[str, float]:
        exit_days = self.first_exit_day.dropna()
        return {
            "exit_probability": float(self.exit_occurred.mean()),
            "survival_probability": float(1.0 - self.exit_occurred.mean()),
            "expected_exit_day": float(exit_days.mean()) if not exit_days.empty else np.nan,
            "median_exit_day": float(exit_days.median()) if not exit_days.empty else np.nan,
            "expected_exit_time_years": float(trading_days_to_years(exit_days.mean(), TRADING_DAYS))
            if not exit_days.empty
            else np.nan,
            "median_exit_time_years": float(trading_days_to_years(exit_days.median(), TRADING_DAYS))
            if not exit_days.empty
            else np.nan,
        }


def calculate_exit_times(paths: pd.DataFrame, risk_profile: str = "Balanced") -> ExitTimeResult:
    if risk_profile not in RISK_PROFILE_RULES:
        raise ValueError(f"Unknown risk profile: {risk_profile}")

    rule = RISK_PROFILE_RULES[risk_profile]
    start = paths.iloc[0]
    running_peak = paths.cummax()
    loss_from_start = paths / start - 1.0
    drawdown = paths / running_peak - 1.0

    breach = (loss_from_start <= -rule["max_loss"]) | (drawdown <= -rule["max_drawdown"])
    exit_occurred = breach.any(axis=0)
    first_exit_day = breach.idxmax(axis=0).where(exit_occurred, other=np.nan)

    return ExitTimeResult(exit_occurred=exit_occurred, first_exit_day=first_exit_day, rule=rule)
