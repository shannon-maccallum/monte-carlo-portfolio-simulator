from __future__ import annotations

import pandas as pd


def trading_days_to_years(days: float, trading_days_per_year: int = 252) -> float:
    return days / trading_days_per_year


def format_trading_day_years(days: float, trading_days_per_year: int = 252) -> str:
    if pd.isna(days):
        return "N/A"
    years = trading_days_to_years(days, trading_days_per_year)
    return f"{days:,.0f} days ({years:.1f} years)"
