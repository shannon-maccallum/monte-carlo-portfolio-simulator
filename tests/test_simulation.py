import numpy as np
import pandas as pd
import pytest

from src.exit_time import calculate_exit_times
from src.plotting import plot_survival_curve
from src.portfolio import Portfolio
from src.risk_metrics import format_summary_table
from src.simulation import MultivariateNormalModel, annualize_daily_simple_return, run_monte_carlo


class ConstantReturnModel:
    name = "constant daily return"

    def __init__(self, daily_return):
        self.daily_return = daily_return

    def simulate_asset_returns(self, n_steps, n_simulations, random_seed=None):
        return np.full((n_simulations, n_steps, 1), self.daily_return)


def test_monte_carlo_shape_is_steps_plus_initial_value():
    returns = pd.DataFrame(
        {
            "VTI": [0.001, -0.002, 0.003, 0.001],
            "BND": [0.0002, 0.0001, -0.0001, 0.0003],
        }
    )
    model = MultivariateNormalModel.fit(returns)
    portfolio = Portfolio(("VTI", "BND"), np.array([0.6, 0.4]))

    result = run_monte_carlo(portfolio, model, starting_value=1000, years=1, n_simulations=25, random_seed=7)

    assert result.paths.shape == (253, 25)
    assert result.portfolio_returns.shape == (252, 25)
    assert (result.paths.iloc[0] == 1000).all()


def test_monte_carlo_rejects_mismatched_portfolio_and_model_order():
    returns = pd.DataFrame(
        {
            "VTI": [0.001, -0.002, 0.003, 0.001],
            "BND": [0.0002, 0.0001, -0.0001, 0.0003],
        }
    )
    model = MultivariateNormalModel.fit(returns)
    portfolio = Portfolio(("BND", "VTI"), np.array([0.4, 0.6]))

    with pytest.raises(ValueError, match="same order as model.asset_names"):
        run_monte_carlo(portfolio, model, starting_value=1000, years=1, n_simulations=25, random_seed=7)


def test_daily_returns_are_compounded_at_daily_frequency():
    portfolio = Portfolio(("VTI",), np.array([1.0]))
    model = ConstantReturnModel(daily_return=0.01)

    result = run_monte_carlo(portfolio, model, starting_value=1000, years=1, n_simulations=1, random_seed=7)

    assert result.ending_values.iloc[0] == pytest.approx(1000 * (1.01 ** 252))


def test_daily_drift_annualization_is_compounded_once():
    assert annualize_daily_simple_return(0.01) == pytest.approx(1.01 ** 252 - 1)


def test_exit_times_detect_threshold_breach():
    paths = pd.DataFrame(
        {
            1: [100, 95, 80],
            2: [100, 102, 103],
        },
        index=[0, 1, 2],
    )

    result = calculate_exit_times(paths, risk_profile="Conservative")

    assert result.exit_occurred[1]
    assert not result.exit_occurred[2]
    assert result.first_exit_day[1] == 2


def test_exit_summary_includes_day_scale_statistics():
    paths = pd.DataFrame(
        {
            1: [100, 95, 80],
            2: [100, 99, 78],
            3: [100, 101, 102],
        },
        index=[0, 1, 2],
    )

    summary = calculate_exit_times(paths, risk_profile="Conservative").summary()

    assert summary["expected_exit_day"] == pytest.approx(2.0)
    assert summary["median_exit_day"] == pytest.approx(2.0)


def test_summary_table_formats_exit_times_as_days_and_years():
    table = format_summary_table(
        {
            "expected_exit_day": 3580,
            "expected_exit_time_years": 3580 / 252,
            "median_exit_day": 3484,
            "median_exit_time_years": 3484 / 252,
        }
    )

    values = dict(zip(table["metric"], table["value"], strict=False))
    assert values["Average exit time"] == "3,580 days (14.2 years)"
    assert values["Median exit time"] == "3,484 days (13.8 years)"
    assert "Expected first-exit time, years" not in values


def test_survival_curve_tracks_probability_tau_greater_than_t():
    first_exit_day = pd.Series([2.0, np.nan, 4.0])

    fig = plot_survival_curve(first_exit_day, horizon_day=4)

    assert fig.data[0].y == pytest.approx([1.0, 1.0, 2 / 3, 2 / 3, 1 / 3])
    assert fig.data[0].customdata == pytest.approx([0, 1 / 252, 2 / 252, 3 / 252, 4 / 252])
    assert "days" in fig.data[0].hovertemplate
    assert "years" in fig.data[0].hovertemplate
