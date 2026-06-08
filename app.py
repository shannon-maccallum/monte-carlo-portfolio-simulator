from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import compute_simple_returns, download_adjusted_prices
from src.exit_time import RISK_PROFILE_RULES, calculate_exit_times
from src.plotting import (
    plot_ending_distribution,
    plot_exit_distribution,
    plot_percentile_bands,
    plot_sample_paths,
    plot_survival_curve,
)
from src.portfolio import Portfolio, load_etf_universe, validate_against_universe
from src.risk_metrics import format_summary_table, summarize_simulation
from src.simulation import MultivariateNormalModel, run_monte_carlo
from src.time_utils import format_trading_day_years


st.set_page_config(page_title="Monte Carlo ETF Portfolio Simulator", layout="wide")

DEFAULT_HISTORICAL_START_DATE = "2015-01-01"
DEFAULT_N_SIMULATIONS = 10_000
DEFAULT_RANDOM_SEED = 42


@st.cache_data(show_spinner=False)
def cached_prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return download_adjusted_prices(tickers, start=start)


def build_portfolio_input(universe: list[str]) -> Portfolio:
    st.sidebar.header("Portfolio Inputs")
    selected = st.sidebar.multiselect(
        "ETF tickers",
        options=universe,
        default=["BND", "VTI", "VEA"],
    )
    if not selected:
        st.stop()

    raw_weights = []
    equal_weight = 1.0 / len(selected)
    for ticker in selected:
        raw_weights.append(
            st.sidebar.number_input(
                f"{ticker} decimal weight",
                min_value=0.0,
                max_value=1.0,
                value=equal_weight,
                step=0.01,
                format="%.4f",
            )
        )

    weight_sum = sum(raw_weights)
    st.sidebar.caption(f"Weight sum: {weight_sum:.4f}")
    if abs(weight_sum - 1.0) > 0.001:
        st.sidebar.error("Weights must be decimals that sum to 1.0, e.g. 0.10, 0.10, 0.10, 0.70.")
        st.stop()

    portfolio = Portfolio(tuple(selected), raw_weights)
    validate_against_universe(portfolio, universe)
    return portfolio


def show_threshold_box(risk_profile: str) -> None:
    rule = RISK_PROFILE_RULES[risk_profile]
    st.info(
        f"{risk_profile} profile thresholds: "
        f"Maximum loss: {rule['max_loss']:.0%}. "
        f"Maximum drawdown: {rule['max_drawdown']:.0%}."
    )


def wrapped_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="wrapped-metric">
            <div class="wrapped-metric-label">{label}</div>
            <div class="wrapped-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_stochastic_process_setup(horizon_years: float, risk_profile: str) -> None:
    with st.expander("Stochastic process formulation", expanded=True):
        st.markdown(
            """
            The simulator treats daily ETF returns as a discrete-time vector-valued stochastic process.
            Historical returns are used only to estimate the model parameters, not to produce a single forecast.
            """
        )
        st.latex(r"\mathbf{R}_t \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})")
        st.latex(r"R_{p,t} = \mathbf{w}^{\top}\mathbf{R}_t")
        st.latex(r"V_t = V_{t-1}(1 + R_{p,t})")
        st.markdown(
            f"""
            **Fixed simulation design:** `{DEFAULT_N_SIMULATIONS:,}` sample paths,
            daily time steps, `{DEFAULT_HISTORICAL_START_DATE}` Yahoo Finance start date,
            `{horizon_years:g}` year horizon, and `{risk_profile}` first-exit thresholds.
            """
        )


def show_exit_time_definition(risk_profile: str) -> None:
    rule = RISK_PROFILE_RULES[risk_profile]
    with st.expander("First-passage / exit-time definition", expanded=True):
        st.markdown(
            """
            The exit time is the first trading day when a simulated portfolio path leaves the acceptable range.
            This turns the project into a first-passage-time problem rather than a point prediction problem.
            """
        )
        st.latex(
            r"\tau = \inf\{t \ge 0 : V_t/V_0 - 1 \le -L \ \mathrm{or}\ V_t/\max_{s \le t}V_s - 1 \le -D\}"
        )
        st.markdown(
            f"""
            For the selected `{risk_profile}` rule, `L = {rule["max_loss"]:.0%}` maximum loss from initial value and
            `D = {rule["max_drawdown"]:.0%}` maximum drawdown from the running peak. The app reports this as
            threshold breach probability, with first-passage terminology used in the definition.
            """
        )


def main() -> None:
    st.title("Monte Carlo ETF Portfolio Simulator")
    st.caption("Sample-path simulation and first-passage analysis for ETF portfolios")
    st.markdown(
        """
        <style>
        .wrapped-metric {
            min-height: 96px;
        }
        .wrapped-metric-label {
            color: rgb(49, 51, 63);
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .wrapped-metric-value {
            color: rgb(49, 51, 63);
            font-size: clamp(1.75rem, 3vw, 2.6rem);
            line-height: 1.12;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    universe = load_etf_universe()
    portfolio = build_portfolio_input(universe)

    st.sidebar.header("Simulation Inputs")
    starting_value = st.sidebar.number_input("Starting value", min_value=1_000.0, value=100_000.0, step=5_000.0)
    horizon_years = st.sidebar.slider("Investment horizon, years", min_value=1.0, max_value=30.0, value=5.0, step=0.5)
    risk_profile = st.sidebar.selectbox(
        "Risk profile",
        options=list(RISK_PROFILE_RULES.keys()),
        index=list(RISK_PROFILE_RULES.keys()).index("Balanced"),
    )

    show_threshold_box(risk_profile)
    show_stochastic_process_setup(horizon_years, risk_profile)
    show_exit_time_definition(risk_profile)

    weights = portfolio.to_frame()
    st.subheader("Normalized Portfolio Weights")
    st.dataframe(weights, use_container_width=True, hide_index=True)
    st.download_button(
        "Download custom portfolio weights",
        data=weights.to_csv(index=False),
        file_name="custom_portfolio.csv",
        mime="text/csv",
    )

    if st.button("Run simulation", type="primary"):
        with st.spinner("Estimating return parameters and generating Monte Carlo sample paths..."):
            prices = cached_prices(portfolio.tickers, DEFAULT_HISTORICAL_START_DATE)
            returns = compute_simple_returns(prices)
            model = MultivariateNormalModel.fit(returns)
            result = run_monte_carlo(
                portfolio=portfolio,
                model=model,
                starting_value=starting_value,
                years=horizon_years,
                n_simulations=DEFAULT_N_SIMULATIONS,
                random_seed=DEFAULT_RANDOM_SEED,
            )
            exit_result = calculate_exit_times(result.paths, risk_profile=risk_profile)
            metrics = summarize_simulation(result, starting_value=starting_value)
            metrics.update(exit_result.summary())
            summary = format_summary_table(metrics)

        st.subheader("Monte Carlo Sample Path Summary")
        cols = st.columns(4)
        cols[0].metric("Mean terminal value", f"${metrics['mean_ending_value']:,.0f}")
        cols[1].metric("Median CAGR", f"{metrics['median_annual_return']:.2%}")
        cols[2].metric("Threshold breach probability", f"{metrics['exit_probability']:.1%}")
        cols[3].metric("Survival probability", f"{metrics['survival_probability']:.1%}")

        cols = st.columns(4)
        cols[0].metric("Median terminal value", f"${metrics['median_ending_value']:,.0f}")
        cols[1].metric("Mean CAGR", f"{metrics['expected_annual_return']:.2%}")
        with cols[2]:
            wrapped_metric("Average exit time", format_trading_day_years(metrics["expected_exit_day"]))
        with cols[3]:
            wrapped_metric("Median exit time", format_trading_day_years(metrics["median_exit_day"]))

        chart_tabs = st.tabs(
            [
                "Sample Paths",
                "Distribution Through Time",
                "Terminal Distribution",
                "Exit-Time Distribution",
                "Survival Curve",
            ]
        )
        with chart_tabs[0]:
            st.plotly_chart(plot_sample_paths(result.paths), use_container_width=True)
        with chart_tabs[1]:
            st.plotly_chart(plot_percentile_bands(result.paths), use_container_width=True)
        with chart_tabs[2]:
            st.plotly_chart(plot_ending_distribution(result.ending_values), use_container_width=True)
        with chart_tabs[3]:
            st.plotly_chart(plot_exit_distribution(exit_result.first_exit_day), use_container_width=True)
        with chart_tabs[4]:
            st.plotly_chart(
                plot_survival_curve(exit_result.first_exit_day, horizon_day=result.paths.index.max()),
                use_container_width=True,
            )

        st.subheader("Stochastic Process Statistics")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.download_button(
            "Download summary table",
            data=summary.to_csv(index=False),
            file_name="simulation_summary.csv",
            mime="text/csv",
        )

        with st.expander("Estimated process parameters"):
            st.markdown(
                """
                These are empirical parameter estimates for the simulated stochastic process.
                They should be interpreted as model inputs, not guaranteed future outcomes.
                """
            )
            parameter_frame = pd.DataFrame(
                {
                    "ticker": returns.columns,
                    "daily_drift": model.mean,
                    "annualized_drift": model.annualized_mean(),
                    "daily_volatility": returns.std().to_numpy(),
                }
            )
            st.dataframe(
                parameter_frame.style.format(
                    {
                        "daily_drift": "{:.4%}",
                        "annualized_drift": "{:.2%}",
                        "daily_volatility": "{:.4%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.write(f"Model: {result.model_name}")
            st.write(f"Historical observations used: {len(returns):,}")


if __name__ == "__main__":
    main()
