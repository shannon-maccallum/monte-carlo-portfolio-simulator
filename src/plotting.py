from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.time_utils import trading_days_to_years


def plot_sample_paths(paths: pd.DataFrame, max_paths: int = 200) -> go.Figure:
    sample = paths.iloc[:, : min(max_paths, paths.shape[1])]
    fig = go.Figure()
    for column in sample.columns:
        fig.add_trace(
            go.Scatter(
                x=sample.index,
                y=sample[column],
                mode="lines",
                line={"width": 1, "color": "rgba(36, 99, 235, 0.12)"},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=paths.index,
            y=paths.median(axis=1),
            mode="lines",
            line={"width": 3, "color": "#111827"},
            name="Median path",
        )
    )
    fig.update_layout(xaxis_title="Trading day", yaxis_title="Portfolio value")
    return fig


def plot_percentile_bands(paths: pd.DataFrame) -> go.Figure:
    percentiles = paths.quantile([0.05, 0.25, 0.50, 0.75, 0.95], axis=1).T
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=percentiles.index, y=percentiles[0.95], line={"width": 0}, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles[0.05],
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.18)",
            line={"width": 0},
            name="5th-95th percentile",
        )
    )
    fig.add_trace(go.Scatter(x=percentiles.index, y=percentiles[0.75], line={"width": 0}, showlegend=False))
    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles[0.25],
            fill="tonexty",
            fillcolor="rgba(20, 184, 166, 0.22)",
            line={"width": 0},
            name="25th-75th percentile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=percentiles.index,
            y=percentiles[0.50],
            mode="lines",
            line={"width": 3, "color": "#111827"},
            name="Median",
        )
    )
    fig.update_layout(xaxis_title="Trading day", yaxis_title="Portfolio value")
    return fig


def plot_ending_distribution(ending_values: pd.Series) -> go.Figure:
    fig = go.Figure(data=[go.Histogram(x=ending_values, nbinsx=60, marker_color="#2563eb")])
    fig.update_layout(xaxis_title="Ending value", yaxis_title="Simulation count", bargap=0.04)
    return fig


def plot_exit_distribution(first_exit_day: pd.Series) -> go.Figure:
    exit_days = first_exit_day.dropna()
    exit_years = exit_days.apply(trading_days_to_years)
    fig = go.Figure(
        data=[
            go.Histogram(
                x=exit_days,
                customdata=exit_years,
                nbinsx=50,
                marker_color="#dc2626",
                hovertemplate="First exit time: %{x:,.0f} days (%{customdata:.1f} years)<br>Count: %{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(xaxis_title="First exit day", yaxis_title="Simulation count", bargap=0.04)
    return fig


def plot_survival_curve(first_exit_day: pd.Series, horizon_day: int) -> go.Figure:
    days = pd.RangeIndex(0, horizon_day + 1, name="day")
    years = [trading_days_to_years(day) for day in days]
    survival_probability = [(first_exit_day.isna() | (first_exit_day > day)).mean() for day in days]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=days,
                y=survival_probability,
                customdata=years,
                mode="lines",
                line={"width": 3, "color": "#0f766e"},
                name="P(tau > t)",
                hovertemplate="t = %{x:,.0f} days (%{customdata:.1f} years)<br>P(tau > t) = %{y:.1%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        xaxis_title="Trading day",
        yaxis_title="Survival probability",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
    )
    return fig
