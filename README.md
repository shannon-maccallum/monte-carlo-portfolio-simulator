# Monte Carlo ETF Portfolio Simulator

This repository is an undergraduate stochastic processes project that simulates ETF portfolio sample paths under uncertainty. The first implementation uses a multivariate normal daily-return process with a historical correlation structure estimated from Yahoo Finance data.

The purpose is not to predict one future portfolio value. The purpose is to define a stochastic process, generate many possible paths from that process, and study the resulting distribution, threshold crossings, and first-exit times.

The project emphasizes:

- Monte Carlo simulation of stochastic return paths
- Distribution of terminal portfolio values
- Mean and median CAGR across sample paths
- Downside risk and drawdown behavior
- Threshold crossing probabilities
- First-passage and survival probabilities
- A Streamlit interface for custom ETF portfolios

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notebook Walkthrough

The project notebook is available at `notebooks/portfolio_monte_carlo_simulation.ipynb`. It is designed as a standalone presentation artifact covering the ETF universe, Yahoo Finance data collection, stochastic process setup, Monte Carlo simulation, exit-time analysis, survival curve, and visuals.

After installing requirements, open it with:

```bash
jupyter notebook notebooks/portfolio_monte_carlo_simulation.ipynb
```

## ETF Universe

The app only allows ETFs listed in `data/etf_universe.csv`. This keeps the project focused and makes the case-study portfolios comparable.

Portfolio weights are entered as decimal allocations that sum to `1.0`, for example:

```text
0.10, 0.10, 0.10, 0.70
```

not percentage values like `10, 10, 10, 70`.

## Current Stochastic Model

Model 1 estimates the daily mean vector and covariance matrix from historical ETF returns:

```text
R_t ~ N(mu, Sigma)
```

For each simulation, correlated daily ETF returns are generated, combined using portfolio weights, and compounded into a portfolio value path:

```text
R_p,t = w'R_t
V_t = V_{t-1}(1 + R_p,t)
```

The app uses a fixed simulation design so results are reproducible:

- 10,000 Monte Carlo paths
- Daily time steps
- Historical data starting 2015-01-01
- User-selected conservative, balanced, or growth first-exit thresholds
- Fixed random seed for reproducible project results

The simulation uses daily estimated returns at each daily time step. Annualized drift values are shown only as diagnostics and are computed by compounding the daily drift:

```text
annualized_drift = (1 + daily_drift)^252 - 1
```

## Exit-Time Analysis

An exit occurs when the simulated path breaches the selected acceptable range.

| Profile | Exit if loss from start exceeds | Exit if drawdown exceeds |
| --- | ---: | ---: |
| Conservative | 10% | 15% |
| Balanced | 18% | 25% |
| Growth | 30% | 40% |

```text
tau = inf{t >= 0 : V_t / V_0 - 1 <= -L or V_t / max(V_s, s <= t) - 1 <= -D}
```

For each path, the simulator stores the first exit day, whether an exit occurred, threshold breach probability before the horizon, average exit time, median exit time, and survival probability.

Exit times are displayed as both trading days and years using:

```text
years = trading_days / 252
```

Example:

```text
3,580 days (14.2 years)
```

The app also plots the empirical survival function:

```text
S(t) = P(tau > t)
```

The selected threshold is displayed directly in the app so the breach probability always has context.

## Project Structure

```text
.
├── app.py
├── data/
│   └── etf_universe.csv
├── notebooks/
│   └── portfolio_monte_carlo_simulation.ipynb
├── paper/
│   └── final_report_outline.md
├── src/
│   ├── data_loader.py
│   ├── exit_time.py
│   ├── plotting.py
│   ├── portfolio.py
│   ├── risk_metrics.py
│   └── simulation.py
└── tests/
    ├── test_portfolio.py
    └── test_simulation.py
```

## Next Modeling Extensions

- Geometric Brownian Motion
- Time-varying volatility
- Regime-switching market states
- Heavy-tailed return distributions
- Case study comparing nine fixed portfolios
