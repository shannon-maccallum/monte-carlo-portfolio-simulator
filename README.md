# Stochastic Portfolio Simulation and Exit-Time Analysis

This repository contains my undergraduate stochastic processes project, which models ETF portfolio performance under uncertainty using Monte Carlo simulation. The model uses a multivariate normal return process with parameters estimated from historical Yahoo Finance data, allowing the simulation to preserve the historical relationships between ETFs.

Rather than attempting to predict a single future portfolio value, the goal of this project is to model portfolio value as a stochastic process, generate thousands of possible future paths, and analyze the resulting distribution of outcomes. By doing so, the project explores not only expected performance but also the likelihood and timing of important risk events.

Key areas of analysis include:

Monte Carlo simulation of portfolio return paths
Distribution of ending portfolio values
Mean and median portfolio growth rates (CAGR)
Portfolio volatility, downside risk, and drawdowns
Threshold breach probabilities
First-passage (exit-time) analysis
Survival probabilities and survival curves
An interactive Streamlit dashboard for custom ETF portfolios

## Open App

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Notebook 

The project notebook is available at `notebooks/portfolio_monte_carlo_simulation.ipynb`

## Exit-Time Analysis

An exit occurs when the simulated path breaches the selected acceptable range.

| Profile | Exit if loss from start exceeds | Exit if drawdown exceeds |
| --- | ---: | ---: |
| Conservative | 10% | 15% |
| Balanced | 18% | 25% |
| Growth | 30% | 40% |


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
