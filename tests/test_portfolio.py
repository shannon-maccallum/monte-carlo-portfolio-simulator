import numpy as np
import pytest

from src.portfolio import Portfolio, validate_against_universe


def test_portfolio_stores_weights_as_decimal_allocations():
    portfolio = Portfolio(("VTI", "BND"), np.array([60, 40]))

    assert portfolio.tickers == ("VTI", "BND")
    assert portfolio.weights.sum() == pytest.approx(1.0)
    assert portfolio.weights.tolist() == pytest.approx([0.6, 0.4])
    assert all(0 <= weight <= 1 for weight in portfolio.weights)


def test_portfolio_rejects_invalid_universe_member():
    portfolio = Portfolio(("VTI", "FAKE"), [0.5, 0.5])

    with pytest.raises(ValueError, match="outside the approved universe"):
        validate_against_universe(portfolio, ["VTI", "BND"])
