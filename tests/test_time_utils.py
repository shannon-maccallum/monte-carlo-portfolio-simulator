import pytest

from src.time_utils import format_trading_day_years, trading_days_to_years


def test_trading_days_to_years_uses_252_day_year():
    assert trading_days_to_years(3580) == pytest.approx(3580 / 252)


def test_format_trading_day_years_uses_commas_and_one_decimal():
    assert format_trading_day_years(3580) == "3,580 days (14.2 years)"
    assert format_trading_day_years(1245) == "1,245 days (4.9 years)"
    assert format_trading_day_years(586) == "586 days (2.3 years)"
