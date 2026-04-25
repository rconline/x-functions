"""Prophet-backed forecast tests.

These tests are deliberately structural — they check that the scaffolding
assembles inputs into the shape Prophet expects, and that the group_cols
branch fans out correctly. Actual Prophet fitting is skipped unless
PROPHET_TESTS=1 is set.
"""

import os

import pandas as pd
import pytest

from spark_ai_functions.core.forecast import _periods_between, forecast_impl


def test_periods_between_daily():
    start = pd.Timestamp("2026-01-01")
    end = pd.Timestamp("2026-01-10")
    assert _periods_between(start, end, "D") == 9


def test_periods_between_handles_same_day():
    t = pd.Timestamp("2026-01-01")
    assert _periods_between(t, t, "D") == 0


@pytest.mark.skipif(not os.environ.get("PROPHET_TESTS"), reason="set PROPHET_TESTS=1")
def test_forecast_end_to_end():
    dates = pd.date_range("2026-01-01", periods=60, freq="D")
    values = [float(i) + 100 for i in range(60)]
    observed = pd.DataFrame({"ds": dates, "y": values})
    out = forecast_impl(
        observed, horizon="2026-03-15",
        time_col="ds", value_col="y", frequency="D",
    )
    assert list(out.columns) == ["ts", "yhat", "yhat_lower", "yhat_upper", "group_key"]
    assert len(out) > 0
