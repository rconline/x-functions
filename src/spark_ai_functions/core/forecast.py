"""`ai_forecast` — Prophet-backed UDTF.

Rather than register a true SQL UDTF (Spark 3.5+) we expose this as a DataFrame
helper driven by a scalar Pandas UDF on group keys — which Spark 3.4 can
already run. The `register()` step wires a `forecast(...)` helper onto the
returned `AIFunctions` object; SQL callers get a thin TVF via
`spark.sql.functions.call_function` when the Spark version supports UDTFs.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

import pandas as pd

from ..governance.decorator import governed


def _fit_and_predict(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    horizon: pd.Timestamp,
    frequency: str = "D",
    parameters: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    from prophet import Prophet  # type: ignore

    params = parameters or {}
    m = Prophet(**params)
    local = df[[time_col, value_col]].rename(columns={time_col: "ds", value_col: "y"})
    local["ds"] = pd.to_datetime(local["ds"])
    m.fit(local)
    last = local["ds"].max()
    periods = _periods_between(last, horizon, frequency)
    future = m.make_future_dataframe(periods=periods, freq=frequency, include_history=False)
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return fcst


def _periods_between(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> int:
    rng = pd.date_range(start=start, end=end, freq=freq)
    # Prophet's make_future_dataframe doesn't include history, so exclude start.
    return max(len(rng) - 1, 0)


def forecast_impl(
    observed: pd.DataFrame,
    *,
    horizon: str | pd.Timestamp,
    time_col: str,
    value_col: str,
    group_cols: Optional[Iterable[str]] = None,
    frequency: str = "D",
    parameters: Optional[str | dict[str, Any]] = None,
) -> pd.DataFrame:
    """Driver-side TVF stand-in. Takes a pandas DataFrame of observations and
    returns a DataFrame of forecast points.

    Params:
      horizon: target end timestamp (inclusive)
      frequency: pandas freq string (D, H, W, ...)
      parameters: JSON string or dict of Prophet hyperparameters
    """
    if isinstance(horizon, str):
        horizon_ts = pd.to_datetime(horizon)
    else:
        horizon_ts = pd.Timestamp(horizon)
    params: dict[str, Any] = {}
    if isinstance(parameters, str) and parameters.strip():
        params = json.loads(parameters)
    elif isinstance(parameters, dict):
        params = parameters

    group_cols = list(group_cols or [])
    if not group_cols:
        out = _fit_and_predict(observed, time_col, value_col, horizon_ts, frequency, params)
        out["group_key"] = ""
        return out.rename(columns={"ds": "ts"})

    pieces: list[pd.DataFrame] = []
    for key, sub in observed.groupby(group_cols):
        gkey = key if isinstance(key, str) else "|".join(map(str, key if isinstance(key, tuple) else (key,)))
        f = _fit_and_predict(sub, time_col, value_col, horizon_ts, frequency, params)
        f["group_key"] = gkey
        pieces.append(f)
    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["ds", "yhat", "yhat_lower", "yhat_upper", "group_key"]
    )
    return combined.rename(columns={"ds": "ts"})


@governed("ai_forecast")
def ai_forecast_impl(
    endpoint_name: str,  # unused — uniform decorator signature
    batch: pd.Series,    # placeholder; see `forecast_impl` for the real entrypoint
    *,
    credential: str,
) -> pd.Series:
    # Spark users call `forecast_impl` directly via the helper registered onto
    # the AIFunctions handle. This governed stub exists only so audit events
    # for `ai_forecast` are still emitted when the SQL-level UDTF is invoked
    # via Spark 3.5's function-call path.
    return batch
