"""Loading and validating input data.

Expected input: one row per campaign per day with the columns below. This is
the grain most ad platform exports and warehouse reporting tables already have.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY_COLUMNS = ["date", "platform", "campaign_id"]
METRIC_COLUMNS = ["impressions", "clicks", "spend", "conversions", "conversion_value"]
REQUIRED_COLUMNS = [*KEY_COLUMNS, "campaign_name", *METRIC_COLUMNS]
BUDGET_COLUMNS = ["campaign_id", "month", "budget"]


class DataError(ValueError):
    """The input data cannot be used. The message says what to fix."""


def _require_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{source}: missing column(s) {missing}; found {list(df.columns)}")


def validate_performance(df: pd.DataFrame, source: str = "input") -> pd.DataFrame:
    """Return a typed, sorted copy of daily performance data, or raise DataError."""
    _require_columns(df, REQUIRED_COLUMNS, source)
    if df.empty:
        raise DataError(f"{source}: no rows")
    out = df[REQUIRED_COLUMNS].copy()

    dates = pd.to_datetime(out["date"], errors="coerce", format="ISO8601")
    if dates.isna().any():
        bad = out.loc[dates.isna(), "date"].head(3).tolist()
        raise DataError(f"{source}: {int(dates.isna().sum())} unparseable date(s), e.g. {bad}")
    out["date"] = dates.dt.normalize()
    for column in ("platform", "campaign_id", "campaign_name"):
        out[column] = out[column].astype("string").str.strip()
        if out[column].isna().any() or (out[column] == "").any():
            raise DataError(f"{source}: empty value(s) in {column}")

    for column in METRIC_COLUMNS:
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any():
            raise DataError(
                f"{source}: {int(values.isna().sum())} non-numeric value(s) in {column}")
        if (values < 0).any():
            raise DataError(f"{source}: {int((values < 0).sum())} negative value(s) in {column}")
        out[column] = values.astype("float64")

    duplicates = out.duplicated(KEY_COLUMNS, keep=False)
    if duplicates.any():
        example = out.loc[duplicates, KEY_COLUMNS].iloc[0].tolist()
        raise DataError(
            f"{source}: {int(duplicates.sum())} rows share a date/platform/campaign_id, "
            f"e.g. {example}. Deduplicate before running detection."
        )
    impossible = out["clicks"] > out["impressions"]
    if impossible.any():
        raise DataError(
            f"{source}: {int(impossible.sum())} row(s) with more clicks than impressions")
    return out.sort_values(KEY_COLUMNS, ignore_index=True)


def load_performance(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")
    # Read ids as text so long numeric campaign ids are not turned into floats.
    raw = pd.read_csv(path, dtype={"campaign_id": "string", "platform": "string"})
    return validate_performance(raw, source=path.name)


def load_budgets(path: str | Path) -> pd.DataFrame:
    """Monthly budgets: campaign_id, month (YYYY-MM), budget."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"File not found: {path}")
    raw = pd.read_csv(path, dtype={"campaign_id": "string", "month": "string"})
    _require_columns(raw, BUDGET_COLUMNS, path.name)
    out = raw[BUDGET_COLUMNS].copy()
    months = pd.to_datetime(out["month"], format="%Y-%m", errors="coerce")
    if months.isna().any():
        raise DataError(f"{path.name}: month must be YYYY-MM")
    out["month"] = months.dt.to_period("M")
    out["budget"] = pd.to_numeric(out["budget"], errors="coerce")
    if out["budget"].isna().any() or (out["budget"] <= 0).any():
        raise DataError(f"{path.name}: budget must be a positive number")
    if out.duplicated(["campaign_id", "month"]).any():
        raise DataError(f"{path.name}: more than one budget for a campaign and month")
    return out
