"""Budget pacing: will each campaign land on its monthly budget?"""

from __future__ import annotations

import math

import pandas as pd

from .config import DetectionConfig

PACING_COLUMNS = [
    "status", "platform", "campaign_id", "campaign_name", "month", "budget", "spend_to_date",
    "daily_run_rate", "days_remaining", "projected_spend", "pace", "projected_variance",
    "budget_runs_out",
]


def pacing_report(
    performance: pd.DataFrame,
    budgets: pd.DataFrame,
    as_of: pd.Timestamp,
    config: DetectionConfig | None = None,
) -> pd.DataFrame:
    """Project month-end spend from spend to date plus the recent daily run rate.

    projected = spend so far this month + (average of the last N days) x days left.
    """
    config = config or DetectionConfig()
    month = as_of.to_period("M")
    days_remaining = (month.end_time.normalize() - as_of).days
    current = budgets[budgets["month"] == month]
    history = performance[performance["date"] <= as_of]

    rows = []
    for budget_row in current.itertuples(index=False):
        campaign = history[history["campaign_id"] == budget_row.campaign_id]
        if campaign.empty:
            continue
        in_month = campaign[campaign["date"].dt.to_period("M") == month]
        spend_to_date = float(in_month["spend"].sum())
        recent = campaign[campaign["date"] > as_of - pd.Timedelta(days=config.pacing_run_rate_days)]
        # Days without a row count as zero spend: a paused campaign has a zero run rate.
        run_rate = float(recent["spend"].sum()) / config.pacing_run_rate_days
        projected = spend_to_date + run_rate * days_remaining
        pace = projected / budget_row.budget

        if pace > config.pacing_over_threshold:
            status = "over"
        elif pace < config.pacing_under_threshold:
            status = "under"
        else:
            status = "on_track"

        runs_out = pd.NaT
        left = budget_row.budget - spend_to_date
        if left <= 0:
            runs_out = as_of
        elif run_rate > 0 and math.ceil(left / run_rate) < days_remaining:
            runs_out = as_of + pd.Timedelta(days=math.ceil(left / run_rate))

        rows.append({
            "status": status,
            "platform": campaign["platform"].iloc[-1],
            "campaign_id": budget_row.campaign_id,
            "campaign_name": campaign["campaign_name"].iloc[-1],
            "month": str(month),
            "budget": float(budget_row.budget),
            "spend_to_date": spend_to_date,
            "daily_run_rate": run_rate,
            "days_remaining": days_remaining,
            "projected_spend": projected,
            "pace": pace,
            "projected_variance": projected - budget_row.budget,
            "budget_runs_out": runs_out,
        })
    report = pd.DataFrame(rows, columns=PACING_COLUMNS)
    if report.empty:
        return report
    report["budget_runs_out"] = pd.to_datetime(report["budget_runs_out"])
    distance = (report["pace"] - 1).abs()
    return report.iloc[distance.sort_values(ascending=False).index].reset_index(drop=True)
