"""Synthetic daily ad data with injected, labelled anomalies.

The labels make it possible to measure the detector (see evaluate.py) instead of
judging it by eye. Counts are drawn from Poisson and binomial distributions and
rates carry extra day-to-day noise, so small campaigns are noisy in the way
real ones are.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

DEFAULT_SEED = 7
DEFAULT_START = date(2026, 7, 1)
DEFAULT_DAYS = 123              # 1 July to 31 October
FIRST_ANOMALY_DAY = 40          # leave a clean run-in so baselines exist
MIN_GAP_DAYS = 12               # between anomalies on one campaign


@dataclass(frozen=True)
class CampaignSpec:
    platform: str
    campaign_id: str
    name: str
    daily_spend: float
    cpc: float
    ctr: float
    cvr: float
    order_value: float
    budget_factor: float = 1.0   # monthly budget relative to planned spend


CAMPAIGNS = [
    CampaignSpec("google_ads", "g-1001", "brand_search", 90, 0.35, 0.220, 0.100, 62),
    CampaignSpec("google_ads", "g-1002", "nonbrand_search", 260, 0.95, 0.050, 0.032, 58, 0.82),
    CampaignSpec("google_ads", "g-1003", "performance_max", 320, 0.60, 0.018, 0.036, 64),
    CampaignSpec("google_ads", "g-1004", "shopping_catalog", 180, 0.48, 0.013, 0.030, 55),
    CampaignSpec("meta_ads", "m-2001", "prospecting_lookalike", 270, 0.72, 0.014, 0.022, 59),
    CampaignSpec("meta_ads", "m-2002", "retargeting_cart", 100, 0.90, 0.022, 0.065, 66, 1.35),
    CampaignSpec("meta_ads", "m-2003", "prospecting_video", 120, 0.55, 0.009, 0.007, 56),
    # Deliberately small: about one conversion a day, too little to judge conversion rate.
    CampaignSpec("meta_ads", "m-2004", "local_test_small", 25, 0.80, 0.012, 0.030, 50),
]

# anomaly type -> (min days, max days, metrics a detector should raise)
ANOMALY_TYPES: dict[str, tuple[int, int, tuple[str, ...]]] = {
    "spend_spike": (1, 3, ("spend",)),
    "delivery_drop": (2, 4, ("spend", "impressions")),
    "cpc_spike": (2, 3, ("cpc",)),
    "ctr_drop": (2, 5, ("ctr", "impressions")),
    "conversion_rate_drop": (2, 4, ("conversion_rate",)),
    "tracking_outage": (2, 4, ("conversion_rate",)),
    "missing_data": (1, 2, ("missing_data",)),
}


def _plan_anomalies(rng: np.random.Generator, days: int, count: int) -> list[tuple[str, int, int]]:
    """Pick ``count`` non-overlapping (type, first day, last day) slots for one campaign."""
    types = rng.choice(list(ANOMALY_TYPES), size=count, replace=False)
    planned: list[tuple[str, int, int]] = []
    for kind in types:
        low, high, _ = ANOMALY_TYPES[kind]
        length = int(rng.integers(low, high + 1))
        for _ in range(200):
            first = int(rng.integers(FIRST_ANOMALY_DAY, days - length - 1))
            last = first + length - 1
            if all(first > b + MIN_GAP_DAYS or last < a - MIN_GAP_DAYS for _, a, b in planned):
                planned.append((str(kind), first, last))
                break
    return planned


def generate(
    seed: int = DEFAULT_SEED,
    start: date = DEFAULT_START,
    days: int = DEFAULT_DAYS,
    anomalies_per_campaign: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (performance, ground_truth, budgets). Same seed, same data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    rows, truth, budget_rows = [], [], []
    months = dates.to_period("M")

    for spec in CAMPAIGNS:
        weekday_effect = np.array([0.96, 0.98, 1.0, 1.02, 1.05, 1.12, 1.07]) \
            * rng.normal(1.0, 0.02, size=7)
        trend = rng.normal(0.0008, 0.0006)
        plan = _plan_anomalies(rng, days, anomalies_per_campaign)
        # A budget is what the campaign is planned to spend that month, before surprises.
        growth = 1 + trend * np.arange(days)
        planned = spec.daily_spend * weekday_effect[dates.day_of_week] * growth
        for month in months.unique():
            budget_rows.append({
                "campaign_id": spec.campaign_id, "month": str(month),
                "budget": round(planned[months == month].sum() * spec.budget_factor, -1),
            })
        for kind, first, last in plan:
            truth.append({
                "platform": spec.platform, "campaign_id": spec.campaign_id,
                "anomaly_type": kind, "start_date": dates[first], "end_date": dates[last],
            })
        strength = {
            "spend_spike": rng.uniform(1.9, 3.0),
            "delivery_drop": rng.uniform(0.1, 0.4),
            "cpc_spike": rng.uniform(1.7, 2.3),
            "ctr_drop": rng.uniform(0.4, 0.6),
            "conversion_rate_drop": rng.uniform(0.3, 0.5),
        }

        for t, day in enumerate(dates):
            active = {kind for kind, first, last in plan if first <= t <= last}
            if "missing_data" in active:
                continue

            demand = weekday_effect[day.day_of_week] * (1 + trend * t) * rng.lognormal(0, 0.07)
            spend = spec.daily_spend * demand
            cpc = spec.cpc * rng.lognormal(0, 0.05)
            ctr = spec.ctr * rng.lognormal(0, 0.06)
            cvr = spec.cvr * rng.lognormal(0, 0.08)

            if "spend_spike" in active:
                spend *= strength["spend_spike"]
            if "delivery_drop" in active:
                spend *= strength["delivery_drop"]
            if "cpc_spike" in active:
                cpc *= strength["cpc_spike"]
            if "ctr_drop" in active:
                ctr *= strength["ctr_drop"]
            if "conversion_rate_drop" in active:
                cvr *= strength["conversion_rate_drop"]

            clicks = max(1, int(rng.poisson(spend / cpc)))
            impressions = max(clicks, round(clicks / ctr))
            conversions = 0 if "tracking_outage" in active else int(rng.binomial(clicks, cvr))
            value_noise = rng.lognormal(0, 0.15 / max(conversions, 1) ** 0.5)
            value = conversions * spec.order_value * value_noise
            rows.append({
                "date": day, "platform": spec.platform, "campaign_id": spec.campaign_id,
                "campaign_name": spec.name, "impressions": impressions, "clicks": clicks,
                "spend": round(spend, 2), "conversions": conversions,
                "conversion_value": round(value, 2),
            })

    performance = pd.DataFrame(rows)
    ground_truth = pd.DataFrame(truth).sort_values(
        ["start_date", "campaign_id"], ignore_index=True)
    budgets = pd.DataFrame(budget_rows)
    return performance, ground_truth, budgets
