"""Measures the detector against the labelled synthetic anomalies.

Scoring is at incident level, which is how a person experiences alerts:

* recall: share of injected anomalies that raised an incident on the right
  campaign, on a relevant metric, overlapping the anomaly's dates;
* precision: share of raised incidents that overlap any injected anomaly on
  that campaign (an anomaly legitimately moves several metrics at once);
* false incidents per campaign-month: how often someone would be interrupted
  for nothing.
"""

from __future__ import annotations

import pandas as pd

from .config import DetectionConfig
from .detect import build_incidents, score
from .schema import validate_performance
from .synthetic import ANOMALY_TYPES, CAMPAIGNS, DEFAULT_DAYS, generate

TOLERANCE = pd.Timedelta(days=1)
CONVERSION_TYPES = ("conversion_rate_drop", "tracking_outage")
VOLUME_TIERS = ("15 or more conversions a day", "5 to 15 conversions a day",
                "fewer than 5 conversions a day")


def _volume_tier(campaign_id: str) -> str | None:
    """Tier by the conversions a campaign is built to get on a normal day."""
    spec = next((c for c in CAMPAIGNS if c.campaign_id == campaign_id), None)
    if spec is None:
        return None
    per_day = spec.daily_spend / spec.cpc * spec.cvr
    return VOLUME_TIERS[0] if per_day >= 15 else VOLUME_TIERS[1] if per_day >= 5 \
        else VOLUME_TIERS[2]


def _overlaps(incidents: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return ((incidents["start_date"] <= end + TOLERANCE)
            & (incidents["end_date"] >= start - TOLERANCE))


def evaluate(incidents: pd.DataFrame, truth: pd.DataFrame, campaign_months: float) -> dict:
    """Compare incidents with ground truth. Returns counts and rates."""
    detected_by_type: dict[str, list[bool]] = {kind: [] for kind in ANOMALY_TYPES}
    matched = pd.Series(False, index=incidents.index)
    by_volume = {tier: {"injected": 0, "detected": 0} for tier in VOLUME_TIERS}

    for anomaly in truth.itertuples(index=False):
        same_campaign = incidents["campaign_id"] == anomaly.campaign_id
        in_window = same_campaign & _overlaps(incidents, anomaly.start_date, anomaly.end_date)
        matched |= in_window
        relevant = incidents["metric"].isin(ANOMALY_TYPES[anomaly.anomaly_type][2])
        found = bool((in_window & relevant).any())
        detected_by_type[anomaly.anomaly_type].append(found)
        tier = _volume_tier(anomaly.campaign_id)
        if anomaly.anomaly_type in CONVERSION_TYPES and tier is not None:
            by_volume[tier]["injected"] += 1
            by_volume[tier]["detected"] += found

    detected = sum(sum(flags) for flags in detected_by_type.values())
    false_incidents = int((~matched).sum())
    return {
        "anomalies": len(truth),
        "detected": detected,
        "recall": detected / len(truth) if len(truth) else float("nan"),
        "incidents": len(incidents),
        "false_incidents": false_incidents,
        "precision": float(matched.mean()) if len(incidents) else float("nan"),
        "false_incidents_per_campaign_month": false_incidents / campaign_months,
        "by_type": {
            kind: {"injected": len(flags), "detected": sum(flags)}
            for kind, flags in detected_by_type.items()
        },
        "conversion_by_volume": by_volume,
    }


def campaign_months() -> float:
    """Campaign-months of data in one default synthetic dataset."""
    return len(CAMPAIGNS) * DEFAULT_DAYS / 30.4


def run_once(seed: int, config: DetectionConfig | None = None) -> dict:
    performance, truth, _ = generate(seed=seed)
    performance = validate_performance(performance, source=f"synthetic seed {seed}")
    incidents = build_incidents(score(performance, config), config)
    return evaluate(incidents, truth, campaign_months=campaign_months())


def benchmark(seeds: list[int], config: DetectionConfig | None = None) -> dict:
    """Pool results over several independently generated datasets."""
    runs = [run_once(seed, config) for seed in seeds]
    total = {key: sum(run[key] for run in runs)
             for key in ("anomalies", "detected", "incidents", "false_incidents")}
    by_type = {
        kind: {field: sum(run["by_type"][kind][field] for run in runs)
               for field in ("injected", "detected")}
        for kind in ANOMALY_TYPES
    }
    by_volume = {
        tier: {field: sum(run["conversion_by_volume"][tier][field] for run in runs)
               for field in ("injected", "detected")}
        for tier in VOLUME_TIERS
    }
    for counts in (*by_type.values(), *by_volume.values()):
        counts["recall"] = counts["detected"] / counts["injected"] if counts["injected"] else None
    return {
        "datasets": len(seeds),
        **total,
        "recall": total["detected"] / total["anomalies"],
        "precision": 1 - total["false_incidents"] / total["incidents"],
        "false_incidents_per_campaign_month":
            total["false_incidents"] / (len(seeds) * campaign_months()),
        "by_type": by_type,
        "conversion_recall_by_volume": by_volume,
    }
