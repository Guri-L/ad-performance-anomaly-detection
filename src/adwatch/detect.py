"""Scores every campaign, metric and day, and turns outliers into incidents."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import Scores, score_level, score_rate
from .config import DetectionConfig

LEVEL_METRICS = ("spend", "impressions", "cpc")
RATE_METRICS = ("ctr", "conversion_rate")
MISSING = "missing_data"

SCORED_COLUMNS = [
    "date", "platform", "campaign_id", "campaign_name", "metric", "observed", "expected",
    "lower", "upper", "z_score", "pct_change", "is_anomaly", "direction", "impact",
    "zero_events",
]
INCIDENT_COLUMNS = [
    "severity", "diagnosis", "platform", "campaign_id", "campaign_name", "metric", "direction",
    "start_date", "end_date", "days", "peak_z", "observed_avg", "expected_avg", "pct_change",
    "estimated_impact",
]


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _trailing_value_per_conversion(value: np.ndarray, conversions: np.ndarray,
                                   window: int) -> np.ndarray:
    """Revenue per conversion over the days before each day (for impact estimates)."""
    out = np.full(len(value), np.nan)
    for t in range(len(value)):
        start = max(0, t - window)
        conv = np.nansum(conversions[start:t])
        if conv > 0:
            out[t] = np.nansum(value[start:t]) / conv
    return out


def _score_campaign(daily: pd.DataFrame, config: DetectionConfig) -> pd.DataFrame:
    """Score one campaign. ``daily`` has one row per calendar day (gaps as NaN)."""
    weekdays = daily["date"].dt.day_of_week.to_numpy()
    col = {c: daily[c].to_numpy(dtype="float64") for c in
           ("impressions", "clicks", "spend", "conversions", "conversion_value")}

    def level(values: np.ndarray, counts: np.ndarray | None = None) -> Scores:
        return score_level(values, weekdays, window=config.window_days,
                           min_history=config.min_history_days,
                           scale_floor=config.level_scale_floor, counts=counts)

    def rate(successes: np.ndarray, trials: np.ndarray) -> Scores:
        return score_rate(successes, trials, window=config.window_days,
                          min_history=config.min_history_days,
                          min_expected=config.rate_min_expected_events)

    cpc = _safe_divide(col["spend"], col["clicks"])
    value_per_conv = _trailing_value_per_conversion(
        col["conversion_value"], col["conversions"], config.window_days)

    # metric -> (observed, scores, z threshold, min change, impact per day, zero-event flag)
    results: dict[str, tuple[np.ndarray, Scores, float, float, np.ndarray, np.ndarray]] = {}
    no_flag = np.zeros(len(daily), dtype=bool)
    nan = np.full(len(daily), np.nan)

    spend_scores = level(col["spend"])
    results["spend"] = (col["spend"], spend_scores, config.level_z_threshold,
                        config.level_min_change, col["spend"] - spend_scores.expected, no_flag)
    results["impressions"] = (
        col["impressions"],
        level(col["impressions"], counts=col["clicks"]),
        config.level_z_threshold, config.level_min_change, nan, no_flag)
    cpc_scores = level(cpc, counts=col["clicks"])
    results["cpc"] = (cpc, cpc_scores, config.level_z_threshold, config.level_min_change,
                      (cpc - cpc_scores.expected) * col["clicks"], no_flag)

    results["ctr"] = (
        _safe_divide(col["clicks"], col["impressions"]),
        rate(col["clicks"], col["impressions"]),
        config.rate_z_threshold, config.rate_min_change, nan, col["clicks"] == 0)
    cvr_scores = rate(col["conversions"], col["clicks"])
    results["conversion_rate"] = (
        _safe_divide(col["conversions"], col["clicks"]), cvr_scores,
        config.rate_z_threshold, config.rate_min_change,
        (col["conversions"] - col["clicks"] * cvr_scores.expected) * value_per_conv,
        col["conversions"] == 0)

    frames = []
    for metric, (observed, scores, z_limit, min_change, impact, zero) in results.items():
        pct = _safe_divide(observed - scores.expected, scores.expected)
        anomaly = (np.abs(scores.z) >= z_limit) & (np.abs(scores.change) >= min_change)
        frames.append(pd.DataFrame({
            "date": daily["date"].to_numpy(),
            "metric": metric,
            "observed": observed,
            "expected": scores.expected,
            "lower": scores.lower,
            "upper": scores.upper,
            "z_score": scores.z,
            "pct_change": pct,
            "is_anomaly": anomaly,
            "direction": np.where(scores.z >= 0, "up", "down"),
            "impact": impact,
            "zero_events": zero,
        }))

    # A day with no row at all, inside the campaign's active span, is its own problem.
    gap = daily["spend"].isna().to_numpy()
    if gap.any():
        frames.append(pd.DataFrame({
            "date": daily["date"].to_numpy()[gap], "metric": MISSING, "observed": np.nan,
            "expected": np.nan, "lower": np.nan, "upper": np.nan, "z_score": np.nan,
            "pct_change": np.nan, "is_anomaly": True, "direction": "down", "impact": np.nan,
            "zero_events": False,
        }))
    return pd.concat(frames, ignore_index=True)


def score(performance: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    """Score validated performance data. Returns one row per campaign, metric and day."""
    config = config or DetectionConfig()
    config.validate()
    parts = []
    for (platform, campaign_id), group in performance.groupby(
            ["platform", "campaign_id"], sort=True):
        days = pd.date_range(group["date"].min(), group["date"].max(), freq="D")
        daily = group.set_index("date").reindex(days).rename_axis("date").reset_index()
        scored = _score_campaign(daily, config)
        scored.insert(1, "platform", platform)
        scored.insert(2, "campaign_id", campaign_id)
        scored.insert(3, "campaign_name", group["campaign_name"].iloc[-1])
        parts.append(scored)
    return pd.concat(parts, ignore_index=True)[SCORED_COLUMNS]


def _diagnose(metric: str, direction: str, all_zero: bool) -> str:
    if metric == MISSING:
        return "Missing data"
    if metric == "conversion_rate" and direction == "down":
        return "Possible tracking outage (zero conversions)" if all_zero \
            else "Conversion rate drop"
    labels = {
        ("spend", "up"): "Spend spike",
        ("spend", "down"): "Spend drop",
        ("impressions", "up"): "Impression surge",
        ("impressions", "down"): "Delivery drop",
        ("cpc", "up"): "CPC increase",
        ("cpc", "down"): "CPC decrease",
        ("ctr", "up"): "CTR jump",
        ("ctr", "down"): "CTR drop",
        ("conversion_rate", "up"): "Conversion rate jump",
    }
    return labels[(metric, direction)]


def _severity(metric: str, peak_z: float, diagnosis: str) -> str:
    if metric == MISSING or diagnosis.startswith("Possible tracking outage"):
        return "high"
    if abs(peak_z) >= 8:
        return "high"
    return "medium" if abs(peak_z) >= 6 else "low"


def build_incidents(scored: pd.DataFrame, config: DetectionConfig | None = None) -> pd.DataFrame:
    """Merge anomalous days into incidents: one per campaign, metric and direction run."""
    config = config or DetectionConfig()
    alerts = scored[scored["is_anomaly"]].sort_values(
        ["platform", "campaign_id", "metric", "direction", "date"])
    if alerts.empty:
        return pd.DataFrame(columns=INCIDENT_COLUMNS)

    keys = ["platform", "campaign_id", "metric", "direction"]
    day_gap = alerts.groupby(keys)["date"].diff().dt.days
    new_run = day_gap.isna() | (day_gap > config.incident_gap_days + 1)
    alerts = alerts.assign(run=new_run.cumsum())

    rows = []
    for _, run in alerts.groupby("run", sort=False):
        first = run.iloc[0]
        peak_z = run["z_score"].loc[run["z_score"].abs().idxmax()] \
            if run["z_score"].notna().any() else float("nan")
        diagnosis = _diagnose(first["metric"], first["direction"], bool(run["zero_events"].all()))
        observed, expected = run["observed"].mean(), run["expected"].mean()
        rows.append({
            "severity": _severity(first["metric"], peak_z, diagnosis),
            "diagnosis": diagnosis,
            "platform": first["platform"],
            "campaign_id": first["campaign_id"],
            "campaign_name": first["campaign_name"],
            "metric": first["metric"],
            "direction": first["direction"],
            "start_date": run["date"].min(),
            "end_date": run["date"].max(),
            "days": len(run),
            "peak_z": peak_z,
            "observed_avg": observed,
            "expected_avg": expected,
            "pct_change": (observed - expected) / expected if expected else float("nan"),
            "estimated_impact": run["impact"].sum(min_count=1),
        })
    incidents = pd.DataFrame(rows, columns=INCIDENT_COLUMNS)
    rank = incidents["severity"].map({"high": 0, "medium": 1, "low": 2})
    order = np.lexsort((-incidents["peak_z"].abs().fillna(np.inf), rank))
    return incidents.iloc[order].reset_index(drop=True)


def filter_recent(incidents: pd.DataFrame, as_of: pd.Timestamp, days: int | None) -> pd.DataFrame:
    """Keep incidents still active in the ``days`` up to and including ``as_of``."""
    kept = incidents[incidents["start_date"] <= as_of]
    if days is not None:
        kept = kept[kept["end_date"] > as_of - pd.Timedelta(days=days)]
    return kept.reset_index(drop=True)
