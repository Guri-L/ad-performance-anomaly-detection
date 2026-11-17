from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adwatch.config import DetectionConfig
from adwatch.detect import INCIDENT_COLUMNS, MISSING, build_incidents, filter_recent, score
from adwatch.pacing import pacing_report
from adwatch.schema import load_performance, validate_performance

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def campaign(days=70, spend=100.0, seed=0, campaign_id="c1", **edits):
    """A steady campaign: ~200 clicks, ~10,000 impressions, ~20 conversions a day."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-07-01", periods=days, freq="D")
    clicks = rng.poisson(200, size=days)
    frame = pd.DataFrame({
        "date": dates,
        "platform": "google_ads",
        "campaign_id": campaign_id,
        "campaign_name": f"campaign {campaign_id}",
        "impressions": (clicks / 0.02).round(),
        "clicks": clicks,
        "spend": spend * rng.lognormal(0, 0.05, size=days),
        "conversions": rng.binomial(clicks, 0.10),
    })
    frame["conversion_value"] = frame["conversions"] * 60.0
    for column, (rows, value) in edits.items():
        frame.loc[rows, column] = value
    return validate_performance(frame)


@pytest.fixture(scope="module")
def sample_incidents():
    scored = score(load_performance(SAMPLE / "ad_performance.csv"))
    return scored, build_incidents(scored)


def test_clean_campaign_raises_nothing():
    assert build_incidents(score(campaign())).empty


def test_empty_result_still_has_all_columns():
    assert list(build_incidents(score(campaign())).columns) == INCIDENT_COLUMNS


def test_consecutive_days_merge_into_one_incident():
    data = campaign()
    data.loc[50:52, "spend"] *= 3
    incidents = build_incidents(score(data))
    spend = incidents[incidents["metric"] == "spend"]
    assert len(spend) == 1
    row = spend.iloc[0]
    assert (row["diagnosis"], row["direction"], row["days"]) == ("Spend spike", "up", 3)
    assert row["start_date"] == pd.Timestamp("2026-08-20")
    assert row["end_date"] == pd.Timestamp("2026-08-22")
    # Extra spend is roughly two extra days' worth for each tripled day.
    assert row["estimated_impact"] == pytest.approx(600, rel=0.15)


def test_separate_runs_stay_separate():
    data = campaign()
    data.loc[[40, 55], "spend"] *= 3
    incidents = build_incidents(score(data))
    assert (incidents["metric"] == "spend").sum() == 2


def test_zero_conversions_are_diagnosed_as_tracking_outage():
    incidents = build_incidents(score(campaign(conversions=(slice(50, 52), 0),
                                               conversion_value=(slice(50, 52), 0.0))))
    row = incidents[incidents["metric"] == "conversion_rate"].iloc[0]
    assert row["diagnosis"] == "Possible tracking outage (zero conversions)"
    assert row["severity"] == "high"
    assert row["estimated_impact"] < -2000  # about 20 conversions x 60 a day, lost


def test_partial_conversion_drop_is_not_called_an_outage():
    data = campaign()
    data.loc[50:52, "conversions"] = (data.loc[50:52, "conversions"] * 0.2).round()
    incidents = build_incidents(score(data))
    assert "Conversion rate drop" in set(incidents["diagnosis"])


def test_gap_inside_active_span_is_reported_as_missing_data():
    data = campaign().drop(index=[45, 46]).reset_index(drop=True)
    incidents = build_incidents(score(data))
    missing = incidents[incidents["metric"] == MISSING].iloc[0]
    assert (missing["severity"], missing["days"]) == ("high", 2)
    assert missing["start_date"] == pd.Timestamp("2026-08-15")


def test_campaigns_are_scored_independently():
    quiet = campaign(campaign_id="quiet")
    noisy = campaign(campaign_id="noisy", seed=1)
    noisy.loc[50, "spend"] *= 4
    both = validate_performance(pd.concat([quiet, noisy]))
    incidents = build_incidents(score(both))
    assert set(incidents["campaign_id"]) == {"noisy"}


def test_adding_later_data_does_not_change_earlier_scores():
    data = campaign()
    early = score(data[data["date"] < "2026-08-20"])
    full = score(data)
    merged = early.merge(full, on=["date", "campaign_id", "metric"], suffixes=("_a", "_b"))
    assert len(merged) == len(early)
    np.testing.assert_array_equal(merged["z_score_a"].to_numpy(), merged["z_score_b"].to_numpy())


def test_incidents_are_ordered_most_severe_first(sample_incidents):
    _, incidents = sample_incidents
    rank = incidents["severity"].map({"high": 0, "medium": 1, "low": 2})
    assert rank.is_monotonic_increasing


def test_sample_data_incidents_match_its_ground_truth(sample_incidents):
    _, incidents = sample_incidents
    truth = pd.read_csv(SAMPLE / "ground_truth.csv", parse_dates=["start_date", "end_date"])
    for kind in ("spend_spike", "delivery_drop", "ctr_drop", "missing_data"):
        for anomaly in truth[truth["anomaly_type"] == kind].itertuples():
            hits = incidents[
                (incidents["campaign_id"] == anomaly.campaign_id)
                & (incidents["start_date"] <= anomaly.end_date)
                & (incidents["end_date"] >= anomaly.start_date)
            ]
            assert not hits.empty, (kind, anomaly.campaign_id, anomaly.start_date)


def test_filter_recent_keeps_incidents_active_in_the_window(sample_incidents):
    _, incidents = sample_incidents
    as_of = pd.Timestamp("2026-10-22")
    recent = filter_recent(incidents, as_of, days=14)
    assert not recent.empty
    assert (recent["start_date"] <= as_of).all()
    assert (recent["end_date"] > as_of - pd.Timedelta(days=14)).all()
    assert len(filter_recent(incidents, as_of, days=None)) > len(recent)


def budgets(amount, month="2026-07"):
    return pd.DataFrame({"campaign_id": ["c1"], "month": [pd.Period(month, "M")],
                         "budget": [amount]})


def flat(days=31, spend=100.0):
    data = campaign(days=days)
    data["spend"] = spend
    return data


def test_pacing_projects_run_rate_over_remaining_days():
    report = pacing_report(flat(), budgets(3100.0), pd.Timestamp("2026-07-10"))
    row = report.iloc[0]
    assert row["spend_to_date"] == pytest.approx(1000)
    assert row["days_remaining"] == 21
    assert row["projected_spend"] == pytest.approx(3100)
    assert (row["status"], row["pace"]) == ("on_track", pytest.approx(1.0))
    assert pd.isna(row["budget_runs_out"])


def test_pacing_flags_overspend_and_dates_the_shortfall():
    row = pacing_report(flat(), budgets(2000.0), pd.Timestamp("2026-07-10")).iloc[0]
    assert row["status"] == "over"
    assert row["projected_variance"] == pytest.approx(1100)
    assert row["budget_runs_out"] == pd.Timestamp("2026-07-20")  # 1,000 left at 100 a day


def test_pacing_flags_underspend():
    row = pacing_report(flat(), budgets(5000.0), pd.Timestamp("2026-07-10")).iloc[0]
    assert (row["status"], row["pace"]) == ("under", pytest.approx(0.62))


def test_recent_run_rate_drives_the_projection_not_the_month_average():
    data = flat()
    data.loc[data["date"] >= "2026-07-08", "spend"] = 300.0  # budget raised a week ago
    row = pacing_report(data, budgets(3100.0), pd.Timestamp("2026-07-14")).iloc[0]
    assert row["daily_run_rate"] == pytest.approx(300)
    assert row["status"] == "over"


def test_paused_campaign_has_zero_run_rate():
    data = flat()
    data = data[data["date"] <= "2026-07-05"]
    row = pacing_report(data, budgets(3100.0), pd.Timestamp("2026-07-20")).iloc[0]
    assert row["daily_run_rate"] == 0
    assert row["projected_spend"] == pytest.approx(500)
    assert row["status"] == "under"


def test_pacing_ignores_budgets_for_other_months_and_unknown_campaigns():
    other_month = pacing_report(flat(), budgets(3100.0, "2026-08"), pd.Timestamp("2026-07-10"))
    assert other_month.empty
    unknown = budgets(3100.0).assign(campaign_id="zzz")
    assert pacing_report(flat(), unknown, pd.Timestamp("2026-07-10")).empty


def test_pacing_thresholds_come_from_config():
    strict = DetectionConfig(pacing_over_threshold=1.02, pacing_under_threshold=0.98)
    row = pacing_report(flat(), budgets(3000.0), pd.Timestamp("2026-07-10"), strict).iloc[0]
    assert row["status"] == "over"
