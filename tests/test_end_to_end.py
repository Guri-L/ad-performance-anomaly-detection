import re
from pathlib import Path

import pandas as pd
import pytest

from adwatch.cli import main
from adwatch.evaluate import benchmark, evaluate
from adwatch.synthetic import ANOMALY_TYPES, MIN_GAP_DAYS, generate

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_generation_is_deterministic():
    first, second = generate(seed=11), generate(seed=11)
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)
    assert not generate(seed=12)[0].equals(first[0])


def test_committed_sample_matches_the_generator(tmp_path):
    assert main(["generate-data", "--out-dir", str(tmp_path)]) == 0
    for name in ("ad_performance.csv", "ground_truth.csv", "budgets.csv"):
        assert (tmp_path / name).read_text() == (SAMPLE / name).read_text(), name


def test_injected_anomalies_are_labelled_and_spaced_out():
    performance, truth, budgets = generate()
    assert set(truth["anomaly_type"]) <= set(ANOMALY_TYPES)
    assert len(truth) == 3 * performance["campaign_id"].nunique()
    for _, group in truth.sort_values("start_date").groupby("campaign_id"):
        gaps = (group["start_date"].iloc[1:].to_numpy() - group["end_date"].iloc[:-1].to_numpy())
        assert (gaps > pd.Timedelta(days=MIN_GAP_DAYS)).all()
    assert budgets.groupby("campaign_id")["month"].nunique().eq(4).all()


def test_evaluate_counts_hits_misses_and_false_incidents():
    truth = pd.DataFrame({
        "campaign_id": ["a", "b"],
        "anomaly_type": ["spend_spike", "tracking_outage"],
        "start_date": pd.to_datetime(["2026-08-10", "2026-08-20"]),
        "end_date": pd.to_datetime(["2026-08-11", "2026-08-22"]),
    })
    incidents = pd.DataFrame({
        "campaign_id": ["a", "a", "b"],
        "metric": ["spend", "cpc", "spend"],
        "start_date": pd.to_datetime(["2026-08-10", "2026-09-01", "2026-08-21"]),
        "end_date": pd.to_datetime(["2026-08-10", "2026-09-01", "2026-08-21"]),
    })
    result = evaluate(incidents, truth, campaign_months=2.0)
    assert result["detected"] == 1                 # the outage raised no conversion incident
    assert result["by_type"]["tracking_outage"] == {"injected": 1, "detected": 0}
    assert result["false_incidents"] == 1          # the cpc incident in September
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["false_incidents_per_campaign_month"] == 0.5


def test_detection_quality_does_not_regress():
    """Loose floors on three datasets; the README reports the 30-dataset figures."""
    result = benchmark([2001, 2002, 2003])
    assert result["recall"] >= 0.70
    assert result["precision"] >= 0.85
    assert result["false_incidents_per_campaign_month"] <= 0.20
    for kind in ("spend_spike", "delivery_drop", "ctr_drop", "missing_data"):
        assert result["by_type"][kind]["recall"] >= 0.95, kind


def run_detect(tmp_path, *extra):
    return main(["detect", "--input", str(SAMPLE / "ad_performance.csv"),
                 "--budgets", str(SAMPLE / "budgets.csv"), "--out-dir", str(tmp_path), *extra])


def test_detect_writes_incidents_pacing_and_report(tmp_path, capsys):
    assert run_detect(tmp_path, "--as-of", "2026-10-22") == 0
    incidents = pd.read_csv(tmp_path / "incidents.csv", parse_dates=["start_date", "end_date"])
    assert not incidents.empty
    assert incidents["end_date"].max() <= pd.Timestamp("2026-10-22")
    pacing = pd.read_csv(tmp_path / "pacing.csv")
    assert set(pacing["status"]) <= {"over", "under", "on_track"}
    assert len(pacing) == 8

    report = (tmp_path / "report.html").read_text()
    assert "Ad performance anomaly report" in report
    assert "data:image/png;base64," in report
    text_only = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", report)
    assert "nan" not in text_only.lower()
    assert "Budget pacing" in capsys.readouterr().out


def test_report_date_hides_everything_after_it(tmp_path):
    assert run_detect(tmp_path, "--as-of", "2026-09-10", "--days", "0") == 0
    incidents = pd.read_csv(tmp_path / "incidents.csv", parse_dates=["start_date"])
    assert incidents["start_date"].max() <= pd.Timestamp("2026-09-10")


def test_fail_on_sets_exit_status(tmp_path):
    assert run_detect(tmp_path, "--as-of", "2026-10-22", "--fail-on", "high") == 1
    # A quiet stretch: nothing in the three days to 12 October.
    assert run_detect(tmp_path, "--as-of", "2026-10-12", "--days", "3", "--fail-on", "low") == 0


def test_bad_input_exits_with_status_2_and_a_message(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("date,spend\n2026-07-01,10\n")
    assert main(["detect", "--input", str(bad), "--out-dir", str(tmp_path)]) == 2
    assert "missing column" in capsys.readouterr().err

    config = tmp_path / "config.yaml"
    config.write_text("nonsense: 1\n")
    assert run_detect(tmp_path, "--config", str(config)) == 2
    assert "unknown setting" in capsys.readouterr().err


def test_evaluate_command_prints_a_summary(capsys):
    assert main(["evaluate", "--datasets", "1", "--first-seed", "5"]) == 0
    out = capsys.readouterr().out
    assert "Recall" in out and "tracking_outage" in out and "conversions a day" in out
