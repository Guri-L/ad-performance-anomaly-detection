from pathlib import Path

import pandas as pd
import pytest

from adwatch.config import ConfigError, DetectionConfig
from adwatch.schema import DataError, load_budgets, load_performance, validate_performance


def frame(**overrides):
    base = {
        "date": ["2026-07-01", "2026-07-02"],
        "platform": ["google_ads", "google_ads"],
        "campaign_id": ["1", "1"],
        "campaign_name": ["brand", "brand"],
        "impressions": [1000, 1100],
        "clicks": [50, 60],
        "spend": [20.0, 22.5],
        "conversions": [3, 4],
        "conversion_value": [150.0, 210.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_valid_data_is_typed_and_sorted():
    out = validate_performance(frame(date=["2026-07-02", "2026-07-01"]))
    assert out["date"].is_monotonic_increasing
    assert str(out["date"].dtype).startswith("datetime64")
    assert out["spend"].dtype == "float64"


def test_missing_column_is_named():
    with pytest.raises(DataError, match=r"missing column\(s\) \['spend'\]"):
        validate_performance(frame().drop(columns="spend"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"date": ["2026-07-01", "not a date"]}, "unparseable date"),
        ({"spend": [20.0, -1.0]}, "negative value"),
        ({"clicks": [50, "many"]}, "non-numeric"),
        ({"date": ["2026-07-01", "2026-07-01"]}, "share a date/platform/campaign_id"),
        ({"clicks": [50, 5000]}, "more clicks than impressions"),
        ({"campaign_name": ["brand", " "]}, "empty value"),
    ],
)
def test_bad_data_is_rejected_with_a_reason(overrides, message):
    with pytest.raises(DataError, match=message):
        validate_performance(frame(**overrides))


def test_empty_input_is_rejected():
    with pytest.raises(DataError, match="no rows"):
        validate_performance(frame().iloc[0:0])


def test_long_numeric_ids_survive_loading(tmp_path):
    path = tmp_path / "perf.csv"
    frame(campaign_id=["120214400518830101", "120214400518830101"]).to_csv(path, index=False)
    assert load_performance(path)["campaign_id"].iloc[0] == "120214400518830101"


def test_missing_file_is_a_data_error(tmp_path):
    with pytest.raises(DataError, match="File not found"):
        load_performance(tmp_path / "nope.csv")


def test_budgets_are_validated(tmp_path):
    path = tmp_path / "budgets.csv"
    path.write_text("campaign_id,month,budget\n1,2026-07,1000\n1,2026-08,1200\n")
    budgets = load_budgets(path)
    assert str(budgets["month"].iloc[0]) == "2026-07"

    path.write_text("campaign_id,month,budget\n1,July,1000\n")
    with pytest.raises(DataError, match="YYYY-MM"):
        load_budgets(path)
    path.write_text("campaign_id,month,budget\n1,2026-07,0\n")
    with pytest.raises(DataError, match="positive"):
        load_budgets(path)
    path.write_text("campaign_id,month,budget\n1,2026-07,10\n1,2026-07,20\n")
    with pytest.raises(DataError, match="more than one budget"):
        load_budgets(path)


def test_defaults_are_valid():
    DetectionConfig().validate()


def test_yaml_overrides_selected_settings(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("level_z_threshold: 6\nwindow_days: 35\n")
    config = DetectionConfig.from_yaml(path)
    assert (config.level_z_threshold, config.window_days) == (6.0, 35)
    assert config.rate_z_threshold == DetectionConfig().rate_z_threshold


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("zthreshold: 3\n", "unknown setting"),
        ("window_days: 5\n", "window_days must be >= min_history_days"),
        ("level_z_threshold: high\n", "must be numbers"),
        ("pacing_over_threshold: 0.9\n", "straddle"),
        ("- a\n- b\n", "expected a mapping"),
    ],
)
def test_bad_config_is_rejected(tmp_path, text, message):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError, match=message):
        DetectionConfig.from_yaml(path)


def test_example_config_documents_the_real_defaults():
    example = Path(__file__).resolve().parents[1] / "config" / "example.yaml"
    assert DetectionConfig.from_yaml(example) == DetectionConfig()
