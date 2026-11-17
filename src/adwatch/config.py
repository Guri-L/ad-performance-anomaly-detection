"""Detection settings. Defaults live here; a YAML file can override any of them."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """A setting is unknown or out of range."""


@dataclass(frozen=True)
class DetectionConfig:
    # Baseline: the trailing days each day is compared against.
    window_days: int = 28
    min_history_days: int = 14

    # Level metrics (spend, impressions, CPC): robust z-score on log values.
    level_z_threshold: float = 4.5
    level_min_change: float = 0.30      # also require a 30% move from expected
    level_scale_floor: float = 0.06     # never assume day-to-day noise below 6%

    # Rate metrics (CTR, conversion rate): overdispersed binomial z-score.
    rate_z_threshold: float = 4.0
    rate_min_change: float = 0.25
    rate_min_expected_events: float = 5.0  # skip days too small to judge

    # Incidents: alerts on the same campaign and metric merge across this gap.
    incident_gap_days: int = 1

    # Budget pacing: projected month-end spend relative to budget.
    pacing_over_threshold: float = 1.10
    pacing_under_threshold: float = 0.85
    pacing_run_rate_days: int = 7

    def validate(self) -> None:
        problems = []
        if self.min_history_days < 7:
            problems.append("min_history_days must be at least 7 (one full week)")
        if self.window_days < self.min_history_days:
            problems.append("window_days must be >= min_history_days")
        for name in ("level_z_threshold", "rate_z_threshold", "level_scale_floor",
                     "rate_min_expected_events"):
            if getattr(self, name) <= 0:
                problems.append(f"{name} must be positive")
        for name in ("level_min_change", "rate_min_change"):
            if not 0 <= getattr(self, name) < 1:
                problems.append(f"{name} must be between 0 and 1")
        if not self.pacing_under_threshold < 1 < self.pacing_over_threshold:
            problems.append("pacing thresholds must straddle 1.0")
        if self.pacing_run_rate_days < 1 or self.incident_gap_days < 0:
            problems.append("pacing_run_rate_days must be >= 1 and incident_gap_days >= 0")
        if problems:
            raise ConfigError("Invalid configuration: " + "; ".join(problems))

    @classmethod
    def from_yaml(cls, path: str | Path) -> DetectionConfig:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path}: expected a mapping of setting names to values")
        known = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(loaded) - set(known))
        if unknown:
            raise ConfigError(f"{path}: unknown setting(s): {', '.join(unknown)}")
        try:
            typed: dict[str, Any] = {
                key: (int(value) if known[key] == "int" else float(value))
                for key, value in loaded.items()
            }
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path}: settings must be numbers ({exc})") from exc
        config = replace(cls(), **typed)
        config.validate()
        return config
