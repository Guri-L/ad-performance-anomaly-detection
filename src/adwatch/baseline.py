"""Expected values and anomaly scores for one campaign's daily series.

Two scoring methods, chosen by what kind of number the metric is:

* Level metrics (spend, impressions, CPC) are positive and move
  multiplicatively, so they are scored on the log scale with a median/MAD
  baseline. Median and MAD ignore up to half the window being unusual, so an
  earlier spike does not hide a later one.
* Rate metrics (CTR, conversion rate) are counts out of trials. Their natural
  noise depends on volume: 2 conversions from 40 clicks says little, 200 from
  4,000 says a lot. They are scored with a binomial z-score, widened by the
  overdispersion actually seen in the window.

Every score for day t uses only days before t. Nothing looks ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

MAD_TO_SIGMA = 1.4826       # scales MAD to a standard deviation for normal data
WEEKDAY_SHRINKAGE = 1.0     # pseudo-observations pulling weekday effects to zero
MEDIAN_VARIANCE_FACTOR = 1.57  # variance of a sample median relative to a mean (pi / 2)
MAX_ABS_Z = 99.0
POOLED_MIN_OWN_Z = 2.0      # a pooled score needs the day itself to be unusual too


@dataclass(frozen=True)
class Scores:
    """Per-day results, aligned with the input. NaN where there is no verdict."""

    expected: FloatArray
    z: FloatArray
    lower: FloatArray   # expected range at +/- 3 sigma, for charts
    upper: FloatArray
    change: FloatArray  # relative change from expected that the score is based on


def _nan(n: int) -> FloatArray:
    return np.full(n, np.nan)


def _mad_sigma(values: FloatArray) -> float:
    return float(MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def score_level(
    values: FloatArray,
    weekdays: NDArray[np.int_],
    *,
    window: int,
    min_history: int,
    scale_floor: float,
    counts: FloatArray | None = None,
) -> Scores:
    """Robust z-score of each day's log value against its trailing window.

    A weekday effect is estimated inside the window (weekends often differ from
    weekdays) and shrunk towards zero, because four Mondays are thin evidence.
    Days that are NaN or not positive are left out of baselines.

    ``counts`` is for metrics built from an event count (CPC and impressions
    both come from clicks here). A count of n varies by about 1/sqrt(n) from
    day to day by chance alone, so the noise estimate is never allowed below
    that. This keeps small campaigns from raising alerts on counting noise.
    """
    n = len(values)
    expected, z, lower, upper, change = _nan(n), _nan(n), _nan(n), _nan(n), _nan(n)
    usable = np.isfinite(values) & (values > 0)
    logs = np.where(usable, np.log(np.where(usable, values, 1.0)), np.nan)

    for t in range(n):
        start = max(0, t - window)
        mask = usable[start:t]
        if mask.sum() < min_history:
            continue
        hist = logs[start:t][mask]
        hist_weekdays = weekdays[start:t][mask]

        overall = np.median(hist)
        effects = np.zeros(7)
        effect_variance = 0.0   # extra uncertainty from estimating today's weekday effect
        for day in range(7):
            same_day = hist[hist_weekdays == day]
            if same_day.size:
                weight = same_day.size / (same_day.size + WEEKDAY_SHRINKAGE)
                effects[day] = weight * (np.median(same_day) - overall)
                if day == weekdays[t]:
                    effect_variance = weight**2 * MEDIAN_VARIANCE_FACTOR / same_day.size

        deseasonalised = hist - effects[hist_weekdays]
        centre = float(np.median(deseasonalised))
        # A forecast is less certain than the fit: the centre and the weekday effect
        # are themselves estimates from a handful of days.
        inflation = np.sqrt(1 + effect_variance + MEDIAN_VARIANCE_FACTOR / hist.size)
        sigma = max(_mad_sigma(deseasonalised) * inflation, scale_floor)
        if counts is not None:
            typical = np.nanmedian(counts[start:t][mask])
            if typical > 0:
                sigma = max(sigma, 1.0 / np.sqrt(typical))
        expected_log = centre + effects[weekdays[t]]

        expected[t] = np.exp(expected_log)
        lower[t] = np.exp(expected_log - 3 * sigma)
        upper[t] = np.exp(expected_log + 3 * sigma)
        if usable[t]:
            z[t] = np.clip((logs[t] - expected_log) / sigma, -MAX_ABS_Z, MAX_ABS_Z)
            change[t] = values[t] / expected[t] - 1
        elif np.isfinite(values[t]):
            z[t] = -MAX_ABS_Z  # a reported zero where a positive value was expected
            change[t] = -1.0
    return Scores(expected, z, lower, upper, change)


def score_rate(
    successes: FloatArray,
    trials: FloatArray,
    *,
    window: int,
    min_history: int,
    min_expected: float,
    horizons: tuple[int, ...] = (1, 3),
) -> Scores:
    """Overdispersed-binomial z-score of each day's rate against its window.

    The baseline rate is the median of the window's daily rates, which a few
    broken days cannot drag down. Real rates vary more from day to day than a
    pure binomial allows, so the variance is multiplied by a dispersion factor
    estimated robustly from the window (never below 1).

    One day is often too little evidence: a campaign expecting 9 conversions a
    day sees zero by chance about once in 8,000 days, which a single-day z of
    -3 does not flag. Three such days in a row are conclusive. So each day is
    also scored on the pooled last ``h`` days for every horizon ``h``, against
    a baseline that ends before those days, and the strongest score is kept.
    A pooled score only counts if the day itself also deviates the same way.
    """
    n = len(successes)
    expected, z, lower, upper, change = _nan(n), _nan(n), _nan(n), _nan(n), _nan(n)
    usable = np.isfinite(successes) & np.isfinite(trials) & (trials > 0)

    for t in range(n):
        if not usable[t]:
            continue
        for horizon in horizons:
            first = t - horizon + 1          # first day of the period being judged
            start = max(0, first - window)
            if first < 0:
                continue
            mask = usable[start:first]
            if mask.sum() < min_history:
                continue
            hist_s, hist_n = successes[start:first][mask], trials[start:first][mask]
            rate = float(np.median(hist_s / hist_n))
            if not 0 < rate < 1:
                continue
            hist_sd = np.sqrt(hist_n * rate * (1 - rate))
            dispersion = max(1.0, _mad_sigma((hist_s - hist_n * rate) / hist_sd) ** 2)

            period = usable[first:t + 1]
            period_s = float(successes[first:t + 1][period].sum())
            period_n = float(trials[first:t + 1][period].sum())
            sd_rate = np.sqrt(rate * (1 - rate) * dispersion / period_n)
            if horizon == 1:
                expected[t] = rate
                lower[t] = max(0.0, rate - 3 * sd_rate)
                upper[t] = min(1.0, rate + 3 * sd_rate)
            if period_n * rate < min_expected:
                continue  # too little volume for a zero or a low count to mean anything
            score = float(np.clip((period_s / period_n - rate) / sd_rate, -MAX_ABS_Z, MAX_ABS_Z))
            if horizon > 1:
                # Without this, the days after an incident ends would stay flagged
                # for as long as the pooled period still contains the bad days.
                own_sd = np.sqrt(rate * (1 - rate) * dispersion / trials[t])
                own_z = (successes[t] / trials[t] - rate) / own_sd
                if own_z * score <= 0 or abs(own_z) < POOLED_MIN_OWN_Z:
                    continue
            if np.isnan(z[t]) or abs(score) > abs(z[t]):
                z[t] = score
                change[t] = (period_s / period_n - rate) / rate
    return Scores(expected, z, lower, upper, change)
