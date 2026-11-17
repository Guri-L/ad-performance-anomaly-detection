import numpy as np
import pytest

from adwatch.baseline import MAX_ABS_Z, score_level, score_rate

LEVEL = dict(window=28, min_history=14, scale_floor=0.06)
RATE = dict(window=28, min_history=14, min_expected=5.0)


def weekdays(n):
    return np.arange(n) % 7


def noisy_level(n=80, seed=0, sigma=0.07):
    rng = np.random.default_rng(seed)
    return 100 * rng.lognormal(0, sigma, size=n)


def test_no_verdict_before_enough_history():
    scores = score_level(noisy_level(), weekdays(80), **LEVEL)
    assert np.isnan(scores.z[:14]).all()
    assert np.isfinite(scores.z[14:]).all()


def test_ordinary_noise_is_not_flagged():
    for seed in range(20):
        scores = score_level(noisy_level(seed=seed), weekdays(80), **LEVEL)
        assert np.nanmax(np.abs(scores.z)) < 4.5, seed


def test_spike_and_drop_are_scored_strongly():
    values = noisy_level()
    values[60] *= 2.5
    values[70] *= 0.3
    scores = score_level(values, weekdays(80), **LEVEL)
    assert scores.z[60] > 6
    assert scores.z[70] < -6
    assert scores.change[60] == pytest.approx(values[60] / scores.expected[60] - 1)


def test_weekend_pattern_is_learned_not_flagged():
    days = weekdays(84)
    values = noisy_level(84, sigma=0.03) * np.where(days >= 5, 2.0, 1.0)
    scores = score_level(values, days, **LEVEL)
    assert np.nanmax(np.abs(scores.z[28:])) < 4.5
    saturday = 28 + 5
    assert scores.expected[saturday] > 1.5 * scores.expected[saturday - 2]


def test_earlier_spike_does_not_hide_a_later_one():
    values = noisy_level()
    values[50:53] *= 3
    values[62] *= 3
    scores = score_level(values, weekdays(80), **LEVEL)
    assert scores.z[62] > 6


def test_reported_zero_is_the_strongest_possible_drop():
    values = noisy_level()
    values[40] = 0.0
    scores = score_level(values, weekdays(80), **LEVEL)
    assert scores.z[40] == -MAX_ABS_Z
    assert scores.change[40] == -1.0


def test_missing_days_are_skipped_not_scored():
    values = noisy_level()
    values[40] = np.nan
    scores = score_level(values, weekdays(80), **LEVEL)
    assert np.isnan(scores.z[40])
    assert np.isfinite(scores.z[41])


def test_scores_never_use_future_days():
    values = noisy_level()
    changed = values.copy()
    changed[50:] *= 5
    before = score_level(values, weekdays(80), **LEVEL)
    after = score_level(changed, weekdays(80), **LEVEL)
    np.testing.assert_array_equal(before.z[:50], after.z[:50])
    np.testing.assert_array_equal(before.expected[:50], after.expected[:50])


def test_counting_noise_floor_protects_small_campaigns():
    rng = np.random.default_rng(3)
    clicks = rng.poisson(12, size=80).astype(float).clip(min=1)
    cpc = 10.0 / clicks  # constant spend: all variation is counting noise
    plain = score_level(cpc, weekdays(80), **LEVEL)
    floored = score_level(cpc, weekdays(80), counts=clicks, **LEVEL)
    assert np.nanmax(np.abs(floored.z)) <= np.nanmax(np.abs(plain.z))
    assert np.nanmax(np.abs(floored.z)) < 4.5


def rate_series(n=80, trials=300, rate=0.03, seed=0):
    rng = np.random.default_rng(seed)
    n_trials = np.full(n, float(trials))
    return rng.binomial(trials, rate, size=n).astype(float), n_trials


def test_ordinary_rate_noise_is_not_flagged():
    for seed in range(20):
        successes, trials = rate_series(seed=seed)
        scores = score_rate(successes, trials, **RATE)
        assert np.nanmax(np.abs(scores.z)) < 4.5, seed


def test_one_quiet_day_is_weak_but_three_are_conclusive():
    """About 9 conversions expected a day: one zero day is not enough, three are."""
    successes, trials = rate_series()
    successes[60:63] = 0
    single_day = score_rate(successes, trials, horizons=(1,), **RATE)
    pooled = score_rate(successes, trials, **RATE)
    assert abs(single_day.z[62]) < 4.0
    assert pooled.z[62] < -4.0


def test_pooled_score_does_not_linger_after_recovery():
    successes, trials = rate_series()
    successes[60:63] = 0
    scores = score_rate(successes, trials, **RATE)
    assert abs(scores.z[63]) < 4.0
    assert abs(scores.z[64]) < 4.0


def test_low_volume_days_get_no_verdict():
    successes, trials = rate_series(trials=40)  # about 1.2 expected a day
    successes[60] = 0
    scores = score_rate(successes, trials, horizons=(1,), **RATE)
    assert np.isnan(scores.z[60])
    assert np.isfinite(scores.expected[60])


def test_overdispersion_widens_the_expected_range():
    rng = np.random.default_rng(1)
    trials = np.full(80, 2000.0)
    steady = rng.binomial(2000, 0.05, size=80).astype(float)
    wobbly = rng.binomial(2000, 0.05 * rng.lognormal(0, 0.3, size=80)).astype(float)
    narrow = score_rate(steady, trials, **RATE)
    wide = score_rate(wobbly, trials, **RATE)
    assert np.nanmean(wide.upper - wide.lower) > 1.5 * np.nanmean(narrow.upper - narrow.lower)


def test_rate_scores_never_use_future_days():
    successes, trials = rate_series()
    changed = successes.copy()
    changed[50:] = 0
    before = score_rate(successes, trials, **RATE)
    after = score_rate(changed, trials, **RATE)
    np.testing.assert_array_equal(before.z[:50], after.z[:50])
