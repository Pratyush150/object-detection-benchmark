"""Latency statistics and the percentile reporting."""

from __future__ import annotations

import numpy as np
import pytest

from detbench.profiling import profile_detector, summarise_samples


def test_percentiles_are_ordered():
    stats = summarise_samples("total", list(range(1, 101)))
    assert stats.minimum <= stats.p50 <= stats.p90 <= stats.p99 <= stats.maximum


def test_mean_hides_a_spike_that_the_p99_exposes():
    # Two hundred frames, four of which stall. The mean moves by under two
    # milliseconds; the p99 moves by seventy. This is the whole argument for
    # specifying frame time at a percentile.
    smooth = [18.0] * 200
    spiky = [18.0] * 196 + [90.0] * 4
    a = summarise_samples("total", smooth)
    b = summarise_samples("total", spiky)
    assert abs(b.mean - a.mean) < 2.0
    assert b.p99 - a.p99 > 50.0


def test_jitter_ratio_is_one_for_a_constant_stream():
    stats = summarise_samples("total", [20.0] * 30)
    assert stats.jitter_ratio == pytest.approx(1.0)


def test_jitter_ratio_grows_with_the_tail():
    stats = summarise_samples("total", [10.0] * 99 + [100.0])
    assert stats.jitter_ratio > 1.0


def test_empty_sample_set_raises():
    with pytest.raises(ValueError):
        summarise_samples("total", [])


def test_stats_dictionary_has_the_expected_keys():
    keys = set(summarise_samples("total", [1.0, 2.0]).as_dict())
    assert keys == {
        "count", "mean_ms", "p50_ms", "p90_ms", "p99_ms",
        "min_ms", "max_ms", "jitter_p99_over_p50",
    }


def test_profile_covers_every_pipeline_stage(synthetic):
    frames = [synthetic.images[i] for i in synthetic.image_ids[:4]]
    profile = profile_detector(synthetic.detector, frames, warmup=1, repeats=2)
    assert set(profile.stages) == {
        "preprocess", "inference", "nms", "postprocess", "total"
    }


def test_profile_sample_count_matches_frames_times_repeats(synthetic):
    frames = [synthetic.images[i] for i in synthetic.image_ids[:4]]
    profile = profile_detector(synthetic.detector, frames, warmup=0, repeats=3)
    assert profile.stages["total"].count == 12


def test_p99_fps_is_never_above_mean_fps(synthetic):
    frames = [synthetic.images[i] for i in synthetic.image_ids[:4]]
    profile = profile_detector(synthetic.detector, frames, warmup=1, repeats=2)
    assert profile.p99_fps <= profile.mean_fps + 1e-9


def test_profile_table_lists_the_stages(synthetic):
    frames = [synthetic.images[i] for i in synthetic.image_ids[:2]]
    text = profile_detector(synthetic.detector, frames, warmup=0).format_table()
    for stage in ("preprocess", "inference", "nms", "total"):
        assert stage in text


def test_profiling_with_no_frames_raises(synthetic):
    with pytest.raises(ValueError):
        profile_detector(synthetic.detector, [])


def test_total_is_at_least_the_sum_of_the_parts(synthetic):
    frames = [synthetic.images[i] for i in synthetic.image_ids[:3]]
    profile = profile_detector(synthetic.detector, frames, warmup=1)
    parts = sum(
        profile.stages[s].mean
        for s in ("preprocess", "inference", "nms", "postprocess")
    )
    assert profile.stages["total"].mean >= parts - 1e-6
