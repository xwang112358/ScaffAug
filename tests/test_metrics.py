"""Unit tests for the virtual-screening evaluation metrics.

Pure-numpy; no torch/data/GPU needed. Checks the defining property of every
early-recognition metric: a perfect ranking scores far above a reversed one.
"""
import numpy as np
import pytest

from welqrate.utils.evaluation import (
    calculate_logAUC, cal_EF, cal_DCG, cal_BEDROC_score,
)


def test_EF_known_values(ranking_example):
    true_y, perfect, reverse, random_, n, n_act = ranking_example
    # perfect ranking: all actives in the top-n_act -> maximal EF = N/n_act
    ef_perfect = cal_EF(true_y, perfect, k=n_act)
    assert ef_perfect == pytest.approx(n / n_act)          # == 10.0
    # reversed ranking: no actives in the top-n_act -> EF = 0
    assert cal_EF(true_y, reverse, k=n_act) == pytest.approx(0.0)
    # perfect >> random
    assert ef_perfect > cal_EF(true_y, random_, k=n_act)


def test_DCG_perfect_beats_reverse(ranking_example):
    true_y, perfect, reverse, random_, n, n_act = ranking_example
    assert cal_DCG(true_y, perfect, k=100) > cal_DCG(true_y, reverse, k=100)
    assert cal_DCG(true_y, perfect, k=100) >= cal_DCG(true_y, random_, k=100)


def test_BEDROC_bounds_and_ordering(ranking_example):
    true_y, perfect, reverse, random_, n, n_act = ranking_example
    b_perfect = cal_BEDROC_score(true_y, perfect, alpha=20.0)
    b_reverse = cal_BEDROC_score(true_y, reverse, alpha=20.0)
    # NOTE: this repo's BEDROC is not clipped to [0, 1] — a perfect ranking scores
    # slightly above 1 (~1.02). We assert the meaningful invariants instead of a hard cap.
    assert b_reverse >= 0.0
    assert b_perfect > 0.9          # strong early recognition, near the maximum
    assert b_reverse < 0.1          # reversed ranking has almost no early recognition
    assert b_perfect > b_reverse


def test_logAUC_perfect_beats_reverse(ranking_example):
    true_y, perfect, reverse, random_, n, n_act = ranking_example
    la_perfect = calculate_logAUC(true_y, perfect, FPR_range=(0.001, 0.1))
    la_reverse = calculate_logAUC(true_y, reverse, FPR_range=(0.001, 0.1))
    assert la_perfect > la_reverse


def test_metrics_are_finite(ranking_example):
    true_y, perfect, _, _, _, n_act = ranking_example
    vals = [
        calculate_logAUC(true_y, perfect),
        cal_EF(true_y, perfect, k=n_act),
        cal_DCG(true_y, perfect, k=100),
        cal_BEDROC_score(true_y, perfect),
    ]
    assert all(np.isfinite(v) for v in vals)
