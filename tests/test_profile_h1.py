from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation.profile_h1 import (
    deterministic_halves,
    paired_profile_metrics,
    pds,
    summarize_profiles,
)


def test_ceiling_halves_are_disjoint_200_vs_200() -> None:
    rows = np.arange(400)
    first, second = deterministic_halves(rows, np.random.default_rng(3))
    assert len(first) == len(second) == 200
    assert np.intersect1d(first, second).size == 0
    assert set(first) | set(second) == set(rows)


def test_profile_metrics_are_paired_and_exclude_target() -> None:
    genes = ["a", "b", "x"]
    targets = ["a", "b"]
    control = np.zeros(3)
    truth = np.array([[100.0, 1.0, 2.0], [3.0, 100.0, 4.0]])
    predicted = truth.copy()
    predicted[0, 0] = -500
    predicted[1, 1] = 900
    result = paired_profile_metrics(predicted, control[None].repeat(2, axis=0), truth, control, targets, genes)
    assert np.allclose(result["expected_cosine"], 1)
    assert np.allclose(result["expected_squared_error"], 0)


def test_official_pds_kernel_scores_zero_profiles_at_chance() -> None:
    genes = ["a", "b", "x", "y"]
    targets = ["a", "b"]
    control = np.zeros(4)
    truth = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]])
    scores = pds(np.zeros_like(truth), truth, control, targets, genes)
    assert scores == {"a": 0.5, "b": 0.5}


def test_official_pds_kernel_scores_shared_direction_at_chance() -> None:
    genes = ["a", "b", "x", "y"]
    targets = ["a", "b"]
    control = np.zeros(4)
    truth = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0]])
    shared = np.array([[0.0, 0.0, 1.0, 1.0]] * 2)
    scores = pds(shared, truth, control, targets, genes)
    assert scores == {"a": 0.5, "b": 0.5}


def test_sampled_mse_audit_removes_null_excess_additively() -> None:
    per_target = pd.DataFrame(
        {
            "expected_cosine": [0.0] * 4,
            "null_cosine": [0.0] * 4,
            "expected_spearman": [0.0] * 4,
            "null_spearman": [0.0] * 4,
            "expected_pds": [0.5] * 4,
            "null_pds": [0.5] * 4,
            "expected_squared_error": [1.0] * 4,
            "null_squared_error": [1.0] * 4,
            "control_squared_error": [1.0] * 4,
            "response_stratum": ["Q1", "Q2", "Q3", "Q4"],
        }
    )
    sampled = per_target.rename(
        columns={
            "expected_squared_error": "sampled_squared_error",
            "null_squared_error": "sampled_null_squared_error",
        }
    ).copy()
    sampled["sampled_squared_error"] = 3.0
    sampled["sampled_null_squared_error"] = 2.0
    sampled["control_squared_error"] = 1.0
    summary = summarize_profiles(per_target, sampled)
    value = summary.loc[summary["arm"] == "sampled_additive_audit", "value"].item()
    assert value == 2.0
