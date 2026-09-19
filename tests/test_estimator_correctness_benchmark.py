"""Exercise independent estimator oracles and their failure gates."""

import numpy as np
import pytest

from benchmarks.estimator_correctness import agree, run_suite


def test_public_estimator_correctness_smoke():
    pytest.importorskip("cvxpy")
    report = run_suite(cases=64)
    assert report["passed"] == 64, [row for row in report["rows"] if row["status"] != "passed"]
    assert report["comparisons"] == 136


def test_raw_clone_oracle_uses_documented_constraint_units():
    pytest.importorskip("cvxpy")
    # A general row with max|A_ij|>1 has different raw and normalized units.
    report = run_suite(case_index=735)
    assert report["passed"] == 1, report
    assert report["comparisons"] == 2


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 1.1])
def test_objective_gate_rejects_bad_values(value):
    with pytest.raises(AssertionError):
        agree(value, 1.0)


def test_corrupt_prediction_fails_cqr_oracle(monkeypatch):
    pytest.importorskip("cvxpy")
    from rehline import CQR_Ridge

    predict = CQR_Ridge.predict
    monkeypatch.setattr(CQR_Ridge, "predict", lambda self, X: predict(self, X) + 1.0)
    report = run_suite(cases=1)
    assert report["passed"] == 0
    assert report["rows"][0]["status"] == "failed"
