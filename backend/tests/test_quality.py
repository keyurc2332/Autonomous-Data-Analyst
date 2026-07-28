"""Quality thresholds. Deterministic -- these decide when tokens get spent."""
import pytest

from app.services.quality import MIN_IMPROVEMENT, assess, is_improvement


def _training(task="classification", metric="f1", value=0.8, n_train=500, **metrics):
    return {
        "task_type": task, "n_train": n_train, "n_test": 100,
        "experiments": [{
            "model_name": "m", "primary_metric": metric,
            "primary_metric_value": value, "metrics": {metric: value, **metrics},
        }],
    }


def test_strong_classification_needs_no_reflection():
    report = assess(_training(value=0.82, roc_auc=0.88))
    assert report.verdict == "strong"
    assert report.needs_reflection is False


def test_low_auc_is_weak_even_with_decent_f1():
    """The Phase 3 result: F1 looked passable, AUC showed it was near chance."""
    report = assess(_training(value=0.617, roc_auc=0.579))
    assert report.verdict == "weak"
    assert report.needs_reflection is True
    assert any("ROC-AUC" in r for r in report.reasons)


def test_low_r2_is_weak():
    report = assess(_training(task="regression", metric="r2", value=0.05))
    assert report.verdict == "weak"
    assert any("R2" in r for r in report.reasons)


def test_small_sample_is_flagged_regardless_of_score():
    report = assess(_training(value=0.9, roc_auc=0.95, n_train=80))
    assert report.verdict == "weak"
    assert any("80 training rows" in r for r in report.reasons)


def test_no_experiments_is_weak():
    assert assess({"task_type": "classification", "experiments": []}).verdict == "weak"


def test_dead_features_detected():
    explanation = {"features": [
        {"feature": "big", "importance": 0.90},
        {"feature": "small", "importance": 0.09},
        {"feature": "inert", "importance": 0.005},
    ]}
    report = assess(_training(value=0.8, roc_auc=0.9), explanation)
    assert report.dead_features == ["inert"]
    assert any("contribute almost nothing" in s for s in report.suggestions)


def test_zero_total_importance_marks_everything_dead():
    explanation = {"features": [{"feature": "a", "importance": 0.0},
                                {"feature": "b", "importance": 0.0}]}
    assert assess(_training(), explanation).dead_features == ["a", "b"]


@pytest.mark.parametrize(
    ("new", "old", "expected"),
    [
        (0.70, 0.60, True),
        (0.62, 0.60, True),                      # exactly the margin
        (0.60 + MIN_IMPROVEMENT / 2, 0.60, False),  # noise-level gain
        (0.55, 0.60, False),                     # regression
        (0.60, 0.60, False),                     # identical
    ],
)
def test_improvement_requires_a_margin(new, old, expected):
    assert is_improvement(new, old) is expected


def test_passed_checks_are_recorded_explicitly():
    """The reflection agent must be told what is fine, not just what is broken.

    Regression: given only "482 training rows" as a bare fact, the model
    invented "too small to build a reliable model" as a justification, even
    though the sample-size check had passed.
    """
    report = assess(_training(value=0.617, roc_auc=0.579, n_train=482))
    assert report.verdict == "weak"

    sample = next(c for c in report.checks if c["name"] == "sample_size")
    assert sample["passed"] is True
    assert "NOT a problem" in sample["detail"]

    # The only failure is class separation.
    failed = [c["name"] for c in report.checks if not c["passed"]]
    assert failed == ["class_separation"]
    assert all("training rows" not in r for r in report.reasons)


def test_failed_sample_size_still_reported():
    report = assess(_training(value=0.9, roc_auc=0.95, n_train=80))
    sample = next(c for c in report.checks if c["name"] == "sample_size")
    assert sample["passed"] is False
    assert any("80 training rows" in r for r in report.reasons)


def test_gate_metric_is_roc_auc_for_classification():
    """Progress must be judged on the metric that raised the concern.

    Regression: the gate fired on ROC-AUC but improvement was measured on F1.
    A retry lifted AUC 0.5796 -> 0.6337 (a real fix for the stated problem)
    yet was rejected because F1 dipped 0.6172 -> 0.6109.
    """
    report = assess(_training(value=0.6172, roc_auc=0.5796))
    assert report.gate_metric == "roc_auc"
    assert report.gate_value == pytest.approx(0.5796)
    assert report.primary_metric == "f1"


def test_gate_falls_back_to_primary_without_auc():
    report = assess(_training(value=0.55))     # no roc_auc supplied
    assert report.gate_metric == "f1"
    assert report.gate_value == pytest.approx(0.55)


def test_regression_gates_on_r2():
    report = assess(_training(task="regression", metric="r2", value=0.1))
    assert report.gate_metric == "r2"


def test_auc_gain_now_counts_as_improvement():
    before = assess(_training(value=0.6172, roc_auc=0.5796))
    after = assess(_training(value=0.6109, roc_auc=0.6337))
    # F1 fell, AUC rose. The gate metric is what decides.
    assert is_improvement(after.gate_value, before.gate_value) is True
    assert is_improvement(after.primary_value, before.primary_value) is False


def test_checks_record_whether_they_gate_the_verdict():
    """Presentation needs to tell a failure apart from a handled finding."""
    report = assess(
        _training(value=0.6, roc_auc=0.58),
        {"features": [{"feature": "a", "importance": 1.0}]},
    )
    by_name = {c["name"]: c for c in report.checks}

    assert by_name["class_separation"]["gating"] is True
    assert by_name["target_leakage"]["gating"] is False
    assert all("gating" in c for c in report.checks)
