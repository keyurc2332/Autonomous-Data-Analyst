"""The propagation contract.

Three bugs shared one shape: a signal added to training or quality, wired into
the database, the UI and the PDF, and never given to the summary. This module
makes that omission fail the suite instead of the report.
"""
import dataclasses

import pytest

from app.agents.findings import FINDINGS, collect
from app.services.quality import QualityReport
from app.services.training import TrainingOutcome

# Fields that carry a *finding* -- something a reader must be told about --
# rather than a measurement already covered by the metrics.
FINDING_FIELDS = {
    "TrainingOutcome": {"additive_leakage", "leaked_features", "sampled_from"},
    "QualityReport": {"dead_features", "reasons"},
}


def _registered(source: str) -> set[str]:
    root = "training" if source == "TrainingOutcome" else "quality"
    return {f.path[-1] for f in FINDINGS if f.path[0] == root}


@pytest.mark.parametrize("source", sorted(FINDING_FIELDS))
def test_every_finding_field_is_registered(source):
    """Adding a signal without registering it must fail here, not in a report."""
    assert FINDING_FIELDS[source] <= _registered(source), (
        f"{FINDING_FIELDS[source] - _registered(source)} exist on {source} but "
        "are not in FINDINGS, so the summary will never mention them."
    )


@pytest.mark.parametrize("source", sorted(FINDING_FIELDS))
def test_registry_does_not_reference_fields_that_do_not_exist(source):
    """The mirror check: a renamed field must not leave a dead registry entry."""
    cls = TrainingOutcome if source == "TrainingOutcome" else QualityReport
    actual = {f.name for f in dataclasses.fields(cls)}
    assert _registered(source) <= actual, (
        f"{_registered(source) - actual} is registered but no longer exists on {source}."
    )


@pytest.mark.parametrize("finding", FINDINGS, ids=lambda f: f.name)
def test_each_finding_renders_when_present(finding):
    """Every registered finding must produce text from a realistic value."""
    samples = {
        "additive_leakage": {"r2": 1.0, "reason": "cnt is a sum.",
                             "contributors": [{"column": "casual", "coefficient": 1.0}]},
        "leaked_features": [{"column": "alive", "score": 1.0, "reason": "restates"}],
        "sampled_from": 53940,
        "dead_features": ["noise"],
        "failed_checks": ["ROC-AUC 0.58 is below 0.65."],
    }
    state: dict = {}
    node = state
    for key in finding.path[:-1]:
        node = node.setdefault(key, {})
    node[finding.path[-1]] = samples[finding.name]

    critical, ordinary = collect(state)
    assert critical + ordinary, f"{finding.name} produced no text"


def test_absent_findings_produce_nothing():
    assert collect({}) == ([], [])
    assert collect({"training": {}, "quality": {}}) == ([], [])


def test_empty_values_are_not_reported():
    """An empty list is not a finding."""
    state = {"training": {"leaked_features": [], "sampled_from": 0},
             "quality": {"dead_features": [], "reasons": []}}
    assert collect(state) == ([], [])


def test_critical_findings_are_separated():
    state = {
        "training": {"additive_leakage": {"reason": "cnt is a sum."}},
        "quality": {"dead_features": ["noise"]},
    }
    critical, ordinary = collect(state)
    assert len(critical) == 1 and "CRITICAL" in critical[0]
    assert len(ordinary) == 1 and "noise" in ordinary[0]
