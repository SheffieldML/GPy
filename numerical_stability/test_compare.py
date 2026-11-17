"""Unit tests for CIP-0011 compare outcome matrix (no dual GPy install required)."""

import pytest

from numerical_stability.compare import (
    IMPROVEMENT,
    IGNORE,
    KNOWN_DEBT,
    REGRESSION,
    STABLE,
    SUITE_DRIFT,
    classify,
    diff_outcomes,
    exit_code_for,
    render_report,
)

pytestmark = pytest.mark.numerical


@pytest.mark.parametrize(
    "baseline,candidate,expected",
    [
        ("xfailed", "passed", IMPROVEMENT),
        ("xfailed", "xpassed", IMPROVEMENT),
        ("passed", "failed", REGRESSION),
        ("passed", "error", REGRESSION),
        ("xfailed", "failed", REGRESSION),
        ("xfailed", "error", REGRESSION),
        ("passed", "passed", STABLE),
        ("xfailed", "xfailed", KNOWN_DEBT),
        ("skipped", "skipped", IGNORE),
        ("failed", "passed", IMPROVEMENT),
        (None, "passed", SUITE_DRIFT),
        ("passed", None, SUITE_DRIFT),
    ],
)
def test_classify_matrix(baseline, candidate, expected):
    assert classify(baseline, candidate) == expected


def test_diff_outcomes_buckets():
    baseline = {
        "a": "xfailed",
        "b": "passed",
        "c": "xfailed",
        "d": "skipped",
    }
    candidate = {
        "a": "passed",
        "b": "failed",
        "c": "xfailed",
        "d": "skipped",
        "e": "passed",
    }
    buckets = diff_outcomes(baseline, candidate)
    assert [n for n, *_ in buckets[IMPROVEMENT]] == ["a"]
    assert [n for n, *_ in buckets[REGRESSION]] == ["b"]
    assert [n for n, *_ in buckets[KNOWN_DEBT]] == ["c"]
    assert [n for n, *_ in buckets[IGNORE]] == ["d"]
    assert [n for n, *_ in buckets[SUITE_DRIFT]] == ["e"]


def test_exit_code_regressions_and_strict_suite():
    buckets = {REGRESSION: [("t", "passed", "failed")]}
    assert exit_code_for(buckets, strict_suite=False) == 1

    buckets = {SUITE_DRIFT: [("t", None, "passed")], IMPROVEMENT: []}
    assert exit_code_for(buckets, strict_suite=False) == 0
    assert exit_code_for(buckets, strict_suite=True) == 1

    buckets = {IMPROVEMENT: [("t", "xfailed", "passed")]}
    assert exit_code_for(buckets, strict_suite=False) == 0


def test_render_report_mentions_refs_and_verdict():
    buckets = {
        IMPROVEMENT: [("n::t", "xfailed", "passed")],
        REGRESSION: [],
    }
    md = render_report(
        buckets,
        baseline_ref="main",
        candidate_ref="HEAD",
        suite_from="baseline",
    )
    assert "main" in md and "HEAD" in md
    assert "**Pass:**" in md
    assert "`n::t`" in md
