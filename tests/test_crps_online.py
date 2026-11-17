"""Regression tests for CRPS with online calibration (issue #39).

``ConformalPredictiveSystem.evaluate`` used to raise ``RuntimeError``
when CRPS was requested with ``online=True``. The score is now computed
per test object, against the calibration set in effect when that object
is predicted.

The reference values below come from an independent analytic integration
of the step CDF defined by each object's cpd grid points -- the same
definition-level oracle used for the batch CRPS tests (issue #40) -- so
these tests check the score against the definition of CRPS, not merely
against the batch path. In particular, the normalized tests would fail
against the old ``get_crps`` formula, which over-counted sigma in the
two partial-interval terms around the observed value (issue #40).
"""
from bisect import insort

import numpy as np
import pytest

from crepes import ConformalPredictiveSystem, WrapRegressor
from crepes.base import calculate_crps


def crps_step_integral(cpd, y):
    """Analytic CRPS of the step CDF through the cpd grid points.

    CRPS(F, y) = int_{-inf}^y F(z)^2 dz + int_y^inf (1 - F(z))^2 dz,
    where F(z) = (j + 1) / N on [cpd_j, cpd_{j + 1}).
    """
    cpd = np.sort(np.asarray(cpd, dtype=float))
    n = len(cpd)
    F = (np.arange(n) + 1) / n
    score = 0.0
    for j in range(n - 1):
        lo, hi = cpd[j], cpd[j + 1]
        if hi <= y:
            score += F[j] ** 2 * (hi - lo)
        elif lo < y:
            score += F[j] ** 2 * (y - lo)
        if lo >= y:
            score += (1 - F[j]) ** 2 * (hi - lo)
        elif hi > y:
            score += (1 - F[j]) ** 2 * (hi - y)
    if y > cpd[-1]:
        score += (y - cpd[-1])
    if y < cpd[0]:
        score += (cpd[0] - y)
    return score


def expected_online_crps(resid_cal, sig_cal, y_hat, y, sig_t=None,
                         bins_cal=None, bins_t=None, warm_start=True):
    """Independent replay of the semi-online CRPS semantics.

    Scores each test object against the calibration set in effect when
    that object is predicted (growing with the true residuals of the
    earlier test objects), using the analytic step-CDF integral as the
    per-object score. Objects predicted with an empty calibration set
    are skipped; if all are skipped, 0 is returned. This replays the
    rolling loop object by object (in test order), independently of the
    implementation in ``calculate_crps_online``.
    """
    if sig_cal is not None:
        alphas_cal = sorted(resid_cal / sig_cal)
    else:
        alphas_cal = sorted(resid_cal)
    if bins_cal is None:
        scores = []
        alphas = list(alphas_cal) if warm_start else []
        for i in range(len(y_hat)):
            if len(alphas) > 0:
                sigma = sig_t[i] if sig_t is not None else 1.0
                cpd = y_hat[i] + sigma * np.array(alphas)
                scores.append(crps_step_integral(cpd, y[i]))
            if sig_t is not None:
                insort(alphas, (y[i] - y_hat[i]) / sig_t[i])
            else:
                insort(alphas, y[i] - y_hat[i])
    else:
        # Group the calibration residuals by bin first, then sort within
        # each bin (sorting the pooled residuals would break the
        # residual-to-bin correspondence).
        if warm_start:
            all_alphas = {}
            for b in np.unique(bins_cal):
                mask = bins_cal == b
                if sig_cal is not None:
                    all_alphas[b] = sorted(resid_cal[mask] / sig_cal[mask])
                else:
                    all_alphas[b] = sorted(resid_cal[mask])
        else:
            all_alphas = {}
        scores = []
        for i in range(len(y_hat)):
            alphas = all_alphas.setdefault(bins_t[i], [])
            if len(alphas) > 0:
                sigma = sig_t[i] if sig_t is not None else 1.0
                cpd = y_hat[i] + sigma * np.array(alphas)
                scores.append(crps_step_integral(cpd, y[i]))
            if sig_t is not None:
                insort(alphas, (y[i] - y_hat[i]) / sig_t[i])
            else:
                insort(alphas, y[i] - y_hat[i])
    if len(scores) > 0:
        return float(np.mean(scores))
    return 0.0


def _standard_data(seed=42, n_cal=100, n_test=30):
    rng = np.random.default_rng(seed)
    resid_cal = rng.normal(0, 1, n_cal)
    y_hat_t = rng.normal(0, 1, n_test)
    y_t = y_hat_t + rng.normal(0, 1, n_test)
    return resid_cal, y_hat_t, y_t


def _heteroscedastic_data(seed=42, n_cal=100, n_test=30):
    rng = np.random.default_rng(seed)
    sig_cal = rng.uniform(0.5, 2.0, n_cal)
    resid_cal = rng.normal(0, 1, n_cal) * sig_cal
    y_hat_t = rng.normal(0, 1, n_test)
    sig_t = rng.uniform(0.5, 2.0, n_test)
    y_t = y_hat_t + rng.normal(0, 1, n_test) * sig_t
    return resid_cal, sig_cal, y_hat_t, sig_t, y_t


def test_online_crps_standard_matches_definition():
    resid_cal, y_hat_t, y_t = _standard_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal)
    result = cps.evaluate(y_hat_t, y_t, metrics=["CRPS"], online=True)
    expected = expected_online_crps(resid_cal, None, y_hat_t, y_t)
    assert result["CRPS"] == pytest.approx(expected)


def test_online_crps_normalized_matches_definition():
    resid_cal, sig_cal, y_hat_t, sig_t, y_t = _heteroscedastic_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal, sigmas=sig_cal)
    result = cps.evaluate(y_hat_t, y_t, sigmas=sig_t, metrics=["CRPS"],
                          online=True)
    expected = expected_online_crps(resid_cal, sig_cal, y_hat_t, y_t,
                                    sig_t=sig_t)
    assert result["CRPS"] == pytest.approx(expected)


def test_online_crps_mondrian_matches_definition():
    resid_cal, y_hat_t, y_t = _standard_data()
    rng = np.random.default_rng(1)
    bins_cal = rng.integers(0, 2, len(resid_cal))
    bins_t = rng.integers(0, 2, len(y_t))
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal, bins=bins_cal)
    result = cps.evaluate(y_hat_t, y_t, bins=bins_t, metrics=["CRPS"],
                          online=True)
    expected = expected_online_crps(resid_cal, None, y_hat_t, y_t,
                                    bins_cal=bins_cal, bins_t=bins_t)
    assert result["CRPS"] == pytest.approx(expected)


def test_online_crps_mondrian_normalized_matches_definition():
    resid_cal, sig_cal, y_hat_t, sig_t, y_t = _heteroscedastic_data()
    rng = np.random.default_rng(1)
    bins_cal = rng.integers(0, 2, len(resid_cal))
    bins_t = rng.integers(0, 2, len(y_t))
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal, sigmas=sig_cal, bins=bins_cal)
    result = cps.evaluate(y_hat_t, y_t, sigmas=sig_t, bins=bins_t,
                          metrics=["CRPS"], online=True)
    expected = expected_online_crps(resid_cal, sig_cal, y_hat_t, y_t,
                                    sig_t=sig_t, bins_cal=bins_cal,
                                    bins_t=bins_t)
    assert result["CRPS"] == pytest.approx(expected)


def test_online_crps_single_object_equals_batch():
    # With a single test object, the online calibration set has not grown
    # yet, so the online score must equal the batch score exactly.
    resid_cal, y_hat_t, y_t = _standard_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal)
    online = cps.evaluate(y_hat_t[:1], y_t[:1], metrics=["CRPS"],
                          online=True)
    batch = cps.evaluate(y_hat_t[:1], y_t[:1], metrics=["CRPS"])
    assert online["CRPS"] == batch["CRPS"]


def test_online_crps_first_object_matches_batch_path():
    # The first test object is scored against the unchanged calibration
    # set, so its per-object score must equal the batch path's score for
    # that object.
    resid_cal, y_hat_t, y_t = _standard_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal)
    cpds_batch = cps.predict_cpds(y_hat_t)
    per_object = calculate_crps(cpds_batch[0:1], np.sort(resid_cal),
                                np.ones(1), y_t[0:1])
    online = cps.evaluate(y_hat_t[:1], y_t[:1], metrics=["CRPS"],
                          online=True)
    assert online["CRPS"] == pytest.approx(per_object)


def test_online_crps_warm_start_false_skips_empty():
    # With warm_start=False, the first object is predicted with an empty
    # calibration set and skipped; the score averages over the rest.
    resid_cal, y_hat_t, y_t = _standard_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal)
    result = cps.evaluate(y_hat_t, y_t, metrics=["CRPS"], online=True,
                          warm_start=False)
    expected = expected_online_crps(resid_cal, None, y_hat_t, y_t,
                                    warm_start=False)
    assert result["CRPS"] == pytest.approx(expected)


def test_online_crps_all_empty_returns_zero():
    # Unfitted system, warm_start=False, a single test object: the only
    # object has an empty calibration set, so the score is 0.
    _, y_hat_t, y_t = _standard_data()
    cps = ConformalPredictiveSystem()
    result = cps.evaluate(y_hat_t[:1], y_t[:1], metrics=["CRPS"],
                          online=True, warm_start=False)
    assert result["CRPS"] == 0


def test_online_crps_default_metrics_exclude_crps():
    # CRPS is available online only when explicitly requested; the
    # default online metrics are unchanged.
    _, y_hat_t, y_t = _standard_data()
    cps = ConformalPredictiveSystem()
    cps.fit(np.random.default_rng(42).normal(0, 1, 100))
    result = cps.evaluate(y_hat_t, y_t, online=True)
    assert "CRPS" not in result


def _replayed_alphas_per_object(resid_cal, sig_cal, y_hat, y, sig_t=None,
                                bins_cal=None, bins_t=None, warm_start=True):
    """Per-object calibration alphas in effect at prediction time.

    Replays the rolling insertion (score, then insert the true residual),
    independently of ``calculate_crps_online``. Returns a list with, for
    each test object, the sorted alphas in effect when it is predicted,
    or None when its calibration set is empty.
    """
    if sig_cal is not None:
        alphas_cal = sorted(resid_cal / sig_cal)
    else:
        alphas_cal = sorted(resid_cal)
    per_object = []
    if bins_cal is None:
        alphas = list(alphas_cal) if warm_start else []
        for i in range(len(y_hat)):
            per_object.append(list(alphas) if len(alphas) > 0 else None)
            if sig_t is not None:
                insort(alphas, (y[i] - y_hat[i]) / sig_t[i])
            else:
                insort(alphas, y[i] - y_hat[i])
    else:
        if warm_start:
            all_alphas = {}
            for b in np.unique(bins_cal):
                mask = bins_cal == b
                if sig_cal is not None:
                    all_alphas[b] = sorted(resid_cal[mask] / sig_cal[mask])
                else:
                    all_alphas[b] = sorted(resid_cal[mask])
        else:
            all_alphas = {}
        for i in range(len(y_hat)):
            alphas = all_alphas.setdefault(bins_t[i], [])
            per_object.append(list(alphas) if len(alphas) > 0 else None)
            if sig_t is not None:
                insort(alphas, (y[i] - y_hat[i]) / sig_t[i])
            else:
                insort(alphas, y[i] - y_hat[i])
    return per_object


def _check_matches_predict_cpds_online(cps, y_hat, y, sig_t, bins_t,
                                       warm_start, per_object_alphas):
    """Cross-check the online score against the library's own CPDs.

    Scores the CPDs actually produced by ``predict_cpds_online`` for each
    object (rather than CPDs rebuilt by the test oracle) and compares the
    mean against ``evaluate(..., online=True)``. This ties the score to
    the calibration sets the library's online prediction path uses, not
    just to the oracle's replay of them.
    """
    cpds_online = cps.predict_cpds_online(y_hat, y, sigmas=sig_t,
                                          bins=bins_t,
                                          warm_start=warm_start)
    scores = []
    for i, alphas in enumerate(per_object_alphas):
        sigma = sig_t[i] if sig_t is not None else 1.0
        if alphas is None:
            assert len(cpds_online[i]) == 0
        else:
            assert np.allclose(cpds_online[i],
                               y_hat[i] + sigma * np.array(alphas))
            scores.append(calculate_crps(
                cpds_online[i].reshape(1, -1), np.array(alphas),
                np.array([sigma]), np.array([y[i]])))
    result = cps.evaluate(y_hat, y, sigmas=sig_t, bins=bins_t,
                          metrics=["CRPS"], online=True,
                          warm_start=warm_start)
    assert result["CRPS"] == pytest.approx(float(np.mean(scores)))


def test_online_crps_matches_predict_cpds_online_standard():
    resid_cal, y_hat_t, y_t = _standard_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal)
    per_object = _replayed_alphas_per_object(resid_cal, None, y_hat_t, y_t)
    _check_matches_predict_cpds_online(cps, y_hat_t, y_t, None, None,
                                       True, per_object)


def test_online_crps_matches_predict_cpds_online_normalized():
    resid_cal, sig_cal, y_hat_t, sig_t, y_t = _heteroscedastic_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal, sigmas=sig_cal)
    per_object = _replayed_alphas_per_object(resid_cal, sig_cal, y_hat_t,
                                             y_t, sig_t=sig_t)
    _check_matches_predict_cpds_online(cps, y_hat_t, y_t, sig_t, None,
                                       True, per_object)


def test_online_crps_matches_predict_cpds_online_mondrian_normalized():
    resid_cal, sig_cal, y_hat_t, sig_t, y_t = _heteroscedastic_data()
    rng = np.random.default_rng(1)
    bins_cal = rng.integers(0, 2, len(resid_cal))
    bins_t = rng.integers(0, 2, len(y_t))
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal, sigmas=sig_cal, bins=bins_cal)
    per_object = _replayed_alphas_per_object(resid_cal, sig_cal, y_hat_t,
                                             y_t, sig_t=sig_t,
                                             bins_cal=bins_cal,
                                             bins_t=bins_t)
    _check_matches_predict_cpds_online(cps, y_hat_t, y_t, sig_t, bins_t,
                                       True, per_object)


def test_online_crps_matches_predict_cpds_online_warm_start_false():
    resid_cal, y_hat_t, y_t = _standard_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal)
    per_object = _replayed_alphas_per_object(resid_cal, None, y_hat_t, y_t,
                                             warm_start=False)
    _check_matches_predict_cpds_online(cps, y_hat_t, y_t, None, None,
                                       False, per_object)


def test_online_crps_warm_start_false_normalized():
    # warm_start=False with sigmas: the second object is scored against
    # a one-element calibration set, exercising the single-alpha CPD path
    # with normalized residuals.
    resid_cal, sig_cal, y_hat_t, sig_t, y_t = _heteroscedastic_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal, sigmas=sig_cal)
    result = cps.evaluate(y_hat_t, y_t, sigmas=sig_t, metrics=["CRPS"],
                          online=True, warm_start=False)
    expected = expected_online_crps(resid_cal, sig_cal, y_hat_t, y_t,
                                    sig_t=sig_t, warm_start=False)
    assert result["CRPS"] == pytest.approx(expected)


def test_online_crps_warm_start_false_mondrian():
    # warm_start=False with Mondrian bins: the first object in each bin
    # is predicted with an empty calibration set and skipped.
    resid_cal, y_hat_t, y_t = _standard_data()
    rng = np.random.default_rng(1)
    bins_cal = rng.integers(0, 2, len(resid_cal))
    bins_t = rng.integers(0, 2, len(y_t))
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal, bins=bins_cal)
    result = cps.evaluate(y_hat_t, y_t, bins=bins_t, metrics=["CRPS"],
                          online=True, warm_start=False)
    expected = expected_online_crps(resid_cal, None, y_hat_t, y_t,
                                    bins_cal=bins_cal, bins_t=bins_t,
                                    warm_start=False)
    assert result["CRPS"] == pytest.approx(expected)


def test_online_crps_unseen_mondrian_bin():
    # A test bin absent from the calibration bins: its first object is
    # predicted with an empty calibration set even with warm_start=True.
    resid_cal, y_hat_t, y_t = _standard_data()
    rng = np.random.default_rng(1)
    bins_cal = rng.integers(0, 2, len(resid_cal))
    bins_t = rng.integers(0, 2, len(y_t))
    bins_t[0] = 2
    bins_t[5] = 2
    assert 2 not in bins_cal
    cps = ConformalPredictiveSystem()
    cps.fit(resid_cal, bins=bins_cal)
    result = cps.evaluate(y_hat_t, y_t, bins=bins_t, metrics=["CRPS"],
                          online=True)
    expected = expected_online_crps(resid_cal, None, y_hat_t, y_t,
                                    bins_cal=bins_cal, bins_t=bins_t)
    assert result["CRPS"] == pytest.approx(expected)
    cpds_online = cps.predict_cpds_online(y_hat_t, y_t, bins=bins_t)
    assert len(cpds_online[0]) == 0
    assert len(cpds_online[5]) > 0


def test_online_crps_y_on_grid_point():
    # y exactly on a CPD grid point exercises the straddling branch of
    # get_crps at its boundary.
    cps = ConformalPredictiveSystem()
    cps.fit(np.array([-1.0, 0.0, 1.0]))
    result = cps.evaluate(np.array([0.0]), np.array([0.0]),
                          metrics=["CRPS"], online=True)
    assert result["CRPS"] == pytest.approx(
        crps_step_integral(np.array([-1.0, 0.0, 1.0]), 0.0))
    # Single-point CPD with y exactly on the point: score 0.
    cps_single = ConformalPredictiveSystem()
    cps_single.fit(np.array([0.0]))
    result = cps_single.evaluate(np.array([0.0]), np.array([0.0]),
                                 metrics=["CRPS"], online=True)
    assert result["CRPS"] == 0.0
    # Single-point CPD with y off the point: |y - cpd|.
    result = cps_single.evaluate(np.array([0.0]), np.array([1.0]),
                                 metrics=["CRPS"], online=True)
    assert result["CRPS"] == pytest.approx(1.0)


def test_online_crps_wrap_regressor():
    # WrapRegressor.evaluate delegates to ConformalPredictiveSystem,
    # so it gains online CRPS without further changes.
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(42)
    n_train, n_cal, n_test = 200, 100, 30
    X_train = rng.normal(0, 1, (n_train, 3))
    y_train = X_train[:, 0] + rng.normal(0, 1, n_train)
    X_cal = rng.normal(0, 1, (n_cal, 3))
    y_cal = X_cal[:, 0] + rng.normal(0, 1, n_cal)
    X_test = rng.normal(0, 1, (n_test, 3))
    y_test = X_test[:, 0] + rng.normal(0, 1, n_test)
    w = WrapRegressor(RandomForestRegressor(n_estimators=10,
                                            random_state=42))
    w.fit(X_train, y_train)
    w.calibrate(X_cal, y_cal, cps=True)
    result = w.evaluate(X_test, y_test, metrics=["CRPS"], online=True)
    assert "CRPS" in result
    assert np.isfinite(result["CRPS"])
