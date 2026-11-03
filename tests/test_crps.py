"""Regression tests for CRPS computation (issue #40).

The CRPS of a conformal predictive distribution was systematically wrong
for observations inside the cpd range in a normalized system (sigma != 1,
from difficulty estimates): in ``get_crps``, the two partial-interval
terms around the observed value ``y`` were multiplied by ``sigma`` once
too often, even
though ``(y - cpd[k])`` and ``(cpd[k + 1] - y)`` are already lengths in the
original units of ``y`` (the cpd values are built as
``y_hat + sigma * alphas``).

The reference values below come from an independent analytic integration
of the step CDF defined by the cpd grid points (equivalently, the energy
form of the CRPS for the discrete uniform distribution on those points).
"""
import numpy as np
import pytest

from crepes import ConformalPredictiveSystem
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


def single_object_crps(alphas, sigma, y, y_hat=0.0):
    cpd = y_hat + sigma * np.asarray(alphas, dtype=float)
    return calculate_crps(cpd.reshape(1, -1), np.asarray(alphas, dtype=float),
                          np.array([sigma]), np.array([y]))


ALPHAS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # normalized residuals, N = 5


@pytest.mark.parametrize("sigma,y", [(1.0, 0.5), (1.0, -1.3), (1.0, 4.0)])
def test_crps_sigma_one_baseline(sigma, y):
    # sigma == 1: locks in the behaviour that was already correct.
    expected = crps_step_integral(sigma * ALPHAS, y)
    assert single_object_crps(ALPHAS, sigma, y) == pytest.approx(expected)


@pytest.mark.parametrize("sigma,y,expected", [
    (2.0, 0.5, 0.90),   # was 1.32 before the fix
    (3.0, 0.5, 1.30),   # was 2.46 before the fix
    (0.5, 0.2, 0.24),   # was 0.18 before the fix
    (2.0, 0.0, 0.80),   # y exactly on a grid point; was 1.12 before the fix
])
def test_crps_sigma_not_one(sigma, y, expected):
    got = single_object_crps(ALPHAS, sigma, y)
    assert got == pytest.approx(expected)
    assert got == pytest.approx(crps_step_integral(sigma * ALPHAS, y))


@pytest.mark.parametrize("y", [5.0, -5.0])
def test_crps_y_outside_range(y):
    # Tail branches: unaffected by the bug, must stay exact.
    assert single_object_crps(ALPHAS, 2.0, y) == pytest.approx(3.40)


@pytest.mark.parametrize("y", [-4.0, 4.0])
def test_crps_y_on_grid_endpoint(y):
    # y exactly on the smallest/largest cpd grid point (sigma = 2, so the
    # grid is [-4, -2, 0, 2, 4]): must match the analytic integral.
    assert single_object_crps(ALPHAS, 2.0, y) == \
        pytest.approx(crps_step_integral(2.0 * ALPHAS, y))


def test_crps_randomized_matches_analytic():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n_cal = int(rng.integers(5, 60))
        alphas = np.sort(rng.normal(0, 1, n_cal))
        sigma = float(10 ** rng.uniform(-0.7, 0.7))  # 0.2 .. 5
        y_hat = float(rng.normal(0, 3))
        y = float(rng.normal(0, 3))
        got = single_object_crps(alphas, sigma, y, y_hat=y_hat)
        expected = crps_step_integral(y_hat + sigma * alphas, y)
        assert got == pytest.approx(expected, abs=1e-10)


def _heteroscedastic_data(seed=42):
    rng = np.random.default_rng(seed)
    n_cal, n_test = 400, 200
    sig_cal = rng.uniform(0.3, 3.0, n_cal)
    resid = rng.normal(0, 1, n_cal) * sig_cal
    y_hat_t = rng.normal(0, 5, n_test)
    sig_t = rng.uniform(0.3, 3.0, n_test)
    y_t = y_hat_t + rng.normal(0, 1, n_test) * sig_t
    return resid, sig_cal, y_hat_t, sig_t, y_t


def test_cps_evaluate_crps_normalized():
    resid, sig_cal, y_hat_t, sig_t, y_t = _heteroscedastic_data()
    cps = ConformalPredictiveSystem()
    cps.fit(resid, sigmas=sig_cal)
    result = cps.evaluate(y_hat_t, y_t, sigmas=sig_t, metrics=["CRPS"])
    cpds = cps.predict_cpds(y_hat_t, sigmas=sig_t)
    expected = np.mean([crps_step_integral(cpds[i], y_t[i])
                        for i in range(len(y_t))])
    assert result["CRPS"] == pytest.approx(expected)


def test_cps_evaluate_crps_mondrian():
    resid, sig_cal, y_hat_t, sig_t, y_t = _heteroscedastic_data()
    rng = np.random.default_rng(1)
    bins_cal = rng.integers(0, 2, len(resid))
    bins_t = rng.integers(0, 2, len(y_t))
    cps = ConformalPredictiveSystem()
    cps.fit(resid, sigmas=sig_cal, bins=bins_cal)
    result = cps.evaluate(y_hat_t, y_t, sigmas=sig_t, bins=bins_t,
                          metrics=["CRPS"])
    cpds = cps.predict_cpds(y_hat_t, sigmas=sig_t, bins=bins_t)
    expected = np.mean([crps_step_integral(cpds[i], y_t[i])
                        for i in range(len(y_t))])
    assert result["CRPS"] == pytest.approx(expected)
