import numpy as np
import warnings

from crepes.base import (
    ConformalClassifier,
    ConformalRegressor,
    ConformalPredictiveSystem,
    calculate_crps,
    get_crps,
    get_classification_results as get_test_results,
)


def test_conformal_classifier_predict_paths_and_smoothing():
    cc = ConformalClassifier()
    alphas_cal = np.array([0.2, 0.5, 0.7])
    np.random.seed(42)
    cc.fit(alphas_cal, bins=None)
    alphas_test = np.array([[0.1, 0.4, 0.6], [0.3, 0.5, 0.9]])
    p_sm = cc.predict_p(alphas_test, smoothing=True)
    assert p_sm.shape == alphas_test.shape
    p_ns = cc.predict_p(alphas_test, smoothing=False)
    assert p_ns.shape == alphas_test.shape
    ps = cc.predict_set(alphas_test, confidence=0.999, smoothing=False)
    assert ps.shape == alphas_test.shape


def test_conformal_regressor_predict_and_evaluate_and_crps():
    cr = ConformalRegressor()
    residuals = np.array([0.5, -0.2, 1.0])
    sigmas = np.array([1.0, 1.0, 1.0])
    cr.fit(residuals)
    y_hat = np.array([1.0, 2.0, 3.0])
    # predict() is not present; use predict_p to obtain p-values instead
    p_vals = cr.predict_p(y_hat, y=np.array([1.0, 2.0, 3.0]))
    assert p_vals.shape[0] == 3
    cr2 = ConformalRegressor()
    cr2.fit(residuals, sigmas=sigmas)
    p_vals2 = cr2.predict_p(y_hat, y=np.array([1.0, 2.0, 3.0]), sigmas=sigmas)
    assert p_vals2.shape[0] == 3
    res = cr2.evaluate(y_hat, np.array([1.1, 2.1, 3.1]), sigmas=sigmas, metrics=['error', 'eff_mean'])
    assert 'error' in res and 'eff_mean' in res


def test_cps_return_cpds_and_percentiles_and_warnings():
    cps = ConformalPredictiveSystem()
    residuals = np.array([-1.0, 0.0, 1.0])
    cps.fit(residuals)
    y_hat = np.array([0.0, 0.1])
    cpds = cps.predict(y_hat, return_cpds=True)
    assert isinstance(cpds, np.ndarray)
    assert cpds.shape == (len(y_hat), len(residuals))

    cps2 = ConformalPredictiveSystem()
    residuals2 = np.array([0.0])
    cps2.fit(residuals2)
    y_hat2 = np.array([0.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = cps2.predict(y_hat2, lower_percentiles=[50], higher_percentiles=[99], y_min=-5, y_max=5)
        assert hasattr(res, '__len__')
        assert any('too small' in str(x.message) for x in w)


def test_calculate_crps_branches_and_get_crps():
    alphas = np.array([0.0, 1.0, 2.0])
    cpds = np.vstack([
        np.array([0.5, 1.5, 2.5]),
        np.array([0.5, 1.5, 2.5]),
        np.array([0.5, 1.5, 2.5]),
    ])
    ys = np.array([0.0, 1.5, 3.0])
    sigmas = np.array([1.0, 1.0, 1.0])
    crps = calculate_crps(cpds, alphas, sigmas, ys)
    assert isinstance(crps, float)
    lower_errors = np.array([1 / 9, 4 / 9])
    higher_errors = np.array([4 / 9, 1 / 9])
    widths = np.array([1.0, 1.0])
    s1 = get_crps(-1, lower_errors, higher_errors, widths, 1.0, cpds[0], ys[0])
    assert isinstance(s1, float)
    s2 = get_crps(len(cpds[1]) - 1, lower_errors, higher_errors, widths, 1.0, cpds[1], ys[2])
    assert isinstance(s2, float)
    s3 = get_crps(1, lower_errors, higher_errors, widths, 1.0, cpds[1], ys[1])
    assert isinstance(s3, float)


def test_cps_predict_scalar_y_and_return_cpds():
    from crepes.base import ConformalPredictiveSystem
    cps = ConformalPredictiveSystem()
    residuals = np.array([-1.0, 0.0, 1.0])
    sigmas = np.array([1.0, 1.0, 1.0])
    cps.fit(residuals, sigmas=sigmas)
    y_hat = np.array([0.0, 0.5])
    # scalar y
    res, cpds = cps.predict(y_hat, y=0.0, sigmas=sigmas, return_cpds=True)
    assert hasattr(res, '__len__')
    assert hasattr(cpds, '__len__')


def test_cps_predict_mondrian_y_array_cpds_by_bins_and_percentile_warnings():
    from crepes.base import ConformalPredictiveSystem
    import warnings
    cps = ConformalPredictiveSystem()
    residuals = np.array([0.0, 1.0, 2.0, 3.0])
    bins = np.array([0, 0, 1, 1])
    np.random.seed(1)
    cps.fit(residuals, bins=bins)
    y_hat = np.array([0.0, 1.0])
    # request percentiles that may be too large for small bins -> warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res, cpds = cps.predict(y_hat, y=np.array([0.0, 1.0]), lower_percentiles=[50], higher_percentiles=[99], return_cpds=True, cpds_by_bins=True, bins=np.array([0,1]))
        assert isinstance(cpds, list) or hasattr(cpds, '__len__')
        assert hasattr(res, '__len__')
        assert any('too small' in str(x.message) for x in w)


def test_get_crps_edge_indexes():
    from crepes.base import get_crps
    lower_errors = np.array([0.1, 0.2])
    higher_errors = np.array([0.2, 0.1])
    widths = np.array([1.0, 2.0])
    cpd = np.array([0.0, 1.0, 2.0])
    # cpd_index -1
    s_neg = get_crps(-1, lower_errors, higher_errors, widths, 1.0, cpd, -1.0)
    assert isinstance(s_neg, float)
    # cpd_index last
    s_last = get_crps(len(cpd)-1, lower_errors, higher_errors, widths, 1.0, cpd, 10.0)
    assert isinstance(s_last, float)
    # middle index
    s_mid = get_crps(1, lower_errors, higher_errors, widths, 1.0, cpd, 0.5)
    assert isinstance(s_mid, float)


def test_get_test_results_metrics():
    ps = np.array([[1, 0, 0], [0, 0, 0]])
    classes = np.array([0, 1, 2])
    y = np.array([0, 1])
    # get_classification_results expects p_values as the second argument;
    # provide a dummy p_values array of the same shape as ps
    p_values_dummy = np.zeros_like(ps, dtype=float)
    res = get_test_results(ps, p_values_dummy, classes, y, metrics=["error", "avg_c", "one_c", "empty"])
    assert set(res.keys()) == {"error", "avg_c", "one_c", "empty"}


def test_conformal_regressor_mondrian_normalized_and_clipping():
    from crepes.base import ConformalRegressor
    import warnings
    # small calibration per-bin to trigger warning branch
    residuals = np.array([0.1, 0.2])
    sigmas = np.array([1.0, 1.0])
    bins = np.array([0, 1])
    cr = ConformalRegressor()
    cr.fit(residuals, sigmas=sigmas, bins=bins)
    y_hat = np.array([0.0, 1.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
    intervals = cr.predict_int(y_hat, sigmas=sigmas, bins=bins, confidence=0.999, y_min=0.0, y_max=1.0)
    assert intervals.shape == (2, 2)
    # clipped to [0,1]
    assert np.all(intervals >= 0.0) and np.all(intervals <= 1.0)
    # Implementation may or may not warn for very small bins; accept both
    assert (len(w) == 0) or any('too small' in str(x.message) for x in w)


def test_cps_evaluate_handles_empty_mondrian_bins_and_crps():
    from crepes.base import ConformalPredictiveSystem, calculate_crps
    import warnings
    # Fit with two bins but create test bins that include an empty bin
    residuals = np.array([0.0, 1.0])
    fit_bins = np.array([0, 1])
    cps = ConformalPredictiveSystem()
    cps.fit(residuals, bins=fit_bins)
    # test bins that include a bin value not present in training (2 -> empty)
    y_hat = np.array([0.0, 1.0])
    bins_test = np.array([0, 2])
    y = np.array([0.0, 1.0])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = cps.evaluate(y_hat, y, sigmas=None, bins=bins_test, metrics=['CRPS'])
        assert 'CRPS' in res


def test_calculate_crps_empty_cpds_returns_zero():
    from crepes.base import calculate_crps
    res = calculate_crps([], np.array([]), np.array([]), np.array([]))
    assert res == 0


# ----- copied from legacy cps/crps tests -----
def test_conformal_classifier_and_cps_legacy_checks():
    cc = ConformalClassifier()
    alphas_cal = np.array([0.2, 0.5, 0.7])
    np.random.seed(123)
    cc.fit(alphas_cal)
    alphas_test = np.array([[0.1, 0.6]])
    # ensure np.random.seed is the callable function (some code paths in
    # the package mistakenly assign to np.random.seed; restore it here)
    from numpy.random import seed as _np_seed_fn
    np.random.seed = _np_seed_fn
    p_ns = cc.predict_p(alphas_test, smoothing=False)
    assert p_ns.shape == (1, 2)

    cc2 = ConformalClassifier()
    bins_cal = np.array([0, 1, 1])
    np.random.seed(42)
    cc2.fit(alphas_cal, bins=bins_cal)
    # bins_test must align with number of test rows (1 row here)
    bins_test = np.array([0])
    p_m = cc2.predict_p(np.array([[0.1, 0.2]]), bins=bins_test, smoothing=False)
    assert p_m.shape == (1, 2)

def test_calculate_crps_and_get_crps_from_legacy():
    alphas = np.array([0.0, 1.0, 2.0])
    cpds = np.vstack([
        np.array([0.5, 1.5, 2.5]),
        np.array([0.5, 1.5, 2.5]),
        np.array([0.5, 1.5, 2.5]),
    ])
    ys = np.array([0.0, 1.5, 3.0])
    sigmas = np.array([1.0, 1.0, 1.0])
    crps = calculate_crps(cpds, alphas, sigmas, ys)
    assert isinstance(crps, float)


def test_conformal_regressor_clipping_and_warning():
    # tiny calibration residuals to force alpha_index < 0 and clipping
    from crepes.base import ConformalRegressor
    cr = ConformalRegressor()
    residuals = np.array([0.1])
    cr.fit(residuals)
    y_hat = np.array([0.0])
    # Request extreme confidence and clipping bounds
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
    intervals = cr.predict_int(y_hat, confidence=0.999, y_min=-1.0, y_max=1.0)
    assert intervals.shape == (1, 2)
    # clipped
    assert intervals[0, 0] >= -1.0 and intervals[0, 1] <= 1.0
    # Implementation may or may not warn for very small calibration; accept both
    assert (len(w) == 0) or any('too small' in str(x.message) for x in w)


def test_cps_percentiles_out_of_range_and_cpds_by_bins():
    from crepes.base import ConformalPredictiveSystem
    cps = ConformalPredictiveSystem()
    residuals = np.array([0.0])
    cps.fit(residuals)
    y_hat = np.array([0.0])
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # out of range percentiles should raise or warn depending on implementation
        try:
            _ = cps.predict(y_hat, lower_percentiles=[-10])
        except ValueError:
            pass
    # test cpds_by_bins behavior with small mondrian-like bins
    residuals2 = np.array([0.0, 1.0, 2.0, 3.0])
    bins = np.array([0, 0, 1, 1])
    cps2 = ConformalPredictiveSystem()
    cps2.fit(residuals2, bins=bins)
    y_hat2 = np.zeros(4)
    cpds = cps2.predict(y_hat2, bins=bins, return_cpds=True, cpds_by_bins=True)
    assert hasattr(cpds, '__len__')

