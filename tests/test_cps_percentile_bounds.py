"""Regression tests for bounds on supported, known-bin CPS percentiles."""

import math
import unittest
import warnings

import numpy as np

from crepes import ConformalPredictiveSystem


def as_percentiles(values):
    if values is None:
        return []
    if np.isscalar(values):
        return [values]
    return list(values)


def manual_percentiles(cpds, lower, higher, y_min=-math.inf, y_max=math.inf):
    """Select order statistics and clip each scalar, independently of CPS."""
    result = []
    for cpd in cpds:
        row = []
        for percentiles, rounding in ((lower, math.floor), (higher, math.ceil)):
            for percentile in as_percentiles(percentiles):
                index = rounding(float(percentile) * (len(cpd) + 1) / 100) - 1
                assert 0 <= index < len(cpd), "test percentile must be supported"
                row.append(min(max(cpd[index], y_min), y_max))
        result.append(row)
    result = np.array(result, dtype=float)
    return result[:, 0] if result.shape[1] == 1 else result


def make_case(no_bins, normalized):
    """Use unequal bin sizes, unsorted residuals, and interleaved test rows."""
    residuals, sigmas, bins = [], [], []
    for b in range(no_bins):
        count = 4 * (b + 1)
        for i in range(count):
            sigma = (0.5, 1.0, 2.0, 4.0)[i % 4] if normalized else 1.0
            score = (i * 7) % count - count // 2 + 2 * b
            residuals.append(score * sigma)
            sigmas.append(sigma)
            bins.append("bin" + str(b))
    test_bins = list(reversed(sorted(set(bins)))) + sorted(set(bins))
    y_hat = [-12.0, 0.0, 8.0, -3.0, 12.0, 0.0, 4.0, -10.0][:len(test_bins)]
    test_sigmas = [(2.0, 0.5, 1.0, 3.0)[i % 4] for i in range(len(y_hat))]
    cps = ConformalPredictiveSystem().fit(
        residuals, sigmas=sigmas if normalized else None, bins=bins
    )
    kwargs = dict(y_hat=y_hat, bins=test_bins)
    if normalized:
        kwargs["sigmas"] = test_sigmas
    cpds = []
    for i, label in enumerate(test_bins):
        scores = sorted(
            residual / sigma
            for residual, sigma, category in zip(residuals, sigmas, bins)
            if category == label
        )
        scale = test_sigmas[i] if normalized else 1.0
        cpds.append([y_hat[i] + scale * score for score in scores])
    return cps, kwargs, cpds


class TestCPSPercentileBounds(unittest.TestCase):
    def setUp(self):
        # Supported ranks and known bins must not take warning/fallback paths.
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.simplefilter("error", UserWarning)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def assert_result(self, actual, expected):
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual(actual.dtype, np.dtype(float))
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_upper_bound_clips_both_interval_endpoints(self):
        cps = ConformalPredictiveSystem().fit([1, 2, 3, 4], bins=["A"] * 4)
        actual = cps.predict_int([0], bins=["A"], confidence=0.5, y_max=2)
        self.assert_result(actual, np.array([[1.0, 2.0]]))

    def test_lower_bound_does_not_invert_interval(self):
        cps = ConformalPredictiveSystem().fit([1, 2, 3, 4], bins=["A"] * 4)
        actual = cps.predict_int([-10], bins=["A"], confidence=0.5, y_min=0)
        self.assert_result(actual, np.array([[0.0, 0.0]]))

    def test_intervals_match_manual_bounds(self):
        for no_bins in (1, 2, 4):
            for normalized in (False, True):
                cps, kwargs, cpds = make_case(no_bins, normalized)
                for y_min, y_max in ((-math.inf, math.inf), (0, math.inf),
                                     (-math.inf, 2), (0, 2), (2, 2)):
                    with self.subTest(bins=no_bins, normalized=normalized,
                                      y_min=y_min, y_max=y_max):
                        expected = manual_percentiles(cpds, 25, 75, y_min, y_max)
                        actual = cps.predict_int(
                            **kwargs, confidence=0.5, y_min=y_min, y_max=y_max
                        )
                        self.assert_result(actual, expected)

    def test_percentile_columns_match_manual_bounds(self):
        requests = ((25, None), (None, 75), (25, 75),
                    ([75, 25, 50], None), (None, [75, 25, 50]),
                    ([75, 25], [50, 75]), ([25], [75]))
        for no_bins in (1, 2, 4):
            for normalized in (False, True):
                cps, kwargs, cpds = make_case(no_bins, normalized)
                for lower, higher in requests:
                    for y_min, y_max in ((-math.inf, math.inf), (0, math.inf),
                                         (-math.inf, 2), (0, 2), (2, 2)):
                        expected = manual_percentiles(cpds, lower, higher,
                                                      y_min, y_max)
                        for method in (cps.predict, cps.predict_percentiles):
                            with self.subTest(bins=no_bins, normalized=normalized,
                                              lower=lower, higher=higher,
                                              y_min=y_min, y_max=y_max,
                                              method=method.__name__):
                                actual = method(
                                    **kwargs, lower_percentiles=lower,
                                    higher_percentiles=higher,
                                    y_min=y_min, y_max=y_max
                                )
                                self.assert_result(actual, expected)

    def test_scalar_and_array_percentile_types(self):
        cps, kwargs, cpds = make_case(1, True)
        for lower, higher in ((25.0, 75.0), (np.int64(25), np.float64(75)),
                              (np.array([25, 50]), np.array([50, 75]))):
            with self.subTest(lower=lower, higher=higher):
                actual = cps.predict_percentiles(
                    **kwargs, lower_percentiles=lower, higher_percentiles=higher,
                    y_min=0, y_max=2
                )
                self.assert_result(actual, manual_percentiles(cpds, lower, higher,
                                                              0, 2))

    def test_p_values_are_not_clipped(self):
        for no_bins in (1, 2, 4):
            for normalized in (False, True):
                cps, kwargs, cpds = make_case(no_bins, normalized)
                for y in (0, [cpd[1] for cpd in cpds]):
                    for smoothing in (False, True):
                        with self.subTest(bins=no_bins, normalized=normalized,
                                          y=y, smoothing=smoothing):
                            actual = cps.predict(
                                **kwargs, y=y, lower_percentiles=[25, 50],
                                higher_percentiles=[50, 75], y_min=2, y_max=3,
                                smoothing=smoothing, seed=123
                            )
                            p_values = cps.predict_p(**kwargs, y=y,
                                                     smoothing=smoothing, seed=123)
                            np.testing.assert_array_equal(actual[:, 0], p_values)
                            self.assertTrue(np.all(actual[:, 0] < 2))
                            if not smoothing:
                                targets = [y] * len(cpds) if np.isscalar(y) else y
                                expected_p = [(1 + sum(value <= target for value in cpd))
                                              / (len(cpd) + 1)
                                              for cpd, target in zip(cpds, targets)]
                                np.testing.assert_array_equal(actual[:, 0], expected_p)
                            self.assert_result(
                                actual[:, 1:],
                                manual_percentiles(cpds, [25, 50], [50, 75], 2, 3)
                            )

    def test_cpds_are_not_clipped(self):
        for no_bins in (1, 2, 4):
            for normalized in (False, True):
                cps, kwargs, expected_cpds = make_case(no_bins, normalized)
                for by_bins in (False, True):
                    with self.subTest(bins=no_bins, normalized=normalized,
                                      by_bins=by_bins):
                        actual, cpds = cps.predict(
                            **kwargs, lower_percentiles=25, higher_percentiles=75,
                            y_min=0, y_max=2, return_cpds=True, cpds_by_bins=by_bins
                        )
                        self.assert_result(actual, manual_percentiles(
                            expected_cpds, 25, 75, 0, 2
                        ))
                        if by_bins:
                            self.assertIsInstance(cpds, list)
                            expected = [np.array([cpd for cpd, category in
                                                  zip(expected_cpds, kwargs["bins"])
                                                  if category == label])
                                        for label in sorted(set(kwargs["bins"]))]
                        else:
                            self.assertIsInstance(cpds, np.ndarray)
                            self.assertEqual(cpds.dtype, np.dtype(object))
                            expected = expected_cpds
                        self.assertEqual(len(cpds), len(expected))
                        for actual_cpd, expected_cpd in zip(cpds, expected):
                            np.testing.assert_array_equal(actual_cpd, expected_cpd)

    def test_p_values_percentiles_and_cpds_together(self):
        cps = ConformalPredictiveSystem().fit([1, 2, 3, 4], bins=["A"] * 4)
        for by_bins in (False, True):
            with self.subTest(by_bins=by_bins):
                actual, cpds = cps.predict(
                    [0], bins=["A"], y=[2], smoothing=False,
                    lower_percentiles=25, higher_percentiles=75,
                    y_min=2, y_max=3, return_cpds=True, cpds_by_bins=by_bins
                )
                self.assert_result(actual, np.array([[0.6, 2.0, 3.0]]))
                expected_cpd = [[1, 2, 3, 4]] if by_bins else [1, 2, 3, 4]
                np.testing.assert_array_equal(cpds[0], expected_cpd)

    def test_bounds_do_not_change_outputs_without_percentiles(self):
        cps = ConformalPredictiveSystem().fit([1, 2, 3, 4], bins=["A"] * 4)
        kwargs = dict(y_hat=[0, -10], bins=["A", "A"])
        self.assertIsNone(cps.predict(**kwargs, y_min=2, y_max=3))
        self.assert_result(cps.predict(**kwargs, y=[2, -8], smoothing=False,
                                       y_min=2, y_max=3), np.array([0.6, 0.6]))
        cpds = cps.predict(**kwargs, y_min=2, y_max=3, return_cpds=True)
        reference = cps.predict_cpds(**kwargs)
        for actual, expected in zip(cpds, ([1, 2, 3, 4], [-9, -8, -7, -6])):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(cpds, reference):
            np.testing.assert_array_equal(actual, expected)

    def test_single_bin_agrees_with_unbinned_system(self):
        for normalized in (False, True):
            residuals = [4.0, -4.0, 1.0, -1.0, 8.0, -8.0, 2.0, -2.0]
            sigmas = [2.0, 1.0, 0.5, 2.0, 4.0, 2.0, 1.0, 0.5] if normalized else None
            test_sigmas = [0.5, 2.0, 4.0] if normalized else None
            standard = ConformalPredictiveSystem().fit(residuals, sigmas=sigmas)
            mondrian = ConformalPredictiveSystem().fit(
                residuals, sigmas=sigmas, bins=["A"] * len(residuals)
            )
            cpds = [[y_hat + scale * score for score in sorted(
                r / s for r, s in zip(residuals, sigmas or [1] * len(residuals))
            )] for y_hat, scale in zip([-10, 0, 10], test_sigmas or [1] * 3)]
            for y_min, y_max in ((-math.inf, math.inf), (0, math.inf),
                                 (-math.inf, 2), (0, 2)):
                with self.subTest(normalized=normalized, y_min=y_min, y_max=y_max):
                    kwargs = dict(y_hat=[-10, 0, 10], sigmas=test_sigmas,
                                  lower_percentiles=[25, 50],
                                  higher_percentiles=[50, 75], y_min=y_min, y_max=y_max)
                    expected = manual_percentiles(cpds, [25, 50], [50, 75],
                                                  y_min, y_max)
                    self.assert_result(standard.predict_percentiles(**kwargs), expected)
                    self.assert_result(mondrian.predict_percentiles(
                        **kwargs, bins=["A"] * 3
                    ), expected)

    def test_unused_calibration_bin_and_numpy_inputs(self):
        cps, kwargs, cpds = make_case(4, True)
        # Every test category is known, but some calibration bins have no test rows.
        kwargs = {key: np.array(value[:1]) for key, value in kwargs.items()}
        actual = cps.predict_percentiles(
            **kwargs, lower_percentiles=[25, 50], higher_percentiles=[50, 75],
            y_min=0, y_max=2
        )
        self.assert_result(actual, manual_percentiles(cpds[:1], [25, 50], [50, 75],
                                                      0, 2))

    def test_calibration_and_input_arrays_are_unchanged(self):
        cps, kwargs, cpds = make_case(2, True)
        kwargs = {key: np.array(value) for key, value in kwargs.items()}
        inputs_before = {key: value.copy() for key, value in kwargs.items()}
        alphas_before = cps.alphas.copy()
        bins_before = cps.bins.copy()
        bin_values_before = cps.binned_alphas[0].copy()
        binned_alphas_before = [alphas.copy() for alphas in cps.binned_alphas[1]]
        actual = cps.predict(**kwargs, lower_percentiles=[25, 50],
                             higher_percentiles=[50, 75], y_min=0, y_max=2)
        self.assert_result(actual, manual_percentiles(cpds, [25, 50], [50, 75], 0, 2))
        for key in kwargs:
            np.testing.assert_array_equal(kwargs[key], inputs_before[key])
        np.testing.assert_array_equal(cps.alphas, alphas_before)
        np.testing.assert_array_equal(cps.bins, bins_before)
        np.testing.assert_array_equal(cps.binned_alphas[0], bin_values_before)
        for actual_alphas, before in zip(cps.binned_alphas[1], binned_alphas_before):
            np.testing.assert_array_equal(actual_alphas, before)


if __name__ == "__main__":
    unittest.main()
