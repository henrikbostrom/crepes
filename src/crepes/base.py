"""Conformal regressors and predictive systems (crepes)

Classes implementing conformal regressors and conformal predictive
systems, which transform point predictions into prediction intervals
and cumulative distribution functions, respectively.

Author: Henrik Boström (bostromh@kth.se)

Copyright 2023 Henrik Boström

License: BSD 3 clause
"""

__version__ = "0.3.0"

import numpy as np
import pandas as pd
import time
import warnings

warnings.simplefilter("always", UserWarning)

class ConformalPredictor():
    """
    The class contains two sub-classes: ConformalRegressor 
    and ConformalPredictiveSystem.
    """
    
    def __init__(self):
        self.alphas = None
        self.fitted = False
        self.normalized = None
        self.mondrian = None
        self.time_fit = None
        self.time_predict = None
        self.time_evaluate = None

        
class ConformalRegressor(ConformalPredictor):
    """
    A conformal regressor transforms point predictions (regression 
    values) into prediction intervals, for a certain confidence level.
    """
    
    def __repr__(self):
        if self.fitted:
            return (f"ConformalRegressor(fitted={self.fitted}, "
                    f"normalized={self.normalized}, "
                    f"mondrian={self.mondrian})")
        else:
            return f"ConformalRegressor(fitted={self.fitted})"
    
    def fit(self, residuals=None, sigmas=None, bins=None):
        """
        Fit conformal regressor.

        Parameters
        ----------
        residuals : array-like of shape (n_values,), default=None
            true values - predicted values
        sigmas: array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories

        Returns
        -------
        self : object
            Fitted ConformalRegressor.

        Examples
        --------
        Assuming that ``y_cal`` and ``y_hat_cal`` are vectors with true
        and predicted targets for some calibration set, then a standard
        conformal regressor can be formed from the residuals:

        .. code-block:: python

           residuals_cal = y_cal - y_hat_cal

           from crepes import ConformalRegressor

           cr_std = ConformalRegressor() 

           cr_std.fit(residuals_cal) 

        Assuming that ``sigmas_cal`` is a vector with difficulty estimates,
        then a normalized conformal regressor can be fitted in the following
        way:

        .. code-block:: python

           cr_norm = ConformalRegressor()
           cr_norm.fit(residuals=residuals_cal, sigmas=sigmas_cal)

        Assuming that ``bins_cals`` is a vector with Mondrian categories 
        (bin labels), then a Mondrian conformal regressor can be fitted in the
        following way:

        .. code-block:: python

           cr_mond = ConformalRegressor()
           cr_mond.fit(residuals=residuals_cal, bins=bins_cal)

        A normalized Mondrian conformal regressor can be fitted in the 
        following way:

        .. code-block:: python

           cr_norm_mond = ConformalRegressor()
           cr_norm_mond.fit(residuals=residuals_cal, sigmas=sigmas_cal, 
                            bins=bins_cal)
        """
        tic = time.time()
        abs_residuals = np.abs(residuals)
        if bins is None:
            self.mondrian = False
            if sigmas is None:
                self.normalized = False
                self.alphas = np.sort(abs_residuals)[::-1]
            else:
                self.normalized = True
                self.alphas = np.sort(abs_residuals/sigmas)[::-1]
        else: 
            self.mondrian = True
            bin_values = np.unique(bins)
            if sigmas is None:            
                self.normalized = False
                self.alphas = (bin_values,[np.sort(
                    abs_residuals[bins==b])[::-1] for b in bin_values])
            else:
                self.normalized = True
                self.alphas = (bin_values, [np.sort(
                    abs_residuals[bins==b]/sigmas[bins==b])[::-1]
                                           for b in bin_values])                
        self.fitted = True
        toc = time.time()
        self.time_fit = toc-tic
        return self

    def predict(self, y_hat=None, sigmas=None, bins=None, confidence=0.95,
                y_min=-np.inf, y_max=np.inf):
        """
        Predict using conformal regressor.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,), default=None
            predicted values
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals

        Returns
        -------
        intervals : ndarray of shape (n_values, 2)
            prediction intervals

        Examples
        --------
        Assuming that ``y_hat_test`` is a vector with predicted targets for a
        test set and ``cr_std`` a fitted standard conformal regressor, then 
        prediction intervals at the 99% confidence level can be obtained by:

        .. code-block:: python

           intervals = cr_std.predict(y_hat=y_hat_test, confidence=0.99)

        Assuming that ``sigmas_test`` is a vector with difficulty estimates for
        the test set and ``cr_norm`` a fitted normalized conformal regressor, 
        then prediction intervals at the default (95%) confidence level can be
        obtained by:

        .. code-block:: python

           intervals = cr_norm.predict(y_hat=y_hat_test, sigmas=sigmas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cr_mond`` a fitted Mondrian conformal 
        regressor, then the following provides prediction intervals at the 
        default confidence level, where the intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = cr_mond.predict(y_hat=y_hat_test, bins=bins_test, 
                                       y_min=0)

        Note
        ----
        In case the specified confidence level is too high in relation to the 
        size of the calibration set, a warning will be issued and the output
        intervals will be of maximum size.
        """
        tic = time.time()
        intervals = np.zeros((len(y_hat),2))
        if not self.mondrian:
            alpha_index = int((1-confidence)*(len(self.alphas)+1))-1
            if alpha_index >= 0:
                alpha = self.alphas[alpha_index]
                if self.normalized:
                    intervals[:,0] = y_hat-alpha*sigmas
                    intervals[:,1] = y_hat+alpha*sigmas
                else:
                    intervals[:,0] = y_hat-alpha
                    intervals[:,1] = y_hat+alpha
            else:
                intervals[:,0] = -np.inf 
                intervals[:,1] = np.inf
                warnings.warn("the no. of calibration examples is too few" \
                              "for the chosen confidence level; the " \
                              "intervals will be of maximum size")
        else:           
            bin_values, bin_alphas = self.alphas
            bin_indexes = [np.argwhere(bins == b).T[0]
                           for b in bin_values]
            alpha_indexes = np.array(
                [int((1-confidence)*(len(bin_alphas[b])+1))-1
                 for b in range(len(bin_values))])
            too_small_bins = np.argwhere(alpha_indexes < 0)
            if len(too_small_bins) > 0:
                if len(too_small_bins[:,0]) < 11:
                    bins_to_show = " ".join([str(bin_values[i]) for i in
                                             too_small_bins[:,0]])
                else:
                    bins_to_show = " ".join([str(bin_values[i]) for i in
                                             too_small_bins[:10,0]]+['...'])
                warnings.warn("the no. of calibration examples is too " \
                              "small for the chosen confidence level " \
                              f"in the following bins: {bins_to_show}; "\
                              "the corresponding intervals will be of " \
                              "maximum size") 
            bin_alpha = np.array([bin_alphas[b][alpha_indexes[b]]
                         if alpha_indexes[b]>=0 else np.inf
                         for b in range(len(bin_values))])
            if self.normalized:
                for b in range(len(bin_values)):
                    intervals[bin_indexes[b],0] = y_hat[bin_indexes[b]] \
                        - bin_alpha[b]*sigmas[bin_indexes[b]]
                    intervals[bin_indexes[b],1] = y_hat[bin_indexes[b]] \
                        + bin_alpha[b]*sigmas[bin_indexes[b]]
            else:
                for b in range(len(bin_values)):
                    intervals[bin_indexes[b],0] = y_hat[bin_indexes[b]] \
                        - bin_alpha[b]
                    intervals[bin_indexes[b],1] = y_hat[bin_indexes[b]] \
                        + bin_alpha[b]                
        if y_min > -np.inf:
            intervals[intervals<y_min] = y_min
        if y_max < np.inf:
            intervals[intervals>y_max] = y_max 
        toc = time.time()
        self.time_predict = toc-tic            
        return intervals

    def evaluate(self, y_hat=None, y=None, sigmas=None, bins=None,
                 confidence=0.95, y_min=-np.inf, y_max=np.inf, metrics=None):
        """
        Evaluate conformal regressor.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,), default=None
            predicted values
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals
        metrics : a string or a list of strings, 
                  default=list of all metrics, i.e., 
                  ["error", "eff_mean", "eff_med", "time_fit", "time_evaluate"]
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics

        Examples
        --------
        Assuming that ``y_hat_test`` and ``y_test`` are vectors with predicted
        and true targets for a test set, ``sigmas_test`` and ``bins_test`` are
        vectors with difficulty estimates and Mondrian categories (bin labels) 
        for the test set, and ``cr_norm_mond`` is a fitted normalized Mondrian
        conformal regressor, then the latter can be evaluated at the default
        confidence level with respect to error and mean efficiency (interval 
        size) by:

        .. code-block:: python

           results = cr_norm_mond.evaluate(y_hat=y_hat_test, y=y_test, 
                                           sigmas=sigmas_test, bins=bins_test,
                                           metrics=["error", "eff_mean"])
        """
        tic = time.time()
        if metrics is None:
            metrics = ["error","eff_mean","eff_med","time_fit","time_evaluate"]
        test_results = {}
        intervals = self.predict(y_hat, sigmas, bins, confidence, y_min, y_max)
        if "error" in metrics:
            test_results["error"] = 1-np.mean(
                np.logical_and(intervals[:,0]<=y,y<=intervals[:,1]))
        if "eff_mean" in metrics:            
            test_results["eff_mean"] = np.mean(intervals[:,1]-intervals[:,0])
        if "eff_med" in metrics:            
            test_results["eff_med"] = np.median(intervals[:,1]-intervals[:,0])
        if "time_fit" in metrics:
            test_results["time_fit"] = self.time_fit
        toc = time.time()
        self.time_evaluate = toc-tic
        if "time_evaluate" in metrics:
            test_results["time_evaluate"] = self.time_evaluate
        return test_results
    
class ConformalPredictiveSystem(ConformalPredictor):
    """
    A conformal predictive system transforms point predictions 
    (regression values) into cumulative distribution functions 
    (conformal predictive distributions).
    """
    
    def __repr__(self):
        if self.fitted:
            return (f"ConformalPredictiveSystem(fitted={self.fitted}, "
                    f"normalized={self.normalized}, "
                    f"mondrian={self.mondrian})")
        else:
            return f"ConformalPredictiveSystem(fitted={self.fitted})"

    def fit(self, residuals=None, sigmas=None, bins=None):
        """
        Fit conformal predictive system.

        Parameters
        ----------
        residuals : array-like of shape (n_values,), default=None
            actual values - predicted values
        sigmas: array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories

        Returns
        -------
        self : object
            Fitted ConformalPredictiveSystem.

        Examples
        --------
        Assuming that ``y_cal`` and ``y_hat_cal`` are vectors with true and
        predicted targets for some calibration set, then a standard conformal
        predictive system can be formed from the residuals:

        .. code-block:: python

           residuals_cal = y_cal - y_hat_cal

           from crepes import ConformalPredictiveSystem

           cps_std = ConformalPredictiveSystem() 

           cps_std.fit(residuals_cal) 

        Assuming that ``sigmas_cal`` is a vector with difficulty estimates,
        then a normalized conformal predictive system can be fitted in the 
        following way:

        .. code-block:: python

           cps_norm = ConformalPredictiveSystem()
           cps_norm.fit(residuals=residuals_cal, sigmas=sigmas_cal)

        Assuming that ``bins_cals`` is a vector with Mondrian categories (bin
        labels), then a Mondrian conformal predictive system can be fitted in
        the following way:

        .. code-block:: python

           cps_mond = ConformalPredictiveSystem()
           cps_mond.fit(residuals=residuals_cal, bins=bins_cal)

        A normalized Mondrian conformal predictive system can be fitted in the
        following way:

        .. code-block:: python

           cps_norm_mond = ConformalPredictiveSystem()
           cps_norm_mond.fit(residuals=residuals_cal, sigmas=sigmas_cal, 
                             bins=bins_cal)
        """

        tic = time.time()
        if bins is None:
            self.mondrian = False
            if sigmas is None:
                self.normalized = False
                self.alphas = np.sort(residuals)
            else:
                self.normalized = True
                self.alphas = np.sort(residuals/sigmas)
        else: 
            self.mondrian = True
            bin_values = np.unique(bins)
            if sigmas is None:            
                self.normalized = False
                self.alphas = (bin_values, [np.sort(
                    residuals[bins==b]) for b in bin_values])
            else:
                self.normalized = True
                self.alphas = (bin_values, [np.sort(
                    residuals[bins==b]/sigmas[bins==b]) for b in bin_values])                
        self.fitted = True
        toc = time.time()
        self.time_fit = toc-tic
        return self
        
    def predict(self, y_hat=None, sigmas=None, bins=None,
                y=None, lower_percentiles=None, higher_percentiles=None,
                y_min=-np.inf, y_max=np.inf, return_cpds=False,
                cpds_by_bins=False):    
        """
        Predict using conformal predictive system.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,), default=None
            predicted values
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        y : float, int or array-like of shape (n_values,), default=None
            values for which p-values should be returned
        lower_percentiles : array-like of shape (l_values,), default=None
            percentiles for which a lower value will be output 
            in case a percentile lies between two values
            (similar to `interpolation="lower"` in `numpy.percentile`)
        higher_percentiles : array-like of shape (h_values,), default=None
            percentiles for which a higher value will be output 
            in case a percentile lies between two values
            (similar to `interpolation="higher"` in `numpy.percentile`)
        y_min : float or int, default=-numpy.inf
            The minimum value to include in prediction intervals.
        y_max : float or int, default=numpy.inf
            The maximum value to include in prediction intervals.
        return_cpds : Boolean, default=False
            specifies whether conformal predictive distributions (cpds)
            should be output or not
        cpds_by_bins : Boolean, default=False
            specifies whether the output cpds should be grouped by bin or not; 
            only applicable when bins is not None and return_cpds = True

        Returns
        -------

        results : ndarray of shape (n_values, p_values+l_values+z_values)
            where p_values = 1 if y is not None and 0 otherwise. A matrix
            where the first column contains p-values, if p_values = 1,
            the following l_values columns contain lower percentiles, and
            the following h_values columns contain higher percentiles.
            Only returned if p_values + l_values + z_values > 0.
        cpds : ndarray of (n_values, c_values), ndarray of (n_values,)
               or list of ndarrays
            conformal predictive distributions. Only returned if 
            return_cpds == True. If bins is None, the distributions are
            represented by a single array, where the number of columns
            (c_values) is determined by the number of residuals of the fitted
            conformal predictive system. Otherwise, the distributions
            are represented by a vector of arrays, if cpds_by_bins = False,
            or a list of arrays, with one element for each bin, if 
            cpds_by_bins = True.

        Examples
        --------
        Assuming that ``y_hat_test`` and ``y_test`` are vectors with predicted
        and true targets, respectively, for a test set and ``cps_std`` a fitted
        standard conformal predictive system, the p-values for the true targets 
        can be obtained by:

        .. code-block:: python

           p_values = cps_std.predict(y_hat=y_hat_test, y=y_test)

        The p-values with respect to some specific value, e.g., 37, can be
        obtained by:

        .. code-block:: python

           p_values = cps_std.predict(y_hat=y_hat_test, y=37)

        Assuming that ``sigmas_test`` is a vector with difficulty estimates for
        the test set and ``cps_norm`` a fitted normalized conformal predictive 
        system, then the 90th and 95th percentiles can be obtained by:

        .. code-block:: python

           percentiles = cps_norm.predict(y_hat=y_hat_test, sigmas=sigmas_test,
                                          higher_percentiles=[90,95])

        In the above example, the nearest higher value is returned, if there is
        no value that corresponds exactly to the requested percentile. If we
        instead would like to retrieve the nearest lower value, we should 
        write:

        .. code-block:: python

           percentiles = cps_norm.predict(y_hat=y_hat_test, sigmas=sigmas_test,
                                          lower_percentiles=[90,95])

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cps_mond`` a fitted Mondrian conformal 
        regressor, then the following returns prediction intervals at the 
        95% confidence level, where the intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = cps_mond.predict(y_hat=y_hat_test, bins=bins_test,
                                        lower_percentiles=2.5,
                                        higher_percentiles=97.5,
                                        y_min=0)

        If we would like to obtain the conformal distributions, we could write
        the following:

        .. code-block:: python

           cpds = cps_norm.predict(y_hat=y_hat_test, sigmas=sigmas_test,
                                   return_cpds=True)

        The output of the above will be an array with a row for each test
        instance and a column for each calibration instance (residual).
        For a Mondrian conformal predictive system, the above will instead
        result in a vector, in which each element is a vector, as the number
        of calibration instances may vary between categories. If we instead
        would like an array for each category, this can be obtained by:

        .. code-block:: python

           cpds = cps_norm.predict(y_hat=y_hat_test, sigmas=sigmas_test,
                                   return_cpds=True, cpds_by_bins=True)

        Note
        ----
        Setting ``cpds_by_bins`` has an effect only for Mondrian conformal 
        predictive systems.

        Note
        ----
        In case the calibration set is too small for the specified lower and
        higher percentiles, a warning will be issued and the output will be 
        ``y_min`` and ``y_max``, respectively.
        """

        tic = time.time()
        if not self.mondrian:
                if self.normalized:
                    cpds = np.array([y_hat[i]+sigmas[i]*self.alphas
                                     for i in range(len(y_hat))])
                else:
                    cpds = np.array([y_hat[i]+self.alphas
                                     for i in range(len(y_hat))])
        else:           
            bin_values, bin_alphas = self.alphas
            bin_indexes = [np.argwhere(bins == b).T[0] for b in bin_values]
            if self.normalized:
                cpds = [np.array([y_hat[i]+sigmas[i]*bin_alphas[b]
                                  for i in bin_indexes[b]])
                        for b in range(len(bin_values))]
            else:
                cpds = [np.array([y_hat[i]+bin_alphas[b] for
                                  i in bin_indexes[b]])
                        for b in range(len(bin_values))]
        no_prec_result_cols = 0
        if isinstance(lower_percentiles, (int, float, np.integer, np.floating)):
            lower_percentiles = [lower_percentiles]
        if isinstance(higher_percentiles, (int, float, np.integer, np.floating)):
            higher_percentiles = [higher_percentiles]
        if lower_percentiles is None:
            lower_percentiles = []
        if higher_percentiles is None:
            higher_percentiles = []
        no_result_columns = \
            (y is not None) + len(lower_percentiles) + len(higher_percentiles)
        if no_result_columns > 0:
            result = np.zeros((len(y_hat),no_result_columns))
        if y is not None:
            no_prec_result_cols += 1
            gammas = np.random.rand(len(y_hat))
            if isinstance(y, (int, float, np.integer, np.floating)):
                if not self.mondrian:
                    result[:,0] = np.array([(len(np.argwhere(cpds[i]<y)) \
                                             + gammas[i])/(len(cpds[i])+1)
                                            for i in range(len(cpds))])
                else:
                    for b in range(len(bin_values)):
                        result[bin_indexes[b],0] = np.array(
                            [(len(np.argwhere(cpds[b][i]<y)) \
                              + gammas[bin_indexes[b]][i])/(len(cpds[b])+1)
                             for i in range(len(cpds[b]))])
            elif type(y) in [list, np.ndarray] and len(y) == len(y_hat):
                if not self.mondrian:
                    result[:,0] = np.array([(len(np.argwhere(cpds[i]<y[i])) \
                                             + gammas[i])/(len(cpds[i])+1)
                                            for i in range(len(cpds))])
                else:
                    for b in range(len(bin_values)):
                        result[bin_indexes[b],0] = \
                            np.array([(len(np.argwhere(
                                cpds[b][i]<y[bin_indexes[b]][i])) \
                                       + gammas[bin_indexes[b]][i]) \
                                      /(len(cpds[b][0])+1)
                                      for i in range(len(cpds[b]))])
            else:
                raise ValueError(("y must either be a single int, float or"
                                  "a list/numpy array of the same length as "
                                  "the residuals"))
        if len(lower_percentiles) > 0:
                if not self.mondrian:
                    lower_indexes = np.array([int(lower_percentile/100 \
                                         * (len(self.alphas)+1))-1
                                     for lower_percentile in lower_percentiles])
                    too_low_indexes = np.argwhere(lower_indexes < 0)
                    if len(too_low_indexes) > 0:
                        lower_indexes[too_low_indexes[:,0]] = 0
                    result[:,no_prec_result_cols:no_prec_result_cols \
                           + len(lower_percentiles)] = cpds[:,lower_indexes]
                    if len(too_low_indexes) > 0:
                        percentiles_to_show = " ".join([
                            str(lower_percentiles[i])
                            for i in too_low_indexes[:,0]])
                        warnings.warn("the no. of calibration examples is " \
                                      "too few for the following lower " \
                                      f"percentiles: {percentiles_to_show}; "\
                                      "the corresponding values are " \
                                      "set to y_min")
                        y_min_columns = [no_prec_result_cols+i
                                         for i in too_low_indexes[:,0]]
                        result[:,y_min_columns] = y_min
                else:
                    too_small_bins = []
                    for b in range(len(bin_values)):
                        lower_indexes = np.array([int(lower_percentile/100 \
                                             * (len(bin_alphas[b])+1))-1
                                         for lower_percentile
                                         in lower_percentiles])
                        too_low_indexes = np.argwhere(lower_indexes < 0)
                        if len(too_low_indexes) > 0:
                            lower_indexes[too_low_indexes[:,0]] = 0
                            too_small_bins.append(str(bin_values[b]))                            
                        result[bin_indexes[b],
                               no_prec_result_cols:no_prec_result_cols \
                               + len(lower_indexes)] = cpds[b][:,lower_indexes]
                        if len(too_low_indexes) > 0:
                            for i in too_low_indexes[:,0]:
                                result[bin_indexes[b],no_prec_result_cols+i] = y_min
                    if len(too_small_bins) > 0:
                        if len(too_small_bins) < 11:
                            bins_to_show = " ".join(too_small_bins)
                        else:
                            bins_to_show = " ".join(
                                too_small_bins[:10]+['...'])
                        warnings.warn("the no. of calibration examples is " \
                                      "too few for some lower percentile" \
                                      "in the following bins:" \
                                      f"{bins_to_show}; "\
                                      "the corresponding values are " \
                                      "set to y_min")
        if y_min > -np.inf:
            result[:,
                   no_prec_result_cols:no_prec_result_cols \
                   + len(lower_percentiles)]\
                   [result[:,no_prec_result_cols:no_prec_result_cols \
                           + len(lower_percentiles)]<y_min] = y_min
        no_prec_result_cols += len(lower_percentiles)
        if len(higher_percentiles) > 0:
                if not self.mondrian:
                    higher_indexes = np.array(
                        [int(np.ceil(higher_percentile/100 \
                                     * (len(self.alphas)+1)))-1
                         for higher_percentile in higher_percentiles],
                        dtype=int)
                    too_high_indexes = np.array(
                        [i for i in range(len(higher_indexes))
                         if higher_indexes[i] > len(self.alphas)-1], dtype=int)
                    if len(too_high_indexes) > 0:
                        percentiles_to_show = " ".join(
                            [str(higher_percentiles[i])
                             for i in too_high_indexes])
                        warnings.warn("the no. of calibration examples is " \
                                      "too few for the following higher " \
                                      f"percentiles: {percentiles_to_show}; "\
                                      "the corresponding values are " \
                                      "set to y_max") 
                    higher_indexes[too_high_indexes] = len(self.alphas)-1
                    result[:,no_prec_result_cols:no_prec_result_cols \
                           + len(higher_indexes)] = cpds[:,higher_indexes]
                    result[:,no_prec_result_cols+too_high_indexes] = y_max
                else:
                    too_small_bins = []
                    for b in range(len(bin_values)):
                        higher_indexes = np.array([
                            int(np.ceil(higher_percentile/100 \
                                        * (len(bin_alphas[b])+1)))-1
                            for higher_percentile in higher_percentiles])
                        too_high_indexes = np.array(
                            [i for i in range(len(higher_indexes))
                             if higher_indexes[i] > len(bin_alphas[b])-1], dtype=int)
                        if len(too_high_indexes) > 0:
                            higher_indexes[too_high_indexes] = -1
                            too_small_bins.append(str(bin_values[b]))                            
                        result[bin_indexes[b],
                               no_prec_result_cols:no_prec_result_cols \
                               + len(higher_indexes)] = cpds[b][:,higher_indexes]
                        if len(too_high_indexes) > 0:
                            for i in too_high_indexes:
                                result[bin_indexes[b],no_prec_result_cols+i] = y_max
                    if len(too_small_bins) > 0:
                        if len(too_small_bins) < 11:
                            bins_to_show = " ".join(too_small_bins)
                        else:
                            bins_to_show = " ".join(
                                too_small_bins[:10]+['...'])
                        warnings.warn("the no. of calibration examples is " \
                                      "too few for some higher percentile" \
                                      "in the following bins:" \
                                      f"{bins_to_show}; "\
                                      "the corresponding values are " \
                                      "set to y_max")
        if y_max < np.inf:
            result[:,no_prec_result_cols:no_prec_result_cols\
                   + len(higher_percentiles)]\
                   [result[:,no_prec_result_cols:no_prec_result_cols \
                           + len(higher_percentiles)]>y_max] = y_max
        toc = time.time()
        self.time_predict = toc-tic            
        if no_result_columns > 0 and return_cpds:
            if not self.mondrian or cpds_by_bins:
                cpds_out = cpds
            else:
                cpds_out = np.empty(len(y_hat), dtype=object)
                for b in range(len(bin_values)):
                    cpds_out[bin_indexes[b]] = [cpds[b][i]
                                                for i in range(len(cpds[b]))]
            return result, cpds_out
        elif no_result_columns > 0:
            return result
        elif return_cpds:
            if not self.mondrian or cpds_by_bins:
                cpds_out = cpds
            else:
                cpds_out = np.empty(len(y_hat), dtype=object)
                for b in range(len(bin_values)):
                    cpds_out[bin_indexes[b]] = [cpds[b][i]
                                                for i in range(len(cpds[b]))]
            return cpds_out

    def evaluate(self, y_hat=None, y=None, sigmas=None, bins=None,
                 confidence=0.95, y_min=-np.inf, y_max=np.inf, metrics=None):
        """
        Evaluate conformal predictive system.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,), default=None,
            predicted values
        y : array-like of shape (n_values,), default=None,
            correct target values
        sigmas : array-like of shape (n_values,), default=None,
            difficulty estimates
        bins : array-like of shape (n_values,), default=None,
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals
        metrics : a string or a list of strings, default=list of all 
            metrics; ["error", "eff_mean","eff_med", "CRPS", "time_fit",
                      "time_evaluate"]
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics

        Examples
        --------
        Assuming that ``y_hat_test`` and ``y_test`` are vectors with predicted
        and true targets for a test set, ``sigmas_test`` and ``bins_test`` are
        vectors with difficulty estimates and Mondrian categories (bin labels) 
        for the test set, and ``cps_norm_mond`` is a fitted normalized Mondrian
        conformal predictive system, then the latter can be evaluated at the 
        default confidence level with respect to error, mean and median 
        efficiency (interval size, given the default confidence level) and 
        continuous-ranked probability score (CRPS) by:

        .. code-block:: python

           results = cps_norm_mond.evaluate(y_hat=y_hat_test, y=y_test, 
                                            sigmas=sigmas_test, bins=bins_test,
                                            metrics=["error", "eff_mean", 
                                                     "eff_med", "CRPS"])
        """

        tic = time.time()
        if metrics is None:
            metrics = ["error","eff_mean","eff_med","CRPS","time_fit",
                       "time_evaluate"]
        lower_percentile = (1-confidence)/2*100
        higher_percentile = (confidence+(1-confidence)/2)*100
        test_results = {}
        if "CRPS" in metrics:
            results, cpds = self.predict(y_hat, sigmas=sigmas, bins=bins, y=y,
                                         lower_percentiles=lower_percentile,
                                         higher_percentiles=higher_percentile,
                                         y_min=y_min, y_max=y_max,
                                         return_cpds=True, cpds_by_bins=True)
            intervals = results[:,[1,2]]
            if not self.mondrian:
                if self.normalized:
                    crps = calculate_crps(cpds, self.alphas, sigmas, y)
                else:
                    crps = calculate_crps(cpds, self.alphas,
                                          np.ones(len(y_hat)), y)
            else:
                bin_values, bin_alphas = self.alphas
                bin_indexes = [np.argwhere(bins == b).T[0]
                               for b in bin_values]
                if self.normalized:
                    crps = np.sum([calculate_crps(cpds[b],
                                                  bin_alphas[b],
                                                  sigmas[bin_indexes[b]],
                                                  y[bin_indexes[b]]) \
                                   * len(bin_indexes[b])
                                   for b in range(len(bin_values))])/len(y)
                else:
                    crps = np.sum([calculate_crps(cpds[b],
                                                  bin_alphas[b],
                                                  np.ones(len(bin_indexes[b])),
                                                  y[bin_indexes[b]]) \
                                   * len(bin_indexes[b])
                                   for b in range(len(bin_values))])/len(y)
        else:
            intervals = self.predict(y_hat, sigmas=sigmas, bins=bins,
                                     lower_percentiles=lower_percentile,
                                     higher_percentiles=higher_percentile,
                                     y_min=y_min, y_max=y_max,
                                     return_CRPS=False)
        if "error" in metrics:
            test_results["error"] = 1-np.mean(np.logical_and(
                intervals[:,0]<=y,y<=intervals[:,1]))
        if "eff_mean" in metrics:            
            test_results["eff_mean"] = np.mean(intervals[:,1]-intervals[:,0])
        if "eff_med" in metrics:            
            test_results["eff_med"] = np.median(intervals[:,1]-intervals[:,0])
        if "CRPS" in metrics:
            test_results["CRPS"] = crps
        if "time_fit" in metrics:
            test_results["time_fit"] = self.time_fit
        toc = time.time()
        self.time_evaluate = toc-tic
        if "time_evaluate" in metrics:
            test_results["time_evaluate"] = self.time_evaluate
        return test_results
        
def calculate_crps(cpds, alphas, sigmas, y):
    """
    Calculate mean continuous-ranked probability score (crps)
    for a set of conformal predictive distributions.

    Parameters
    ----------
    cpds : array-like of shape (n_values, c_values)
        conformal predictive distributions
    alphas : array-like of shape (c_values,)
        sorted (normalized) residuals of the calibration examples 
    sigmas : array-like of shape (n_values,),
        difficulty estimates
    y : array-like of shape (n_values,)
        correct target values
        
    Returns
    -------
    crps : float
        mean continuous-ranked probability score for the conformal
        predictive distributions 
    """
    widths = np.array([alphas[i+1]-alphas[i] for i in range(len(alphas)-1)])
    cum_probs = np.cumsum([1/len(alphas) for i in range(len(alphas)-1)])
    lower_errors = cum_probs**2
    higher_errors = (1-cum_probs)**2
    cpd_indexes = [np.argwhere(cpds[i]<y[i]) for i in range(len(y))]
    cpd_indexes = [-1 if len(c)==0 else c[-1][0] for c in cpd_indexes]
    return np.mean([get_crps(cpd_indexes[i], lower_errors, higher_errors,
                             widths, sigmas[i], cpds[i], y[i])
                    for i in range(len(y))])
        
def get_crps(cpd_index, lower_errors, higher_errors, widths, sigma, cpd, y):
    """
    Calculate continuous-ranked probability score (crps) for a single
    conformal predictive distribution. 

    Parameters
    ----------
    cpd_index : int
        highest index for which y is higher than the corresponding cpd value
    lower_errors : array-like of shape (c_values-1,)
        values to add to crps for values less than y
    higher_errors : array-like of shape (c_values-1,)
        values to add to crps for values higher than y
    widths : array-like of shape (c_values-1,),
        differences between consecutive pairs of sorted (normalized) residuals 
        of the calibration examples 
    sigma : int or float
        difficulty estimate for single object
    cpd : array-like of shape (c_values,)
        conformal predictive distyribution
    y : int or float
        correct target value
        
    Returns
    -------
    crps : float
        continuous-ranked probability score
    """
    if cpd_index == -1:
        score = np.sum(higher_errors*widths*sigma)+(cpd[0]-y) 
    elif cpd_index == len(cpd)-1:
        score = np.sum(lower_errors*widths*sigma)+(y-cpd[-1]) 
    else:
        score = np.sum(lower_errors[:cpd_index]*widths[:cpd_index]*sigma) +\
            np.sum(higher_errors[cpd_index+1:]*widths[cpd_index+1:]*sigma) +\
            lower_errors[cpd_index]*(y-cpd[cpd_index])*sigma +\
            higher_errors[cpd_index]*(cpd[cpd_index+1]-y)*sigma
    return score
