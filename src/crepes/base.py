"""Conformal classifiers, regressors, and predictive systems (crepes)

Classes implementing conformal classifiers, regressors, and predictive
systems, on top of any standard classifier and regressor, transforming
the original predictions into well-calibrated p-values and cumulative
distribution functions, or prediction sets and intervals with coverage
guarantees.

Author: Henrik Boström (bostromh@kth.se)

Copyright 2024 Henrik Boström

License: BSD 3 clause

"""

__version__ = "0.7.0"

import numpy as np
import pandas as pd
import time
import warnings

from crepes.extras import hinge, MondrianCategorizer

warnings.simplefilter("always", UserWarning)

class ConformalPredictor():
    """
    The class contains three sub-classes: :class:`.ConformalClassifier`,
    :class:`.ConformalRegressor`, and :class:`.ConformalPredictiveSystem`.
    """
    
    def __init__(self):
        self.fitted = False
        self.mondrian = None
        self.time_fit = None
        self.time_predict = None
        self.time_evaluate = None
        self.alphas = None
        self.normalized = None

class ConformalClassifier(ConformalPredictor):
    """
    A conformal classifier transforms non-conformity scores into p-values
    or prediction sets for a certain confidence level.
    """
    
    def __repr__(self):
        if self.fitted:
            return (f"ConformalClassifier(fitted={self.fitted}, "
                    f"mondrian={self.mondrian})")
        else:
            return f"ConformalClassifier(fitted={self.fitted})"
    
    def fit(self, alphas, bins=None):
        """
        Fit conformal classifier.

        Parameters
        ----------
        alphas : array-like of shape (n_samples,)
            non-conformity scores
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories

        Returns
        -------
        self : object
            Fitted ConformalClassifier.

        Examples
        --------
        Assuming that ``alphas_cal`` is a vector with non-conformity scores,
        then a standard conformal classifier is formed in the following way:

        .. code-block:: python

           from crepes import ConformalClassifier

           cc_std = ConformalClassifier() 

           cc_std.fit(alphas_cal) 

        Assuming that ``bins_cals`` is a vector with Mondrian categories 
        (bin labels), then a Mondrian conformal classifier is fitted in the
        following way:

        .. code-block:: python

           cc_mond = ConformalClassifier()
           cc_mond.fit(alphas_cal, bins=bins_cal)
        """
        tic = time.time()
        if bins is None:
            self.mondrian = False
            self.alphas = np.sort(alphas)[::-1]
        else: 
            self.mondrian = True
            bin_values = np.unique(bins)
            self.alphas = (bin_values, [np.sort(alphas[bins==b])[::-1]
                                        for b in bin_values])
        self.fitted = True
        toc = time.time()
        self.time_fit = toc-tic
        return self

    def predict_p(self, alphas, bins=None, confidence=0.95):
        """
        Obtain (smoothed) p-values from conformal classifier.

        Parameters
        ----------
        alphas : array-like of shape (n_samples, n_classes)
            non-conformity scores
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level

        Returns
        -------
        p-values : ndarray of shape (n_samples, n_classes)
            p-values

        Examples
        --------
        Assuming that ``alphas_test`` is a vector with non-conformity scores
        for a test set and ``cc_std`` a fitted standard conformal classifier, 
        then p-values for the test is obtained by:

        .. code-block:: python

           p_values = cc_std.predict_p(alphas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cc_mond`` a fitted Mondrian conformal 
        classifier, then the following provides p-values for the test set:

        .. code-block:: python

           p_values = cc_mond.predict_p(alphas_test, bins=bins_test)
        """
        tic = time.time()
        if not self.mondrian:
            p_values = np.array(
                [[(np.sum(self.alphas > alpha) + np.random.rand()*(
                    np.sum(self.alphas == alpha)+1))/(len(self.alphas)+1)
                  for alpha in alpha_row] for alpha_row in alphas])
        else:           
            bin_values, bin_alphas = self.alphas
            bin_indexes = np.array([np.argwhere(bin_values == bins[i])[0][0]
                                    for i in range(len(bins))])
            p_values = np.array([[(np.sum(bin_alphas[bin_indexes[i]] > alpha) \
                                   + np.random.rand()*(np.sum(bin_alphas[
                                       bin_indexes[i]] == alpha)+1))/(
                                           len(bin_alphas[bin_indexes[i]])+1)
                                  for alpha in alphas[i]]
                                 for i in range(len(alphas))])
        toc = time.time()
        self.time_predict = toc-tic            
        return p_values
    
    def predict_set(self, alphas, bins=None, confidence=0.95, smoothing=False):
        """
        Obtain prediction sets using conformal classifier.

        Parameters
        ----------
        alphas : array-like of shape (n_samples, n_classes)
            non-conformity scores
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level
        smoothing : bool, default=False
           use smoothed p-values

        Returns
        -------
        prediction sets : ndarray of shape (n_samples, n_classes)
            prediction sets

        Examples
        --------
        Assuming that ``alphas_test`` is a vector with non-conformity scores
        for a test set and ``cc_std`` a fitted standard conformal classifier, 
        then prediction sets at the default (95%) confidence level are
        obtained by:

        .. code-block:: python

           prediction_sets = cc_std.predict_set(alphas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cc_mond`` a fitted Mondrian conformal 
        classifier, then the following provides prediction sets for the test set,
        at the 90% confidence level:

        .. code-block:: python

           p_values = cc_mond.predict_set(alphas_test, 
                                          bins=bins_test,
                                          confidence=0.9)

        Note
        ----
        Using smoothed p-values substantially increases computation time and
        hardly has any effect on the predictions sets, except for when having 
        small calibration sets.
        """
        tic = time.time()
        if smoothing:
            p_values = self.predict_p(alphas, bins)
            prediction_sets = (p_values >= 1-confidence).astype(int)
        elif bins is None:
            alpha_index = int((1-confidence)*(len(self.alphas)+1))-1
            if alpha_index >= 0:
                alpha_value = self.alphas[alpha_index]
                prediction_sets = (alphas <= alpha_value).astype(int)
            else:
                prediction_sets = np.ones(alphas.shape)
                warnings.warn("the no. of calibration examples is " \
                              "too small for the chosen confidence level; " \
                              "all labels are included in the prediction sets")
        else:
            bin_values, bin_alphas = self.alphas
            alpha_indexes = np.array(
                [int((1-confidence)*(len(bin_alphas[b])+1))-1
                 for b in range(len(bin_values))])
            alpha_values = [bin_alphas[b][alpha_indexes[b]]
                            if alpha_indexes[b] >= 0
                            else -np.inf for b in range(len(bin_values))]
            bin_indexes = np.array([np.argwhere(bin_values == bins[i])[0][0]
                                    for i in range(len(bins))])
            prediction_sets = np.array(
                [alphas[i] <= alpha_values[bin_indexes[i]]
                 for i in range(len(alphas))], dtype=int)
            if (alpha_indexes < 0).any():
                warnings.warn("the no. of calibration examples in some bins is" \
                              " too small for the chosen confidence level; " \
                              "all labels are included in the corresponding" \
                              "prediction sets")
        toc = time.time()
        self.time_predict = toc-tic            
        return prediction_sets

    def evaluate(self, alphas, classes, y, bins=None, confidence=0.95,
                 smoothing=False, metrics=None):
        """
        Evaluate conformal classifier.

        Parameters
        ----------
        alphas : array-like of shape (n_samples, n_classes)
            non-conformity scores
        classes : array-like of shape (n_classes,)
            class names
        y : array-like of shape (n_samples,)
            correct class labels
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        confidence : float in range (0,1), default=0.95
            confidence level
        smoothing : bool, default=False
           use smoothed p-values
        metrics : a string or a list of strings, 
                  default = list of all metrics, i.e., ["error", "avg_c", 
                  "one_c", "empty", "time_fit", "time_evaluate"]
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics, where "error" is the 
            fraction of prediction sets not containing the true class label,
            "avg_c" is the average no. of predicted class labels, "one_c" is
            the fraction of singleton prediction sets, "empty" is the fraction
            of empty prediction sets, "time_fit" is the time taken to fit the 
            conformal classifier, and "time_evaluate" is the time taken for the
            evaluation 

        Examples
        --------
        Assuming that ``alphas`` is an array containing non-conformity scores 
        for all classes for the test objects, ``classes`` and ``y_test`` are 
        vectors with the class names and true class labels for the test set, 
        respectively, and ``cc`` is a fitted standard conformal classifier, 
        then the latter can be evaluated at the default confidence level with 
        respect to error and average number of labels in the prediction sets by:

        .. code-block:: python

           results = cc.evaluate(alphas, y_test, metrics=["error", "avg_c"])

        Note
        ----
        Using smoothed p-values substantially increases computation time and
        hardly has any effect on the results, except for when having small 
        calibration sets.
        """
        if metrics is None:
            metrics = ["error", "avg_c", "one_c", "empty", "time_fit",
                       "time_evaluate"]
        tic = time.time()
        prediction_sets = self.predict_set(alphas, bins, confidence, smoothing)
        test_results = get_test_results(prediction_sets, classes, y, metrics)
        toc = time.time()
        self.time_evaluate = toc-tic
        if "time_fit" in metrics:
            test_results["time_fit"] = self.time_fit
        if "time_evaluate" in metrics:
            test_results["time_evaluate"] = self.time_evaluate
        return test_results

def get_test_results(prediction_sets, classes, y, metrics):
    test_results = {}
    class_indexes = np.array(
        [np.argwhere(classes == y[i])[0][0] for i in range(len(y))])        
    if "error" in metrics:
        test_results["error"] = 1-np.sum(
            prediction_sets[np.arange(len(y)), class_indexes]) / len(y)
    if "avg_c" in metrics:            
        test_results["avg_c"] = np.sum(prediction_sets) / len(y)
    if "one_c" in metrics:            
        test_results["one_c"] = np.sum(
            [np.sum(p) == 1 for p in prediction_sets]) / len(y)
    if "empty" in metrics:            
        test_results["empty"] = np.sum(
            [np.sum(p) == 0 for p in prediction_sets]) / len(y)
    return test_results

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
    
    def fit(self, residuals, sigmas=None, bins=None):
        """
        Fit conformal regressor.

        Parameters
        ----------
        residuals : array-like of shape (n_values,)
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
           cr_norm.fit(residuals_cal, sigmas=sigmas_cal)

        Assuming that ``bins_cals`` is a vector with Mondrian categories 
        (bin labels), then a Mondrian conformal regressor can be fitted in the
        following way:

        .. code-block:: python

           cr_mond = ConformalRegressor()
           cr_mond.fit(residuals_cal, bins=bins_cal)

        A normalized Mondrian conformal regressor can be fitted in the 
        following way:

        .. code-block:: python

           cr_norm_mond = ConformalRegressor()
           cr_norm_mond.fit(residuals_cal, sigmas=sigmas_cal, 
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

    def predict(self, y_hat, sigmas=None, bins=None, confidence=0.95,
                y_min=-np.inf, y_max=np.inf):
        """
        Predict using conformal regressor.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
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

           intervals = cr_std.predict(y_hat_test, confidence=0.99)

        Assuming that ``sigmas_test`` is a vector with difficulty estimates for
        the test set and ``cr_norm`` a fitted normalized conformal regressor, 
        then prediction intervals at the default (95%) confidence level can be
        obtained by:

        .. code-block:: python

           intervals = cr_norm.predict(y_hat_test, sigmas=sigmas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cr_mond`` a fitted Mondrian conformal 
        regressor, then the following provides prediction intervals at the 
        default confidence level, where the intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = cr_mond.predict(y_hat_test, bins=bins_test, 
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
                warnings.warn("the no. of calibration examples is too small" \
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

    def evaluate(self, y_hat, y, sigmas=None, bins=None,
                 confidence=0.95, y_min=-np.inf, y_max=np.inf, metrics=None):
        """
        Evaluate conformal regressor.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct target values
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

           results = cr_norm_mond.evaluate(y_hat_test, y_test, 
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
                np.logical_and(intervals[:,0]<=y, y<=intervals[:,1]))
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

    def fit(self, residuals, sigmas=None, bins=None):
        """
        Fit conformal predictive system.

        Parameters
        ----------
        residuals : array-like of shape (n_values,)
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
           cps_norm.fit(residuals_cal, sigmas=sigmas_cal)

        Assuming that ``bins_cals`` is a vector with Mondrian categories (bin
        labels), then a Mondrian conformal predictive system can be fitted in
        the following way:

        .. code-block:: python

           cps_mond = ConformalPredictiveSystem()
           cps_mond.fit(residuals_cal, bins=bins_cal)

        A normalized Mondrian conformal predictive system can be fitted in the
        following way:

        .. code-block:: python

           cps_norm_mond = ConformalPredictiveSystem()
           cps_norm_mond.fit(residuals_cal, sigmas=sigmas_cal, 
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
        
    def predict(self, y_hat, sigmas=None, bins=None,
                y=None, lower_percentiles=None, higher_percentiles=None,
                y_min=-np.inf, y_max=np.inf, return_cpds=False,
                cpds_by_bins=False):    
        """
        Predict using conformal predictive system.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
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
        results : ndarray of shape (n_values, n_cols) or (n_values,)
            the shape is (n_values, n_cols) if n_cols > 1 and otherwise
            (n_values,), where n_cols = p_values+l_values+h_values where 
            p_values = 1 if y is not None and 0 otherwise, l_values are the
            number of lower percentiles, and h_values are the number of higher
            percentiles. Only returned if n_cols > 0.
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

           p_values = cps_std.predict(y_hat_test, y=y_test)

        The p-values with respect to some specific value, e.g., 37, can be
        obtained by:

        .. code-block:: python

           p_values = cps_std.predict(y_hat_test, y=37)

        Assuming that ``sigmas_test`` is a vector with difficulty estimates for
        the test set and ``cps_norm`` a fitted normalized conformal predictive 
        system, then the 90th and 95th percentiles can be obtained by:

        .. code-block:: python

           percentiles = cps_norm.predict(y_hat_test, sigmas=sigmas_test,
                                          higher_percentiles=[90,95])

        In the above example, the nearest higher value is returned, if there is
        no value that corresponds exactly to the requested percentile. If we
        instead would like to retrieve the nearest lower value, we should 
        write:

        .. code-block:: python

           percentiles = cps_norm.predict(y_hat_test, sigmas=sigmas_test,
                                          lower_percentiles=[90,95])

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cps_mond`` a fitted Mondrian conformal 
        regressor, then the following returns prediction intervals at the 
        95% confidence level, where the intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = cps_mond.predict(y_hat_test, bins=bins_test,
                                        lower_percentiles=2.5,
                                        higher_percentiles=97.5,
                                        y_min=0)

        If we would like to obtain the conformal distributions, we could write
        the following:

        .. code-block:: python

           cpds = cps_norm.predict(y_hat_test, sigmas=sigmas_test,
                                   return_cpds=True)

        The output of the above will be an array with a row for each test
        instance and a column for each calibration instance (residual).
        For a Mondrian conformal predictive system, the above will instead
        result in a vector, in which each element is a vector, as the number
        of calibration instances may vary between categories. If we instead
        would like an array for each category, this can be obtained by:

        .. code-block:: python

           cpds = cps_norm.predict(y_hat_test, sigmas=sigmas_test,
                                   return_cpds=True, cpds_by_bins=True)

        Note
        ----
        In case the calibration set is too small for the specified lower and
        higher percentiles, a warning will be issued and the output will be 
        ``y_min`` and ``y_max``, respectively.

        Note
        ----
        Setting ``return_cpds=True`` may consume a lot of memory, as a matrix is
        generated for which the number of elements is the product of the number 
        of calibration and test objects, unless a Mondrian approach is employed; 
        for the latter, this number is reduced by increasing the number of bins.

        Note
        ----
        Setting ``cpds_by_bins=True`` has an effect only for Mondrian conformal 
        predictive systems.
        """

        tic = time.time()
        if self.mondrian:
            bin_values, bin_alphas = self.alphas
            bin_indexes = [np.argwhere(bins == b).T[0] for b in bin_values]
        no_prec_result_cols = 0
        if isinstance(lower_percentiles, (int, float, np.integer, np.floating)):
            lower_percentiles = [lower_percentiles]
        if isinstance(higher_percentiles, (int, float, np.integer, np.floating)):
            higher_percentiles = [higher_percentiles]
        if lower_percentiles is None:
            lower_percentiles = []
        if higher_percentiles is None:
            higher_percentiles = []
        if (np.array(lower_percentiles) > 100).any() or \
           (np.array(lower_percentiles) < 0).any() or \
           (np.array(higher_percentiles) > 100).any() or \
           (np.array(higher_percentiles) < 0).any():
            raise ValueError("All percentiles must be in the range [0,100]")
        no_result_columns = \
            (y is not None) + len(lower_percentiles) + len(higher_percentiles)
        if no_result_columns > 0:
            result = np.zeros((len(y_hat),no_result_columns))
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            no_prec_result_cols += 1
            gammas = np.random.rand(len(y_hat))
            if isinstance(y, (int, float, np.integer, np.floating)):
                if not self.mondrian:
                    if self.normalized:
                        result[:,0] = np.array(
                            [(len(np.argwhere(
                                (y_hat[i]+sigmas[i]*self.alphas)<y)) \
                              + gammas[i])/(len(self.alphas)+1)
                             for i in range(len(y_hat))])
                    else:
                        result[:,0] = np.array(
                            [(len(np.argwhere((y_hat[i]+self.alphas)<y)) \
                              + gammas[i])/(len(self.alphas)+1)
                             for i in range(len(y_hat))])                      
                else:
                    for b in range(len(bin_values)):
                        if self.normalized:
                            result[bin_indexes[b],0] = np.array(
                                [(len(np.argwhere(
                                    (y_hat[i]+sigmas[i]*bin_alphas[b])<y)) \
                                  + gammas[i])/(len(bin_alphas[b])+1)
                                 for i in bin_indexes[b]])
                        else:
                            result[bin_indexes[b],0] = np.array(
                                [(len(np.argwhere(
                                    (y_hat[i]+bin_alphas[b])<y)) \
                                  + gammas[i])/(len(bin_alphas[b])+1)
                                 for i in bin_indexes[b]])
            elif isinstance(y, (list, np.ndarray)) and len(y) == len(y_hat):
                if not self.mondrian:
                    if self.normalized:
                        result[:,0] = np.array(
                            [(len(np.argwhere(
                                (y_hat[i]+sigmas[i]*self.alphas)<y[i])) \
                              + gammas[i])/(len(self.alphas)+1)
                             for i in range(len(y_hat))])
                    else:
                        result[:,0] = np.array(
                            [(len(np.argwhere((y_hat[i]+self.alphas)<y[i])) \
                              + gammas[i])/(len(self.alphas)+1)
                             for i in range(len(y_hat))])
                else:
                    for b in range(len(bin_values)):
                        if self.normalized:
                            result[bin_indexes[b],0] = np.array(
                                [(len(np.argwhere(
                                    (y_hat[i]+sigmas[i]*bin_alphas[b])<y[i])) \
                                  + gammas[i])/(len(bin_alphas[b])+1)
                                 for i in bin_indexes[b]])
                        else:
                            result[bin_indexes[b],0] = np.array(
                                [(len(np.argwhere(
                                    (y_hat[i]+bin_alphas[b])<y[i])) \
                                  + gammas[i])/(len(bin_alphas[b])+1)
                                 for i in bin_indexes[b]])
            else:
                raise ValueError(("y must either be a single int, float or"
                                  "a list/numpy array of the same length as "
                                  "the residuals"))
        percentile_indexes = []
        y_min_columns = []
        y_max_columns = []
        if len(lower_percentiles) > 0:
            if not self.mondrian:
                lower_indexes = np.array([int(lower_percentile/100 \
                                              * (len(self.alphas)+1))-1
                                          for lower_percentile in lower_percentiles])
                too_low_indexes = np.argwhere(lower_indexes < 0)
                if len(too_low_indexes) > 0:
                    lower_indexes[too_low_indexes[:,0]] = 0
                    percentiles_to_show = " ".join([
                        str(lower_percentiles[i])
                        for i in too_low_indexes[:,0]])
                    warnings.warn("the no. of calibration examples is " \
                                  "too small for the following lower " \
                                  f"percentiles: {percentiles_to_show}; "\
                                  "the corresponding values are " \
                                  "set to y_min")
                    y_min_columns = [no_prec_result_cols+i
                                     for i in too_low_indexes[:,0]]
                percentile_indexes = lower_indexes
            else:
                too_small_bins = []
                binned_lower_indexes = []
                for b in range(len(bin_values)):
                    lower_indexes = np.array([int(lower_percentile/100 \
                                                  * (len(bin_alphas[b])+1))-1
                                              for lower_percentile
                                              in lower_percentiles])
                    binned_lower_indexes.append(lower_indexes)
                    too_low_indexes = np.argwhere(lower_indexes < 0)
                    if len(too_low_indexes) > 0:
                        lower_indexes[too_low_indexes[:,0]] = 0
                        too_small_bins.append(str(bin_values[b]))                            
                        y_min_columns.append([no_prec_result_cols+i
                                              for i in too_low_indexes[:,0]])
                    else:
                        y_min_columns.append([])
                percentile_indexes = [binned_lower_indexes]
                if len(too_small_bins) > 0:
                    if len(too_small_bins) < 11:
                        bins_to_show = " ".join(too_small_bins)
                    else:
                        bins_to_show = " ".join(
                            too_small_bins[:10]+['...'])
                    warnings.warn("the no. of calibration examples is " \
                                  "too small for some lower percentile " \
                                  "in the following bins:" \
                                  f"{bins_to_show}; "\
                                  "the corresponding values are " \
                                  "set to y_min")                   
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
                    higher_indexes[too_high_indexes] = len(self.alphas)-1
                    percentiles_to_show = " ".join(
                        [str(higher_percentiles[i])
                         for i in too_high_indexes])
                    warnings.warn("the no. of calibration examples is " \
                                  "too small for the following higher " \
                                  f"percentiles: {percentiles_to_show}; "\
                                  "the corresponding values are " \
                                  "set to y_max")
                    y_max_columns = [no_prec_result_cols+len(lower_percentiles)+i
                                     for i in too_high_indexes]
                if len(percentile_indexes) == 0:
                    percentile_indexes = higher_indexes
                else:
                    percentile_indexes = np.concatenate((lower_indexes,
                                                         higher_indexes))
            else:
                too_small_bins = []
                binned_higher_indexes = []
                for b in range(len(bin_values)):
                    higher_indexes = np.array([
                        int(np.ceil(higher_percentile/100 \
                                    * (len(bin_alphas[b])+1)))-1
                        for higher_percentile in higher_percentiles])
                    binned_higher_indexes.append(higher_indexes)
                    too_high_indexes = np.array(
                        [i for i in range(len(higher_indexes))
                         if higher_indexes[i] > len(bin_alphas[b])-1],
                        dtype=int)
                    if len(too_high_indexes) > 0:
                        higher_indexes[too_high_indexes] = -1
                        too_small_bins.append(str(bin_values[b]))
                        y_max_columns.append([no_prec_result_cols + \
                                              len(lower_percentiles)+i
                                              for i in too_high_indexes])
                    else:
                        y_max_columns.append([])
                if len(percentile_indexes) == 0:
                    percentile_indexes = [binned_higher_indexes]
                else:
                    percentile_indexes.append(binned_higher_indexes)
                if len(too_small_bins) > 0:
                    if len(too_small_bins) < 11:
                        bins_to_show = " ".join(too_small_bins)
                    else:
                        bins_to_show = " ".join(
                            too_small_bins[:10]+['...'])
                    warnings.warn("the no. of calibration examples is " \
                                  "too small for some higher percentile " \
                                  "in the following bins:" \
                                  f"{bins_to_show}; "\
                                  "the corresponding values are " \
                                  "set to y_max")
        if len(percentile_indexes) > 0:
            if not self.mondrian:
                if self.normalized:
                    result[:,no_prec_result_cols:no_prec_result_cols \
                           + len(percentile_indexes)] = np.array(
                               [(y_hat[i] + sigmas[i] * \
                                 self.alphas)[percentile_indexes]
                                for i in range(len(y_hat))])
                else:
                    result[:,no_prec_result_cols:no_prec_result_cols \
                           + len(percentile_indexes)] = np.array(
                               [(y_hat[i]+self.alphas)[percentile_indexes]
                                for i in range(len(y_hat))])
                if len(y_min_columns) > 0:
                    result[:,y_min_columns] = y_min
                if len(y_max_columns) > 0:
                    result[:,y_max_columns] = y_max
            else:
                if len(percentile_indexes) == 1:
                    percentile_indexes = percentile_indexes[0]
                else:
                    percentile_indexes = [np.concatenate(
                        (percentile_indexes[0][b],percentile_indexes[1][b]))
                                          for b in range(len(bin_values))]
                if self.normalized:
                    for b in range(len(bin_values)):
                        if len(bin_indexes[b]) > 0:
                            result[bin_indexes[b],
                                   no_prec_result_cols:no_prec_result_cols \
                                   + len(percentile_indexes[b])] = \
                                       np.array([(y_hat[i] + sigmas[i] * \
                                                  bin_alphas[b])[
                                                      percentile_indexes[b]]
                                        for i in bin_indexes[b]])
                else:
                    for b in range(len(bin_values)):
                        if len(bin_indexes[b]) > 0:
                            result[bin_indexes[b],
                                   no_prec_result_cols:no_prec_result_cols \
                                   + len(percentile_indexes[b])] = np.array(
                                       [(y_hat[i]+bin_alphas[b])[
                                           percentile_indexes[b]]
                                        for i in bin_indexes[b]])
                if len(y_min_columns) > 0:
                    for b in range(len(bin_values)):
                        if len(bin_indexes[b]) > 0 and \
                           len(y_min_columns[b]) > 0:
                                result[bin_indexes[b],y_min_columns[b]] = y_min
                if len(y_max_columns) > 0:
                    for b in range(len(bin_values)):
                        if len(bin_indexes[b]) > 0 and \
                           len(y_max_columns[b]) > 0:
                            result[bin_indexes[b],y_max_columns[b]] = y_max
            if y_min > -np.inf:
                result[:,
                       no_prec_result_cols:no_prec_result_cols \
                       + len(percentile_indexes)]\
                       [result[:,no_prec_result_cols:no_prec_result_cols \
                               + len(percentile_indexes)]<y_min] = y_min
            if y_max < np.inf:
                result[:,no_prec_result_cols:no_prec_result_cols\
                       + len(percentile_indexes)]\
                       [result[:,no_prec_result_cols:no_prec_result_cols \
                               + len(percentile_indexes)]>y_max] = y_max
            no_prec_result_cols += len(percentile_indexes)
        toc = time.time()
        self.time_predict = toc-tic            
        if no_result_columns > 0 and result.shape[1] == 1:
            result = result[:,0]
        if return_cpds:
            if not self.mondrian:
                if self.normalized:
                    cpds = np.array([y_hat[i]+sigmas[i]*self.alphas
                                     for i in range(len(y_hat))])
                else:
                    cpds = np.array([y_hat[i]+self.alphas
                                     for i in range(len(y_hat))])
            else:           
                if self.normalized:
                    cpds = [np.array([y_hat[i]+sigmas[i]*bin_alphas[b]
                                      for i in bin_indexes[b]])
                            for b in range(len(bin_values))]
                else:
                    cpds = [np.array([y_hat[i]+bin_alphas[b] for
                                      i in bin_indexes[b]])
                            for b in range(len(bin_values))]
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

    def evaluate(self, y_hat, y, sigmas=None, bins=None,
                 confidence=0.95, y_min=-np.inf, y_max=np.inf, metrics=None):
        """
        Evaluate conformal predictive system.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
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

           results = cps_norm_mond.evaluate(y_hat_test, y_test, 
                                            sigmas=sigmas_test, bins=bins_test,
                                            metrics=["error", "eff_mean", 
                                                     "eff_med", "CRPS"])

        Note
        ----
        The use of the metric ``CRPS`` may consume a lot of memory, as a matrix
        is generated for which the number of elements is the product of the 
        number of calibration and test objects, unless a Mondrian approach is 
        employed; for the latter, this number is reduced by increasing the number 
        of bins.
        """
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y_hat, pd.Series):
            y_hat = y_hat.values
        tic = time.time()
        test_results = {}
        lower_percentile = (1-confidence)/2*100
        higher_percentile = (confidence+(1-confidence)/2)*100
        if metrics is None:
            metrics = ["error","eff_mean","eff_med","CRPS","time_fit",
                       "time_evaluate"]
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
                                     return_cpds=False)
        if "error" in metrics:
            test_results["error"] = 1-np.mean(np.logical_and(
                intervals[:,0]<=y, y<=intervals[:,1]))
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
    if len(cpds) > 0:
        widths = np.array([alphas[i+1]-alphas[i] for i in range(len(alphas)-1)])
        cum_probs = np.cumsum([1/len(alphas) for i in range(len(alphas)-1)])
        lower_errors = cum_probs**2
        higher_errors = (1-cum_probs)**2
        cpd_indexes = [np.argwhere(cpds[i]<y[i]) for i in range(len(y))]
        cpd_indexes = [-1 if len(c)==0 else c[-1][0] for c in cpd_indexes]
        result = np.mean([get_crps(cpd_indexes[i], lower_errors, higher_errors,
                                   widths, sigmas[i], cpds[i], y[i])
                          for i in range(len(y))])
    else:
        result = 0
    return result

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
    
class WrapRegressor():
    """
    A learner wrapped with a :class:`.ConformalRegressor`
    or :class:`.ConformalPredictiveSystem`.
    """
    
    def __init__(self, learner):
        self.cr = None
        self.cps = None
        self.calibrated = False
        self.learner = learner
        self.de = None
        self.mc = None
        
    def __repr__(self):
        if self.calibrated:
            if self.cr is not None:
                return (f"WrapRegressor(learner={self.learner}, "
                        f"calibrated={self.calibrated}, "
                        f"predictor={self.cr})")
            else:
                return (f"WrapRegressor(learner={self.learner}, "
                        f"calibrated={self.calibrated}, "
                        f"predictor={self.cps})")                
        else:
            return f"WrapRegressor(learner={self.learner}, calibrated={self.calibrated})"
        
    def fit(self, X, y, **kwargs):
        """
        Fit learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,),
            target values
        kwargs : optional arguments
           any additional arguments are forwarded to the
           ``fit`` method of the ``learner`` object

        Returns
        -------
        None

        Examples
        --------
        Assuming ``X_train`` and ``y_train`` to be an array and vector 
        with training objects and labels, respectively, a random
        forest may be wrapped and fitted by:

        .. code-block:: python

           from sklearn.ensemble import RandomForestRegressor
           from crepes import WrapRegressor

           rf = WrapRegressor(RandomForestRegressor())
           rf.fit(X_train, y_train)
           
        Note
        ----
        The learner, which can be accessed by ``rf.learner``, may be fitted 
        before as well as after being wrapped.

        Note
        ----
        All arguments, including any additional keyword arguments, to 
        :meth:`.fit` are forwarded to the ``fit`` method of the learner.        
        """
        if isinstance(y, pd.Series):
            y = y.values
        self.learner.fit(X, y, **kwargs)
    
        
    def predict(self, X):
        """
        Predict with learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects

        Returns
        -------
        y : array-like of shape (n_samples,),
            values predicted using the ``predict`` 
            method of the ``learner`` object.

        Examples
        --------
        Assuming ``w`` is a :class:`.WrapRegressor` object for which the wrapped
        learner ``w.learner`` has been fitted, (point) predictions of the 
        learner can be obtained for a set of test objects ``X_test`` by:

        .. code-block:: python

           y_hat = w.predict(X_test)
           
        The above is equivalent to:

        .. code-block:: python

           y_hat = w.learner.predict(X_test)
        """
        return self.learner.predict(X)

    def calibrate(self, X, y, de=None, mc=None, oob=False, cps=False):
        """
        Fit a :class:`.ConformalRegressor` or 
        :class:`.ConformalPredictiveSystem` using the wrapped learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,),
            target values
        de: :class:`crepes.extras.DifficultyEstimator`, default=None
            object used for computing difficulty estimates
        mc: function or :class:`crepes.extras.MondrianCategorizer`, default=None
            function or :class:`crepes.extras.MondrianCategorizer` for computing Mondrian categories
        oob : bool, default=False
           use out-of-bag estimation
        cps : bool, default=False
            if cps=False, the method fits a :class:`.ConformalRegressor`
            and otherwise, a :class:`.ConformalPredictiveSystem`

        Returns
        -------
        self : object
            The :class:`.WrapRegressor` object is updated with a fitted 
            :class:`.ConformalRegressor` or :class:`.ConformalPredictiveSystem`

        Examples
        --------
        Assuming ``X_cal`` and ``y_cal`` to be an array and vector, 
        respectively, with objects and labels for the calibration set,
        and ``w`` is a :class:`.WrapRegressor` object for which the learner 
        has been fitted, a standard conformal regressor is formed by:

        .. code-block:: python

           w.calibrate(X_cal, y_cal) 

        Assuming that ``de`` is a fitted difficulty estimator,
        a normalized conformal regressor is obtained by: 

        .. code-block:: python

           w.calibrate(X_cal, y_cal, de=de)

        Assuming that ``get_categories`` is a function that returns categories
        (bin labels), a Mondrian conformal regressor is obtained by:

        .. code-block:: python

           w.calibrate(X_cal, y_cal, mc=get_categories)

        A normalized Mondrian conformal regressor is generated in the
        following way:

        .. code-block:: python

           w.calibrate(X_cal, y_cal, de=de, mc=get_categories)

        By providing the option ``oob=True``, the conformal regressor
        will be calibrating using out-of-bag predictions, allowing
        the full set of training objects (``X_train``) and labels (``y_train``)
        to be used, e.g.,

        .. code-block:: python

           w.calibrate(X_train, y_train, oob=True)

        By providing the option ``cps=True``, each of the above calls will instead 
        generate a :class:`.ConformalPredictiveSystem`, e.g.,

        .. code-block:: python

           w.calibrate(X_cal, y_cal, de=de, cps=True)

        Note
        ----
        Enabling out-of-bag calibration, i.e., setting ``oob=True``, requires 
        that the wrapped learner has an attribute ``oob_prediction_``, which 
        e.g., is the case for a ``sklearn.ensemble.RandomForestRegressor``, if
        enabled when created, e.g., ``RandomForestRegressor(oob_score=True)``

        Note
        ----
        The use of out-of-bag calibration, as enabled by ``oob=True``, 
        does not come with the theoretical validity guarantees of the regular
        (inductive) conformal regressors and predictive systems, due to that
        calibration and test instances are not handled in exactly the same way.
        """
        if isinstance(y, pd.Series):
            y = y.values
        if oob:
            residuals = y - self.learner.oob_prediction_
        else:
            residuals = y - self.predict(X)
        if de is None:
            sigmas = None
        else:
            sigmas = de.apply(X)
        self.de = de
        if mc is None:
            bins = None
        elif isinstance(mc, MondrianCategorizer):
            bins = mc.apply(X)
        else:
            bins = mc(X)
        self.mc = mc
        if not cps:
            self.cr = ConformalRegressor()
            self.cr.fit(residuals, sigmas=sigmas, bins=bins)
            self.cps = None
        else:
            self.cps = ConformalPredictiveSystem()
            self.cps.fit(residuals, sigmas=sigmas, bins=bins)
            self.cr = None
        self.calibrated = True
        return self

    def predict_int(self, X, confidence=0.95, y_min=-np.inf, y_max=np.inf):
        """
        Predict interval using fitted :class:`.ConformalRegressor` or
        :class:`.ConformalPredictiveSystem`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals

        Returns
        -------
        intervals : ndarray of shape (n_samples, 2)
            prediction intervals

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects and ``w`` is a 
        :class:`.WrapRegressor` object that has been calibrated, i.e., 
        :meth:`.calibrate` has been applied, prediction intervals at the 
        99% confidence level can be obtained by:

        .. code-block:: python

           intervals = w.predict_int(X_test, confidence=0.99)

        The following provides prediction intervals at the default confidence 
        level (95%), where the intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = w.predict_int(X_test, y_min=0)

        Note
        ----
        In case the specified confidence level is too high in relation to the 
        size of the calibration set, a warning will be issued and the output
        intervals will be of maximum size.
        """
        if not self.calibrated:
            raise RuntimeError(("predict_int requires that calibrate has been"
                                "called first"))
        else:
            y_hat = self.learner.predict(X)
            if self.de is None:
                sigmas = None
            else:
                sigmas = self.de.apply(X)
            if self.mc is None:
                bins = None
            elif isinstance(self.mc, MondrianCategorizer):
                bins = self.mc.apply(X)
            else:
                bins = self.mc(X)
            if self.cr is not None:
                return self.cr.predict(y_hat, sigmas=sigmas, bins=bins,
                                       confidence=confidence,
                                       y_min=y_min, y_max=y_max)
            else:
                lower_percentile = (1-confidence)/2*100
                higher_percentile = (confidence+(1-confidence)/2)*100
                return self.cps.predict(y_hat, sigmas=sigmas, bins=bins,
                                        lower_percentiles=lower_percentile,
                                        higher_percentiles=higher_percentile,
                                        y_min=y_min, y_max=y_max)

    def predict_cps(self, X, y=None, lower_percentiles=None,
                    higher_percentiles=None, y_min=-np.inf, y_max=np.inf,
                    return_cpds=False, cpds_by_bins=False):
        """
        Predict using :class:`.ConformalPredictiveSystem`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : float, int or array-like of shape (n_samples,), default=None
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
        results : ndarray of shape (n_samples, n_cols) or (n_samples,)
            the shape is (n_samples, n_cols) if n_cols > 1 and otherwise
            (n_samples,), where n_cols = p_values+l_values+h_values where 
            p_values = 1 if y is not None and 0 otherwise, l_values are the
            number of lower percentiles, and h_values are the number of higher
            percentiles. Only returned if n_cols > 0.
        cpds : ndarray of (n_samples, c_values), ndarray of (n_samples,)
               or list of ndarrays
            conformal predictive distributions. Only returned if 
            return_cpds == True. For non-Mondrian conformal predictive systems,
            the distributions are represented by a single array, where the 
            number of columns (c_values) is determined by the number of 
            residuals of the fitted conformal predictive system. For Mondrian
            conformal predictive systems, the distributions are represented by
            a vector of arrays, if cpds_by_bins = False, or a list of arrays, 
            with one element for each Mondrian category, if cpds_by_bins = True.

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects, ``y_test`` is a 
        vector with true targets, ``w`` is a :class:`.WrapRegressor` object 
        calibrated with the option ``cps=True``, p-values for the true targets 
        can be obtained by:

        .. code-block:: python

           p_values = w.predict_cps(X_test, y=y_test)

        P-values with respect to some specific value, e.g., 37, can be
        obtained by:

        .. code-block:: python

           p_values = w.predict_cps(X_test, y=37)

        The 90th and 95th percentiles can be obtained by:

        .. code-block:: python

           percentiles = w.predict_cps(X_test, higher_percentiles=[90,95])

        In the above example, the nearest higher value is returned, if there is
        no value that corresponds exactly to the requested percentile. If we
        instead would like to retrieve the nearest lower value, we should 
        write:

        .. code-block:: python

           percentiles = w.predict_cps(X_test, lower_percentiles=[90,95])

        The following returns prediction intervals at the 95% confidence level,
        where the intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = w.predict_cps(X_test,
                                     lower_percentiles=2.5,
                                     higher_percentiles=97.5,
                                     y_min=0)

        If we would like to obtain the conformal distributions, we could write
        the following:

        .. code-block:: python

           cpds = w.predict_cps(X_test, return_cpds=True)

        The output of the above will be an array with a row for each test
        instance and a column for each calibration instance (residual).
        If the learner is wrapped with a Mondrian conformal predictive system, 
        the above will instead result in a vector, in which each element is a
        vector, as the number of calibration instances may vary between 
        categories. If we instead would like an array for each category, this 
        can be obtained by:

        .. code-block:: python

           cpds = w.predict_cps(X_test, return_cpds=True, cpds_by_bins=True)

        Note
        ----
        This method is available only if the learner has been wrapped with a
        :class:`.ConformalPredictiveSystem`, i.e., :meth:`.calibrate`
        has been called with the option ``cps=True``.

        Note
        ----
        In case the calibration set is too small for the specified lower and
        higher percentiles, a warning will be issued and the output will be 
        ``y_min`` and ``y_max``, respectively.

        Note
        ----
        Setting ``return_cpds=True`` may consume a lot of memory, as a matrix is
        generated for which the number of elements is the product of the number 
        of calibration and test objects, unless a Mondrian approach is employed; 
        for the latter, this number is reduced by increasing the number of bins.

        Note
        ----
        Setting ``cpds_by_bins=True`` has an effect only for Mondrian conformal 
        predictive systems.
        """
        if isinstance(y, pd.Series):
            y = y.values
        if self.cps is None:
            raise RuntimeError(("predict_cps requires that calibrate has been"
                                "called first with cps=True"))
        else:
            y_hat = self.learner.predict(X)
            if self.de is None:
                sigmas = None
            else:
                sigmas = self.de.apply(X)
            if self.mc is None:
                bins = None
            elif isinstance(self.mc, MondrianCategorizer):
                bins = self.mc.apply(X)
            else:
                bins = self.mc(X)
            return self.cps.predict(y_hat, sigmas=sigmas, bins=bins,
                                    y=y, lower_percentiles=lower_percentiles,
                                    higher_percentiles=higher_percentiles,
                                    y_min=y_min, y_max=y_max,
                                    return_cpds=return_cpds,
                                    cpds_by_bins=cpds_by_bins)

    def evaluate(self, X, y, confidence=0.95,
                 y_min=-np.inf, y_max=np.inf, metrics=None):
        """
        Evaluate :class:`.ConformalRegressor` or 
        :class:`.ConformalPredictiveSystem`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           set of objects
        y : array-like of shape (n_samples,)
            correct target values
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals
        metrics : a string or a list of strings, default=list of all 
            metrics; for a learner wrapped with a conformal regressor
            these are "error", "eff_mean","eff_med", "time_fit", and
            "time_evaluate", while if wrapped with a conformal predictive
            system, the metrics also include "CRPS". 
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics     

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects, ``y_test`` is a 
        vector with true targets, and ``w`` is a calibrated 
        :class:`.WrapRegressor` object, then the latter can be evaluated at 
        the 90% confidence level with respect to error, mean and median 
        efficiency (interval size) by:

        .. code-block:: python

           results = w.evaluate(X_test, y_test, confidence=0.9,
                                metrics=["error", "eff_mean", "eff_med"])

        Note
        ----
        If included in the list of metrics, "CRPS" (continuous-ranked
        probability score) will be ignored if the :class:`.WrapRegressor` object
        has been calibrated with the (default) option ``cps=False``, i.e., the 
        learner is wrapped with a :class:`.ConformalRegressor`.

        Note
        ----
        The use of the metric ``CRPS`` may consume a lot of memory, as a matrix
        is generated for which the number of elements is the product of the 
        number of calibration and test objects, unless a Mondrian approach is 
        employed; for the latter, this number is reduced by increasing the number 
        of categories.

        Note
        ----
        The reported result for ``time_fit`` only considers fitting the
        conformal regressor or predictive system; not for fitting the
        learner.
        """
        if isinstance(y, pd.Series):
            y = y.values
        if not self.calibrated:
            raise RuntimeError(("evaluate requires that calibrate has been"
                                "called first"))
        else:
            y_hat = self.learner.predict(X)
            if self.de is None:
                sigmas = None
            else:
                sigmas = self.de.apply(X)
            if self.mc is None:
                bins = None
            elif isinstance(self.mc, MondrianCategorizer):
                bins = self.mc.apply(X)
            else:
                bins = self.mc(X)
            if self.cr is not None:
                return self.cr.evaluate(y_hat, y, sigmas=sigmas,
                                        bins=bins, confidence=confidence,
                                        y_min=y_min, y_max=y_max)
            else:
                return self.cps.evaluate(y_hat, y, sigmas=sigmas,
                                         bins=bins, confidence=confidence,
                                         y_min=y_min, y_max=y_max)

class WrapClassifier():
    """
    A learner wrapped with a :class:`.ConformalClassifier`.
    """
    
    def __init__(self, learner):
        self.cc = None
        self.nc = None
        self.calibrated = False
        self.learner = learner

    def __repr__(self):
        if self.calibrated:
            return (f"WrapClassifier(learner={self.learner}, "
                    f"calibrated={self.calibrated}, "
                    f"predictor={self.cc})")
        else:
            return f"WrapClassifier(learner={self.learner}, calibrated={self.calibrated})"
        
    def fit(self, X, y, **kwargs):
        """
        Fit learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,),
            target values
        kwargs : optional arguments
           any additional arguments are forwarded to the
           ``fit`` method of the ``learner`` object

        Returns
        -------
        None

        Examples
        --------
        Assuming ``X_train`` and ``y_train`` to be an array and vector 
        with training objects and labels, respectively, a random
        forest may be wrapped and fitted by:

        .. code-block:: python

           from sklearn.ensemble import RandomForestClassifier
           from crepes import WrapClassifier

           rf = Wrap(RandomForestClassifier())
           rf.fit(X_train, y_train)
           
        Note
        ----
        The learner, which can be accessed by ``rf.learner``, may be fitted 
        before as well as after being wrapped.

        Note
        ----
        All arguments, including any additional keyword arguments, to 
        :meth:`.fit` are forwarded to the ``fit`` method of the learner.        
        """
        if isinstance(y, pd.Series):
            y = y.values
        self.learner.fit(X, y, **kwargs)
    
        
    def predict(self, X):
        """
        Predict with learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
            set of objects

        Returns
        -------
        y : array-like of shape (n_samples,),
            values predicted using the ``predict`` 
            method of the ``learner`` object.

        Examples
        --------
        Assuming ``w`` is a :class:`.WrapClassifier` object for which the 
        wrapped learner ``w.learner`` has been fitted, (point) 
        predictions of the learner can be obtained for a set
        of test objects ``X_test`` by:

        .. code-block:: python

           y_hat = w.predict(X_test)
           
        The above is equivalent to:

        .. code-block:: python

           y_hat = w.learner.predict(X_test)
        """
        return self.learner.predict(X)

    def predict_proba(self, X):
        """
        Predict with learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects

        Returns
        -------
        y : array-like of shape (n_samples, n_classes),
            predicted probabilities using the ``predict_proba`` 
            method of the ``learner`` object.

        Examples
        --------
        Assuming ``w`` is a :class:`.WrapClassifier` object for which the 
        wrapped learner ``w.learner`` has been fitted, predicted
        probabilities of the learner can be obtained for a set
        of test objects ``X_test`` by:

        .. code-block:: python

           probabilities = w.predict_proba(X_test)
           
        The above is equivalent to:

        .. code-block:: python

           probabilities = w.learner.predict_proba(X_test)
        """
        return self.learner.predict_proba(X)
    
    def calibrate(self, X, y, oob=False, class_cond=False, nc=hinge, mc=None):
        """
        Fit a :class:`.ConformalClassifier` using the wrapped learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,),
            target values
        oob : bool, default=False
           use out-of-bag estimation
        class_cond : bool, default=False
            if class_cond=True, the method fits a Mondrian
            :class:`.ConformalClassifier` using the class
            labels as categories
        nc : function, default = :func:`crepes.extras.hinge`
            function to compute non-conformity scores
        mc: function or :class:`crepes.extras.MondrianCategorizer`, default=None
            function or :class:`crepes.extras.MondrianCategorizer` for computing Mondrian 
            categories

        Returns
        -------
        self : object
            Wrap object updated with a fitted :class:`.ConformalClassifier`

        Examples
        --------
        Assuming ``X_cal`` and ``y_cal`` to be an array and vector, 
        respectively, with objects and labels for the calibration set,
        and ``w`` is a :class:`.WrapClassifier` object for which the learner 
        has been fitted, a standard conformal classifier can be formed by:

        .. code-block:: python

           w.calibrate(X_cal, y_cal) 

        Assuming that ``get_categories`` is a function that returns a vector of
        Mondrian categories (bin labels), a Mondrian conformal classifier can
        be generated by:

        .. code-block:: python

           w.calibrate(X_cal, y_cal, mc=get_categories)

        By providing the option ``oob=True``, the conformal classifier
        will be calibrating using out-of-bag predictions, allowing
        the full set of training objects (``X_train``) and labels (``y_train``)
        to be used, e.g.,

        .. code-block:: python

           w.calibrate(X_train, y_train, oob=True)

        By providing the option ``class_cond=True``, a Mondrian conformal classifier
        will be formed using the class labels as categories, e.g.,

        .. code-block:: python

           w.calibrate(X_cal, y_cal, class_cond=True)

        Note
        ----
        Any Mondrian categorizer specified by the ``mc`` argument will be 
        ignored by :meth:`.calibrate`, if ``class_cond=True``, as the latter 
        implies that Mondrian categories are formed using the labels in ``y``. 

        Note
        ----
        Enabling out-of-bag calibration, i.e., setting ``oob=True``, requires 
        that the wrapped learner has an attribute ``oob_decision_function_``, 
        which e.g., is the case for a ``sklearn.ensemble.RandomForestClassifier``, 
        if enabled when created, e.g., ``RandomForestClassifier(oob_score=True)``

        Note
        ----
        The use of out-of-bag calibration, as enabled by ``oob=True``, does not 
        come with the theoretical validity guarantees of the regular (inductive) 
        conformal classifiers, due to that calibration and test instances are not
        handled in exactly the same way.
        """
        if isinstance(y, pd.Series):
            y = y.values
        self.cc = ConformalClassifier()
        self.nc = nc
        self.mc = mc
        self.class_cond = class_cond
        if oob:
            alphas = nc(self.learner.oob_decision_function_, self.learner.classes_, y)
        else:
            alphas = nc(self.learner.predict_proba(X), self.learner.classes_, y)
        if class_cond:
            self.cc.fit(alphas, bins=y)
        else:
            if isinstance(mc, MondrianCategorizer):
                bins = mc.apply(X)
                self.cc.fit(alphas, bins=bins)
            elif mc is not None:
                bins = mc(X)
                self.cc.fit(alphas, bins=bins)
            else:
                self.cc.fit(alphas)
        self.calibrated = True
        return self

    def predict_p(self, X):
        """
        Obtain (smoothed) p-values using conformal classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects

        Returns
        -------
        p-values : ndarray of shape (n_samples, n_classes)
            p-values

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects and ``w`` is a 
        :class:`.WrapClassifier` object that has been calibrated, i.e., 
        :meth:`.calibrate` has been applied, the p-values for the test 
        objects are obtained by:

        .. code-block:: python

           p_values = w.predict_p(X_test)
        """
        tic = time.time()
        alphas = self.nc(self.learner.predict_proba(X))
        if self.class_cond:
            p_values = np.array([
                self.cc.predict_p(alphas,
                                  np.full(len(X),
                                          self.learner.classes_[c]))[:, c]
                for c in range(len(self.learner.classes_))]).T
        else:
            if isinstance(self.mc, MondrianCategorizer):
                bins = self.mc.apply(X)
                p_values = self.cc.predict_p(alphas, bins)
            elif self.mc is not None:
                bins = self.mc(X)
                p_values = self.cc.predict_p(alphas, bins)
            else:
                p_values = self.cc.predict_p(alphas)
        toc = time.time()
        self.time_predict = toc-tic            
        return p_values

    def predict_set(self, X, confidence=0.95, smoothing=False):
        """
        Obtain prediction sets using conformal classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        confidence : float in range (0,1), default=0.95
            confidence level
        smoothing : bool, default=False
           use smoothed p-values

        Returns
        -------
        prediction sets : ndarray of shape (n_values, n_classes)
            prediction sets, where the value 1 (0) indicates
            that the class label is included (excluded), i.e.,
            the corresponding p-value is less than 1-confidence

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects and ``w`` is a 
        :class:`.WrapClassifier` object that has been calibrated, i.e., 
        :meth:`.calibrate` has been applied, the prediction sets for the 
        test objects at the 99% confidence level are obtained by:

        .. code-block:: python

           prediction_sets = w.predict_set(X_test, confidence=0.99)

        Note
        ----
        Using smoothed p-values substantially increases computation time and
        hardly has any effect on the predictions sets, except for when having 
        small calibration sets.        
        """
        tic = time.time()
        alphas = self.nc(self.learner.predict_proba(X))
        if self.class_cond:
            prediction_set = np.array([
                self.cc.predict_set(alphas,
                                    np.full(len(X),
                                            self.learner.classes_[c]),
                                    confidence, smoothing)[:, c]
                for c in range(len(self.learner.classes_))]).T
        else:
            if isinstance(self.mc, MondrianCategorizer):
                bins = self.mc.apply(X)
            elif self.mc is not None:
                bins = self.mc(X)
            else:
                bins = None
            prediction_set = self.cc.predict_set(alphas, bins, confidence,
                                                 smoothing)
        toc = time.time()
        self.time_predict = toc-tic            
        return prediction_set

    def evaluate(self, X, y, confidence=0.95, smoothing=False,
                 metrics=None):
        """
        Evaluate :class:`.ConformalClassifier`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           set of objects
        y : array-like of shape (n_samples,)
            correct target values
        confidence : float in range (0,1), default=0.95
            confidence level
        smoothing : bool, default=False
           use smoothed p-values
        metrics : a string or a list of strings, 
                  default=list of all metrics, i.e., ["error", "avg_c", "one_c",
                  "empty", "time_fit", "time_evaluate"]
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics, where "error" is the 
            fraction of prediction sets not containing the true class label,
            "avg_c" is the average no. of predicted class labels, "one_c" is
            the fraction of singleton prediction sets, "empty" is the fraction
            of empty prediction sets, "time_fit" is the time taken to fit the 
            conformal classifier, and "time_evaluate" is the time taken for the
            evaluation 
     

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects, ``y_test`` is a 
        vector with true targets, and ``w`` is a calibrated 
        :class:`.WrapClassifier` object, then the latter can be evaluated at 
        the 90% confidence level with respect to error, average prediction set
        size and fraction of singleton predictions by:

        .. code-block:: python

           results = w.evaluate(X_test, y_test, confidence=0.9,
                                metrics=["error", "avg_c", "one_c"])

        Note
        ----
        The reported result for ``time_fit`` only considers fitting the
        conformal regressor or predictive system; not for fitting the
        learner.

        Note
        ----
        Using smoothed p-values substantially increases computation time and
        hardly has any effect on the results, except for when having small 
        calibration sets.
        """
        if isinstance(y, pd.Series):
            y = y.values
        if not self.calibrated:
            raise RuntimeError(("evaluate requires that calibrate has been"
                                "called first"))
        else:
            if metrics is None:
                metrics = ["error", "avg_c", "one_c", "empty", "time_fit",
                           "time_evaluate"]
            tic = time.time()
            prediction_sets = self.predict_set(X, confidence, smoothing)
            test_results = get_test_results(prediction_sets,
                                            self.learner.classes_, y, metrics)
            toc = time.time()
            self.time_evaluate = toc-tic
            if "time_fit" in metrics:
                test_results["time_fit"] = self.cc.time_fit
            if "time_evaluate" in metrics:
                test_results["time_evaluate"] = self.time_evaluate
            return test_results
