"""Conformal classifiers, regressors, and predictive systems (crepes)

Classes implementing conformal classifiers, regressors, and predictive
systems, on top of any standard classifier and regressor, transforming
the original predictions into well-calibrated p-values and cumulative
distribution functions, or prediction sets and intervals with coverage
guarantees.

Author: Henrik Boström (bostromh@kth.se)

Copyright 2025 Henrik Boström

License: BSD 3 clause

"""

__version__ = "0.8.0"

import numpy as np
import pandas as pd
from bisect import insort
from scipy.stats import kstest
import time
import warnings

from crepes.extras import (
    hinge,
    MondrianCategorizer
    )
                          
warnings.simplefilter("always", UserWarning)

class ConformalPredictor():
    """
    The class contains three sub-classes: :class:`.ConformalClassifier`,
    :class:`.ConformalRegressor`, and :class:`.ConformalPredictiveSystem`.
    """
    
    def __init__(self):
        self.fitted = False
        self.mondrian = None
        self.alphas = None
        self.bins = None
        self.normalized = None
        self.binned_alphas = None
        self.time_fit = None
        self.time_predict = None
        self.time_evaluate = None
        self.seed = None

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
    
    def fit(self, alphas, bins=None, seed=None):
        """
        Fit conformal classifier.

        Parameters
        ----------
        alphas : array-like of shape (n_samples,)
            non-conformity scores
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        seed : int, default=None
           set random seed

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

        Note
        ----
        By providing a random seed, e.g., ``seed=123``, calls to the methods
        ``predict_p``, ``predict_set`` and ``evaluate`` of the
        :class:`.ConformalClassifier` object will be deterministic.
        """
        tic = time.time()
        self.alphas = alphas
        if bins is None:
            self.bins = None
            self.mondrian = False
        else: 
            self.bins = bins
            self.mondrian = True            
        self.seed = seed
        self.fitted = True
        self.fitted_ = True
        toc = time.time()
        self.time_fit = toc-tic
        return self

    def predict_p(self, alphas, bins=None, all_classes=True, classes=None,
                  y=None, smoothing=True, seed=None):
        """
        Obtain (smoothed or non-smoothed) p-values from conformal classifier.

        Parameters
        ----------
        alphas : array-like of shape (n_samples, n_classes)
            non-conformity scores
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        all_classes : bool, default=True
            return p-values for all classes
        classes : array-like of shape (n_classes,), default=None
            class names, used only if all_classes=False
        y : array-like of shape (n_samples,), default=None
            correct class labels, used only if all_classes=False
        smoothing : bool, default=True
           return smoothed p-values
        seed : int, default=None
           set random seed

        Returns
        -------
        p-values : ndarray of shape (n_samples, n_classes)
            p-values

        Examples
        --------
        Assuming that ``alphas_test`` is a vector with non-conformity scores
        for a test set and ``cc_std`` a fitted standard conformal classifier, 
        then p-values for the test set is obtained by:

        .. code-block:: python

           p_values = cc_std.predict_p(alphas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin
        labels) for the test set and ``cc_mond`` a fitted Mondrian conformal 
        classifier, then the following provides (smoothed) p-values for the
        test set:

        .. code-block:: python

           p_values = cc_mond.predict_p(alphas_test, bins=bins_test)

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given when calling ``fit``.
        """
        tic = time.time()
        if type(alphas) == list:
            alphas = np.array(alphas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        p_values = p_values_batch(self.alphas, alphas, self.bins, bins,
                                  smoothing, seed)
        if not all_classes:
            class_indexes = np.array(
                [np.argwhere(classes == y[i])[0][0] for i in range(len(y))])
            p_values = p_values[np.arange(len(y)), class_indexes]
        toc = time.time()
        self.time_predict = toc-tic            
        return p_values
    
    def predict_p_online(self, alphas, classes, y, bins=None, all_classes=True,
                         smoothing=True, seed=None, warm_start=True):
        """
        Obtain (smoothed or non-smoothed) p-values from conformal classifier,
        computed using online calibration.

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
        all_classes : bool, default=True
            return p-values for all classes
        smoothing : bool, default=True
           return smoothed p-values
        seed : int, default=None
           set random seed
        warm_start : bool, default=True
           extend original calibration set

        Returns
        -------
        p-values : ndarray of shape (n_samples, n_classes)
            p-values

        Examples
        --------
        Assuming that ``alphas_test`` is a vector with non-conformity scores
        for a test set, ``classes`` is a vector with class names, ``y_test``
        is a vector with the correct class labels for the test set, and
        ``cc_std`` a fitted standard conformal classifier, 
        then p-values for the test set is obtained by:

        .. code-block:: python

           p_values = cc_std.predict_p_online(alphas_test, classes, y_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cc_mond`` a fitted Mondrian conformal 
        classifier, then the following provides (smoothed) p-values for the
        test set:

        .. code-block:: python

           p_values = cc_mond.predict_p_online(alphas_test, classes, y_test,
                                               bins=bins_test)

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.
        """
        tic = time.time()
        if type(alphas) == list:
            alphas = np.array(alphas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        if warm_start:
            alphas_cal = self.alphas
            bins_cal = self.bins
        else:
            alphas_cal = None
            bins_cal = None
        p_values = p_values_online_classification(alphas, classes, y, bins,
                                                  alphas_cal, bins_cal,
                                                  all_classes, smoothing, seed)
        toc = time.time()
        self.time_predict = toc-tic            
        return p_values

    def predict_set(self, alphas, bins=None, confidence=0.95, smoothing=True,
                    seed=None):
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
        smoothing : bool, default=True
           use smoothed p-values
        seed : int, default=None
           set random seed

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
        classifier, then the following provides prediction sets for the test
        set, at the 90% confidence level:

        .. code-block:: python

           p_values = cc_mond.predict_set(alphas_test, 
                                          bins=bins_test,
                                          confidence=0.9)

        Note
        ----
        The use of smoothed p-values increases computation time and typically
        has a minor effect on the predictions sets, except for small
        calibration sets.

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.
        """
        tic = time.time()
        if type(alphas) == list:
            alphas = np.array(alphas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        p_values = p_values_batch(self.alphas, alphas, self.bins, bins,
                                  smoothing, seed)
        prediction_sets = (p_values >= 1-confidence).astype(int)
        toc = time.time()
        self.time_predict = toc-tic            
        return prediction_sets

    def predict_set_online(self, alphas, classes, y, bins=None, confidence=0.95,
                           smoothing=True, seed=None, warm_start=True):
        """
        Obtain prediction sets using conformal classifier,
        computed using online calibration.

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
        smoothing : bool, default=True
           use smoothed p-values
        seed : int, default=None
           set random seed
        warm_start : bool, default=True
           extend original calibration set

        Returns
        -------
        prediction sets : ndarray of shape (n_samples, n_classes)
            prediction sets

        Examples
        --------
        Assuming that ``alphas_test`` is a vector with non-conformity scores
        for a test set, ``classes`` is a vector with class names, ``y`` is
        a vector with the correct class labels for the test set, and ``cc_std``
        a fitted standard conformal classifier, then prediction sets at the
        default (95%) confidence level are obtained using online calibration by:

        .. code-block:: python

           prediction_sets = cc_std.predict_set_online(alphas_test, classes,
                                                       y_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cc_mond`` a fitted Mondrian conformal 
        classifier, then the following provides prediction sets for the test
        set, at the 90% confidence level:

        .. code-block:: python

           p_values = cc_mond.predict_set_online(alphas_test, classes, y_test, 
                                                 bins=bins_test, confidence=0.9)

        Note
        ----
        The use of smoothed p-values increases computation time and typically
        has a minor effect on the predictions sets, except for small
        calibration sets.

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.
        """
        tic = time.time()
        if type(alphas) == list:
            alphas = np.array(alphas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        if warm_start:
            alphas_cal = self.alphas
            bins_cal = self.bins
        else:
            alphas_cal = None
            bins_cal = None
        p_values = p_values_online_classification(alphas, classes, y, bins,
                                                  alphas_cal, bins_cal,
                                                  True, smoothing, seed)
        prediction_sets = (p_values >= 1-confidence).astype(int)
        toc = time.time()
        self.time_predict = toc-tic            
        return prediction_sets
    
    def evaluate(self, alphas, classes, y, bins=None, confidence=0.95,
                 smoothing=True, metrics=None, seed=None, online=False,
                 warm_start=True):
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
        smoothing : bool, default=True
           use smoothed p-values
        metrics : a string or a list of strings, 
                  default = list of all metrics, i.e., ["error", "avg_c", 
                  "one_c", "empty", "ks_test", "time_fit", "time_evaluate"]
        seed : int, default=None
           set random seed
        online : bool, default=False
           compute p-values using online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics, where "error" is the 
            fraction of prediction sets not containing the true class label,
            "avg_c" is the average no. of predicted class labels, "one_c" is
            the fraction of singleton prediction sets, "empty" is the fraction
            of empty prediction sets, "ks_test" is the p-value for the
            Kolmogorov-Smirnov test of uniformity of predicted p-values,
            "time_fit" is the time taken to fit the conformal classifier,
            and "time_evaluate" is the time taken for the evaluation 

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
        The use of smoothed p-values increases computation time and typically
        has a minor effect on the results, except for small calibration sets.

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given when calling ``fit``.        
        """
        if metrics is None:
            metrics = ["error", "avg_c", "one_c", "empty", "ks_test",
                       "time_fit", "time_evaluate"]
        tic = time.time()
        if type(alphas) == list:
            alphas = np.array(alphas)
        if type(bins) == list:
            bins = np.array(bins)
        if type(classes) == list:
            classes = np.array(classes)            
        if seed is None:
            seed = self.seed
        if not online:
            p_values = self.predict_p(alphas, bins, True, classes, y,
                                      smoothing, seed)
            prediction_sets = (p_values >= 1-confidence).astype(int)        
        else:
            p_values = self.predict_p_online(alphas, classes, y, bins, True,
                                             smoothing, seed, warm_start)
            prediction_sets = (p_values >= 1-confidence).astype(int)
        test_results = get_classification_results(prediction_sets, p_values,
                                                  classes, y, metrics)
        toc = time.time()
        self.time_evaluate = toc-tic
        if "time_fit" in metrics:
            test_results["time_fit"] = self.time_fit
        if "time_evaluate" in metrics:
            test_results["time_evaluate"] = self.time_evaluate
        return test_results
    
def get_classification_results(prediction_sets, p_values, classes, y, metrics):
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
    if "ks_test" in metrics:            
        test_results["ks_test"] = kstest(p_values[np.arange(len(y)),
                                                  class_indexes],
                                         "uniform").pvalue
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
        if type(residuals) == list:
            residuals = np.array(residuals)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        abs_residuals = np.abs(residuals)
        if bins is None:
            self.mondrian = False
            self.bins = None
            if sigmas is None:
                self.normalized = False
                self.alphas = np.sort(abs_residuals)[::-1]
            else:
                self.normalized = True
                self.alphas = np.sort(abs_residuals/sigmas)[::-1]
        else: 
            self.mondrian = True
            self.bins = bins
            if sigmas is None:
                self.alphas = abs_residuals
            else:
                self.alphas = abs_residuals/sigmas
            bin_values = np.unique(bins)
            if sigmas is None:            
                self.normalized = False
                self.binned_alphas = (bin_values,[np.sort(
                    abs_residuals[bins==b])[::-1] for b in bin_values])
            else:
                self.normalized = True
                self.binned_alphas = (bin_values, [np.sort(
                    abs_residuals[bins==b]/sigmas[bins==b])[::-1]
                                           for b in bin_values])                
        self.fitted = True
        self.fitted_ = True
        toc = time.time()
        self.time_fit = toc-tic
        return self

    def predict_p(self, y_hat, y, sigmas=None, bins=None, smoothing=True,
                  seed=None):
        """
        Obtain (smoothed or non-smoothed) p-values from conformal regressor.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            labels
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        smoothing : bool, default=True
           return smoothed p-values
        seed : int, default=None
           set random seed

        Returns
        -------
        p-values : ndarray of shape (n_samples, n_classes)
            p-values

        Examples
        --------
        Assuming that ``y_hat`` and ``y_test`` are vectors with predicted and correct
        labels for a test set and ``cr_std`` a fitted standard conformal regressor,
        then p-values are obtained by:

        .. code-block:: python

           p_values = cr_std.predict_p(y_hat, y_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cr_mond`` a fitted Mondrian conformal 
        regressor, then the following provides (smoothed) p-values:

        .. code-block:: python

           p_values = cr_mond.predict_p(y_hat, y, bins=bins_test)

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.
        """
        if not self.fitted:
            raise RuntimeError(("Batch predictions requires a fitted "
                                "conformal regressor"))
        tic = time.time()
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(y) == list:
            y = np.array(y)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        if sigmas is None:
            alphas = np.abs(y - y_hat)
        else:
            alphas = np.abs(y - y_hat)/sigmas
        p_values = p_values_batch(self.alphas, alphas, self.bins, bins,
                                  smoothing, seed)
        toc = time.time()
        self.time_predict = toc-tic            
        return p_values

    def predict_p_online(self, y_hat, y, t=None, sigmas=None, bins=None,
                         smoothing=True, seed=None, warm_start=True):
        """
        Obtain (smoothed or non-smoothed) p-values from conformal regressor,
        computed using online calibration.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct labels, used as targets if t=None
        t : int, float or array-like of shape (n_samples,), default=None
            targets
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        smoothing : bool, default=True
           return smoothed p-values
        seed : int, default=None
           set random seed
        warm_start : bool, default=True
           extend original calibration set
        
        Returns
        -------
        p-values : ndarray of shape (n_samples, n_classes)
            p-values

        Examples
        --------
        Assuming that ``y_hat`` and ``y_test`` are vectors with predicted and correct
        labels for a test set and ``cr_std`` a fitted standard conformal regressor,
        then p-values for the correct labels are obtained by online calibration by:

        .. code-block:: python

           p_values = cr_std.predict_p_online(y_hat, y_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cr_mond`` a fitted Mondrian conformal 
        regressor, then the following provides (smoothed) p-values:

        .. code-block:: python

           p_values = cr_mond.predict_p_online(y_hat, y, bins=bins_test)

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.
        """
        tic = time.time()
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(y) == list:
            y = np.array(y)
        if type(t) == list:
            t = np.array(t)
        elif isinstance(t, (int, float, np.integer, np.floating)):
            t = np.full(len(y), t)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        if sigmas is None:
            alphas = np.abs(y - y_hat)
        else:
            alphas = np.abs(y - y_hat)/sigmas
        if t is None:
            alphas_target = None
        elif sigmas is None:
            alphas_target = np.abs(t - y_hat)
        else:
            alphas_target = np.abs(t - y_hat)/sigmas
        p_values = p_values_online_regression(alphas, alphas_target, bins,
                                              self.alphas, self.bins,
                                              smoothing, seed)
        toc = time.time()
        self.time_predict = toc-tic            
        return p_values

    def predict_int(self, y_hat, sigmas=None, bins=None, confidence=0.95,
                    y_min=-np.inf, y_max=np.inf):
        """
        Obtain prediction intervals from conformal regressor.

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

           intervals = cr_std.predict_int(y_hat_test, confidence=0.99)

        Assuming that ``sigmas_test`` is a vector with difficulty estimates for
        the test set and ``cr_norm`` a fitted normalized conformal regressor, 
        then prediction intervals at the default (95%) confidence level can be
        obtained by:

        .. code-block:: python

           intervals = cr_norm.predict_int(y_hat_test, sigmas=sigmas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cr_mond`` a fitted Mondrian conformal 
        regressor, then the following provides prediction intervals at the 
        default confidence level, where the intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = cr_mond.predict_int(y_hat_test, bins=bins_test, 
                                           y_min=0)

        Note
        ----
        In case the specified confidence level is too high in relation to the 
        size of the calibration set, a warning will be issued and the output
        intervals will be of maximum size.
        """
        if not self.fitted:
            raise RuntimeError(("Batch predictions requires a fitted "
                                "conformal regressor"))
        tic = time.time()
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        intervals = np.zeros((len(y_hat),2))
        if not self.mondrian:
            alpha_index = int((1-confidence)*(len(self.alphas)+1))-1
            if alpha_index >= 0:
                alpha = self.alphas[alpha_index]
                if self.normalized:
                    intervals[:,0] = y_hat - alpha*sigmas
                    intervals[:,1] = y_hat + alpha*sigmas
                else:
                    intervals[:,0] = y_hat - alpha
                    intervals[:,1] = y_hat + alpha
            else:
                intervals[:,0] = -np.inf 
                intervals[:,1] = np.inf
                warnings.warn("the no. of calibration examples is too small" \
                              "for the chosen confidence level; the " \
                              "intervals will be of maximum size")
        else:           
            bin_values, bin_alphas = self.binned_alphas
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

    def predict_int_online(self, y_hat, y, sigmas=None, bins=None,
                           confidence=0.95, y_min=-np.inf, y_max=np.inf,
                           warm_start=True):
        """
        Obtain prediction intervals from conformal regressor, where the
        intervals are formed using online calibration.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct labels
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
        warm_start : bool, default=True
           extend original calibration set

        Returns
        -------
        intervals : ndarray of shape (n_values, 2)
            prediction intervals

        Examples
        --------
        Assuming that ``y_hat_test`` is a vector with predicted targets and
        ``y_test`` is a vector with correct targets for a test set and
        ``cr_std`` is a fitted standard conformal regressor, then 
        prediction intervals at the 99% confidence level can be obtained using
        online calibration by:

        .. code-block:: python

           intervals = cr_std.predict_int_online(y_hat_test, y_test,
                                                 confidence=0.99)

        Assuming that ``sigmas_test`` is a vector with difficulty estimates for
        the test set and ``cr_norm`` a fitted normalized conformal regressor, 
        then prediction intervals at the default (95%) confidence level can be
        obtained by:

        .. code-block:: python

           intervals = cr_norm.predict_int_online(y_hat_test, y_test,
                                                  sigmas=sigmas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cr_mond`` a fitted Mondrian conformal 
        regressor, then the following provides prediction intervals at the 
        default confidence level, where the intervals are lower-bounded by 0:

        .. code-block:: python

           intervals = cr_mond.predict_int_online(y_hat_test, y_test,
                                                  bins=bins_test, y_min=0)

        Note
        ----
        In case the specified confidence level is too high in relation to the 
        size of the calibration set, the output intervals will be of maximum
        size.
        """
        tic = time.time()
        intervals = np.zeros((len(y_hat),2))
        if bins is None:
            if warm_start and self.alphas is not None:
                alphas_cal = list(self.alphas)
            else:
                alphas_cal = []
            if sigmas is not None:
                for i in range(len(y_hat)):
                    alpha_index = int((1-confidence)*(len(alphas_cal)+1))-1
                    if alpha_index >= 0:
                        alpha = alphas_cal[alpha_index]
                        intervals[i,0] = y_hat[i]-alpha*sigmas[i]
                        intervals[i,1] = y_hat[i]+alpha*sigmas[i]
                    else:
                        intervals[i,0] = -np.inf 
                        intervals[i,1] = np.inf
                    insort(alphas_cal, np.abs(y[i]-y_hat[i])/sigmas[i],
                           key=lambda x: -x)
            else:
                for i in range(len(y_hat)):
                    alpha_index = int((1-confidence)*(len(alphas_cal)+1))-1
                    if alpha_index >= 0:
                        alpha = alphas_cal[alpha_index]
                        intervals[i,0] = y_hat[i]-alpha
                        intervals[i,1] = y_hat[i]+alpha
                    else:
                        intervals[i,0] = -np.inf 
                        intervals[i,1] = np.inf
                    insort(alphas_cal, np.abs(y[i]-y_hat[i]),
                           key=lambda x: -x)
        else:
            if warm_start and self.binned_alphas is not None:
                bin_values_cal, bin_alphas_cal = self.binned_alphas
                all_alphas_cal = {bin_values_cal[i] : list(bin_alphas_cal[i])
                                  for i in range(len(bin_values_cal))}
            else:
                all_alphas_cal = {}
            bin_values, bin_indexes = np.unique(bins, return_inverse=True)
            for b in range(len(bin_values)):
                alphas_cal = all_alphas_cal.get(bin_values[b], [])
                orig_indexes = np.arange(len(bins))[bin_indexes == b]
                if sigmas is not None:
                    for i in orig_indexes:
                        alpha_index = int((1-confidence)*(len(alphas_cal)+1))-1
                        if alpha_index >= 0:
                            alpha = alphas_cal[alpha_index]
                            intervals[i,0] = y_hat[i] - alpha*sigmas[i]
                            intervals[i,1] = y_hat[i] + alpha*sigmas[i]
                        else:
                            intervals[i,0] = -np.inf 
                            intervals[i,1] = np.inf
                        insort(alphas_cal, np.abs(y[i]-y_hat[i])/sigmas[i],
                               key=lambda x: -x)
                else:
                    for i in orig_indexes:
                        alpha_index = int((1-confidence)*(len(alphas_cal)+1))-1
                        if alpha_index >= 0:
                            alpha = alphas_cal[alpha_index]
                            intervals[i,0] = y_hat[i] - alpha
                            intervals[i,1] = y_hat[i] + alpha
                        else:
                            intervals[i,0] = -np.inf 
                            intervals[i,1] = np.inf
                        insort(alphas_cal, np.abs(y[i]-y_hat[i]),
                               key=lambda x: -x)
        if y_min > -np.inf:
            intervals[intervals<y_min] = y_min
        if y_max < np.inf:
            intervals[intervals>y_max] = y_max 
        toc = time.time()
        self.time_predict = toc-tic            
        return intervals
        
    def evaluate(self, y_hat, y, sigmas=None, bins=None, confidence=0.95,
                 y_min=-np.inf, y_max=np.inf, metrics=None, smoothing=True,
                 seed=None, online=False, warm_start=True):
        """
        Evaluate conformal regressor.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct labels
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
                  ["error", "eff_mean", "eff_med", "ks_test",
                   "time_fit", "time_evaluate"]
        smoothing : bool, default=True
           employ smoothed p-values
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics, where "error" is the 
            fraction of prediction intervals not containing the true label,
            "eff_mean" is the mean length of prediction intervals,
            "eff_med" is the median length of the prediction intervals, 
            "ks_test" is the p-value for the Kolmogorov-Smirnov test of
            uniformity of predicted p-values, "time_fit" is the time taken
            to fit the conformal regressor, and "time_evaluate" is the time
            taken for the evaluation         
        
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
        if not self.fitted and not online:
            raise RuntimeError(("Batch evaluation requires a fitted "
                                "conformal regressor"))
        tic = time.time()
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(y) == list:
            y = np.array(y)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        if not online and not self.normalized:
            sigmas = None
        if not online and not self.mondrian:
            bins = None
        if metrics is None:
            metrics = ["error", "eff_mean", "eff_med", "ks_test",
                       "time_fit", "time_evaluate"]
        test_results = {}
        if not online:
            intervals = self.predict_int(y_hat, sigmas, bins, confidence,
                                         y_min, y_max)
        else:
            intervals = self.predict_int_online(y_hat, y, sigmas, bins,
                                                confidence, y_min, y_max,
                                                warm_start)
        if "error" in metrics:
            test_results["error"] = 1-np.mean(
                np.logical_and(intervals[:,0]<=y, y<=intervals[:,1]))
        if "eff_mean" in metrics:            
            test_results["eff_mean"] = np.mean(intervals[:,1] - intervals[:,0])
        if "eff_med" in metrics:            
            test_results["eff_med"] = np.median(intervals[:,1] - intervals[:,0])
        if "ks_test" in metrics:
            if not online:
                p_values = self.predict_p(y_hat, y, sigmas, bins, smoothing,
                                          seed)
            else:
                p_values = self.predict_p_online(y_hat, y, None, sigmas, bins,
                                                 smoothing, seed, warm_start)
            test_results["ks_test"] = kstest(p_values, "uniform").pvalue
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

    def fit(self, residuals, sigmas=None, bins=None, seed=None):
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
        seed : int, default=None
           set random seed

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

        Note
        ----
        By providing a random seed, e.g., ``seed=123``, calls to the methods
        ``predict`` and ``evaluate`` of the :class:`.ConformalPredictiveSystem`
        object will be deterministic.        
        """
        tic = time.time()
        if type(residuals) == list:
            residuals = np.array(residuals)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
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
            self.bins = bins
            if sigmas is None:
                self.alphas = residuals
            else:
                self.alphas = residuals/sigmas
            bin_values = np.unique(bins)
            if sigmas is None:            
                self.normalized = False
                self.binned_alphas = (bin_values, [np.sort(
                    residuals[bins==b]) for b in bin_values])
            else:
                self.normalized = True
                self.binned_alphas = (bin_values, [np.sort(
                    residuals[bins==b]/sigmas[bins==b]) for b in bin_values])
        self.fitted = True
        self.fitted_ = True
        self.seed = seed
        toc = time.time()
        self.time_fit = toc-tic
        return self
    
    def predict_p(self, y_hat, y, sigmas=None, bins=None, smoothing=True,
                  seed=None):    
        """
        Obtain p-values from conformal predictive system.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : int, float or array-like of shape (n_values,)
            labels
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        smoothing : bool, default=True
           return smoothed p-values
        seed : int, default=None
           set random seed

        Returns
        -------
        p_values : ndarray of shape (n_values,)

        Examples
        --------
        Assuming that ``y_hat_test`` and ``y_test`` are vectors with predicted
        and true targets, respectively, for a test set and ``cps_std`` a fitted
        standard conformal predictive system, the p-values for the true targets 
        can be obtained by:

        .. code-block:: python

           p_values = cps_std.predict(y_hat_test, y=y_test)

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.

        Note
        ----
        If smoothing is disabled, i.e., ``smoothing=False``, then setting a
        value for ``seed`` has no effect.
        """
        p_values = self.predict(y_hat, sigmas, bins, y, smoothing=smoothing,
                                seed=seed)
        return p_values

    def predict_p_online(self, y_hat, y, t=None, sigmas=None, bins=None,
                         smoothing=True, seed=None, warm_start=True):
        """
        Obtain (smoothed or non-smoothed) p-values from conformal predictive
        system, computed using online calibration.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct labels, used as targets if t=None
        t : int, float or array-like of shape (n_samples,), default=None
            targets
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_samples,), default=None
            Mondrian categories
        smoothing : bool, default=True
           return smoothed p-values
        seed : int, default=None
           set random seed
        warm_start : bool, default=True
           extend original calibration set
        
        Returns
        -------
        p-values : ndarray of shape (n_samples,)
            p-values

        Examples
        --------
        Assuming that ``y_hat_test`` and ``y_test`` are vectors with predicted
        and true targets, respectively, for a test set and ``cps_std`` a fitted
        standard conformal predictive system, the p-values for the true targets, 
        computed using online calibration, can be obtained by:

        .. code-block:: python

           p_values = cps_std.predict_p_online(y_hat_test, y_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cps_mond`` a fitted Mondrian conformal 
        predictive system, then the following provides (smoothed) p-values for
        the test set:

        .. code-block:: python

           p_values = cps_mond.predict_p_online(y_hat_test, y_test,
                                                bins=bins_test)

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.

        Note
        ----
        If smoothing is disabled, i.e., ``smoothing=False``, then setting a
        value for ``seed`` has no effect.        
        """
        tic = time.time()
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(y) == list:
            y = np.array(y)
        if type(t) == list:
            t = np.array(t)
        elif isinstance(t, (int, float, np.integer, np.floating)):
            t = np.full(len(y), t)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        if sigmas is None:
            alphas = y - y_hat
        else:
            alphas = (y - y_hat)/sigmas
        if t is None:
            alphas_target = None
        elif sigmas is None:
            alphas_target = t - y_hat
        else:
            alphas_target = (t - y_hat)/sigmas
        p_values = p_values_online_regression(alphas, alphas_target, bins,
                                              self.alphas, self.bins,
                                              smoothing, seed)
        toc = time.time()
        self.time_predict = toc-tic            
        return p_values

    def predict_int(self, y_hat, sigmas=None, bins=None, confidence=0.95,
                y_min=-np.inf, y_max=np.inf):    
        """
        Obtain prediction intervals from conformal predictive system.

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
            The minimum value to include in prediction intervals.
        y_max : float or int, default=numpy.inf
            The maximum value to include in prediction intervals.

        Returns
        -------
        intervals : ndarray of shape (n_values, 2)

        Examples
        --------
        Assuming that ``y_hat_test`` and ``y_test`` are vectors with predicted
        and true targets, respectively, for a test set and ``cps_std`` a fitted
        standard conformal predictive system, the p-values for the true targets 
        can be obtained by:

        .. code-block:: python

           p_values = cps_std.predict_int(y_hat_test, y=y_test)

        Note
        ----
        In case the calibration set is too small for the specified confidence
        level, a warning will be issued and the output will be 
        ``y_min`` and ``y_max``, respectively.

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.        
        """
        lower_percentile = (1-confidence)/2*100
        higher_percentile = (confidence+(1-confidence)/2)*100
        intervals = self.predict(y_hat, sigmas, bins, 
                                 lower_percentiles=lower_percentile,
                                 higher_percentiles=higher_percentile,
                                 y_min=y_min, y_max=y_max)
        return intervals

    def predict_int_online(self, y_hat, y, sigmas=None, bins=None,
                           confidence=0.95, y_min=-np.inf, y_max=np.inf,
                           warm_start=True):
        """
        Obtain prediction intervals from conformal predictive system, where
        the intervals are formed using online calibration.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct labels
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
        warm_start : bool, default=True
           extend original calibration set

        Returns
        -------
        intervals : ndarray of shape (n_values, 2)
            prediction intervals

        Examples
        --------
        Assuming that ``y_hat_test`` is a vector with predicted targets and
        ``y_test`` is a vector with the correct targets for a test set and
        ``cps_std`` a fitted standard conformal predictive system, then
        prediction intervals at the 99% confidence level can be obtained using
        online calibration by:

        .. code-block:: python

           intervals = cps_std.predict_int_online(y_hat_test, y_test,
                                                  confidence=0.99)

        Assuming that ``sigmas_test`` is a vector with difficulty estimates for
        the test set and ``cps_norm`` a fitted normalized conformal predictive
        system, then prediction intervals at the default (95%) confidence level
        can be obtained using online calibration by:

        .. code-block:: python

           intervals = cps_norm.predict_int_online(y_hat_test, y_test,
                                                   sigmas=sigmas_test)

        Assuming that ``bins_test`` is a vector with Mondrian categories (bin 
        labels) for the test set and ``cps_mond`` a fitted Mondrian conformal 
        predictive system, then the following provides prediction intervals at
        the default confidence level, where the intervals are lower-bounded by
        0:

        .. code-block:: python

           intervals = cps_mond.predict_int_online(y_hat_test, y_test,
                                                   bins=bins_test, y_min=0)

        Note
        ----
        In case the specified confidence level is too high in relation to the 
        size of the calibration set, the output intervals will be of maximum
        size.
        """
        tic = time.time()
        intervals = np.zeros((len(y_hat),2))
        if bins is None:
            if warm_start and self.alphas is not None:
                alphas_cal = list(self.alphas)
            else:
                alphas_cal = []
            if sigmas is not None:
                for i in range(len(y_hat)):
                    index_low = int((1-confidence)/2*(len(alphas_cal)+1))-1
                    index_high = len(alphas_cal)-index_low-1
                    if index_low >= 0:
                        alpha_low = alphas_cal[index_low]
                        alpha_high = alphas_cal[index_high]
                        intervals[i,0] = y_hat[i] + alpha_low*sigmas[i]
                        intervals[i,1] = y_hat[i] + alpha_high*sigmas[i]
                    else:
                        intervals[i,0] = -np.inf 
                        intervals[i,1] = np.inf 
                    insort(alphas_cal, (y[i]-y_hat[i])/sigmas[i])
            else:
                for i in range(len(y_hat)):
                    index_low = int((1-confidence)/2*(len(alphas_cal)+1))-1
                    index_high = len(alphas_cal)-index_low-1
                    if index_low >= 0:
                        alpha_low = alphas_cal[index_low]
                        alpha_high = alphas_cal[index_high]
                        intervals[i,0] = y_hat[i] + alpha_low
                        intervals[i,1] = y_hat[i] + alpha_high
                    else:
                        intervals[i,0] = -np.inf 
                        intervals[i,1] = np.inf 
                    insort(alphas_cal, (y[i]-y_hat[i]))
        else:
            if warm_start and self.binned_alphas is not None:
                bin_values_cal, bin_alphas_cal = self.binned_alphas
                all_alphas_cal = {bin_values_cal[i] : list(bin_alphas_cal[i])
                                  for i in range(len(bin_values_cal))}
            else:
                all_alphas_cal = {}
            bin_values, bin_indexes = np.unique(bins, return_inverse=True)
            for b in range(len(bin_values)):
                alphas_cal = all_alphas_cal.get(bin_values[b], [])
                orig_indexes = np.arange(len(bins))[bin_indexes == b]
                if sigmas is not None:
                    for i in orig_indexes:
                        index_low = int((1-confidence)/2*(len(alphas_cal)+1))-1
                        index_high = len(alphas_cal)-index_low-1
                        if index_low >= 0:
                            alpha_low = alphas_cal[index_low]
                            alpha_high = alphas_cal[index_high]
                            intervals[i,0] = y_hat[i] + alpha_low*sigmas[i]
                            intervals[i,1] = y_hat[i] + alpha_high*sigmas[i]
                        else:
                            intervals[i,0] = -np.inf 
                            intervals[i,1] = np.inf 
                        insort(alphas_cal, (y[i]-y_hat[i])/sigmas[i])
                else:
                    for i in orig_indexes:
                        index_low = int((1-confidence)/2*(len(alphas_cal)+1))-1
                        index_high = len(alphas_cal)-index_low-1
                        if index_low >= 0:
                            alpha_low = alphas_cal[index_low]
                            alpha_high = alphas_cal[index_high]
                            intervals[i,0] = y_hat[i] + alpha_low
                            intervals[i,1] = y_hat[i] + alpha_high
                        else:
                            intervals[i,0] = -np.inf 
                            intervals[i,1] = np.inf 
                        insort(alphas_cal, (y[i]-y_hat[i]))
        if y_min > -np.inf:
            intervals[intervals<y_min] = y_min
        if y_max < np.inf:
            intervals[intervals>y_max] = y_max 
        toc = time.time()
        self.time_predict = toc-tic            
        return intervals
    
    def predict_percentiles(self, y_hat, sigmas=None, bins=None,
                            lower_percentiles=None, higher_percentiles=None,
                            y_min=-np.inf, y_max=np.inf):
        """
        Obtain percentiles with conformal predictive system.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        lower_percentiles : float, int, or array-like of shape (l_values,),
                            default=None
            percentiles for which a lower value will be output in case
            a percentile lies between two values (equivalent to
            `interpolation="lower"` in `numpy.percentile`)
        higher_percentiles : float, int, or array-like of shape (h_values,),
                             default=None
            percentiles for which a higher value will be output in case
            a percentile lies between two values (equivalent to
            `interpolation="higher"` in `numpy.percentile`)
        y_min : float or int, default=-numpy.inf
            The minimum value to include
        y_max : float or int, default=numpy.inf
            The maximum value to include

        Returns
        -------
        percentiles : ndarray of shape (n_values, l_values + h_values)
            percentiles

        Examples
        --------
        Assuming that ``y_hat_test`` is a vector with predicted targets
        for a test set and ``cps_std`` a fitted standard conformal
        predictive system, then percentiles can be obtained by:

        .. code-block:: python

           p_values = cps_std.predict_percentiles(y_hat_test,
                                                  lower_percentiles=2.5,
                                                  higher_percentiles=97.5)

        Note
        ----
        In case the calibration set is too small for the specified percentiles
        level, a warning will be issued and the output will be ``y_min`` and
        ``y_max``, respectively.
        """
        percentiles = self.predict(y_hat, sigmas, bins,
                                   lower_percentiles=lower_percentiles,
                                   higher_percentiles=higher_percentiles,
                                   y_min=y_min, y_max=y_max)
        return percentiles

    def predict_percentiles_online(self, y_hat, y, sigmas=None, bins=None,
                                   lower_percentiles=None,
                                   higher_percentiles=None,
                                   y_min=-np.inf, y_max=np.inf,
                                   warm_start=True):
        """
        Obtain percentiles from conformal predictive system, computed using
        online calibration.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct labels
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        lower_percentiles : float, int, or array-like of shape (l_values,),
                            default=None
            percentiles for which a lower value will be output in case
            a percentile lies between two values (equivalent to
            `interpolation="lower"` in `numpy.percentile`)
        higher_percentiles : float, int, or array-like of shape (h_values,),
                             default=None
            percentiles for which a higher value will be output in case
            a percentile lies between two values (equivalent to
            `interpolation="higher"` in `numpy.percentile`)
        y_min : float or int, default=-numpy.inf
            The minimum value to include
        y_max : float or int, default=numpy.inf
            The maximum value to include
        warm_start : bool, default=True
           extend original calibration set

        Returns
        -------
        percentiles : ndarray of shape (n_values, l_values + h_values)
            percentiles

        Examples
        --------
        Assuming that ``y_hat_test`` and ``y_test`` are vectors with predicted
        and correct targets, respectively, for a test set and ``cps_std`` a
        fitted standard conformal predictive system, then percentiles computed
        using online calibration can be obtained by:

        .. code-block:: python

           p_values = cps_std.predict_percentiles_online(y_hat_test, y_test,
                                                         lower_percentiles=2.5,
                                                         higher_percentiles=97.5)

        Note
        ----
        In case the calibration set is too small for the specified percentiles
        level, the output values will be ``y_min`` and ``y_max``, respectively.
        """
        tic = time.time()
        if isinstance(lower_percentiles, (int, float, np.integer, np.floating)):
            lower_percentiles = np.array([lower_percentiles])
        elif lower_percentiles is None:
            lower_percentiles = np.array([])
        elif isinstance(lower_percentiles, list):
            lower_percentiles = np.array(lower_percentiles)
        if isinstance(higher_percentiles, (int, float, np.integer, np.floating)):
            higher_percentiles = np.array([higher_percentiles])
        elif higher_percentiles is None:
            higher_percentiles = np.array([])
        elif isinstance(higher_percentiles, list):
            higher_percentiles = np.array(higher_percentiles)
        if (lower_percentiles > 100).any() or \
           (lower_percentiles < 0).any() or \
           (higher_percentiles > 100).any() or \
           (higher_percentiles < 0).any():
            raise ValueError("All percentiles must be in the range [0,100]")
        lower_percentiles /= 100
        higher_percentiles /= 100
        no_low = len(lower_percentiles)
        no_high = len(higher_percentiles)
        percentiles = np.zeros((len(y_hat), no_low + no_high))
        if bins is None:
            if warm_start and self.alphas is not None:
                alphas_cal = list(self.alphas)
            else:
                alphas_cal = []
            for i in range(len(y_hat)):
                if len(alphas_cal) > 0:
                    if sigmas is not None:
                        cpd = y_hat[i] + sigmas[i]*np.array(alphas_cal)
                    else:
                        cpd = y_hat[i] + np.array(alphas_cal)
                    low_indexes = [int(p*(len(alphas_cal) + 1)) - 1
                                   for p in lower_percentiles]
                    percentiles[i, :no_low] = [cpd[j] if j >=0 else
                                               -np.inf for j in low_indexes]
                    high_indexes = [int(np.ceil(p*(len(alphas_cal) + 1))) - 1
                                    for p in higher_percentiles]
                    percentiles[i, no_low:] = [cpd[j] if j < len(alphas_cal)
                                               else np.inf for j in high_indexes]
                else:
                    percentiles[i, :no_low] = np.full(no_low, -np.inf)
                    percentiles[i, no_low:] = np.full(no_high, np.inf)
                if sigmas is not None:
                    insort(alphas_cal, (y[i]-y_hat[i])/sigmas[i])
                else:
                    insort(alphas_cal, (y[i]-y_hat[i]))
        else:
            if warm_start and self.binned_alphas is not None:
                bin_values_cal, bin_alphas_cal = self.binned_alphas
                all_alphas_cal = {bin_values_cal[i] : list(bin_alphas_cal[i])
                                  for i in range(len(bin_values_cal))}
            else:
                all_alphas_cal = {}
            bin_values, bin_indexes = np.unique(bins, return_inverse=True)
            for b in range(len(bin_values)):
                alphas_cal = all_alphas_cal.get(bin_values[b], [])
                orig_indexes = np.arange(len(bins))[bin_indexes == b]
                for i in orig_indexes:
                    if len(alphas_cal) > 0:
                        if sigmas is not None:
                            cpd = y_hat[i] + sigmas[i]*np.array(alphas_cal)
                        else:
                            cpd = y_hat[i] + np.array(alphas_cal)
                        low_indexes = [int(p*(len(alphas_cal) + 1)) - 1
                                       for p in lower_percentiles]
                        percentiles[i, :no_low] = [cpd[j] if j >=0 else -np.inf
                                                   for j in low_indexes]
                        high_indexes = [int(np.ceil(
                            p*(len(alphas_cal) + 1))) - 1
                                        for p in higher_percentiles]
                        percentiles[i, no_low:] = [cpd[j] if j < len(alphas_cal)
                                                   else np.inf
                                                   for j in high_indexes]
                    else:
                        percentiles[i, :no_low] = np.full(no_low, -np.inf)
                        percentiles[i, no_low:] = np.full(no_high, np.inf)
                    if sigmas is not None:
                        insort(alphas_cal, (y[i]-y_hat[i])/sigmas[i])
                    else:
                        insort(alphas_cal, (y[i]-y_hat[i]))
        if y_min > -np.inf:
            too_small = np.argwhere(percentiles < y_min)
            percentiles[too_small[:, 0], too_small[:, 1]] = y_min
        if y_max < np.inf:
            too_large = np.argwhere(percentiles > y_max)
            percentiles[too_large[:, 0], too_large[:, 1]] = y_max
        toc = time.time()
        self.time_predict = toc-tic            
        return percentiles
    
    def predict_cpds(self, y_hat, sigmas=None, bins=None,
                     cpds_by_bins=False):    
        """
        Obtain conformal predictive distributions from conformal predictive
        system.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        cpds_by_bins : Boolean, default=False
            specifies whether the output cpds should be grouped by bin or not; 

        Returns
        -------
        cpds : ndarray of shape (n_values, c_values) or (n_values,)
               or list of ndarrays
            conformal predictive distributions. If bins is None, the
            distributions are represented by a single array, where the
            number of columns (c_values) is determined by the number of
            residuals of the fitted conformal predictive system. Otherwise,
            the distributions are represented by a vector of arrays,
            if cpds_by_bins = False, or a list of arrays, with one element
            for each bin, if cpds_by_bins = True.

        Examples
        --------
        Assuming that ``y_hat_test`` is a vector with predicted targets
        for a test set and ``cps_std`` a fitted standard conformal predictive
        system, conformal predictive distributions (cpds) can be obtained by:

        .. code-block:: python

           cpds = cps_std.predict_cpds(y_hat_test)

        Note
        ----
        The returned array may be very large as its size is the product of the
        number of calibration and test objects, unless a Mondrian approach is
        employed; for the latter, this number is reduced by increasing the
        number of bins.

        Note
        ----
        Setting ``cpds_by_bins=True`` has an effect only for Mondrian conformal 
        predictive systems.
        """
        cpds = self.predict(y_hat, sigmas, bins, return_cpds=True,
                            cpds_by_bins=cpds_by_bins)
        return cpds

    def predict_cpds_online(self, y_hat, y, sigmas=None, bins=None,
                            warm_start=True):
        """
        Obtain conformal predictive distributions from conformal predictive
        system, computed using online calibration.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct labels
        sigmas : array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories
        warm_start : bool, default=True
           extend original calibration set

        Returns
        -------
        cpds : ndarray of shape (n_values,)
            conformal predictive distributions

        Examples
        --------
        Assuming that ``y_hat_test`` and ``y_test`` are vectors with predicted
        and correct targets for a test set and ``cps_std`` a fitted standard
        conformal predictive system, then conformal predictive distributions
        can be obtained using online calibration by:

        .. code-block:: python

           cpds = cps_std.predict_cpds_online(y_hat_test, y_test)

        Note
        ----
        The returned vector of vectors may be very large; the largest element
        may be of the same size as the concatenation of the calibration and
        test sets.
        """
        tic = time.time()
        cpds = np.empty(len(y_hat), dtype=object)
        if bins is None:
            if warm_start and self.alphas is not None:
                alphas_cal = list(self.alphas)
            else:
                alphas_cal = []
            if sigmas is not None:
                for i in range(len(y_hat)):
                    if len(alphas_cal) > 0:
                        cpds[i] = y_hat[i] + sigmas[i]*np.array(alphas_cal)
                    else:
                        cpds[i] = np.array([])
                    insort(alphas_cal, (y[i]-y_hat[i])/sigmas[i])
            else:
                for i in range(len(y_hat)):
                    if len(alphas_cal) > 0:
                        cpds[i] = y_hat[i] + np.array(alphas_cal)
                    else:
                        cpds[i] = np.array([])
                    insort(alphas_cal, (y[i]-y_hat[i]))
        else:
            if warm_start and self.binned_alphas is not None:
                bin_values_cal, bin_alphas_cal = self.binned_alphas
                all_alphas_cal = {bin_values_cal[i] : list(bin_alphas_cal[i])
                                  for i in range(len(bin_values_cal))}
            else:
                all_alphas_cal = {}
            bin_values, bin_indexes = np.unique(bins, return_inverse=True)
            for b in range(len(bin_values)):
                alphas_cal = all_alphas_cal.get(bin_values[b], [])
                orig_indexes = np.arange(len(bins))[bin_indexes == b]
                if sigmas is not None:
                    for i in orig_indexes:
                        if len(alphas_cal) > 0:
                            cpds[i] = y_hat[i] + sigmas[i]*np.array(alphas_cal)
                        else:
                            cpds[i] = np.array([])
                        insort(alphas_cal, (y[i]-y_hat[i])/sigmas[i])
                else:
                    for i in orig_indexes:
                        if len(alphas_cal) > 0:
                            cpds[i] = y_hat[i] + np.array(alphas_cal)
                        else:
                            cpds[i] = np.array([])
                        insort(alphas_cal, (y[i]-y_hat[i]))
        toc = time.time()
        self.time_predict = toc-tic            
        return cpds
    
    def predict(self, y_hat, sigmas=None, bins=None,
                y=None, lower_percentiles=None, higher_percentiles=None,
                y_min=-np.inf, y_max=np.inf, return_cpds=False,
                cpds_by_bins=False, smoothing=True, seed=None):    
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
        smoothing : bool, default=True
           return smoothed p-values
        seed : int, default=None
           set random seed

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

        In the above example, the nearest higher value is returned, if there
        is no value that corresponds exactly to the requested percentile.
        If we instead would like to retrieve the nearest lower value, we
        should write:

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

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.        
        """
        tic = time.time()
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(y) == list:
            y = np.array(y)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
        if self.mondrian:
            bin_values, bin_alphas = self.binned_alphas
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
                if smoothing:
                    if not self.mondrian:
                        if self.normalized:
                            result[:, 0] = np.array(
                                [(np.sum(
                                    y_hat[i] + sigmas[i]*self.alphas < y) \
                                  + (np.sum(
                                      y_hat[i] + sigmas[i]*self.alphas == y) \
                                     + 1) \
                                  * gammas[i])/(len(self.alphas) + 1)
                                 for i in range(len(y_hat))])
                        else:
                            result[:, 0] = np.array(
                                [(np.sum(
                                    y_hat[i] + self.alphas < y) \
                                  + (np.sum(
                                      y_hat[i] + self.alphas == y) + 1) \
                                  * gammas[i])/(len(self.alphas) + 1)
                                 for i in range(len(y_hat))])
                    else:
                        for b in range(len(bin_values)):
                            if self.normalized:
                                result[bin_indexes[b], 0] = np.array(
                                    [(np.sum(
                                        y_hat[i] + sigmas[i]*bin_alphas[b] < y) \
                                      + (np.sum(
                                          y_hat[i] + \
                                          sigmas[i]*bin_alphas[b] == y) + 1) \
                                      * gammas[i])/(len(bin_alphas[b]) + 1)
                                     for i in bin_indexes[b]])
                            else:
                                result[bin_indexes[b], 0] = np.array(
                                    [(np.sum(
                                        y_hat[i] + bin_alphas[b] < y) \
                                      + (np.sum(
                                          y_hat[i] + bin_alphas[b] == y) + 1) \
                                      * gammas[i])/(len(bin_alphas[b]) + 1)
                                     for i in bin_indexes[b]])
                else:                    
                    if not self.mondrian:
                        if self.normalized:
                            result[:, 0] = np.array(
                                [(np.sum(
                                    y_hat[i] + sigmas[i]*self.alphas <= y) + 1) \
                                 / (len(self.alphas) + 1)
                                 for i in range(len(y_hat))])
                        else:
                            result[:, 0] = np.array(
                                [(np.sum(
                                    y_hat[i] + self.alphas <= y) +1) \
                                 / (len(self.alphas) + 1)
                                 for i in range(len(y_hat))])
                    else:
                        for b in range(len(bin_values)):
                            if self.normalized:
                                result[bin_indexes[b], 0] = np.array(
                                    [(np.sum(
                                        y_hat[i]+sigmas[i]*bin_alphas[b] <= y) \
                                      + 1) \
                                      / (len(bin_alphas[b]) + 1)
                                     for i in bin_indexes[b]])
                            else:
                                result[bin_indexes[b], 0] = np.array(
                                    [(np.sum(
                                        y_hat[i] + bin_alphas[b] <= y) + 1) \
                                      / (len(bin_alphas[b]) + 1)
                                     for i in bin_indexes[b]])
            elif isinstance(y, (list, np.ndarray)) and len(y) == len(y_hat):
                if smoothing:
                    if not self.mondrian:
                        if self.normalized:
                            result[:, 0] = np.array(
                                [(np.sum(
                                    y_hat[i] + sigmas[i]*self.alphas < y[i]) \
                                  + (np.sum(
                                      y_hat[i] + sigmas[i]*self.alphas == y[i]) \
                                     + 1) \
                                  * gammas[i])/(len(self.alphas) + 1)
                                 for i in range(len(y_hat))])
                        else:
                            result[:, 0] = np.array(
                                [(np.sum(
                                    y_hat[i] + self.alphas < y[i]) \
                                  + (np.sum(
                                      y_hat[i] + self.alphas == y[i]) + 1) \
                                  * gammas[i])/(len(self.alphas) + 1)
                                 for i in range(len(y_hat))])
                    else:
                        for b in range(len(bin_values)):
                            if self.normalized:
                                result[bin_indexes[b], 0] = np.array(
                                    [(np.sum(
                                        y_hat[i] \
                                        + sigmas[i]*bin_alphas[b] < y[i]) \
                                      + (np.sum(
                                          y_hat[i] + \
                                          sigmas[i]*bin_alphas[b] == y[i]) + 1) \
                                      * gammas[i])/(len(bin_alphas[b]) + 1)
                                     for i in bin_indexes[b]])
                            else:
                                result[bin_indexes[b], 0] = np.array(
                                    [(np.sum(
                                        y_hat[i] + bin_alphas[b] < y[i]) \
                                      + (np.sum(
                                          y_hat[i] + bin_alphas[b] == y[i]) \
                                         + 1) \
                                      * gammas[i])/(len(bin_alphas[b]) + 1)
                                     for i in bin_indexes[b]])
                else:                    
                    if not self.mondrian:
                        if self.normalized:
                            result[:, 0] = np.array(
                                [(np.sum(
                                    y_hat[i] + sigmas[i]*self.alphas <= y[i]) \
                                  + 1) \
                                 / (len(self.alphas) + 1)
                                 for i in range(len(y_hat))])
                        else:
                            result[:, 0] = np.array(
                                [(np.sum(
                                    y_hat[i] + self.alphas <= y[i]) +1) \
                                 / (len(self.alphas) + 1)
                                 for i in range(len(y_hat))])
                    else:
                        for b in range(len(bin_values)):
                            if self.normalized:
                                result[bin_indexes[b], 0] = np.array(
                                    [(np.sum(
                                        y_hat[i] + \
                                        sigmas[i]*bin_alphas[b] <= y[i]) + 1) \
                                      / (len(bin_alphas[b]) + 1)
                                     for i in bin_indexes[b]])
                            else:
                                result[bin_indexes[b], 0] = np.array(
                                    [(np.sum(
                                        y_hat[i] + bin_alphas[b] <= y[i]) + 1) \
                                      / (len(bin_alphas[b]) + 1)
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
                                          for lower_percentile in \
                                          lower_percentiles])
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
                    y_max_columns = [no_prec_result_cols \
                                     + len(lower_percentiles)+i
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
        if seed is not None:
            np.random.set_state(random_state)

    def evaluate(self, y_hat, y, sigmas=None, bins=None, confidence=0.95,
                 y_min=-np.inf, y_max=np.inf, metrics=None, smoothing=True,
                 seed=None, online=False, warm_start=True):
        """
        Evaluate conformal predictive system.

        Parameters
        ----------
        y_hat : array-like of shape (n_values,)
            predicted values
        y : array-like of shape (n_values,)
            correct labels
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
        metrics : a string or a list of strings, default=list of all 
            applicable metrics; ["error", "eff_mean","eff_med", "CRPS",
            "ks_test", "time_fit", "time_evaluate"]
        smoothing : bool, default=True
           employ smoothed p-values
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True
            
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics, where "error" is the 
            fraction of prediction intervals not containing the true label,
            "eff_mean" is the mean length of prediction intervals,
            "eff_med" is the median length of the prediction intervals,
            "CRPS" is the continuous ranked probability score,
            "ks_test" is the p-value for the Kolmogorov-Smirnov test of
            uniformity of predicted p-values, "time_fit" is the time taken
            to fit the conformal predictive system, and "time_evaluate" is
            the time taken for the evaluation         

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
        The use of the metric ``CRPS`` may require a lot of memory, as a matrix
        is generated for which the number of elements is the product of the 
        number of calibration and test objects, unless a Mondrian approach is 
        employed; for the latter, this number is reduced by increasing the
        number of bins.

        Note
        ----
        The metric ``CRPS`` is only available for batch evaluation, i.e., when
        ``online=False``.
        
        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.        
        """
        if not self.fitted and not online:
            raise RuntimeError(("Batch evaluation requires a fitted "
                                "conformal predictive system"))
        tic = time.time()
        if metrics is None and not online:
            metrics = ["error", "eff_mean", "eff_med", "CRPS", "ks_test",
                       "time_fit", "time_evaluate"]
        elif metrics is None and online:
            metrics = ["error", "eff_mean", "eff_med", "ks_test", "time_fit",
                       "time_evaluate"]
        if "CRPS" in metrics and online:
            raise RuntimeError(("CRPS not available as metric for "
                                "online calibration"))
        if type(y_hat) == list:
            y_hat = np.array(y_hat)
        if type(y) == list:
            y = np.array(y)
        if type(sigmas) == list:
            sigmas = np.array(sigmas)
        if type(bins) == list:
            bins = np.array(bins)
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(y_hat, pd.Series):
            y_hat = y_hat.values
        if not online and not self.normalized:
            sigmas = None
        if not online and not self.mondrian:
            bins = None
        if not online:
            intervals = self.predict_int(y_hat, sigmas, bins, confidence,
                                         y_min, y_max)
        else:
            intervals = self.predict_int_online(y_hat, y, sigmas, bins,
                                                confidence, y_min, y_max,
                                                warm_start)
        test_results = {}
        if "error" in metrics:
            test_results["error"] = 1-np.mean(
                np.logical_and(intervals[:,0]<=y, y<=intervals[:,1]))
        if "eff_mean" in metrics:            
            test_results["eff_mean"] = np.mean(intervals[:,1]-intervals[:,0])
        if "eff_med" in metrics:            
            test_results["eff_med"] = np.median(intervals[:,1]-intervals[:,0])
        if "time_fit" in metrics:
            test_results["time_fit"] = self.time_fit
        if "CRPS" in metrics and not online:
            cpds = self.predict_cpds(y_hat, sigmas, bins,
                                     cpds_by_bins=True)    
            if not self.mondrian:
                if self.normalized:
                    crps = calculate_crps(cpds, self.alphas, sigmas, y)
                else:
                    crps = calculate_crps(cpds, self.alphas,
                                          np.ones(len(y_hat)), y)
            else:
                bin_values, bin_alphas = self.binned_alphas
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
            test_results["CRPS"] = crps
        if "ks_test" in metrics:
            if not online:
                p_values = self.predict_p(y_hat, y, sigmas, bins, smoothing,
                                          seed)
            else:
                p_values = self.predict_p_online(y_hat, y, None, sigmas, bins,
                                                 smoothing, seed, warm_start)
            test_results["ks_test"] = kstest(p_values, "uniform").pvalue
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
        correct labels
        
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
        conformal predictive distribution
    y : int or float
        correct labels
        
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
        score = np.sum(lower_errors[:cpd_index]*widths[:cpd_index]*sigma) \
            + np.sum(higher_errors[cpd_index+1:]*widths[cpd_index+1:]*sigma) \
            + lower_errors[cpd_index]*(y-cpd[cpd_index])*sigma \
            + higher_errors[cpd_index]*(cpd[cpd_index+1]-y)*sigma
    return score

def p_values_batch(alphas_cal, alphas_test, bins_cal=None, bins_test=None,
                   smoothing=True, seed=None):
    """
    Given non-conformity scores for the calibration set, provides (smoothed or
    non-smoothed) p-values for non-conformity scores for test set, optionally
    with assigned Mondrian categories.

    Parameters
    ----------
    alphas_cal : array-like of shape (n_samples,)
        non-conformity scores for calibration set
    alphas_test : array-like of shape (n_samples,) or (n_samples, n_classes)
        non-conformity scores for test set
    bins_cal : array-like of shape (n_samples,), default=None
        Mondrian categories for calibration set
    bins_test : array-like of shape (n_samples,), default=None
        Mondrian categories for test set
    smoothing : bool, default=True
        return smoothed p-values
    seed : int, default=None
        set random seed

    Returns
    -------
    p-values : array-like of shape (n_samples,) or (n_samples, n_classes)
        p-values 
    """
    p_values = np.zeros(alphas_test.shape)
    if type(alphas_cal) == list:
        alphas_cal = np.array(alphas_cal)
    if type(alphas_test) == list:
        alphas_test = np.array(alphas_test)
    if type(bins_cal) == list:
        bins_cal = np.array(bins_cal)
    if type(bins_test) == list:
        bins_test = np.array(bins_test)
    if seed is not None:
        random_state = np.random.get_state()
        np.random.seed(seed)
    if bins_cal is None:
        q = len(alphas_cal)
        if smoothing:
            if len(alphas_test.shape) > 1:
                thetas = np.random.rand(alphas_test.shape[0],
                                        alphas_test.shape[1])
                p_values = np.array([
                    [(np.sum(alphas_cal > alphas_test[i,c]) + thetas[i,c]*(
                        np.sum(alphas_cal == alphas_test[i,c])+1))/(q+1)
                     for c in range(alphas_test.shape[1])]
                for i in range(len(alphas_test))])
            else:
                thetas = np.random.rand(len(alphas_test))
                p_values = np.array([
                    (np.sum(alphas_cal > alphas_test[i]) + thetas[i]*(
                        np.sum(alphas_cal == alphas_test[i])+1))/(q+1)
                    for i in range(len(alphas_test))])
        else:
            if len(alphas_test.shape) > 1:
                p_values = np.array([[(
                    np.sum(alphas_cal >= alphas_test[i,c])+1)/(q+1)
                                      for c in range(alphas_test.shape[1])]
                                     for i in range(len(alphas_test))])
            else:
                p_values = np.array([
                    (np.sum(alphas_cal >= alphas_test[i])+1)/(q+1)
                    for i in range(len(alphas_test))])
    else:
        p_values = np.zeros(alphas_test.shape)
        bin_values, bin_indexes = np.unique(np.hstack((bins_cal, bins_test)),
                                            return_inverse=True)
        bin_indexes_cal = bin_indexes[:len(bins_cal)]
        bin_indexes_test = bin_indexes[len(bins_cal):]
        if smoothing:
            for b in range(len(bin_values)):
                bin_alphas_cal = alphas_cal[bin_indexes_cal == b]
                q = len(bin_alphas_cal)
                bin_alphas_test = alphas_test[bin_indexes_test == b]
                if len(bin_alphas_test.shape) > 1:
                    thetas = np.random.rand(bin_alphas_test.shape[0],
                                            bin_alphas_test.shape[1])
                    bin_p_values = np.array([[(
                        np.sum(bin_alphas_cal > bin_alphas_test[i,c]) + \
                        thetas[i,c]*(np.sum(bin_alphas_cal == \
                                            bin_alphas_test[i,c])+1))/(q+1)
                                              for c in range(
                                                      alphas_test.shape[1])]
                                             for i in range(
                                                     len(bin_alphas_test))])
                else:
                    thetas = np.random.rand(len(bin_alphas_test))
                    bin_p_values = np.array([
                        (np.sum(bin_alphas_cal > bin_alphas_test[i]) + \
                         thetas[i]*(np.sum(bin_alphas_cal == \
                                           bin_alphas_test[i])+1))/(q+1)
                        for i in range(len(bin_alphas_test))])
                orig_indexes = np.arange(len(alphas_test))[
                    bin_indexes_test == b]
                if len(orig_indexes) > 0:
                    p_values[orig_indexes] = bin_p_values
        else:
            for b in range(len(bin_values)):
                bin_alphas_cal = alphas_cal[bin_indexes_cal == b]
                q = len(bin_alphas_cal)
                bin_alphas_test = alphas_test[bin_indexes_test == b]
                if len(bin_alphas_test.shape) > 1:
                    bin_p_values = np.array([[(np.sum(
                        bin_alphas_cal >= bin_alphas_test[i,c])+1)/(q+1)
                                              for c in range(
                                                      alphas_test.shape[1])]
                                             for i in range(
                                                     len(bin_alphas_test))])
                else:
                    bin_p_values = np.array([(np.sum(
                        bin_alphas_cal >= bin_alphas_test[i])+1)/(q+1)
                                             for i in range(
                                                     len(bin_alphas_test))])
                orig_indexes = np.arange(len(alphas_test))[bin_indexes_test == b]
                if len(orig_indexes) > 0:
                    p_values[orig_indexes] = bin_p_values
    if seed is not None:
        np.random.set_state(random_state)
    return p_values

def p_values_online_classification(alphas, classes, y, bins=None,
                                   alphas_cal=None, bins_cal=None,
                                   all_classes=True, smoothing=True,
                                   seed=None):
    """
    Provides (smoothed or non-smoothed) p-values, computed using online
    calibration, for a sequence of alphas and correct class labels, optionally
    with assigned Mondrian categories.

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
    alphas_cal : array-like of shape (n_cal,)
        non-conformity scores for calibration set
    bins_cal : array-like of shape (n_cal,)
        Mondrian categories for calibration set
    all_classes : bool, default=True
        return p-values for all classes
    smoothing : bool, default=True
        return smoothed p-values
    seed : int, default=None
        set random seed

    Returns
    -------
    p-values : array-like of shape (n_samples,) or (n_samples, n_classes)
        p-values 
    """
    if type(alphas) == list:
        alphas = np.array(alphas)
    if type(alphas_cal) == list:
        alphas_cal = np.array(alphas_cal)
    if type(bins) == list:
        bins = np.array(bins)
    if seed is not None:
        random_state = np.random.get_state()
        np.random.seed(seed)
    class_indexes = np.array(
        [np.argwhere(classes == y[i])[0][0] for i in range(len(y))])        
    if alphas_cal is not None:
        all_alphas = np.hstack((alphas_cal, alphas[np.arange(len(alphas)),
                                                 class_indexes]))
        start = len(alphas_cal)
    else:
        all_alphas = alphas[np.arange(len(alphas)), class_indexes]
        start = 0
    if all_classes:
        p_values = np.zeros(alphas.shape)
    else:
        p_values = np.zeros(len(alphas))
    if bins_cal is not None:
        all_bins = np.hstack((bins_cal, bins))
    else:
        all_bins = bins        
    if bins is None:
        if smoothing:
            if all_classes:
                thetas = np.random.rand(alphas.shape[0], alphas.shape[1])
                p_values = np.array([[(
                    np.sum(all_alphas[:start+q] > alphas[q,c]) \
                    + thetas[q,c] * (np.sum(
                        all_alphas[:start+q] == alphas[q,c])+1))/(start + q + 1)
                                      for c in range(alphas.shape[1])]
                                     for q in range(len(alphas))])
            else:
                thetas = np.random.rand(len(alphas))
                p_values = np.array([
                    (np.sum(all_alphas[:start+q] > all_alphas[start+q]) \
                     + thetas[q] * (np.sum(
                         all_alphas[:start+q+1] == all_alphas[start+q]))) \
                    / (start + q + 1)
                    for q in range(len(alphas))])
        else:
            if all_classes:
                p_values = np.array(
                    [[(np.sum(all_alphas[:start+q] >= alphas[q,c]) + 1) \
                      / (start + q + 1) for c in range(alphas.shape[1])]
                     for q in range(len(alphas))])
            else:
                p_values = np.array(
                    [np.sum(all_alphas[:start+q+1] >= all_alphas[q])/ \
                     (start + q + 1) for q in range(len(alphas))])
    else:
        bin_values, bin_indexes = np.unique(all_bins, return_inverse=True)
        if smoothing:
            for b in range(len(bin_values)):
                bin_all_alphas = all_alphas[bin_indexes == b]
                bin_alphas = alphas[bin_indexes[start:] == b]
                bin_start = len(bin_all_alphas) - len(bin_alphas)
                if all_classes:
                    thetas = np.random.rand(bin_alphas.shape[0],
                                            bin_alphas.shape[1])
                    bin_p_values = np.array([[
                        (np.sum(bin_all_alphas[:bin_start+q] > bin_alphas[q,c]) \
                         + thetas[q,c] * (np.sum(bin_all_alphas[
                             :bin_start+q] == bin_alphas[q,c]) + 1)) \
                        / (bin_start + q + 1)
                        for c in range(bin_alphas.shape[1])]
                                             for q in range(len(bin_alphas))])
                else:
                    thetas = np.random.rand(len(bin_alphas))
                    bin_p_values = np.array([
                        (np.sum(bin_all_alphas[:bin_start+q] > \
                                bin_all_alphas[bin_start+q]) \
                         + thetas[q] * (
                             np.sum(bin_all_alphas[:bin_start+q] == \
                                    bin_all_alphas[bin_start+q]) + 1)) \
                        / (bin_start + q + 1)
                        for q in range(len(bin_alphas))])
                orig_indexes = np.arange(len(alphas))[bin_indexes[start:] == b]
                if len(orig_indexes) > 0:
                    p_values[orig_indexes] = bin_p_values
        else:
            for b in range(len(bin_values)):
                bin_all_alphas = all_alphas[bin_indexes == b]
                bin_alphas = alphas[bin_indexes[start:] == b]
                bin_start = len(bin_all_alphas) - len(bin_alphas)
                if all_classes:
                    p_values_bin = np.array([[
                        (np.sum(bin_all_alphas[
                            :bin_start+q] >= bin_alphas[q,c]) + 1) \
                        / (bin_start + q + 1)
                        for c in range(bin_alphas.shape[1])]
                                             for q in range(len(bin_alphas))])
                else:
                    p_values_bin = np.array([
                        (np.sum(bin_all_alphas[
                            :bin_start+q] >= \
                                bin_all_alphas[bin_start+q]) + 1) \
                        / (bin_start + q + 1) for q in range(len(bin_alphas))])
                orig_indexes = np.arange(len(alphas))[bin_indexes[start:] == b]
                if len(orig_indexes) > 0:
                    p_values[orig_indexes] = bin_p_values
    if seed is not None:
        np.random.set_state(random_state)
    return p_values

def p_values_online_class_cond(alphas, classes, y, alphas_cal=None, y_cal=None,
                               all_classes=True, smoothing=True, seed=None):
    """
    Provides (smoothed or non-smoothed) p-values, computed using online
    calibration, for a sequence of alphas and correct class labels, optionally
    with assigned Mondrian categories.

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
    alphas_cal : array-like of shape (n_cal,)
        non-conformity scores for calibration set
    bins_cal : array-like of shape (n_cal,)
        Mondrian categories for calibration set
    all_classes : bool, default=True
        return p-values for all classes
    smoothing : bool, default=True
        return smoothed p-values
    seed : int, default=None
        set random seed

    Returns
    -------
    p-values : array-like of shape (n_samples,) or (n_samples, n_classes)
        p-values 
    """
    if type(alphas) == list:
        alphas = np.array(alphas)
    if type(alphas_cal) == list:
        alphas_cal = np.array(alphas_cal)
    if seed is not None:
        random_state = np.random.get_state()
        np.random.seed(seed)
    if y_cal is not None:
        all_y = np.hstack((y_cal, y))
        start = len(y_cal)
    else:
        all_y = y
        start = 0
    class_indexes = np.array(
        [np.argwhere(classes == all_y[i])[0][0] for i in range(len(all_y))])
    if alphas_cal is not None:
        all_alphas = np.hstack((alphas_cal, alphas[np.arange(len(alphas)),
                                                   class_indexes[start:]]))
        class_vectors = np.zeros((len(all_y)+1, alphas.shape[1]))
        class_vectors[np.arange(len(all_y))+1, class_indexes] = 1
        class_counts = np.cumsum(class_vectors, axis=0)[start:].astype(int)
    else:
        all_alphas = alphas[np.arange(len(alphas)), class_indexes]
        class_vectors = np.zeros((alphas.shape[0]+1, alphas.shape[1]))
        class_vectors[np.arange(len(alphas))+1, class_indexes] = 1
        class_counts = np.cumsum(class_vectors, axis=0).astype(int)
    bin_alphas = {c : all_alphas[class_indexes == c]
                  for c in range(len(classes))}
    if all_classes:
        if smoothing:
            thetas = np.random.rand(len(alphas), len(classes))
            p_values = np.array([
                [(np.sum(bin_alphas[c][:class_counts[q,c]] > alphas[q,c]) \
                 + thetas[q,c]*(np.sum(
                     bin_alphas[c][:class_counts[q,c]] == alphas[q,c])+1)) \
                 / (class_counts[q,c]+1)
                 for c in range(len(classes))]
                for q in range(len(alphas))])
        else:
            p_values = np.array([
                [(np.sum(bin_alphas[c][:class_counts[q,c]] >= alphas[q,c])+1) \
                 / (class_counts[q,c]+1)
                 for c in range(len(classes))]
                for q in range(len(alphas))])
    else:
        c = class_indexes[start:]
        if smoothing:
            thetas = np.random.rand(len(alphas))
            p_values = np.array([
                (np.sum(bin_alphas[c[q]][
                    :class_counts[q,c[q]]] > alphas[q,c[q]]) \
                 + thetas[q]*(np.sum(
                     bin_alphas[c[q]][
                         :class_counts[q,c[q]]] == alphas[q,c[q]])+1)) \
                /(class_counts[q,c[q]]+1)
                for q in range(len(alphas))])
        else:
            p_values = np.array([
                (np.sum(bin_alphas[c[q]][
                    :class_counts[q,c[q]]] >= alphas[q,c[q]])+1) \
                /(class_counts[q,c[q]]+1)
                for q in range(len(alphas))])
    if seed is not None:
        np.random.set_state(random_state)
    return p_values

def p_values_online_regression(alphas, alphas_target=None, bins=None,
                               alphas_cal=None, bins_cal=None, smoothing=True,
                               seed=None):
    """
    Provides (smoothed or non-smoothed) p-values, computed using online
    calibration, for a sequence of alphas, optionally with assigned Mondrian
    categories.

    Parameters
    ----------
    alphas : array-like of shape (n_samples,)
        non-conformity scores to use for online calibration,
        p-values are provided for these if alphas_target=None
    alphas_target : int, float or array-like of shape (n_samples,), default=None
        non-conformity scores to provide p-values for
    bins : array-like of shape (n_samples,), default=None
        Mondrian categories
    alphas_cal : array-like of shape (n_cal,)
        non-conformity scores for calibration set
    bins_cal : array-like of shape (n_cal,)
        Mondrian categories for calibration set
    smoothing : bool, default=True
        return smoothed p-values
    seed : int, default=None
        set random seed

    Returns
    -------
    p-values : array-like of shape (n_samples,)
        p-values
    """
    if type(alphas) == list:
        alphas = np.array(alphas)
    if type(alphas_target) == list:
        alphas_target = np.array(alphas_target)
    elif isinstance(alphas_target, (int, float, np.integer, np.floating)):
        alphas_target = np.full(len(alphas), alphas_target)
    elif alphas_target is None:
        alphas_target = alphas
    if type(alphas_cal) == list:
        alphas_cal = np.array(alphas_cal)
    if type(bins) == list:
        bins = np.array(bins)
    if seed is not None:
        random_state = np.random.get_state()
        np.random.seed(seed)
    if alphas_cal is not None:
        all_alphas = np.hstack((alphas_cal, alphas))
        start = len(alphas_cal)
    else:
        all_alphas = alphas
        start = 0
    if bins_cal is not None:
        all_bins = np.hstack((bins_cal, bins))
    else:
        all_bins = bins      
    if bins is None:
        if smoothing:
            thetas = np.random.rand(len(alphas))
            p_values = np.array([(np.sum(
                all_alphas[:start+q] > alphas_target[q]) \
                                  + thetas[q] * (
                                      np.sum(all_alphas[:start+q] \
                                             == alphas_target[q]) + 1)) \
                                 / (start + q + 1) for q in range(len(alphas))])
        else:
            p_values = np.array([(np.sum(
                all_alphas[:start+q] >= alphas_target[q]) + 1)/(start + q + 1)
                                 for q in range(len(alphas))])
    else:
        p_values = np.zeros(len(alphas))
        bin_values, bin_indexes = np.unique(all_bins, return_inverse=True)
        if smoothing:
            for b in range(len(bin_values)):
                bin_all_alphas = all_alphas[bin_indexes == b]
                bin_alphas = alphas_target[bin_indexes[start:] == b]
                bin_start = len(bin_all_alphas) - len(bin_alphas)
                thetas = np.random.rand(len(bin_alphas))
                bin_p_values = np.array([
                    (np.sum(bin_all_alphas[:bin_start+q] > bin_alphas[q]) \
                     + thetas[q] * (np.sum(
                         bin_all_alphas[:bin_start+q] \
                         == bin_alphas[q]) + 1)) / (bin_start + q + 1)
                    for q in range(len(bin_alphas))])
                orig_indexes = np.arange(len(alphas))[bin_indexes[start:] == b]
                if len(orig_indexes) > 0:
                    p_values[orig_indexes] = bin_p_values
        else:
            for b in range(len(bin_values)):
                bin_all_alphas = all_alphas[bin_indexes == b]
                bin_alphas = alphas_target[bin_indexes[start:] == b]
                bin_start = len(bin_all_alphas) - len(bin_alphas)
                bin_p_values = np.array([
                    (np.sum(bin_all_alphas[:bin_start+q] >= bin_alphas[q]) + 1) \
                    / (bin_start + q + 1)
                    for q in range(len(bin_alphas))])
                orig_indexes = np.arange(len(alphas))[bin_indexes[start:] == b]
                if len(orig_indexes) > 0:
                    p_values[orig_indexes] = bin_p_values            
    if seed is not None:
        np.random.set_state(random_state)
    return p_values
    
class WrapClassifier():
    """
    A learner wrapped with a :class:`.ConformalClassifier`.
    """
    
    def __init__(self, learner):
        self.cc = None
        self.nc = None
        self.calibrated = False
        self.learner = learner
        self.seed = None

    def __repr__(self):
        if self.calibrated:
            return (f"WrapClassifier(learner={self.learner}, "
                    f"calibrated={self.calibrated}, "
                    f"predictor={self.cc})")
        else:
            return (f"WrapClassifier(learner={self.learner}, "
                    f"calibrated={self.calibrated})")
        
    def fit(self, X, y, **kwargs):
        """
        Fit learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,),
            labels
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
        self.fitted_ = True
                
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
    
    def calibrate(self, X=[], y=[], oob=False, class_cond=False, nc=hinge,
                  mc=None, seed=None):
        """
        Fit a :class:`.ConformalClassifier` using the wrapped learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=[]
           set of objects
        y : array-like of shape (n_samples,), default=[]
            labels
        oob : bool, default=False
           use out-of-bag estimation
        class_cond : bool, default=False
            if class_cond=True, the method fits a Mondrian
            :class:`.ConformalClassifier` using the class
            labels as categories
        nc : function, default = :func:`crepes.extras.hinge`
            function to compute non-conformity scores
        mc: function or :class:`crepes.extras.MondrianCategorizer`, default=None
            function or :class:`crepes.extras.MondrianCategorizer` for computing
            Mondrian categories
        seed : int, default=None
           set random seed

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

        By providing the option ``class_cond=True``, a Mondrian conformal
        classifier will be formed using the class labels as categories,
        e.g.,

        .. code-block:: python

           w.calibrate(X_cal, y_cal, class_cond=True)

        Note
        ----
        Any Mondrian categorizer specified by the ``mc`` argument will be 
        ignored by :meth:`.calibrate`, if ``class_cond=True``, as the latter 
        implies that Mondrian categories are formed using the labels in ``y``.

        Note
        ----
        By providing a random seed, e.g., ``seed=123``, the call to
        ``calibrate`` as well as calls to the methods ``predict_set``,
        ``predict_p`` and ``evaluate`` of the :class:`.WrapClassifier`
        object will be deterministic.

        Note
        ----
        Enabling out-of-bag calibration, i.e., setting ``oob=True``, requires 
        that the wrapped learner has an attribute ``oob_decision_function_``, 
        which e.g., as for a ``sklearn.ensemble.RandomForestClassifier``,
        if enabled when created, e.g., ``RandomForestClassifier(oob_score=True)``

        Note
        ----
        The use of out-of-bag calibration, as enabled by ``oob=True``, does not 
        come with the theoretical validity guarantees of the regular (inductive) 
        conformal classifiers, due to that calibration and test instances are
        not handled in exactly the same way.
        """
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
            self.seed = seed
        if isinstance(y, pd.Series):
            y = y.values
        self.cc = ConformalClassifier()
        self.nc = nc
        self.mc = mc
        self.class_cond = class_cond
        if len(y) > 0:
            if oob:
                alphas = nc(self.learner.oob_decision_function_,
                            self.learner.classes_, y)
            else:
                alphas = nc(self.learner.predict_proba(X),
                            self.learner.classes_, y)
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
            self.calibrated_ = True
        else:
            self.cc.fit([])
        if seed is not None:
            np.random.set_state(random_state)
        return self

    def predict_p(self, X, y=None, all_classes=True, smoothing=True, seed=None,
                  online=False, warm_start=True):
        """
        Obtain (smoothed or non-smoothed) p-values using conformal classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,), default=None
            correct class labels; used only if online=True
            or all_classes=False
        all_classes : bool, default=True
            return p-values for all classes
        smoothing : bool, default=True
           use smoothed p-values
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True

        Returns
        -------
        p-values : ndarray of shape (n_samples, n_classes)
            p-values

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects and ``w`` is a 
        :class:`.WrapClassifier` object that has been calibrated, i.e., 
        :meth:`.calibrate` has been applied, the (smoothed) p-values for
        the test objects are obtained by:

        .. code-block:: python

           p_values = w.predict_p(X_test)

        Assuming that ``y_test`` a vector of correct labels for the test
        objects, then p-values for the test objects are obtained using
        online calibration by:

        .. code-block:: python

           p_values = w.predict_p(X_test, y_test, online=True)

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``calibrate``.
        """
        if not self.calibrated and not online:
            raise RuntimeError(("Batch predictions requires a calibrated "
                                "conformal classifier"))
        tic = time.time()
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
        alphas = self.nc(self.learner.predict_proba(X))
        classes = self.learner.classes_
        if not online:
            if self.class_cond:
                p_values = np.array([
                    self.cc.predict_p(alphas,
                                      np.full(len(X), classes[c]),
                                      smoothing=smoothing)[:, c]
                    for c in range(len(classes))]).T
                if not all_classes:
                    class_indexes = np.array(
                        [np.argwhere(classes == y[i])[0][0]
                         for i in range(len(y))])
                    p_values = p_values[np.arange(len(y)), class_indexes]
            else:
                if isinstance(self.mc, MondrianCategorizer):
                    bins = self.mc.apply(X)
                elif self.mc is not None:
                    bins = self.mc(X)
                else:
                    bins = None
                p_values = self.cc.predict_p(alphas, bins, all_classes, classes,
                                             y, smoothing)
        else:
            if self.class_cond:
                if warm_start:
                    alphas_cal = self.cc.alphas
                    y_cal = self.cc.bins
                else:
                    alphas_cal = None
                    y_cal = None               
                p_values = p_values_online_class_cond(alphas, classes, y,
                                                      alphas_cal, y_cal,
                                                      all_classes, smoothing,
                                                      seed)
            else:
                if isinstance(self.mc, MondrianCategorizer):
                    bins = self.mc.apply(X)
                elif self.mc is not None:
                    bins = self.mc(X)
                else:
                    bins = None
                p_values = self.cc.predict_p_online(alphas, classes, y, bins,
                                                    all_classes, smoothing,
                                                    seed, warm_start)
        if seed is not None:
            np.random.set_state(random_state)
        toc = time.time()
        self.time_predict = toc-tic            
        return p_values
    
    def predict_set(self, X, y=None, confidence=0.95, smoothing=True,
                    seed=None, online=False, warm_start=True):
        """
        Obtain prediction sets using conformal classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,), default=None
            correct class labels; used only if online=True
        confidence : float in range (0,1), default=0.95
            confidence level
        smoothing : bool, default=True
           use smoothed p-values
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True
        
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
        :meth:`.calibrate` has been applied, then prediction sets for the 
        test objects at the 99% confidence level are obtained by:

        .. code-block:: python

           prediction_sets = w.predict_set(X_test, confidence=0.99)

        Assuming that ``y_test`` a vector of correct labels for the test
        objects, then prediction sets for the test objects at the default
        (95%) confidence level are obtained using online calibration by:

        .. code-block:: python

           prediction_sets = w.predict_set(X_test, y_test, online=True)

        Note
        ----
        The use of smoothed p-values increases computation time and typically
        has a minor effect on the predictions sets, except for small calibration
        sets.        

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``calibrate``.
        """
        if not self.calibrated and not online:
            raise RuntimeError(("Batch predictions requires a calibrated "
                                "conformal classifier"))
        tic = time.time()
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
        alphas = self.nc(self.learner.predict_proba(X))
        classes = self.learner.classes_
        if not online:
            if self.class_cond:
                prediction_sets = np.array([
                    self.cc.predict_set(alphas,
                                        np.full(len(X),
                                                classes[c]),
                                        confidence, smoothing)[:, c]
                    for c in range(len(classes))]).T
            else:
                if isinstance(self.mc, MondrianCategorizer):
                    bins = self.mc.apply(X)
                elif self.mc is not None:
                    bins = self.mc(X)
                else:
                    bins = None
                prediction_sets = self.cc.predict_set(alphas, bins,
                                                      confidence, smoothing)
        else:
            if self.class_cond:
                if warm_start:
                    alphas_cal = self.cc.alphas
                    y_cal = self.cc.bins
                else:
                    alphas_cal = None
                    y_cal = None               
                p_values = p_values_online_class_cond(alphas, classes, y,
                                                      alphas_cal, y_cal,
                                                      True, smoothing, seed)
                prediction_sets = (p_values >= 1-confidence).astype(int)
            else:
                if isinstance(self.mc, MondrianCategorizer):
                    bins = self.mc.apply(X)
                elif self.mc is not None:
                    bins = self.mc(X)
                else:
                    bins = None
                prediction_sets = self.cc.predict_set_online(alphas, classes, y,
                                                             bins, confidence,
                                                             smoothing, seed,
                                                             warm_start)
        if seed is not None:
            np.random.set_state(random_state)
        toc = time.time()
        self.time_predict = toc-tic            
        return prediction_sets
    
    def evaluate(self, X, y, confidence=0.95, smoothing=True,
                 metrics=None, seed=None, online=False, warm_start=True):
        """
        Evaluate the conformal classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           set of objects
        y : array-like of shape (n_samples,)
            correct labels
        confidence : float in range (0,1), default=0.95
            confidence level
        smoothing : bool, default=True
           use smoothed p-values
        metrics : a string or a list of strings, 
                  default=list of all metrics, i.e., ["error", "avg_c", "one_c",
                  "empty", "ks_test", "time_fit", "time_evaluate"]
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True           
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics, where "error" is the 
            fraction of prediction sets not containing the true class label,
            "avg_c" is the average no. of predicted class labels, "one_c" is
            the fraction of singleton prediction sets, "empty" is the fraction
            of empty prediction sets, "ks_test" is the p-value for the
            Kolmogorov-Smirnov test of uniformity of predicted p-values,
            "time_fit" is the time taken to fit the conformal classifier,
            and "time_evaluate" is the time taken for the evaluation 

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
        The use of smoothed p-values increases computation time and typically
        has a minor effect on the results, except for small calibration sets.

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``calibrate``.        
        """
        if not self.calibrated and not online:
            raise RuntimeError(("Batch evaluation requires a calibrated "
                                "conformal classifier"))
        if isinstance(y, pd.Series):
            y = y.values
        if metrics is None:
            metrics = ["error", "avg_c", "one_c", "empty", "ks_test",
                       "time_fit", "time_evaluate"]
        tic = time.time()
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
        p_values = self.predict_p(X, y, True, smoothing, seed, online,
                                  warm_start)
        prediction_sets = (p_values >= 1-confidence).astype(int)
        test_results = get_classification_results(prediction_sets,
                                                  p_values,
                                                  self.learner.classes_,
                                                  y, metrics)
        if seed is not None:
            np.random.set_state(random_state)
        toc = time.time()
        self.time_evaluate = toc-tic
        if "time_fit" in metrics:
            test_results["time_fit"] = self.cc.time_fit
        if "time_evaluate" in metrics:
            test_results["time_evaluate"] = self.time_evaluate
        return test_results

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
        self.seed = None
        
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
            return (f"WrapRegressor(learner={self.learner}, "
                    f"calibrated={self.calibrated})")
        
    def fit(self, X, y, **kwargs):
        """
        Fit learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,),
            labels
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
        self.fitted_ = True
    
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
        Assuming ``w`` is a :class:`.WrapRegressor` object for which the
        wrapped learner ``w.learner`` has been fitted, (point) predictions
        of the learner can be obtained for a set of test objects ``X_test``
        by:

        .. code-block:: python

           y_hat = w.predict(X_test)
           
        The above is equivalent to:

        .. code-block:: python

           y_hat = w.learner.predict(X_test)
        """
        return self.learner.predict(X)

    def calibrate(self, X=[], y=[], de=None, mc=None, oob=False, cps=False,
                  seed=None):
        """
        Fit a :class:`.ConformalRegressor` or 
        :class:`.ConformalPredictiveSystem` using the wrapped learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=[]
           set of objects
        y : array-like of shape (n_samples,), default=[]
            labels
        de: :class:`crepes.extras.DifficultyEstimator`, default=None
            object used for computing difficulty estimates
        mc: function or :class:`crepes.extras.MondrianCategorizer`, default=None
            function or :class:`crepes.extras.MondrianCategorizer` for computing
            Mondrian categories
        oob : bool, default=False
           use out-of-bag estimation
        cps : bool, default=False
            if cps=False, the method fits a :class:`.ConformalRegressor`
            and otherwise, a :class:`.ConformalPredictiveSystem`
        seed : int, default=None
           set random seed

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

        By providing the option ``cps=True``, each of the above calls will
        instead generate a :class:`.ConformalPredictiveSystem`, e.g.,

        .. code-block:: python

           w.calibrate(X_cal, y_cal, de=de, cps=True)

        Note
        ----
        By providing a random seed, e.g., ``seed=123``, the call to
        ``calibrate`` as well as calls to the methods ``predict_int``,
        ``predict_cps`` and ``evaluate`` of the :class:`.WrapRegressor`
        object will be deterministic.
        
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
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
            self.seed = seed
        if isinstance(y, pd.Series):
            y = y.values
        self.de = de
        self.mc = mc
        if len(y) > 0:
            if oob:
                residuals = y - self.learner.oob_prediction_
            else:
                residuals = y - self.predict(X)
            if de is None:
                sigmas = None
            else:
                sigmas = de.apply(X)
            if mc is None:
                bins = None
            elif isinstance(mc, MondrianCategorizer):
                bins = mc.apply(X)
            else:
                bins = mc(X)
            if not cps:
                self.cr = ConformalRegressor()
                self.cr.fit(residuals, sigmas=sigmas, bins=bins)
                self.cps = None
            else:
                self.cps = ConformalPredictiveSystem()
                self.cps.fit(residuals, sigmas=sigmas, bins=bins)
                self.cr = None
            self.calibrated = True
            self.calibrated_ = True
        else:
            if not cps:
                self.cr = ConformalRegressor()
                self.cr.fit([])
                self.cps = None
            else:
                self.cps = ConformalPredictiveSystem()
                self.cps.fit([])
                self.cr = None
        if seed is not None:
            np.random.set_state(random_state)
        return self

    def predict_p(self, X, y=None, t=None, smoothing=True, seed=None,
                  online=False, warm_start=True):
        """
        Return (smoothed or non-smoothed) p-values for provided targets,
        using fitted :class:`.ConformalRegressor` or
        :class:`.ConformalPredictiveSystem`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,), default=None
            correct labels, used for online calibration if
            online=True, and used as targets if t=None
        t : int, float or array-like of shape (n_samples,), default=None
            targets
        smoothing : bool, default=True
           return smoothed p-values
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True
        
        Returns
        -------
        p_values : ndarray of shape (n_samples,)
            p_values

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects, ``y_test`` is the
        set of correct labels and ``w`` is a :class:`.WrapRegressor` object
        that has been calibrated, i.e., :meth:`.calibrate` has been applied,
        then (smoothed) p-values are obtained by:

        .. code-block:: python

           p_values = w.predict_p(X_test, y_test)

        Given a single or vector of targets ``t``, p-values can be obtained
        using online calibration by:

        .. code-block:: python

           p_values = w.predict_p(X_test, y_test, t, online=True)
        
        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given when calling ``calibrate``.
        """
        if not self.calibrated and not online:
            raise RuntimeError(("Batch predictions requires a calibrated "
                                "regressor or predictive system"))
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
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
        if not online:
            if t is not None:
                y = t
            if self.cr is not None:
                p_values = self.cr.predict_p(y_hat, y, sigmas, bins, smoothing,
                                             seed)
            else:
                p_values = self.cps.predict_p(y_hat, y, sigmas, bins, smoothing,
                                              seed)
        else:
            if self.cr is not None:
                p_values = self.cr.predict_p_online(y_hat, y, t, sigmas, bins,
                                                    smoothing, seed, warm_start)
            else:
                p_values = self.cps.predict_p_online(y_hat, y, t, sigmas, bins,
                                                     smoothing, seed, warm_start)
        if seed is not None:
            np.random.set_state(random_state)
        return p_values
    
    def predict_int(self, X, y=None, confidence=0.95, y_min=-np.inf,
                    y_max=np.inf, seed=None, online=False, warm_start=True):
        """
        Obtain prediction intervals with fitted :class:`.ConformalRegressor`
        or :class:`.ConformalPredictiveSystem`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,), default=None
            correct labels; used only if online=True
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True

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

        Assuming ``y_test`` is a vector containing the correct labels for the
        test objects, intervals (at the default confidence level) are provided
        using online calibration by:

        .. code-block:: python

           intervals = w.predict_int(X_test, y_test, online=True)
        
        Note
        ----
        In case the specified confidence level is too high in relation to the 
        size of the calibration set, the output intervals will be of maximum
        size.

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given when calling ``calibrate``.
        """
        if not self.calibrated and not online:
            raise RuntimeError(("Batch predictions requires a calibrated "
                                "regressor or predictive system"))
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
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
        if not online:
            if self.cr is not None:
                intervals = self.cr.predict_int(y_hat, sigmas=sigmas, bins=bins,
                                                confidence=confidence,
                                                y_min=y_min, y_max=y_max)
            else:
                intervals = self.cps.predict_int(y_hat, sigmas=sigmas, bins=bins,
                                                 confidence=confidence,
                                                 y_min=y_min, y_max=y_max)
        else:
            if self.cr is not None:
                intervals = self.cr.predict_int_online(y_hat, y, sigmas, bins,
                                                       confidence, y_min, y_max,
                                                       warm_start)
            else:
                intervals = self.cps.predict_int_online(y_hat, y, sigmas, bins,
                                                        confidence, y_min, y_max,
                                                        warm_start)            
        if seed is not None:
            np.random.set_state(random_state)
        return intervals

    def predict_percentiles(self, X, y=None, lower_percentiles=None,
                            higher_percentiles=None,
                            y_min=-np.inf, y_max=np.inf, seed=None,
                            online=False, warm_start=True):
        """
        Obtain percentiles with conformal predictive system.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,), default=None
            correct labels; used only if online=True
        lower_percentiles : array-like of shape (l_values,), default=None
            percentiles for which a lower value will be output 
            in case a percentile lies between two values
            (similar to `interpolation="lower"` in `numpy.percentile`)
        higher_percentiles : array-like of shape (h_values,), default=None
            percentiles for which a higher value will be output 
            in case a percentile lies between two values
            (similar to `interpolation="higher"` in `numpy.percentile`)
        y_min : float or int, default=-numpy.inf
            The minimum value to include
        y_max : float or int, default=numpy.inf
            The maximum value to include
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True

        Returns
        -------
        percentiles : ndarray of shape (n_values, n_percentiles)

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects and ``cps`` is a 
        :class:`.WrapRegressor` object that has been calibrated while
        enabling the generation of a conformal predictive system, i.e., 
        :meth:`.calibrate` has been called with ``cps=True``, percentiles
        can be obtained by:

        .. code-block:: python

           percentiles = cps.predict_percentiles(X_test, lower_percentiles=2.5,
                                               higher_percentiles=97.5)

        Multiple (lower and higher) percentiles may be requested by:
        .. code-block:: python

           percentiles = cps.predict_percentiles(X_test,
                                                 lower_percentiles=[2.5,5],
                                                 higher_percentiles=[95,97.5])

        Assuming ``y_test`` is a vector containing the correct labels for the
        test objects, percentiles are provided using online calibration by:

        .. code-block:: python

           intervals = cps.predict_percentiles(X_test, y_test,
                                               higher_percentiles=[90,95,99],
                                               online=True)        
        """
        if not self.calibrated and not online:
            raise RuntimeError(("Batch prediction of percentiles requires a "
                                "calibrated conformal predictive system"))
        if self.cps is None:
            raise RuntimeError(("Prediction of percentiles requires a prior "
                                "call to calibrate with cps=True"))
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
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
        if not online:
            percentiles = self.cps.predict_percentiles(y_hat, sigmas, bins,
                                                       lower_percentiles,
                                                       higher_percentiles,
                                                       y_min, y_max)
        else:
            percentiles = self.cps.predict_percentiles_online(y_hat, y, sigmas,
                                                              bins,
                                                              lower_percentiles,
                                                              higher_percentiles,
                                                              y_min, y_max,
                                                              warm_start)
        if seed is not None:
            np.random.set_state(random_state)
        return percentiles
        
    def predict_cpds(self, X, y=None, seed=None, online=False, warm_start=True):
        """
        Obtain conformal predictive distributions from conformal predictive
        system.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features),
           set of objects
        y : array-like of shape (n_samples,), default=None
            correct labels; used only if online=True
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True

        Returns
        -------
        cpds : ndarray of shape (n_values, c_values) or (n_values,)
            conformal predictive distributions. If online=False and
            bins is None, the distributions are represented by a single
            array, where the number of columns (c_values) is determined
            by the number of residuals of the fitted conformal predictive
            system. Otherwise, the output is a vector of arrays.

        Examples
        --------
        Assuming that ``X_test`` is a set of test objects and ``cps`` is a 
        :class:`.WrapRegressor` object that has been calibrated while
        enabling the generation of a conformal predictive system, i.e., 
        :meth:`.calibrate` has been called with ``cps=True``, conformal
        predictive distributions (cpds) can be obtained by:

        .. code-block:: python

           cpds = cps.predict_cpds(X_test)

        Assuming ``y_test`` is a vector containing the correct labels for the
        test objects, cpds can be generated using online calibration by:

        .. code-block:: python

           cpds = cps.predict_cpds(X_test, y_test, online=True)        
        
        Note
        ----
        The returned array may be very large as its size is the product of the
        number of calibration and test objects, unless a Mondrian approach is
        employed; for the latter, this number is reduced by increasing the
        number of bins. For online calibration, the largest element in the
        vector may be of the same size as the concatenation of the calibration
        and test sets.

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``fit``.        
        """
        if not self.calibrated and not online:
            raise RuntimeError(("Batch prediction of cpds requires a "
                                "calibrated conformal predictive system"))
        if self.cps is None:
            raise RuntimeError(("Prediction of cpds requires a prior call "
                                "to calibrate with cps=True"))
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
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
        if not online:
            cpds = self.cps.predict_cpds(y_hat, sigmas, bins,
                                         cpds_by_bins=False)
        else:
            cpds = self.cps.predict_cpds_online(y_hat, y, sigmas, bins,
                                                warm_start)
        if seed is not None:
            np.random.set_state(random_state)
        return cpds
    
    def predict_cps(self, X, y=None, lower_percentiles=None,
                    higher_percentiles=None, y_min=-np.inf, y_max=np.inf,
                    return_cpds=False, cpds_by_bins=False,
                    smoothing=True, seed=None):
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
        smoothing : bool, default=True
           return smoothed p-values
        seed : int, default=None
           set random seed

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
        Setting ``return_cpds=True`` may consume a lot of memory, as a matrix
        is generated for which the number of elements is the product of the
        number of calibration and test objects, unless a Mondrian approach is
        employed; for the latter, this number is reduced by increasing the
        number of bins.

        Note
        ----
        Setting ``cpds_by_bins=True`` has an effect only for Mondrian conformal 
        predictive systems.

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given in the call to ``calibrate``.        
        """
        if self.cps is None:
            raise RuntimeError(("predict_cps requires that calibrate has been "
                                "called first with cps=True"))
        if isinstance(y, pd.Series):
            y = y.values
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
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
        result = self.cps.predict(y_hat, sigmas=sigmas, bins=bins,
                                  y=y, lower_percentiles=lower_percentiles,
                                  higher_percentiles=higher_percentiles,
                                  y_min=y_min, y_max=y_max,
                                  return_cpds=return_cpds,
                                  cpds_by_bins=cpds_by_bins,
                                  smoothing=smoothing,
                                  seed=seed)
        if seed is not None:
            np.random.set_state(random_state)
        return result

    def evaluate(self, X, y, confidence=0.95, y_min=-np.inf, y_max=np.inf,
                 metrics=None, seed=None, online=False, warm_start=True):
        """
        Evaluate :class:`.ConformalRegressor` or 
        :class:`.ConformalPredictiveSystem`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           set of objects
        y : array-like of shape (n_samples,)
            correct labels
        confidence : float in range (0,1), default=0.95
            confidence level
        y_min : float or int, default=-numpy.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=numpy.inf
            maximum value to include in prediction intervals
        metrics : a string or a list of strings, default=list of all 
            metrics; for a learner wrapped with a conformal regressor
            these are "error", "eff_mean","eff_med", "ks_test", "time_fit",
            and "time_evaluate", while if wrapped with a conformal predictive
            system, the metrics also include "CRPS". 
        seed : int, default=None
           set random seed
        online : bool, default=False
           employ online calibration
        warm_start : bool, default=True
           extend original calibration set; used only if online=True
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics, where "error" is the 
            fraction of prediction intervals not containing the true label,
            "eff_mean" is the mean length of prediction intervals,
            "eff_med" is the median length of the prediction intervals,
            "CRPS" is the continuous ranked probability score,
            "ks_test" is the p-value for the Kolmogorov-Smirnov test of
            uniformity of predicted p-values, "time_fit" is the time taken
            to fit the conformal regressor/predictive system, and
            "time_evaluate" is the time taken for the evaluation         

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
        The metric ``CRPS`` is only available for batch evaluation, i.e., when
        ``online=False``, and will be ignored if the :class:`.WrapRegressor`
        object has been calibrated with the (default) option ``cps=False``,
        i.e., the learner is wrapped with a :class:`.ConformalRegressor`.

        Note
        ----
        The use of the metric ``CRPS`` may require a lot of memory, as a
        matrix is generated for which the number of elements is the product
        of the number of calibration and test objects, unless a Mondrian
        approach is employed; for the latter, this number is reduced by
        increasing the number of categories.

        Note
        ----
        The reported result for ``time_fit`` only considers fitting the
        conformal regressor or predictive system; not for fitting the
        learner.

        Note
        ----
        If a value for ``seed`` is given, it will take precedence over any
        ``seed`` value given when calling ``calibrate``.
        """
        if not self.calibrated and not online:
            raise RuntimeError(("batch evaluation requires a calibrated "
                                "conformal regressor"))
        if isinstance(y, pd.Series):
            y = y.values
        if seed is None:
            seed = self.seed
        if seed is not None:
            random_state = np.random.get_state()
            np.random.seed(seed)
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
            result = self.cr.evaluate(y_hat, y, sigmas, bins, confidence,
                                      y_min, y_max, metrics, seed,
                                      online, warm_start)
        else:
            result = self.cps.evaluate(y_hat, y, sigmas,
                                       bins, confidence,
                                       y_min, y_max, metrics, seed,
                                       online, warm_start)
        if seed is not None:
            np.random.set_state(random_state)
        return result
