"""Conformal regressors and predictive systems (crepes)

Classes implementing conformal regressors and conformal predictive
systems, which transform point predictions into prediction intervals
and cumulative distribution functions, respectively.

Author: Henrik Boström (bostromh@kth.se)

Copyright 2022 Henrik Boström

License: BSD 3 clause
"""

import numpy as np
import pandas as pd
import time

__version__ = "0.1.0"

class ConformalPredictor():
    """
    Conformal Predictor.

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
    Conformal Regressor.

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
            actual values - predicted values
        sigmas: array-like of shape (n_values,), default=None
            difficulty estimates
        bins : array-like of shape (n_values,), default=None
            Mondrian categories

        Returns
        -------
        self : object
            Fitted ConformalRegressor.
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
        y_min : float or int, default=-np.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=np.inf
            maximum value to include in prediction intervals

        Returns
        -------
        intervals : ndarray of shape (n_values, 2)
            prediction intervals
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
                # If the no. of calibration examples is too small for
                # the chosen confidence level, then the intervals will
                # be of maximum size
        else:           
            bin_values, bin_alphas = self.alphas
            bin_indexes = [np.argwhere(bins == b).T[0] for b in bin_values]
            alpha_indexes = [int((1-confidence)*(len(bin_alphas[b])+1))-1
                             for b in range(len(bin_values))]
            bin_alpha = [bin_alphas[b][alpha_indexes[b]]
                         if alpha_indexes[b]>=0 else np.inf
                         for b in range(len(bin_values))]
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
        y_min : float or int, default=-np.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=np.inf
            maximum value to include in prediction intervals
        metrics : a string or a list of strings, 
                  default=list of all metrics, i.e., 
                  ["error", "efficiency", "time_fit", "time_evaluate"]
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics
        """

        tic = time.time()
        if metrics is None:
            metrics = ["error","efficiency","time_fit","time_evaluate"]
        test_results = {}
        intervals = self.predict(y_hat, sigmas, bins, confidence, y_min, y_max)
        if "error" in metrics:
            test_results["error"] = 1-np.mean(
                np.logical_and(intervals[:,0]<=y,y<=intervals[:,1]))
        if "efficiency" in metrics:            
            test_results["efficiency"] = np.mean(intervals[:,1]-intervals[:,0])
        if "time_fit" in metrics:
            test_results["time_fit"] = self.time_fit
        toc = time.time()
        self.time_evaluate = toc-tic
        if "time_evaluate" in metrics:
            test_results["time_evaluate"] = self.time_evaluate
        return test_results
    
class ConformalPredictiveSystem(ConformalPredictor):
    """
    Conformal Predictive System.

    A conformal predictive system transforms point predictions 
    (regression values) into cumulative distributions (conformal 
    predictive distributions).
    
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
            values for which cumulative probabilities should be returned
        lower_percentiles : array-like of shape (l_values,), default=None
            percentiles for which a lower value will be output 
            in case a percentile lies between two values
            (similar to `interpolation="lower"` in `numpy.percentile`)
        higher_percentiles : array-like of shape (h_values,), default=None
            percentiles for which a higher value will be output 
            in case a percentile lies between two values
            (similar to `interpolation="higher"` in `numpy.percentile`)
        y_min : float or int, default=-np.inf
            The minimum value to include in prediction intervals.
        y_max : float or int, default=np.inf
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
        if type(lower_percentiles) in [int, float]:
            lower_percentiles = [lower_percentiles]
        if type(higher_percentiles) in [int, float]:
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
            if type(y) in [int, float]:
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
                    lower_indexes = [int(lower_percentile/100 \
                                         * (len(self.alphas)+1))-1
                                     for lower_percentile in lower_percentiles]
                    result[:,no_prec_result_cols:no_prec_result_cols \
                           + len(lower_percentiles)] = cpds[:,lower_indexes]
                    y_min_columns = [no_prec_result_cols+i
                                     for i in range(len(lower_indexes))
                                     if lower_indexes[i]<0]
                    result[:,y_min_columns] = y_min
                else:
                    for b in range(len(bin_values)):
                        lower_indexes = [int(lower_percentile/100 \
                                             * (len(bin_alphas[b])+1))-1
                                         for lower_percentile
                                         in lower_percentiles]                
                        result[bin_indexes[b],
                               no_prec_result_cols:no_prec_result_cols \
                               + len(lower_indexes)] = cpds[b][:,lower_indexes]
                    y_min_columns = [no_prec_result_cols+i
                                     for i in range(len(lower_indexes))
                                     if lower_indexes[i]<0]
                    result[:,y_min_columns] = y_min                    
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
                    higher_indexes[too_high_indexes] = len(self.alphas)-1
                    result[:,no_prec_result_cols:no_prec_result_cols \
                           + len(higher_indexes)] = cpds[:,higher_indexes]
                    result[:,no_prec_result_cols+too_high_indexes] = y_max
                else:
                    for b in range(len(bin_values)):
                        higher_indexes = [
                            int(np.ceil(higher_percentile/100 \
                                        * (len(bin_alphas[b])+1)))-1
                            for higher_percentile in higher_percentiles]
                        result[bin_indexes[b],
                               no_prec_result_cols:no_prec_result_cols \
                               + len(higher_indexes)] = cpds[b]\
                                   [:,higher_indexes]
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
        y_min : float or int, default=-np.inf
            minimum value to include in prediction intervals
        y_max : float or int, default=np.inf
            maximum value to include in prediction intervals
        metrics : a string or a list of strings, default=list of all 
            metrics; ["error", "efficiency", "CRPS", "time_fit", 
                      "time_evaluate"]
        
        Returns
        -------
        results : dictionary with a key for each selected metric 
            estimated performance using the metrics
        """

        tic = time.time()
        if metrics is None:
            metrics = ["error","efficiency","CRPS","time_fit","time_evaluate"]
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
        if "efficiency" in metrics:            
            test_results["efficiency"] = np.mean(intervals[:,1]-intervals[:,0])
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
