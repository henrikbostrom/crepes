"""Conformal regressors and predictive systems (crepes) fillings

Helper class and functions to generate difficulty estimates, with and
without out-of-bag predictions, and Mondrian categories (bins) for
conformal regressors and conformal predictive systems.

Author: Henrik Boström (bostromh@kth.se)

Copyright 2023 Henrik Boström

License: BSD 3 clause

"""

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

class DifficultyEstimator():
    """
    A difficulty estimator outputs scores for objects to be used by 
    normalized conformal regressors and predictive systems.
    """
    
    def __repr__(self):
        if self.fitted and self.estimator_type == "knn":
            return (f"DifficultyEstimator(fitted={self.fitted}, "
                    f"type={self.estimator_type}, "
                    f"k={self.k}, "
                    f"target={self.target_type}, "
                    f"scaler={self.scaler}, "                    
                    f"beta={self.beta}, "
                    f"oob={self.oob}"             
                    ")")
        elif self.fitted and self.estimator_type == "variance":
            return (f"DifficultyEstimator(fitted={self.fitted}, "
                    f"type={self.estimator_type}, "
                    f"scaler={self.scaler}, "                    
                    f"beta={self.beta}, "
                    f"oob={self.oob}"             
                    ")")
        else:
            return f"DifficultyEstimator(fitted={self.fitted})"
    
    def fit(self, X=None, y=None, residuals=None, learner=None, k=25, 
            scaler=False, beta=0.01, oob=False):
        """
        Fit difficulty estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
           set of objects
        y : array-like of shape (n_values,), default=None
            target values
        residuals : array-like of shape (n_values,), default=None
            true target values - predicted values
        learner : an object with attribute ``learner.estimators_``, default=None
           an ensemble model where each model m in ``learner.estimators_`` has a
           method ``m.predict``
        k: int, default=25
           number of neighbors (used only if learner=None)
        scaler : bool, default=True
           use min-max-scaler on the difficulty estimates
        beta : int or float, default=0.01 
           value to add to the difficulty estimates (after scaling)
        oob : bool, default=False
           use out-of-bag estimation

        Returns
        -------
        self : object
            Fitted DifficultyEstimator.

        Examples
        --------
        Assuming that ``X_prop_train`` is a proper training set, 
        then a difficulty estimator using the distances to the k 
        nearest neighbors can be formed in the following way 
        (here using the default ``k=25``):
        
        .. code-block:: python

           from crepes.fillings import DifficultyEstimator

           de_knn_dist = DifficultyEstimator() 
           de_knn_dist.fit(X_prop_train)

        Assuming that ``y_prop_train`` is a vector with target values 
        for the proper training set, then a difficulty estimator using 
        standard deviation of the targets of the k nearest neighbors 
        is formed by: 

        .. code-block:: python

           de_knn_std = DifficultyEstimator() 
           de_knn_std.fit(X_prop_train, y=y_prop_train)

        Assuming that ``X_prop_res`` is a vector with residuals 
        for the proper training set, then a difficulty estimator using 
        the mean of the absolute residuals of the k nearest neighbors 
        is formed by: 

        .. code-block:: python

           de_knn_res = DifficultyEstimator() 
           de_knn_res.fit(X_prop_train, residuals=X_prop_res)

        Assuming that ``learner_prop`` is a trained model for which
        ``learner.estimators_`` is a collection of base models, each 
        implementing the ``predict`` method; this holds e.g., for 
        ``RandomForestRegressor``, a difficulty estimator using the variance
        of the predictions of the constituent models is formed by: 

        .. code-block:: python

           de_var = DifficultyEstimator() 
           de_var.fit(learner=learner_prop)

        The difficulty estimates may be normalized (using min-max scaling) by
        setting ``scaler=True``. It should be noted that this comes with a 
        computational cost; for estimators based on the k-nearest neighbor, 
        a leave-one-out protocol is employed to find the minimum and maximum 
        distances that are used by the scaler. This also requires that a set 
        of objects is provided for the variance-based approach (to allow for 
        finding the minimum and maximum values). Hence, if normalization is to
        be employed for the latter, objects have to be included:

        .. code-block:: python

           de_var = DifficultyEstimator() 
           de_var.fit(X_proper_train, learner=learner_prop, scaler=True)

        The ``DifficultyEstimator`` can also support the construction of conformal
        regressors and predictive systems that employ out-of-bag calibration. 
        For the k-nearest neighbor approaches, the difficulty of each object 
        in the provided training set will be computed using a leave-one-out 
        procedure, while for the variance-based approach the out-of-bag predictions 
        will be employed. This is enabled by setting ``oob=True`` when calling the 
        ``fit`` method, which also requires the (full) training set (``X_train``),
        and for the variance-based approach a corresponding trained model 
        (``learner_full``) to be provided: 

        .. code-block:: python

           de_var_oob = DifficultyEstimator() 
           de_var_oob.fit(X_train, learner=learner_full, scaler=True, oob=True)

        A small value (beta) is added to the difficulty estimates. The default 
        is ``beta=0.01``. In order to make the beta value have the same effect 
        across different estimators, you may consider normalizing the difficulty 
        estimates (using min-max scaling) by setting ``scaler=True``. Note that 
        beta is added after the normalization, which means that the range of
        scores after normalization will be [0+beta, 1+beta]. Below, we use 
        ``beta=0.001`` together with 10 neighbors (``k=10``):

        .. code-block:: python
        
           de_knn_mod = DifficultyEstimator() 
           de_knn_dist.fit(X_prop_train, k=10, beta=0.001, scaler=True)

        Note
        ----
        The use of out-of-bag calibration, as enabled by ``oob=True``, 
        does not come with the theoretical validity guarantees of the regular
        (inductive) conformal regressors and predictive systems, due to that 
        calibration and test instances are not handled in exactly the same way.
        """
        self.y = y
        self.residuals = residuals
        self.learner = learner
        self.k = k
        self.beta = beta
        self.scaler = scaler
        self.oob = oob
        if self.learner is None:
            self.estimator_type = "knn"
            if self.residuals is None:
                if self.y is None:
                    self.target_type = "none"
                else:
                    self.target_type = "labels"
            else:
                self.target_type = "residuals"
        else:
            self.estimator_type = "variance"
            try:
                self.learner.estimators_
            except:
                raise ValueError(
                    "learner is missing the attribute estimators_")
            if self.oob:
                try:
                    self.learner.estimators_[0].random_state
                except:
                    raise ValueError(
                        ("learner.estimators_ is missing the attribute "
                         "random_state"))
        if self.estimator_type == "knn":
            if X is None:
                raise ValueError("X=None is not allowed for k-nearest"
                                 " neighbor estimators")
            nn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1)
            nn_scaler = MinMaxScaler(clip=True)
            nn_scaler.fit(X)
            X_scaled = nn_scaler.transform(X)
            nn.fit(X_scaled)
            self.nn = nn
            self.nn_scaler = nn_scaler
            if self.oob or self.scaler:
                if self.target_type == "none":
                    distances, neighbor_indexes = nn.kneighbors(
                        return_distance=True)
                    sigmas = np.array([np.sum(distances[i])
                                       for i in range(len(distances))]) 
                elif self.target_type == "labels":
                    neighbor_indexes = nn.kneighbors(return_distance=False)
                    sigmas = np.array([np.std(y[indexes])
                                       for indexes in neighbor_indexes])
                else: # self.target_type == "residuals"
                    neighbor_indexes = nn.kneighbors(return_distance=False)
                    sigmas = np.array([np.mean(np.abs(residuals[indexes]))
                                       for indexes in neighbor_indexes])
                if self.scaler:
                    sigma_scaler = MinMaxScaler(clip=True)
                    sigma_scaler.fit(sigmas[:,None])
                    self.sigma_scaler = sigma_scaler
                    sigmas = self.sigma_scaler.transform(sigmas[:,None])[:,0]
                if self.oob:
                    self.sigmas = sigmas
        else: # self.estimator_type == "variance":
            if X is None and (self.oob or self.scaler):
                raise ValueError("X=None is allowed only if oob=False and "
                                 "scaler=False for variance estimator")
            if self.oob or self.scaler:
                predictions = np.array([model.predict(X)
                                        for model in self.learner.estimators_])
                oob_masks = np.array([
                    get_oob(self.learner.estimators_[i].random_state, len(X))
                    for i in range(len(self.learner.estimators_))])
                sigmas = np.array([np.var(predictions[oob_masks[:,i],i])
                                   for i in range(len(X))])
            if self.scaler:
                sigma_scaler = MinMaxScaler(clip=True)
                sigma_scaler.fit(sigmas[:,None])
                self.sigma_scaler = sigma_scaler
                sigmas = self.sigma_scaler.transform(sigmas[:,None])[:,0] 
            if self.oob:
                    self.sigmas = sigmas
        self.fitted = True
        return self

    def apply(self, X=None):
        """
        Apply difficulty estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
           set of objects

        Returns
        -------
        sigmas : array-like of shape (n_samples,)
            difficulty estimates 

        Examples
        --------
        Assuming ``de`` to be a fitted ``DifficultyEstimator``, i.e., for which
        the ``fit`` method has earlier been successfully called, then difficulty
        estimates for a set of objects ``X`` is obtained by:

        .. code-block:: python
        
           difficulty_estimates = de.apply(X)

        If ``de_oob`` is a ``DifficultyEstimator`` that has been fitted with
        the option ``oob=True`` and a training set, then a call to the ``apply`` 
        method without any objects will return the estimates for the training 
        set:

        .. code-block:: python
        
           oob_difficulty_estimates = de.apply()

        For a difficulty estimator employing any of the k-nearest neighbor 
        approaches, the above will return an estimate for the difficulty 
        of each object in the training set computed using a leave-one-out 
        procedure, while for the variance-based approach the out-of-bag 
        predictions will instead be used. 
        """
        if X is None:
            if not self.oob:
                raise ValueError("X=None is allowed only if oob=True")
            sigmas = self.sigmas
        elif self.estimator_type == "knn":
            X_scaled = self.nn_scaler.transform(X)
            if self.target_type == "none":
                distances, neighbor_indexes = self.nn.kneighbors(
                    X_scaled, return_distance=True)
                sigmas = np.array([np.sum(distances[i])
                                   for i in range(len(distances))])
            elif self.target_type == "labels":
                neighbor_indexes = self.nn.kneighbors(X_scaled,
                                                      return_distance=False)
                sigmas = np.array([np.std(self.y[indexes])
                                   for indexes in neighbor_indexes])
            else: # self.target_type == "residuals"
                neighbor_indexes = self.nn.kneighbors(X_scaled,
                                                      return_distance=False)
                sigmas = np.array([np.mean(np.abs(self.residuals[indexes]))
                                   for indexes in neighbor_indexes])
            if self.scaler:
                sigmas = self.sigma_scaler.transform(sigmas[:,None])[:,0]
        else: # self.estimator_type == "variance"
            if self.oob:
                predictions = np.array([model.predict(X)
                                        for model in self.learner.estimators_])
                oob_masks = np.array([
                    get_oob(self.learner.estimators_[i].random_state, len(X))
                    for i in range(len(self.learner.estimators_))])
                sigmas = np.array([np.var(predictions[oob_masks[:,i],i])
                                   for i in range(len(X))])
            else:    
                sigmas = np.var([model.predict(X) for
                                 model in self.learner.estimators_],
                            axis=0)
            if self.scaler:
                sigmas = self.sigma_scaler.transform(sigmas[:,None])[:,0]            
        return sigmas + self.beta

def get_oob(seed, n_samples):
    """
    Provides out-of-bag samples from a random seed and sample size.

    Parameters
    ----------
    seed : int
        random seed
    n_samples : int
        sample size
        
    Returns
    -------
    oob : array-like of shape (n_samples,)
        binary vector indicating which samples are out-of-bag and not 
    """
    return np.bincount(np.random.RandomState(seed).randint(0, n_samples,
                                                           n_samples),
                       minlength=n_samples) == 0

def binning(values=None, bins=10):
    """
    Provides bins for a set of values.

    Parameters
    ----------
    values : array-like of shape (n_samples,), default=None
        set of values
    bins : int or array-like of shape (n_bins,), default=10
        number of bins to use for equal-sized binning or threshold values 
        to use for binning
        
    Returns
    -------
    assigned_bins : array-like of shape (n_samples,)
        bins to which values have been assigned
    boundaries : array-like of shape (bins+1,)
        threshold values for the bins; the first is always -np.inf and
        the last is np.inf. Returned only if bins is an int.

    Examples
    --------
    Assuming that ``sigmas`` is a vector with difficulty estimates,
    then Mondrian categories (bins) can be formed by finding thresholds
    for 20 equal-sized bins by:

    .. code-block:: python

       from crepes.fillings import binning
        
       bins, bin_thresholds = binning(values=sigmas, bins=20)

    The above will result in that ``bins`` is assigned a vector
    of the same length as ``sigmas`` with label names (integers
    from 0 to 19), while ``bin_thresholds`` define the boundaries
    for the bins. The latter can be used to assign bin labels
    to another vector, e.g., ``sigmas_test``, by providing the thresholds 
    as input to ``binning``:

    .. code-block:: python
        
       test_bins  = binning(values=sigmas_test, bins=bin_thresholds)

    Here the output is just a vector ``test_bins`` with label names
    of the same length as ``sigmas_test``.

    Note
    ----
    A very small random number is added to each value when forming bins
    for the purpose of tie-breaking.
    """
    mod_values = values+np.random.rand(len(values))*1e-9
    # Adding a very small random number, which a.s. avoids ties
    # without affecting performance
    if type(bins) == int:
        assigned_bins, bin_boundaries = pd.qcut(mod_values,bins,
                                                labels=False,retbins=True,
                                                duplicates="drop",
                                                precision=12)
        bin_boundaries[0] = -np.inf
        bin_boundaries[-1] = np.inf
        return assigned_bins, bin_boundaries
    else:
        assigned_bins = pd.cut(mod_values,bins,labels=False,retbins=False)
        return assigned_bins
    
