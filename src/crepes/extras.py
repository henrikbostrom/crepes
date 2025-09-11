"""Conformal classifiers, regressors, and predictive systems (crepes) extras

Functions for generating non-conformity scores and Mondrian categories
(bins), and classes for generating difficulty estimates and Mondrian 
categorizers, with and without out-of-bag predictions.

Author: Henrik Boström (bostromh@kth.se)

Copyright 2025 Henrik Boström

License: BSD 3 clause

"""

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

def hinge(X_prob, classes=None, y=None):
    """
    Computes non-conformity scores for conformal classifiers.

    Parameters
    ----------
    X_prob : array-like of shape (n_samples, n_classes)
        predicted class probabilities
    classes : array-like of shape (n_classes,), default=None
        class names
    y : array-like of shape (n_samples,), default=None
        correct target values

    Returns
    -------
    scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
        non-conformity scores. The shape is (n_samples, n_classes)
        if classes and y are None.

    Examples
    --------
    Assuming that ``X_prob`` is an array with predicted probabilities and
    ``classes`` and ``y`` are vectors with the class names (in order) and
    correct class labels, respectively, the non-conformity scores are generated
    by:

    .. code-block:: python

       from crepes.extras import hinge
        
       alphas = hinge(X_prob, classes, y)

    The above results in that ``alphas`` is assigned a vector of the same length
    as ``X_prob`` with a non-conformity score for each object, here 
    defined as 1 minus the predicted probability for the correct class label.
    These scores can be used when fitting a :class:`.ConformalClassifier` or
    calibrating a :class:`.WrapClassifier`. Non-conformity scores for test 
    objects, for which ``y`` is not known, can be obtained from the corresponding
    predicted probabilities (``X_prob_test``) by:

    .. code-block:: python

       alphas_test = hinge(X_prob_test)

    The above results in that ``alphas_test`` is assigned an array of the same
    shape as ``X_prob_test`` with non-conformity scores for each class in the 
    columns for each test object.
    """
    if y is not None:
        if isinstance(y, pd.Series):
            y = y.values
        class_indexes = np.array(
            [np.argwhere(classes == y[i])[0][0] for i in range(len(y))])
        result = 1-X_prob[np.arange(len(y)),class_indexes]
    else:
        result = 1-X_prob
    return result

def margin(X_prob, classes=None, y=None):
    """Computes non-conformity scores for conformal classifiers.

    Parameters
    ----------
    X_prob : array-like of shape (n_samples, n_classes)
        predicted class probabilities
    classes : array-like of shape (n_classes,), default=None
        class names
    y : array-like of shape (n_samples,), default=None
        correct target values

    Returns
    -------
    scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
        non-conformity scores. The shape is (n_samples, n_classes)
        if classes and y are None.

    Examples
    --------
    Assuming that ``X_prob`` is an array with predicted probabilities and
    ``classes`` and ``y`` are vectors with the class names (in order) and
    correct class labels, respectively, the non-conformity scores are generated 
    by:

    .. code-block:: python

       from crepes.extras import margin
        
       alphas = margin(X_prob, classes, y)

    The above results in that ``alphas`` is assigned a vector of the same length 
    as ``X_prob`` with a non-conformity score for each object, here
    defined as the highest predicted probability for a non-correct class label 
    minus the predicted probability for the correct class label. These scores can
    be used when fitting a :class:`.ConformalClassifier` or calibrating a 
    :class:`.WrapClassifier`. Non-conformity scores for test objects, for which 
    ``y`` is not known, can be obtained from the corresponding predicted 
    probabilities (``X_prob_test``) by:

    .. code-block:: python

       alphas_test = margin(X_prob_test)

    The above results in that ``alphas_test`` is assigned an array of the same
    shape as ``X_prob_test`` with non-conformity scores for each class in the 
    columns for each test object.

    """
    if y is not None:
        if isinstance(y, pd.Series):
            y = y.values
        class_indexes = np.array(
            [np.argwhere(classes == y[i])[0][0] for i in range(len(y))])
        result = np.array([
            (np.max(X_prob[i, [j != class_indexes[i]
                               for j in range(X_prob.shape[1])]])
             - X_prob[i, class_indexes[i]]) for i in range(len(X_prob))])
    else:
        result = np.array([
            [(np.max(X_prob[i, [j != c for j in range(X_prob.shape[1])]])
             - X_prob[i, c]) for c in range(X_prob.shape[1])]
            for i in range(len(X_prob))])
    return result

def binning(values, bins=10):
    """
    Provides bins for a set of values.

    Parameters
    ----------
    values : array-like of shape (n_samples,)
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

       from crepes.extras import binning
        
       bins, bin_thresholds = binning(sigmas, bins=20)

    The above results in that ``bins`` is assigned a vector
    of the same length as ``sigmas`` with label names (integers
    from 0 to 19), while ``bin_thresholds`` define the boundaries
    for the bins. The latter can be used to assign bin labels
    to another vector, e.g., ``sigmas_test``, by providing the thresholds 
    as input to :meth:`binning`:

    .. code-block:: python
        
       test_bins  = binning(sigmas_test, bins=bin_thresholds)

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
    if isinstance(bins, int):
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

class MondrianCategorizer():
    """
    A MondrianCategorizer outputs categories for objects to be used by 
    Mondrian conformal classifiers, regressors and predictive systems.
    """
    
    def __init__(self):
        self.fitted = False
        self.f = None
        self.de = None
        self.learner = None
        self.oob = False
        self.bin_thresholds = None

    def __repr__(self):
        if self.f is not None:
            return (f"MondrianCategorizer(fitted={self.fitted}, "
                    f"f={self.f.__name__}, "
                    f"no_bins={len(self.bin_thresholds)-1})")
        elif self.de is not None:
            return (f"MondrianCategorizer(fitted={self.fitted}, "
                    f"de={self.de}, no_bins={len(self.bin_thresholds)-1})")
        elif self.learner is not None:
            return (f"MondrianCategorizer(fitted={self.fitted}, "
                    f"learner={self.learner}, "
                    f"no_bins={len(self.bin_thresholds)-1})")
        else:
            return f"MondrianCategorizer(fitted={self.fitted})"
    
    def fit(self, X=None, f=None, de=None, learner=None, no_bins=10, oob=False):
        """
        Fit Mondrian categorizer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
            set of objects
        f : function which given an array-like of shape (n_samples, n_features)
            should return a vector of shape (n_samples,) of type int or float, 
            default=None
            function used to compute Mondrian categories
        de : a :class:`.DifficultyEstimator`, default=None
            a fitted difficulty estimator (used only if f is not None)
        learner : an object with the method ``learner.predict``, default=None
            a fitted regression model (used only if de and f are not None) 
        no_bins : int, default=10
           no. of Mondrian categories
        oob : bool, default=False
           use out-of-bag estimation (not used if f is not None)

        Returns
        -------
        self : object
            Fitted MondrianCategorizer.

        Examples
        --------
        Assuming that ``X_train`` is an array of shape (n_samples, n_features)
        and ``get_values`` is a function that given ``X_train`` returns a vector
        of values of shape (n_samples,), then a Mondrian categorizer can
        be formed in the following way, where the boundaries for the Mondrian
        categories are found by partitioning the values in the vector into five
        equal-sized bins:
        
        .. code-block:: python

           from crepes.extras import MondrianCategorizer

           mc = MondrianCategorizer()
           mc.fit(X, f=get_values, no_bins=5)
        """
        if f is not None:
            if X is not None:
                scores = f(X)
                bins, bin_thresholds = binning(scores, bins=no_bins)
                self.bin_thresholds = bin_thresholds
            else:
                raise ValueError("X must be provided since f is not None")
            self.f = f
        elif de is not None:
            if oob:
                scores = de.apply()
                self.oob = True
            else:
                if X is not None:
                    scores = de.apply(X)
                else:
                    raise ValueError(("X must be provided since de is not None"
                                      "and oob=False"))
            self.de = de
            bins, bin_thresholds = binning(scores, bins=no_bins)
            self.bin_thresholds = bin_thresholds    
        elif learner is not None:
            if oob:
                scores = learner.oob_prediction_
                self.oob = True
            else:
                if X is not None:
                    scores = learner.predict(X)
                else:
                    raise ValueError(("X must be provided since learner is "
                                      "not None and oob=False"))
            self.learner = learner
            bins, bin_thresholds = binning(scores, bins=no_bins)
            self.bin_thresholds = bin_thresholds
        else:
            raise ValueError("One of f, de, and learner must not be None")
        self.fitted = True
        self.fitted_ = True
        return self

    def apply(self, X):
        """
        Apply Mondrian categorizer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
           set of objects

        Returns
        -------
        bins : array-like of shape (n_samples,)
            Mondrian categories 

        Examples
        --------
        Assuming ``mc`` to be a fitted :class:`.MondrianCategorizer`, i.e., for
        which :meth:`.fit` has earlier been called, Mondrian categories for a
        set of objects ``X`` is obtained by:

        .. code-block:: python
        
           categories = mc.apply(X)

        Note
        ----
        The array used when calling :meth:`.fit` must have the same number of
        columns (``n_features``) as the array used as input to :meth:`.apply`.
        """
        if self.f is not None:
            if self.bin_thresholds is None:
                bins = self.f(X)
            else:
                scores = self.f(X)
                bins = binning(scores, bins=self.bin_thresholds)
        elif self.de is not None:
            scores = self.de.apply(X)
            bins = binning(scores, bins=self.bin_thresholds)
        elif self.learner is not None:
            if self.oob:
                predictions = np.array([model.predict(X)
                                        for model in self.learner.estimators_])
                oob_masks = np.array([
                    get_oob(self.learner.estimators_[i].random_state, len(X))
                    for i in range(len(self.learner.estimators_))])
                scores = np.array([np.mean(predictions[oob_masks[:,i],i])
                                   for i in range(len(X))])
            else:
                scores = learner.predict(X)
            bins = binning(scores, bins=self.bin_thresholds)
        return bins
                
class DifficultyEstimator():
    """
    A difficulty estimator outputs scores for objects to be used by 
    normalized conformal regressors and predictive systems.
    """
    
    def __init__(self):
        self.fitted = False
        self.estimator_type = None

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
    
    def fit(self, X=None, f=None, y=None, residuals=None, learner=None,
            k=25, scaler=False, beta=0.01, oob=False):
        """
        Fit difficulty estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), default=None
           set of objects
        f : function which given an array-like of shape (n_samples, n_features)
            should return a vector of shape (n_samples,) of type int or float, 
            default=None
            function used to compute difficulty estimates
        y : array-like of shape (n_samples,), default=None
            target values
        residuals : array-like of shape (n_samples,), default=None
            true target values - predicted values
        learner : an object with attribute ``learner.estimators_``, default=None
           an ensemble model where each model m in ``learner.estimators_`` has
           a method ``m.predict`` (used only if f=None)
        k: int, default=25
           number of neighbors (used only if f=None and learner=None)
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

           from crepes.extras import DifficultyEstimator

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

        Difficulty estimates may also be computed by an externally defined
        function. Assuming that ``diff_model`` is a fitted regression model,
        for which the ``predict`` method gives estimates of the absolute
        error for the objects in ``X_proper_train``, then normalized difficulty
        estimates can be obtained from the following difficulty estimator:

        .. code-block:: python

           de_mod = DifficultyEstimator() 
           de_mod.fit(X_proper_train, f=diff_model.predict, scaler=True)
        
        The :class:`.DifficultyEstimator` can also support the construction of 
        conformal regressors and predictive systems that employ out-of-bag 
        calibration. For the k-nearest neighbor approaches, the difficulty of
        each object in the provided training set will be computed using a 
        leave-one-out procedure, while for the variance-based approach the 
        out-of-bag predictions will be employed. This is enabled by setting 
        ``oob=True`` when calling the :meth:`.fit` method, which also requires 
        the (full) training set (``X_train``), and for the variance-based 
        approach a corresponding trained model (``learner_full``) to be 
        provided: 

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
           de_knn_mod.fit(X_prop_train, k=10, beta=0.001, scaler=True)

        Note
        ----
        The use of out-of-bag calibration, as enabled by ``oob=True``, 
        does not come with the theoretical validity guarantees of the regular
        (inductive) conformal regressors and predictive systems, due to that 
        calibration and test instances are not handled in exactly the same way.
        """
        self.f = f
        if isinstance(y, pd.Series):
            y = y.values
        self.y = y
        self.residuals = residuals
        self.learner = learner
        self.k = k
        self.beta = beta
        self.scaler = scaler
        self.oob = oob
        if self.f is not None:
            self.estimator_type = "function"
        elif self.learner is not None:
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
        else:
            self.estimator_type = "knn"
            if self.residuals is None:
                if self.y is None:
                    self.target_type = "none"
                else:
                    self.target_type = "labels"
            else:
                self.target_type = "residuals"
            if X is None:
                raise ValueError("X=None is not allowed for k-nearest"
                                 " neighbor estimators")
            
        if self.estimator_type == "function":
            if X is None and self.scaler:
                raise ValueError("X=None is allowed only if scaler=False"
                                 " for function estimators")
            if self.oob:
                raise ValueError("oob=True is not allowed for function"
                                 " estimators")
            if self.scaler:
                sigmas = self.f(X)
                sigma_scaler = MinMaxScaler(clip=True)
                sigma_scaler.fit(sigmas[:,None])
                self.sigma_scaler = sigma_scaler

        if self.estimator_type == "variance":
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

        if self.estimator_type == "knn":
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
                    
        self.fitted = True
        self.fitted_ = True
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
        Assuming ``de`` to be a fitted :class:`.DifficultyEstimator`, i.e., for
        which :meth:`.fit` has earlier been called, difficulty estimates for a
        set of objects ``X`` is obtained by:

        .. code-block:: python
        
           difficulty_estimates = de.apply(X)

        If ``de_oob`` is a :class:`.DifficultyEstimator` that has been fitted 
        with the option ``oob=True`` and a training set, then a call to 
        :meth:`.apply` without any objects will return the estimates for the 
        training set:

        .. code-block:: python
        
           oob_difficulty_estimates = de_oob.apply()

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
        elif self.estimator_type == "variance":
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
        else: # self.estimator_type == "function"
            sigmas = self.f(X)
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
