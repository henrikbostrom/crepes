"""Conformal regressors and predictive systems (crepes) fillings

Helper functions to generate difficulty estimates, with and without
out-of-bag predictions, and Mondrian categories for conformal regressors
and conformal predictive systems.

Author: Henrik Boström (bostromh@kth.se)

Copyright 2023 Henrik Boström

License: BSD 3 clause

"""

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

def sigma_variance(X=None, learner=None, beta=0.01):
    """
    Provides difficulty estimates for a set of objects
    using the variance of the predictions by a learner.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), default=None
        set of objects
    learner : an object with the attribute learner.estimators_, default=None
        an ensemble model where each model m in learner.estimators_ has a
        method m.predict
    beta : int or float, default=0.01 
        value to add to the difficulty estimates
        
    Returns
    -------
    sigmas : array-like of shape (n_samples,)
        difficulty estimates 
    """
    try:
        learner.estimators_
    except:
        raise ValueError("The learner is missing the attribute estimators_")
    return np.var([model.predict(X) for model in learner.estimators_],
                  axis=0) + beta

def sigma_variance_oob(X=None, learner=None, beta=0.01):
    """
    Provides difficulty estimates for a set of objects
    using the variance of the out-of-bag predictions by a learner.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), default=None
        set of objects
    learner : an object with the attribute learner.estimators_, default=None
        an ensemble model where each model m in learner.estimators_ has an
        attribute m.random_state
    beta : int or float, default=0.01 
        value to add to the difficulty estimates
        
    Returns
    -------
    sigmas : array-like of shape (n_samples,)
        difficulty estimates 
    """
    try:
        learner.estimators_
    except:
        raise ValueError("The learner is missing the attribute estimators_")
    try:
        learner.estimators_[0].random_state
    except:
        raise ValueError(("The learner.estimators_ is missing the attribute"
                          "random_state"))
    predictions = np.array([model.predict(X) for model in learner.estimators_])
    oob_masks = np.array([get_oob(learner.estimators_[i].random_state, len(X))
                          for i in range(len(learner.estimators_))])
    return np.array([np.var(predictions[oob_masks[:,i],i])
                     for i in range(len(X))]) + beta

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

def sigma_knn(X=None, X_ref=None, y_ref=None, residuals=None, k=25, beta=0.01):
    """
    Provides difficulty estimates for a set of objects using the 
    distance to, target standard deviation or absolute residuals of the 
    nearest neighbors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), default=None
        set of objects
    X_ref : array-like of shape (n_ref_samples, n_features), default=None
        set of reference objects
    y_ref : array-like of shape (n_ref_samples,), default=None
        target values for the reference objects X_ref
    residuals : array-like of shape (n_ref_samples,), default=None
        residuals for the reference objects X_ref
    k: int, default=25
        number of neighbors
    beta : int or float, default=0.01 
        value to add to the difficulty estimates
        
    Returns
    -------
    sigmas : array-like of shape (n_samples,)
        difficulty estimates
    """
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    scaler = MinMaxScaler(clip=True)
    scaler.fit(X_ref)
    X_ref_scaled = scaler.transform(X_ref)
    nn.fit(X_ref_scaled)
    X_scaled = scaler.transform(X)
    if residuals is None and y_ref is None:
        distances, neighbor_indexes = nn.kneighbors(X_scaled,
                                                    return_distance=True)
        sigmas = np.array([np.sum(distances[i])
                           for i in range(len(distances))]) + beta
    elif residuals is None:
        neighbor_indexes = nn.kneighbors(X_scaled, return_distance=False)
        sigmas = np.array([np.std(y_ref[indexes])
                           for indexes in neighbor_indexes]) + beta
    else:
        neighbor_indexes = nn.kneighbors(X_scaled, return_distance=False)
        sigmas = np.array([np.mean(np.abs(residuals[indexes]))
                           for indexes in neighbor_indexes]) + beta
    return sigmas

def sigma_knn_oob(X=None, y=None, residuals=None, k=25, beta=0.01):
    """
    Provides difficulty estimates for a set of objects using the 
    distance to, target standard deviation or absolute residuals of 
    the nearest neighbors.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features), default=None
        set of objects
    X_ref : array-like of shape (n_ref_samples, n_features), default=None
        set of reference objects
    y_ref : array-like of shape (n_ref_samples,), default=None
        target values for the reference objects X_ref
    residuals : array-like of shape (n_ref_samples,), default=None
        residuals for the reference objects X_ref
    k: int, default=25
        number of neighbors
    beta : int or float, default=0.01 
        value to add to the difficulty estimates
        
    Returns
    -------
    sigmas : array-like of shape (n_samples,)
        difficulty estimates
    """
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    scaler = MinMaxScaler(clip=True)
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    nn.fit(X_scaled)
    if residuals is None and y is None:
        distances, neighbor_indexes = nn.kneighbors(return_distance=True)
        sigmas = np.array([np.sum(distances[i])
                           for i in range(len(distances))]) + beta
    elif residuals is None:
        neighbor_indexes = nn.kneighbors(return_distance=False)
        sigmas = np.array([np.std(y[indexes])
                           for indexes in neighbor_indexes]) + beta
    else:
        neighbor_indexes = nn.kneighbors(return_distance=False)
        sigmas = np.array([np.mean(np.abs(residuals[indexes]))
                           for indexes in neighbor_indexes]) + beta
    return sigmas

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
    
