"""Conformal regressors and predictive systems (crepes) fillings

Helper functions to generate residuals and sigmas, with and without
out-of-bag calibration, for conformal regressors and conformal
predictive systems.

Author : Henrik Boström (bostromh@kth.se)

Copyright 2021 Henrik Boström

License: BSD 3 clause

"""

# To do:
#
# - "min-bin-size" as alternative to "no_bins" for the helper function "binning"
# - error messages
# - commenting and documentation 

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

def sigma_variance(X=None, learner=None, beta=0.01):
    try:
        learner.estimators_
    except:
        raise ValueError("The learner is missing the attribute estimators_")
    return np.var([model.predict(X) for model in learner.estimators_], axis=0) + beta

def sigma_variance_oob(X=None, learner=None, beta=0.01):
    try:
        learner.estimators_
    except:
        raise ValueError("The learner is missing the attribute estimators_")
    try:
        learner.estimators_[0].random_state
    except:
        raise ValueError("The learner.estimators_ is missing the attribute random_state")
    predictions = np.array([model.predict(X) for model in learner.estimators_])
    oob_masks = np.array([get_oob_mask(learner.estimators_[i].random_state,len(X))
                          for i in range(len(learner.estimators_))])
    return np.array([np.var(predictions[oob_masks[:,i],i]) for i in range(len(X))]) + beta

def get_oob_mask(seed, n):
    return np.bincount(np.random.RandomState(seed).randint(0, n, n),
                       minlength=n) == 0

def sigma_knn(X=None, residuals=None, X_test=None, k=5, beta=0.01):
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    scaler = MinMaxScaler(clip=True)
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    nn.fit(X_scaled)
    if X_test is None:
        neighbor_indexes = nn.kneighbors(return_distance=False)
    else:
        X_test_scaled = scaler.transform(X_test)
        neighbor_indexes = nn.kneighbors(X_test_scaled, return_distance=False)
    return np.array([np.mean(np.abs(residuals[indexes])) for indexes in neighbor_indexes]) + beta

def binning(values=None, bins=10):
    mod_values = values+np.random.rand(len(values))*1e-9 # Adding a very small random number, which a.s. avoids ties
                                                         # without affecting performance
    if type(bins) == int:
        assigned_bins, bin_boundaries = pd.qcut(mod_values,bins+1,labels=False,retbins=True,duplicates="drop",precision=12)
        bin_boundaries[0] = -np.inf
        bin_boundaries[-1] = np.inf
        return assigned_bins, bin_boundaries
    else:
        assigned_bins = pd.cut(mod_values,bins,labels=False,retbins=False)
        return assigned_bins
