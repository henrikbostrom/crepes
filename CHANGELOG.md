# Changelog
	
## v0.6.1 (21/08/2023)

### Features

- The function `margin` for computing non-conformity scores for conformal classifiers has been added to `crepes.extras`.
	
### Fixes

- Fixed a bug in the `DifficultyEstimator` class (in `crepes.extras`), which caused an error when trying to display a non-fitted object. Thanks to @tuvelofstrom for pointing this out.

- Fixed an error in the documentation for the function `hinge`. 
	
- The Jupyter notebooks `crepes_nb_wrap.ipynb` and `crepes_nb.ipynb` have been updated to illustrate the new `margin` function.
	
## v0.6.0 (28/06/2023)

### Features

- The classes `ConformalClassifier` and `WrapClassifier` have been added to `crepes`, allowing for generation of standard and Mondrian conformal classifiers, which produce p-values and prediction sets. The `calibrate` method of `WrapClassifier` allows for easily generating class-conditional conformal classifiers and using out-of-bag calibration. See [the documentation](https://crepes.readthedocs.io/en/latest/crepes.html) for the interface to objects of the class through the `calibrate`, `predict_p` and `predict_set` methods, in addition to the `fit`, `predict` and `predict_proba` methods of the wrapped learner. The method `evaluate` allows for evaluating the predictive performance using a set of standard metrics.

- The function `hinge` for computing non-conformity scores for conformal classifiers has been added to `crepes.extras`.	
	
### Fixes

- The class `Wrap` has changed name to `WrapRegressor` and the arguments to the `calibrate` method of this class have been changed to be in line with the `calibrate` method of `WrapClassifier`. 	

- The Jupyter notebooks `crepes_nb_wrap.ipynb` and `crepes_nb.ipynb` have been updated and extended
	
## v0.5.1 (22/06/2023)

### Fix

- Fixed a bug in the ``evaluate`` method of ``ConformalPredictiveSystem``, which caused an error when using ``CRPS`` as a single metric, i.e., when providing ``metrics=["CRPS"]`` as input. Thanks to @Zeeshan-Khaliq for pointing this out.
	
## v0.5.0 (02/06/2023)

### Feature

- The full cpds matrix is calculated only if requested to be output (``return_cpds=True``) by the ``predict`` method of ``ConformalPredictiveSystem`` or if the set of metrics include "CRPS" for the ``evaluate`` method. This allows large test and calibration sets to be handled without excessive use of memory in other cases. Thanks to @christopherjluke and @SebastianLeborg for highlighting and discussing the problem.

### Fixes

- Default values for mandatory arguments for the methods ``fit``, ``predict`` and ``evaluate`` methods of ``ConformalRegressor`` and ``ConformalPredictiveSystem``, as well as the function ``binning`` in ``crepes.extras``, are no longer provided

- ``y_min`` and ``y_max`` correctly inserted for all percentiles

- The ``evaluate`` method for ``ConformalPredictiveSystem`` fixed to work correctly even if CRPS not included in metrics, and if all test objects belong to the same Mondrian category
	
- Incorrect values for percentiles will render an error message
	
## v0.4.0 (16/05/2023)

### Feature

- The class `Wrap` has been added to `crepes`, allowing for easily extending the underlying learner with methods for forming, and making predictions with, conformal regressors and predictive systems. See [the documentation](https://crepes.readthedocs.io/en/latest/crepes.html) for the interface to objects of the class through the `calibrate`, `predict_int` and `predict_cps` methods, in addition to the `fit` and `predict` methods of the wrapped learner.

### Fixes

- A Jupyter notebook `crepes_nb_wrap.ipynb` has been added to the documentation to illustrate the use of the `Wrap` class.

- The output result array of a conformal predictive system is converted to a vector if the array contains one column only.

- The documentation has been updated and now includes links to classes and methods.
	
- `crepes.fillings` has been renamed to `crepes.extras`
	
## v0.3.0 (11/05/2023)

### Features

- The class `DifficultyEstimator` was added to `crepes.fillings`, incorporating functionality provided by the previous functions `sigma_knn`, `sigma_knn_oob`, `sigma_variance`, and `sigma_variance_oob`, which now are superfluous and have been removed from `crepes.fillings`. See [the documentation](https://crepes.readthedocs.io/en/latest/crepes.fillings.html) for the interface to objects of the class through the `fit` and `apply` methods.

- An option to normalize difficulty estimates, by providing `scaler=True` to the `fit` method of `DifficultyEstimator`, has been included.

- An option to install the package from [conda-forge](https://anaconda.org/conda-forge/crepes) has been included.

### Fixes

- The Jupyter notebook `crepes_nb.ipynb` has been updated to incorporate the above features

- The documentation of the [crepes package](https://crepes.readthedocs.io/en/latest/crepes.html) and the [crepes.fillings module](https://crepes.readthedocs.io/en/latest/crepes.fillings.html) has been updated with links to source code, additional examples and notes.
	
## v0.2.0 (28/04/2023)

### Features

- Modified `sigma_knn` to allow for calculating difficulty in three ways; using distances only, using standard deviation of the target and using the absolute residuals of the nearest neighbors.
- Added `sigma_knn_oob` in `crepes.fillings`
- Renamed the performance metric `efficiency` to `eff_mean` (mean efficiency) and added `eff_med` (median efficiency) to the `evaluate` method in `ConformalRegressor` and `ConformalPredictiveSystem`
- Added warning messages for the case that the calibration set  is too small for the specified confidence level or lower/higher percentiles [thanks to @Geethen for highlighting this]
- Added examples in comments
- The documentation has been generated using Sphinx and resides in [crepes.readthedocs.io](https://crepes.readthedocs.io/en/latest/)

### Fixes

- Extended type checks to include NumPy floats and integers [thanks to @patpizio for pointing this out]
- Corrected a bug in the assignment of min/max values for Mondrian conformal predictive systems
- The Jupyter notebook with examples has been updated, changed name to `crepes_nb.ipynb` and moved to the docs folder 
- Changed the default `k` to 25 in `sigma_knn`

## v0.1.0 (28/06/2022)

### Feature

- Added the parameter `cpds_by_bins` to the `predict` method of `ConformalPredictiveSystem`

### Fixes

- Comments updated and added for all classes and functions
- Line widths for code and comments adjusted to meet PEP 8
- Renamed some parameter names
- The function `binning` in `crepes.fillings` updated to produce the correct number of bins
- The Jupyter notebook `crepes.ipynb` has been updated and extended

## v0.0.1 (17/11/2021)

- First release of `crepes`
