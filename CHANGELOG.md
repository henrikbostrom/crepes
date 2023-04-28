# Changelog

## v0.2.0 (28/04/2023)

### Features

- Modified `sigma_knn` to allow for calculating difficulty in three ways; using distances only, using standard deviation of the target and using the absolute residuals of the nearest neighbors.
- Added `sigma_knn_oob` in `crepes.fillings`
- Renamed the performance metric `efficiency` to `eff_mean` (mean efficiency) and added `eff_med` (median efficiency) to the `evaluate` method in `ConformalRegressor` and `ConformalPredictiveSystem`
- Added warning messages for the case that the calibration set  is too small for the specified confidence level or lower/higher percentiles [thanks to Geethen for highlighting this]
- Added examples in comments
- The documentation has been generated using Sphinx and resides in [crepes.readthedocs.io](https://crepes.readthedocs.io/en/latest/)

### Fixes

- Extended type checks to include NumPy floats and integers [thanks to patpizio for pointing this out]
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