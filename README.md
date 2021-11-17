# crepes

## Conformal regressors and predictive systems

Install with:

`pip install crepes`

The package implements *conformal regressors*, which transform point
predictions (produced by any underlying regression model) into
prediction intervals, for specified levels of confidence.

The package also implements *conformal predictive systems*, which
transform the point predictions into cumulative distributions
(conformal predictive distributions), e.g., allowing prediction
intervals to be extracted as well as probabilities for the target
value falling below specified thresholds.

The package implements standard, normalized and Mondrian conformal
regressors and predictive systems, and allows for using both
built-in and tailored difficulty estimates and Mondrian categories.

For examples of how to use the package, see [this Jupyter
notebook](crepes.ipynb).

Developed by Henrik Boström (bostromh@kth.se), 2021
