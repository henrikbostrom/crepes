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

The main package implements standard, normalized and Mondrian conformal
regressors and predictive systems; see documentation
[here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.html).

The package allows for using any difficulty estimates and Mondrian categories
that you like, but a module, called `crepes.fillings` is provided with
some prebuilt options; see documentation
[here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.fillings.html).

For examples of how to use the package, see [this Jupyter
notebook](https://github.com/henrikbostrom/crepes/blob/main/crepes.ipynb).

Author: Henrik Boström (bostromh@kth.se)
Copyright 2021 Henrik Boström
License: BSD 3 clause
