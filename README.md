# crepes

### Conformal regressors and predictive systems

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

The main package `crepes` implements standard, normalized and Mondrian conformal
regressors and predictive systems, and allows you to use your own difficulty
estimates and Mondrian categories. There is also a separate module,
called `crepes.fillings`, which provides some standard options.

For (many) examples of how to use the package, see [this Jupyter
notebook](https://github.com/henrikbostrom/crepes/blob/main/crepes.ipynb).

See detailed documentation for the `crepes` package
[here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.html)
and for the `crepes.fillings` module 
[here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.fillings.html).

Author: Henrik Boström (bostromh@kth.se)
Copyright 2021 Henrik Boström
License: BSD 3 clause
