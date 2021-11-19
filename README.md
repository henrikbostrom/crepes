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
called `crepes.fillings`, which provides some standard options for these.

For (many) examples of how to use the package, see [this Jupyter
notebook](https://github.com/henrikbostrom/crepes/blob/main/crepes.ipynb).

See detailed documentation for the `crepes` package
[here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.html)
and for the `crepes.fillings` module 
[here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.fillings.html).

#### References

<a id="1">[1]</a>
Vovk, V., Gammerman, A. and Shafer, G., 2005. Algorithmic learning in
a random world. Springer

<a id="2">[2]</a>
Papadopoulos, H., Proedrou, K., Vovk, V. and Gammerman, A., 2002,
August. Inductive confidence machines for regression. In European
Conference on Machine Learning (pp. 345-356). Springer, Berlin,
Heidelberg.

<a id="3">[3]</a>
Johansson, U., Boström, H., Löfström, T. and Linusson, H.,
2014. Regression conformal prediction with random forests. Machine
learning, 97(1-2), pp. 155-176.

<a id="4">[4]</a>
Boström, H., Linusson, H., Löfström, T. and Johansson, U.,
2017. Accelerating difficulty estimation for conformal regression
forests. Annals of Mathematics and Artificial Intelligence, 81(1-2),
pp.125-144.

<a id="5">[5]</a>
Boström, H. and Johansson, U., 2020. Mondrian conformal regressors. In
Conformal and Probabilistic Prediction and Applications. PMLR, 128, pp. 114-133.

<a id="6">[6]</a>
Vovk, V., Petej, I., Nouretdinov, I., Manokhin, V. and Gammerman, A.,
2020. Computationally efficient versions of conformal predictive
distributions. Neurocomputing, 397, pp.292-308.

<a id="7">[7]</a>

Boström, H., Johansson, U. and Löfström, T., 2021. Mondrian conformal
predictive distributions. In Conformal and Probabilistic Prediction
and Applications. PMLR, 152, pp. 24-38.

<a id="8">[8]</a>

<a id="9">[9]</a>

<a id="10">[10]</a>

---

Author: Henrik Boström (bostromh@kth.se)
Copyright 2021 Henrik Boström
License: BSD 3 clause
