# crepes

### Conformal regressors and predictive systems

`crepes` is a Python package for generating *conformal regressors*, which transform point
predictions produced by any underlying regression model into prediction intervals for specified levels of confidence. The package also implements *conformal predictive systems*, which
transform the point predictions into cumulative distribution functions.

The `crepes` package implements standard, normalized and Mondrian conformal
regressors and predictive systems. While the package allows you to use your own difficulty
estimates and Mondrian categories, there is also a separate module,
called `crepes.fillings`, which provides some standard options for these.

#### Installation

Install with: `pip install crepes`

#### Documentation

For documentation of the `crepes` package, see 
[here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.html).

For documentation of the `crepes.fillings` module, see
[here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.fillings.html).

For examples of how to use the package and module, see [this Jupyter
notebook](https://github.com/henrikbostrom/crepes/blob/main/crepes.ipynb).

#### References

<a id="1">[1]</a>
Vovk, V., Gammerman, A. and Shafer, G., 2005. Algorithmic learning in
a random world. Springer
[Link](https://link.springer.com/book/10.1007/b106715)

<a id="2">[2]</a> Papadopoulos, H., Proedrou, K., Vovk, V. and
Gammerman, A., 2002. Inductive confidence machines for regression. European Conference on Machine Learning, pp. 345-356.
[Link](https://link.springer.com/chapter/10.1007/3-540-36755-1_29)

<a id="3">[3]</a>
Johansson, U., Boström, H., Löfström, T. and Linusson, H.,
2014. Regression conformal prediction with random forests. Machine learning, 97(1-2), pp. 155-176.
[Link](https://link.springer.com/article/10.1007/s10994-014-5453-0)

<a id="4">[4]</a>
Boström, H., Linusson, H., Löfström, T. and Johansson, U.,
2017. Accelerating difficulty estimation for conformal regression
forests. Annals of Mathematics and Artificial Intelligence, 81(1-2), pp.125-144.
[Link](https://link.springer.com/article/10.1007/s10472-017-9539-9)

<a id="5">[5]</a>
Boström, H. and Johansson, U., 2020. Mondrian conformal regressors. In Conformal and Probabilistic Prediction and Applications. PMLR, 128, pp. 114-133.
[Link](https://proceedings.mlr.press/v128/bostrom20a.html)

<a id="6">[6]</a>
Vovk, V., Petej, I., Nouretdinov, I., Manokhin, V. and Gammerman, A.,
2020. Computationally efficient versions of conformal predictive distributions. Neurocomputing, 397, pp.292-308.
[Link](https://www.aminer.org/pub/5e09aac9df1a9c0c416c9b70/computationally-efficient-versions-of-conformal-predictive-distributions)

<a id="7">[7]</a>
Boström, H., Johansson, U. and Löfström, T., 2021. Mondrian conformal
predictive distributions. In Conformal and Probabilistic Prediction and Applications. PMLR, 152, pp. 24-38.
[Link](https://proceedings.mlr.press/v152/bostrom21a.html)

- - -

Author: Henrik Boström (bostromh@kth.se)

Copyright 2021 Henrik Boström

License: BSD 3 clause
