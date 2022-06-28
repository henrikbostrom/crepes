# `crepes`: Conformal Regressors and Predictive Systems

`crepes` is a Python package for generating *conformal regressors*, which transform point predictions of any underlying regression model into prediction intervals for specified levels of confidence. The package also implements *conformal predictive systems*, which transform the point predictions into cumulative distribution functions.

The `crepes` package implements standard, normalized and Mondrian conformal regressors and predictive systems. While the package allows you to use your own difficulty estimates and Mondrian categories, there is also a separate module, called `crepes.fillings`, which provides some standard options for these.

## Installation

Install with: `pip install crepes`

## Quick start

### Conformal regressors

We first import the main class and two helper functions:

```python
from crepes import ConformalRegressor
from crepes.fillings import sigma_knn, binning
```

We will illustrate the above using a dataset from www.openml.org and a `RandomForestRegressor` from `sklearn`:

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

dataset = fetch_openml(name="house_sales", version=3)
X = dataset.data.values.astype(float)
y = dataset.target.values.astype(float)
```

We now first split the dataset into a training and a test set, and then further split the training set into a proper training set and a calibration set. Finally, we fit a random forest to the proper training set and apply it to obtain point predictions for the test set (nothing new here).

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                            test_size=0.25)

learner = RandomForestRegressor() 
learner.fit(X_prop_train, y_prop_train)
y_hat_test = learner.predict(X_test)
```

Now, we will make use of the calibration set. We apply the model also to the calibration objects and calculate the residuals.
```python
y_hat_cal = learner.predict(X_cal)
residuals_cal = y_cal - y_hat_cal
```

The latter is just what we need to fit a standard conformal regressor:
```python
cr_std = ConformalRegressor()
cr_std.fit(residuals=residuals_cal)
```

We can now apply the conformal regressor to get prediction intervals for the test objects, using the point predictions as input, where the probability of not including the correct target in an interval is 1-confidence:
```python
std_intervals = cr_std.predict(y_hat=y_hat_test, confidence=0.99)
```

The output is a NumPy array, specifying the lower and upper bound of each interval:

```numpy
array([[-353379.  ,  939231.  ],
       [-251874.3 , 1040735.7 ],
       [-138329.5 , 1154280.5 ],
       ...,
       [-389128.68,  903481.32],
       [-313003.  ,  979607.  ],
       [ -90551.53, 1202058.47]])
```

We may request that the intervals are cut to exclude impossible values, in this case below 0, and if we also rely on the default confidence level (0.95), the output intervals will be a bit tighter:

```python
intervals_std = cr_std.predict(y_hat=y_hat_test, y_min=0)
```

```numpy
array([[  7576.18, 578275.82],
       [109080.88, 679780.52],
       [222625.68, 793325.32],
       ...,
       [     0.  , 542526.14],
       [ 47952.18, 618651.82],
       [270403.65, 841103.29]])
```

The above intervals are not normalized, i.e., they are all of the same size (at least before they are cut). We could make the intervals more informative through normalization using difficulty estimates; more difficult instances will be assigned wider intervals.

We will here use the helper function `sigma_knn` for this purpose. It estimates the difficulty by the mean absolute errors of the k (default `k=5`) nearest neighbors to each instance in the calibration set. A small value (beta) is added to the estimates, which may be given through a (named) argument to the function; below we just use the default, i.e., `beta=0.01`.

```python
sigmas_cal = sigma_knn(X=X_cal, residuals=residuals_cal)
```

The difficulty estimates and residuals of the calibration examples can now be used to form a normalized conformal regressor:

```python
cr_norm = ConformalRegressor()
cr_norm.fit(residuals=residuals_cal, sigmas=sigmas_cal)
```

To generate prediction intervals for the test set using the normalized conformal regressor, we need difficulty estimates for the test set too, which we get using the calibration objects and residuals. 

```python
sigmas_test = sigma_knn(X=X_cal, residuals=residuals_cal, X_test=X_test)
```

Now we can obtain the prediction intervals, using the point predictions and difficulty estimates for the test set:

```python
intervals_norm = cr_norm.predict(y_hat=y_hat_test, sigmas=sigmas_test, 
                                 y_min=0)
```

```numpy
array([[     0.        , 645527.3140099 ],
       [100552.5573358 , 688308.8426642 ],
       [206605.7263972 , 809345.2736028 ],
       ...,
       [ 55388.60029434, 458964.03970566],
       [252094.62400964, 414509.37599036],
       [305546.225071  , 805960.714929  ]])
```

Depending on the employed difficulty estimator, the normalized intervals may sometimes be unreasonably large, in the sense that they may be several times larger than any previously observed error. Moreover, if the difficulty estimator is not very informative, e.g., completely random, the varying interval sizes may give a false impression of that we can expect lower prediction errors for instances with tighter intervals. Ideally, a difficulty estimator providing little or no informatoion on the expected error should instead lead to more uniformly distributed interval sizes.

A Mondrian conformal regressor can be used to address these problems, by dividing the object space into non-overlapping so-called Mondrian categories, and forming a (standard) conformal regressor for each category. The category membership of the objects can be provided as an additional argument, named `bins`, for the `fit` method.

Here we will use the helper function `binning` to form Mondrian categories by equal-sized binning of the difficulty estimates; the function will return labels for the calibration objects as well as thresholds for the bins, which are later to be used when binning the test objects:

```python
bins_cal, bin_thresholds = binning(values=sigmas_cal, bins=20)
```

A Mondrian conformal regressor is obtained from the residuals and Mondrian category labels:

```python
cr_mond = ConformalRegressor()
cr_mond.fit(residuals=residuals_cal, bins=bins_cal)
```

In order to use the Mondrian conformal regressor on the test objects, we need to get the labels of the Mondrian categories for these:

```python
bins_test = binning(values=sigmas_test, bins=bin_thresholds)
```

Now we have everything we need to get the prediction intervals:

```python
intervals_mond = cr_mond.predict(y_hat=y_hat_test, bins=bins_test, y_min=0)
```

```numpy
array([[     0.  , 592782.5 ],
       [146648.15, 642213.25],
       [260192.95, 755758.05],
       ...,
       [ 38332.66, 476019.98],
       [198148.5 , 468455.5 ],
       [329931.17, 781575.77]])
```

### Conformal predictive systems

The interface to a `ConformalPredictiveSystem` is very similar to that of a conformal regressor; by providing just the residuals, we get a standard conformal predictive system, by providing also difficulty estimates, we get a normalized conformal predictive system and by providing labels for Mondrian categories (bins), we get a Mondrian conformal predictive system.

The main difference to conformal regressors concerns the output; instead of prediction intervals, conformal predictive systems produce complete cumulative distribution functions (conformal predictive distributions). From these we can generate prediction intervals, but we can also obtain calibrated point predictions, as well as p values for given target values.

Let us fit a Mondrian normalized conformal predictive system, using the above residuals and difficulty estimates (sigmas), and where the Mondrian categories (bins) are formed using the point predictions for the calibration set:

```python
from crepes import ConformalPredictiveSystem

bins_cal, bin_thresholds = binning(values=y_hat_cal, bins=5)

cps_mond_norm = ConformalPredictiveSystem().fit(residuals=residuals_cal,
                                                sigmas=sigmas_cal,
                                                bins=bins_cal)
```

We already have the point predictions and the difficulty estimates for the test objects, so we just need the category labels according to the new bin thresholds:

```python
bins_test = binning(values=y_hat_test, bins=bin_thresholds)
```

We can now extract prediction intervals from the conformal predictive distributions for the test objects: 

```python
intervals = cps_mond_norm.predict(y_hat=y_hat_test,
                                  sigmas=sigmas_test,
                                  bins=bins_test,
                                  lower_percentiles=2.5,
                                  higher_percentiles=97.5,
                                  y_min=0)
```

```numpy
array([[     0.        , 537757.93618585],
       [177348.62535049, 655015.98985999],
       [253618.31669927, 783707.98804461],
       ...,
       [ 73466.09003216, 397289.46238233],
       [273315.68901744, 405309.55870912],
       [274035.55188125, 789701.43635318]])
```

We can also get the p values for the true target values; they should be uniformly distributed, if the test objects are drawn from the same underlying distribution as the calibration examples.

```python
p_values = cps_mond_norm.predict(y_hat=y_hat_test,
                                 sigmas=sigmas_test,
                                 bins=bins_test,
                                 y=y_test)
```

```numpy
array([[0.3262945 ],
       [0.12184386],
       [0.82948135],
       ...,
       [0.75042278],
       [0.61815831],
       [0.70252814]])
```

We may request that the predict method returns the full conformal predictive distribution (CPD) for each test instance, as defined by the threshold values, by setting `return_cpds=True`. The format of the distributions vary with the type of conformal predictive system; for a standard and normalized CPS, the output is an array with a row for each test instance and a column for each calibration instance (residual), while for a Mondrian CPS, the default output is a vector containing one CPD per test instance, since the number of values may vary between categories.

```python
cpds = cps_mond_norm.predict(y_hat=y_hat_test,
                             sigmas=sigmas_test,
                             bins=bins_test,
                             return_cpds=True)
```

The resulting vector of arrays is not displayed here, but we instead provide a plot for the CPD of a random test instance:

![cpd](https://user-images.githubusercontent.com/7838741/176182669-1cb739dc-dc38-4c30-80b1-624c2f6b4b83.png)

## Examples

For additional examples of how to use the package and module, including how to use out-of-bag predictions rather than having to rely on dividing the training set into a proper training and calibration set, see [this Jupyter notebook](https://github.com/henrikbostrom/crepes/blob/main/crepes.ipynb).

## Documentation

For documentation of the `crepes` package, see [here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.html).

For documentation of the `crepes.fillings` module, see [here](http://htmlpreview.github.io/?https://github.com/henrikbostrom/crepes/blob/main/docs/crepes.fillings.html).

## Citing crepes

If you use `crepes` for a scientific publication, you may cite the following paper:

Boström, H., 2022. crepes: a Python Package for Generating Conformal Regressors and Predictive Systems. In Conformal and Probabilistic Prediction and Applications. PMLR, 179.

Bibtex entry:

```bibtex
@InProceedings{crepes,
  title = 	 {crepes: a Python Package for Generating Conformal Regressors and Predictive Systems},
  author =       {Bostr\"om, Henrik},
  booktitle = 	 {Proceedings of the Eleventh Symposium on Conformal and Probabilistic Prediction and Applications},
  year = 	 {2022},
  editor = 	 {Johansson, Ulf and Boström, Henrik and An Nguyen, Khuong and Luo, Zhiyuan and Carlsson, Lars},
  volume = 	 {179},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR}
}
```

## References

<a id="1">[1]</a> Vovk, V., Gammerman, A. and Shafer, G., 2005. Algorithmic learning in a random world. Springer [Link](https://link.springer.com/book/10.1007/b106715)

<a id="2">[2]</a> Papadopoulos, H., Proedrou, K., Vovk, V. and Gammerman, A., 2002. Inductive confidence machines for regression. European Conference on Machine Learning, pp. 345-356. [Link](https://link.springer.com/chapter/10.1007/3-540-36755-1_29)

<a id="3">[3]</a> Johansson, U., Boström, H., Löfström, T. and Linusson, H., 2014. Regression conformal prediction with random forests. Machine learning, 97(1-2), pp. 155-176. [Link](https://link.springer.com/article/10.1007/s10994-014-5453-0)

<a id="4">[4]</a> Boström, H., Linusson, H., Löfström, T. and Johansson, U., 2017. Accelerating difficulty estimation for conformal regression forests. Annals of Mathematics and Artificial Intelligence, 81(1-2), pp.125-144. [Link](https://link.springer.com/article/10.1007/s10472-017-9539-9)

<a id="5">[5]</a> Boström, H. and Johansson, U., 2020. Mondrian conformal regressors. In Conformal and Probabilistic Prediction and Applications. PMLR, 128, pp. 114-133. [Link](https://proceedings.mlr.press/v128/bostrom20a.html)

<a id="6">[6]</a> Vovk, V., Petej, I., Nouretdinov, I., Manokhin, V. and Gammerman, A., 2020. Computationally efficient versions of conformal predictive distributions. Neurocomputing, 397, pp.292-308. [Link](https://www.aminer.org/pub/5e09aac9df1a9c0c416c9b70/computationally-efficient-versions-of-conformal-predictive-distributions)

<a id="7">[7]</a> Boström, H., Johansson, U. and Löfström, T., 2021. Mondrian conformal predictive distributions. In Conformal and Probabilistic Prediction and Applications. PMLR, 152, pp. 24-38. [Link](https://proceedings.mlr.press/v152/bostrom21a.html)

- - -

Author: Henrik Boström (bostromh@kth.se)
Copyright 2022 Henrik Boström
License: BSD 3 clause
