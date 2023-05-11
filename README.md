# `crepes`

`crepes` is a Python package for generating *conformal regressors*, which transform point predictions of any underlying regression model into prediction intervals for specified levels of confidence. The package also implements *conformal predictive systems*, which transform the point predictions into cumulative distribution functions.

The `crepes` package implements standard, normalized and Mondrian conformal regressors and predictive systems. While the package allows you to use your own difficulty estimates and Mondrian categories, there is also a separate module, called `crepes.fillings`, which provides some standard options for these.

## Installation

From [PyPI](https://pypi.org/project/crepes/)

```bash
pip install crepes
```

From [conda-forge](https://anaconda.org/conda-forge/crepes)

```bash
conda install -c conda-forge crepes
```

## Documentation

For complete documentation of the `crepes` package, see [here](https://crepes.readthedocs.io/en/latest/).

## Quickstart

### Conformal regressors

We first import the main class from `crepes` and a helper class and function from `crepes.fillings`:

```python
from crepes import ConformalRegressor
from crepes.fillings import DifficultyEstimator, binning
```

We will illustrate the above using a dataset from [www.openml.org](https://www.openml.org) and a `RandomForestRegressor` from [sklearn](https://scikit-learn.org):

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

The output is a [NumPy](https://numpy.org) array, specifying the lower and upper bound of each interval:

```numpy
array([[-171902.2 ,  953866.2 ],
       [-276818.01,  848950.39],
       [  22679.37, 1148447.77],
       ...,
       [ 242954.02, 1368722.42],
       [-308093.73,  817674.67],
       [-227057.4 ,  898711.  ]])
```

We may request that the intervals are cut to exclude impossible values, in this case below 0, and if we also rely on the default confidence level (0.95), the output intervals will be a bit tighter:

```python
intervals_std = cr_std.predict(y_hat=y_hat_test, y_min=0)
```

```numpy
array([[ 152258.55,  629705.45],
       [  47342.74,  524789.64],
       [ 346840.12,  824287.02],
       ...,
       [ 567114.77, 1044561.67],
       [  16067.02,  493513.92],
       [  97103.35,  574550.25]])
```

The above intervals are not normalized, i.e., they are all of the same size (at least before they are cut). We could make the intervals more informative through normalization using difficulty estimates; more difficult instances will be assigned wider intervals.

We will use a `DifficultyEstimator` for this purpose. Here it estimates the difficulty by the standard deviation of the target of the k (default `k=25`) nearest neighbors in the proper training set to each instance in the calibration set. A small value (beta) is added to the estimates, which may be given through an argument to the function; below we just use the default, i.e., `beta=0.01`.

```python
de = DifficultyEstimator()
de.fit(X_prop_train, y=y_prop_train)

sigmas_cal = de.apply(X_cal)
```

The difficulty estimates and residuals of the calibration examples can now be used to form a normalized conformal regressor:

```python
cr_norm = ConformalRegressor()
cr_norm.fit(residuals=residuals_cal, sigmas=sigmas_cal)
```

To generate prediction intervals for the test set using the normalized conformal regressor, we need difficulty estimates for the test set too, which we get using the same difficulty estimator as for the calibration set:

```python
sigmas_test = de.apply(X_test)
```

Now we can obtain the prediction intervals, using the point predictions and difficulty estimates for the test set:

```python
intervals_norm = cr_norm.predict(y_hat=y_hat_test, sigmas=sigmas_test, 
                                 y_min=0)
```

```numpy
array([[ 226719.06607977,  555244.93392023],
       [ 173767.90753715,  398364.47246285],
       [ 124690.70166966, 1046436.43833034],
       ...,
       [ 607949.71540572, 1003726.72459428],
       [ 188671.3752278 ,  320909.5647722 ],
       [ 145340.39076824,  526313.20923176]])
```

Depending on the employed difficulty estimator, the normalized intervals may sometimes be unreasonably large, in the sense that they may be several times larger than any previously observed error. Moreover, if the difficulty estimator is not very informative, e.g., completely random, the varying interval sizes may give a false impression of that we can expect lower prediction errors for instances with tighter intervals. Ideally, a difficulty estimator providing little or no information on the expected error should instead lead to more uniformly distributed interval sizes.

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
array([[ 206379.7 ,  575584.3 ],
       [ 144014.65,  428117.73],
       [  17965.57, 1153161.57],
       ...,
       [ 653865.22,  957811.22],
       [ 174264.87,  335316.07],
       [ 140587.46,  531066.14]])
```

### Conformal predictive systems

The interface to a `ConformalPredictiveSystem` is very similar to that of a conformal regressor; by providing just the residuals, we get a standard conformal predictive system, by providing also difficulty estimates, we get a normalized conformal predictive system and by providing labels for Mondrian categories (bins), we get a Mondrian conformal predictive system.

The main difference to conformal regressors concerns the output; instead of prediction intervals, conformal predictive systems produce complete cumulative distribution functions (conformal predictive distributions). From these we can generate prediction intervals, but we can also obtain percentiles, calibrated point predictions, as well as p-values for given target values.

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
array([[ 245826.3422693 ,  517315.83618985],
       [ 145348.03415848,  392968.15587997],
       [ 148774.65461212, 1034300.84195976],
       ...,
       [ 589200.5725957 , 1057013.89102007],
       [ 171938.29382952,  317732.31611141],
       [ 167498.01540504,  482328.98552632]])
```

We can also get the p-values for the true target values; they should be uniformly distributed, if the test objects are drawn from the same underlying distribution as the calibration examples.

```python
p_values = cps_mond_norm.predict(y_hat=y_hat_test,
                                 sigmas=sigmas_test,
                                 bins=bins_test,
                                 y=y_test)
```

```numpy
array([[0.98603614],
       [0.87178256],
       [0.44201984],
       ...,
       [0.05688804],
       [0.09473604],
       [0.31069913]])
```

We may request that the predict method returns the full conformal predictive distribution (CPD) for each test instance, as defined by the threshold values, by setting `return_cpds=True`. The format of the distributions vary with the type of conformal predictive system; for a standard and normalized CPS, the output is an array with a row for each test instance and a column for each calibration instance (residual), while for a Mondrian CPS, the default output is a vector containing one CPD per test instance, since the number of values may vary between categories.

```python
cpds = cps_mond_norm.predict(y_hat=y_hat_test,
                             sigmas=sigmas_test,
                             bins=bins_test,
                             return_cpds=True)
```

The resulting vector of arrays is not displayed here, but we instead provide a plot for the CPD of a random test instance:

![cpd](https://user-images.githubusercontent.com/7838741/235081969-328d7a23-26c9-4799-a246-8c35fd7ac88e.png)

## Examples

For additional examples of how to use the package and module, including how to use out-of-bag predictions rather than having to rely on dividing the training set into a proper training and calibration set, see [the documentation](https://crepes.readthedocs.io/en/latest/) and [this Jupyter notebook](https://github.com/henrikbostrom/crepes/blob/main/docs/crepes_nb.ipynb).

## Citing crepes

If you use `crepes` for a scientific publication, you may cite the following paper:

Boström, H., 2022. crepes: a Python Package for Generating Conformal Regressors and Predictive Systems. In Conformal and Probabilistic Prediction and Applications. PMLR, 179. [Link](https://copa-conference.com/papers/COPA2022_paper_11.pdf)

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

<a id="8">[8]</a> Vovk, V., 2022. Universal predictive systems. Pattern Recognition. 126: pp. 108536 [Link](https://dl.acm.org/doi/abs/10.1016/j.patcog.2022.108536)


- - -

Author: Henrik Boström (bostromh@kth.se)
Copyright 2023 Henrik Boström
License: BSD 3 clause
