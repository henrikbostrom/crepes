<p align="center"><a href="https://crepes.readthedocs.io"><img alt="crepes" src="https://github-production-user-asset-6210df.s3.amazonaws.com/7838741/358183921-fcdf2887-3d30-4042-bd93-dbccde52f6a8.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240815%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240815T094123Z&X-Amz-Expires=300&X-Amz-Signature=8b49a7808767c6ac6cf65110fab083349e66fa5d6c4a330d84a6e2882f9b5176&X-Amz-SignedHeaders=host&actor_id=7838741&key_id=0&repo_id=428992317"></a></p>

<p align="center">
<a href="https://pypi.org/project/crepes/"><img src="https://img.shields.io/badge/pypi package-0.7.0-brightgreen" alt="PyPI version" height=20 align="center"></a>
<a href="https://anaconda.org/conda-forge/crepes"><img src="https://img.shields.io/badge/conda--forge-0.7.0-orange" alt="conda-forge version" height=20 align="center"></a>
<a href="https://pepy.tech/project/crepes"><img src="https://static.pepy.tech/badge/crepes?dummy=unused" alt="Downloads" height=20 align="center"></a>
<a href="https://crepes.readthedocs.io/en/latest"><img src="https://readthedocs.org/projects/crepes/badge/?version=latest" alt="docs status" height=20 align="center"></a> 
<a href="https://github.com/henrikbostrom/crepes/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--clause-blue" alt="License" height=20 align="center"></a>
<a href="https://github.com/henrikbostrom/crepes/blob/main/CHANGELOG.md"><img src="https://img.shields.io/github/release-date/henrikbostrom/crepes" alt="Release date" height=20 align="center"></a>
</p>

<br>

`crepes` is a Python package that implements conformal classifiers,
regressors, and predictive systems on top of any standard classifier
and regressor, turning the original predictions into
well-calibrated p-values and cumulative distribution functions, or
prediction sets and intervals with coverage guarantees.

The `crepes` package implements standard and Mondrian conformal
classifiers as well as standard, normalized and Mondrian conformal
regressors and predictive systems. While the package allows you to use
your own functions to compute difficulty estimates, non-conformity
scores and Mondrian categories, there is also a separate module,
called `crepes.extras`, which provides some standard options for
these.

## Installation

From [PyPI](https://pypi.org/project/crepes/)

```bash
pip install crepes
```

From [conda-forge](https://anaconda.org/conda-forge/crepes)

```bash
conda install conda-forge::crepes
```

## Documentation

For the complete documentation, see [crepes.readthedocs.io](https://crepes.readthedocs.io/en/latest/).

## Quickstart

Let us illustrate how we may use `crepes` to generate and apply
conformal classifiers with a dataset from
[www.openml.org](https://www.openml.org), which we first split into a
training and a test set using `train_test_split` from
[sklearn](https://scikit-learn.org), and then further split the
training set into a proper training set and a calibration set:

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

dataset = fetch_openml(name="qsar-biodeg", parser="auto")

X = dataset.data.values.astype(float)
y = dataset.target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                            test_size=0.25)
```

We now "wrap" a random forest classifier, fit it to the proper
training set, and fit a standard conformal classifier through the
`calibrate` method:

```python
from crepes import WrapClassifier
from sklearn.ensemble import RandomForestClassifier

rf = WrapClassifier(RandomForestClassifier(n_jobs=-1))

rf.fit(X_prop_train, y_prop_train)

rf.calibrate(X_cal, y_cal)
```

We may now produce p-values for the test set (an array with as many
columns as there are classes):

```python
rf.predict_p(X_test)
```

```numpy
array([[0.00427104, 0.74842304],
       [0.07874355, 0.2950549 ],
       [0.50529983, 0.01557963],
       ...,
       [0.8413356 , 0.00201167],
       [0.84402215, 0.00654927],
       [0.29601955, 0.07766093]])
```

We can also get prediction sets, represented by binary vectors
indicating presence (1) or absence (0) of the class labels that
correspond to the columns, here at the 90% confidence level:

```python
rf.predict_set(X_test, confidence=0.9)
```

```numpy
array([[0, 1],
       [0, 1],
       [1, 0],
       ...,
       [1, 0],
       [1, 0],
       [1, 0]])
```

Since we have access to the true class labels, we can evaluate the
conformal classifier (here using all available metrics which is the
default), at the 99% confidence level:

```python
rf.evaluate(X_test, y_test, confidence=0.99)
```

```python
{'error': 0.005681818181818232,
 'avg_c': 1.691287878787879,
 'one_c': 0.3087121212121212,
 'empty': 0.0,
 'time_fit': 2.3365020751953125e-05,
 'time_evaluate': 0.017678260803222656}
```

To control the error level across different groups of objects of
interest, we may use so-called Mondrian conformal classifiers. A
Mondrian conformal classifier if formed by providing a function or a
`MondrianCategorizer` (defined in `crepes.extras`) as an additional
argument, named `mc`, for the `calibrate` method.

For illustration, we will use the predicted labels of the underlying
model to form the categories. Note that the prediction sets are generated
for the test objects using the same categorization (under the hood).

```python
rf_mond = WrapClassifier(rf.learner)

rf_mond.calibrate(X_cal, y_cal, mc=rf_mond.predict)

rf_mond.predict_set(X_test)
```

```numpy
array([[0, 1],
       [1, 1],
       [1, 0],
       ...,
       [1, 0],
       [1, 0],
       [1, 1]])
```

We may also form the categories using a `MondrianCategorizer`, which
may be fitted in several different ways. Below we show how to form
categories by (equal-sized) binning of the first feature value, using
five bins (instead of the default which is 10); note that we need
objects to get the threshold values for the categories (bins). 

```python
from crepes.extras import MondrianCategorizer

def get_values(X):
    return X[:,0]

mc = MondrianCategorizer()
mc.fit(X_cal, f=get_values, no_bins=5)

rf_mond = WrapClassifier(rf.learner)
rf_mond.calibrate(X_cal, y_cal, mc=mc)

rf_mond.predict_set(X_test)
```

```numpy
array([[0, 1],
       [1, 1],
       [1, 0],
       ...,
       [1, 0],
       [1, 0],
       [1, 1]])
```

For conformal classifiers that employ learners that use bagging, like
random forests, we may consider an alternative strategy to dividing
the original training set into a proper training and calibration set;
we may use the out-of-bag (OOB) predictions, which allow us to use the
full training set for both model building and calibration. It should
be noted that this strategy does not come with the theoretical
validity guarantee of the above (inductive) conformal classifiers, due
to that calibration and test instances are not handled in exactly the
same way. In practice, however, conformal classifiers based on
out-of-bag predictions rarely fail to meet the coverage requirements.

Below we show how to enable this in conjunction with a specific type
of Mondrian conformal classifier, a so-called class-conditional
conformal classifier, which uses the class labels as Mondrian
categories:

```python
rf = WrapClassifier(RandomForestClassifier(n_jobs=-1, n_estimators=500, oob_score=True))

rf.fit(X_train, y_train)

rf.calibrate(X_train, y_train, class_cond=True, oob=True)

rf.evaluate(X_test, y_test, confidence=0.9)
```

```python
{'error': 0.10795454545454541,
 'avg_c': 1.0984848484848484,
 'one_c': 0.9015151515151515,
 'empty': 0.0,
 'time_fit': 0.0001518726348876953,
 'time_evaluate': 0.06513118743896484}
```

Let us also illustrate how `crepes` can be used to generate conformal
regressors and predictive systems. Again, we import a dataset from
[www.openml.org](https://www.openml.org), which we split into a
training and a test set and then further split the training set into a
proper training set and a calibration set:

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

dataset = fetch_openml(name="house_sales", version=3, parser="auto")

X = dataset.data.values.astype(float)
y = dataset.target.values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                            test_size=0.25)
```

Let us now "wrap" a `RandomForestRegressor` from
[sklearn](https://scikit-learn.org) using the class `WrapRegressor`
from `crepes` and fit it (in the usual way) to the proper training
set:

```python
from sklearn.ensemble import RandomForestRegressor
from crepes import WrapRegressor

rf = WrapRegressor(RandomForestRegressor())
rf.fit(X_prop_train, y_prop_train)
```

We may now fit a conformal regressor using the calibration set through
the `calibrate` method:

```python
rf.calibrate(X_cal, y_cal)
```

The conformal regressor can now produce prediction intervals for the
test set, here using a confidence level of 99%:

```python
rf.predict_int(X_test, confidence=0.99)
```

```numpy
array([[   8260.53, 1065083.53],
       [ -54858.5 , 1001964.5 ],
       [  -7779.25, 1049043.75],
       ...,
       [ 297229.8 , 1354052.8 ],
       [-270260.  ,  786563.  ],
       [-185146.94,  871676.06]])
```

The output is a [NumPy](https://numpy.org) array with a row for each
test instance, and where the two columns specify the lower and upper
bound of each prediction interval.

We may request that the intervals are cut to exclude impossible
values, in this case below 0, and if we also rely on the default
confidence level (0.95), the output intervals will be a bit tighter:

```python
rf.predict_int(X_test, y_min=0)
```

```numpy
array([[ 288602.83,  784741.23],
       [ 225483.8 ,  721622.2 ],
       [ 272563.05,  768701.45],
       ...,
       [ 577572.1 , 1073710.5 ],
       [  10082.3 ,  506220.7 ],
       [  95195.36,  591333.76]])
```

The above intervals are not normalized, i.e., they are all of the same
size (at least before they are cut). We could make them more
informative through normalization using difficulty estimates; objects
considered more difficult will be assigned wider intervals.

We will use a `DifficultyEstimator` from the `crepes.extras` module
for this purpose. Here we estimate the difficulty by the standard
deviation of the target of the k (default `k=25`) nearest neighbors in
the proper training set to each object in the calibration set. A small
value (beta) is added to the estimates, which may be given through an
argument to the function; below we just use the default, i.e.,
`beta=0.01`.

We first fit the difficulty estimator and then calibrate the conformal
regressor, using the calibration objects and labels together the
difficulty estimator:

```python
from crepes.extras import DifficultyEstimator

de = DifficultyEstimator()
de.fit(X_prop_train, y=y_prop_train)

rf.calibrate(X_cal, y_cal, de=de)
```

To obtain prediction intervals, we just have to provide test objects
to the `predict_int` method, as the difficulty estimates will be
computed by the incorporated difficulty estimator:

```python
rf.predict_int(X_test, y_min=0)
```

```numpy
array([[ 222036.82862012,  851307.23137988],
       [ 316413.83821721,  630692.16178279],
       [ 384784.44135415,  656480.05864585],
       ...,
       [ 110527.74801848, 1540754.85198152],
       [ 174799.94131735,  341503.05868265],
       [ 274305.55734858,  412223.56265142]])
```

Depending on the employed difficulty estimator, the normalized
intervals may sometimes be unreasonably large, in the sense that they
may be several times larger than any previously observed
error. Moreover, if the difficulty estimator is uninformative, e.g.,
completely random, the varying interval sizes may give a false
impression of that we can expect lower prediction errors for instances
with tighter intervals. Ideally, a difficulty estimator providing
little or no information on the expected error should instead lead to
more uniformly distributed interval sizes.

A Mondrian conformal regressor can be used to address these problems,
by dividing the object space into non-overlapping so-called Mondrian
categories, and forming a (standard) conformal regressor for each
category. We may form a Mondrian conformal regressor by providing a
function or a `MondrianCategorizer` (defined in `crepes.extras`) as an
additional argument, named `mc`, for the `calibrate` method.

Here we employ a `MondrianCategorizer`; it may be fitted in several
different ways, and below we show how to form categories by binning of
the difficulty estimates into 20 bins, using the difficulty estimator
fitted above.

```python
from crepes.extras import MondrianCategorizer

mc_diff = MondrianCategorizer()
mc_diff.fit(X_cal, de=de, no_bins=20)

rf.calibrate(X_cal, y_cal, mc=mc_diff)
```

When making predictions, the test objects will be assigned to Mondrian categories
according to the incorporated `MondrianCategorizer` (or labeling function):

```python
rf.predict_int(X_test, y_min=0)
```

```numpy
array([[ 242624.89,  830719.17],
       [ 329358.5 ,  617747.5 ],
       [ 371028.  ,  670236.5 ],
       ...,
       [      0.  , 1730501.3 ],
       [ 157022.53,  359280.47],
       [ 266456.61,  420072.51]])
```

We could very easily switch from conformal regressors to conformal
predictive systems. The latter produce cumulative distribution
functions (conformal predictive distributions). From these we can
generate prediction intervals, but we can also obtain percentiles,
calibrated point predictions, as well as p-values for given target
values. Let us see how we can go ahead to do that.

Well, there is only one thing above that changes: just provide
`cps=True` to the `calibrate` method.

We can, for example, form normalized Mondrian conformal predictive
systems, by providing both a Mondrian categorizer and difficulty estimator
to the `calibrate` method. Here we will consider Mondrian categories formed
from binning the point predictions:

```python
mc_pred = MondrianCategorizer()
mc_pred.fit(X_cal, f=rf.predict, no_bins=5)

rf.calibrate(X_cal, y_cal, de=de, mc=mc_pred, cps=True)
```

We can now make predictions with the conformal predictive system,
through the method `predict_cps`.  The output of this method can be
controlled quite flexibly; here we request prediction intervals with
95% confidence to be output:

```python
rf.predict_cps(X_test, lower_percentiles=2.5, higher_percentiles=97.5, y_min=0)
```

```numpy
array([[ 240114.65604157,  869014.03528742],
       [ 339706.24924814,  609239.58260891],
       [ 404920.87940518,  637934.16698199],
       ...,
       [      0.        , 1947549.10314688],
       [ 173038.55234664,  335836.19025193],
       [ 280187.36965593,  399290.04471503]])
```

If we would like to take a look at the p-values for the true targets (these should be uniformly distributed), we can do the following:

```python
rf.predict_cps(X_test, y=y_test)
```

```numpy
array([0.38424814, 0.54023864, 0.28727364, ..., 0.35291685, 0.6110545 ,
       0.60037036])
```

We may request that the `predict_cps` method returns the full
conformal predictive distribution (CPD) for each test instance, as
defined by the threshold values, by setting `return_cpds=True`. The
format of the distributions vary with the type of conformal predictive
system; for a standard and normalized CPS, the output is an array with
a row for each test instance and a column for each calibration
instance (residual), while for a Mondrian CPS, the default output is a
vector containing one CPD per test instance, since the number of
values may vary between categories.

```python
cpds = rf.predict_cps(X_test, return_cpds=True)
```

The resulting vector of arrays is not displayed here, but we instead provide a plot for the CPD of a random test instance:

![cpd](https://user-images.githubusercontent.com/7838741/235081969-328d7a23-26c9-4799-a246-8c35fd7ac88e.png)

## Examples

For additional examples of how to use the package and module, see [the documentation](https://crepes.readthedocs.io/en/latest/), [this Jupyter notebook using WrapClassifier and WrapRegressor](https://github.com/henrikbostrom/crepes/blob/main/docs/crepes_nb_wrap.ipynb), and [this Jupyter notebook using ConformalClassifier, ConformalRegressor, and ConformalPredictiveSystem](https://github.com/henrikbostrom/crepes/blob/main/docs/crepes_nb.ipynb).

## Citing crepes

If you use `crepes` for a scientific publication, you are kindly requested to cite the following paper:

Boström, H., 2022. crepes: a Python Package for Generating Conformal Regressors and Predictive Systems. In Conformal and Probabilistic Prediction and Applications. PMLR, 179. [Link](https://proceedings.mlr.press/v179/bostrom22a.html)

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
Copyright 2024 Henrik Boström
License: BSD 3 clause
