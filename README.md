<p align="center"><a href="https://crepes.readthedocs.io"><img alt="crepes" src="https://github.com/henrikbostrom/crepes/blob/main/docs/crepes_logo.png"></a></p>

<p align="center">
<a href="https://pypi.org/project/crepes/"><img src="https://img.shields.io/badge/pypi package-0.8.0-brightgreen" alt="PyPI version" height=20 align="center"></a>
<a href="https://anaconda.org/conda-forge/crepes"><img src="https://img.shields.io/badge/conda--forge-0.8.0-orange" alt="conda-forge version" height=20 align="center"></a>
<a href="https://pepy.tech/project/crepes"><img src="https://static.pepy.tech/badge/crepes?dummy=unused" alt="Downloads" height=20 align="center"></a>
<a href="https://crepes.readthedocs.io/en/latest"><img src="https://readthedocs.org/projects/crepes/badge/?version=latest" alt="docs status" height=20 align="center"></a> 
<a href="https://github.com/henrikbostrom/crepes/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--clause-blue" alt="License" height=20 align="center"></a>
<a href="https://github.com/henrikbostrom/crepes/blob/main/CHANGELOG.md"><img src="https://img.shields.io/github/release-date/henrikbostrom/crepes" alt="Release date" height=20 align="center"></a>
</p>

<br>

`crepes` is a Python package for conformal prediction that implements conformal classifiers,
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
{'error': 0.007575757575757569,
 'avg_c': 1.6325757575757576,
 'one_c': 0.36742424242424243,
 'empty': 0.0,
 'ks_test': 0.0033578466103315894,
 'time_fit': 1.9073486328125e-06,
 'time_evaluate': 0.04798746109008789}
```

To control the error level across different groups of objects of
interest, we may use so-called Mondrian conformal classifiers. A
Mondrian conformal classifier is formed by providing a function or a
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

The class-conditional conformal classifier is a special type of Mondrian
conformal classifier, for which the categories are formed by the true labels;
we can generate one by setting `class_cond=True` in the call to `calibrate`

```python
rf_classcond = WrapClassifier(rf.learner)

rf_classcond.calibrate(X_cal, y_cal, class_cond=True)

rf_classcond.evaluate(X_test, y_test, confidence=0.99)
```

```python
{'error': 0.0018939393939394478,
 'avg_c': 1.740530303030303,
 'one_c': 0.25946969696969696,
 'empty': 0.0,
 'ks_test': 0.11458837583733483,
 'time_fit': 7.152557373046875e-07,
 'time_evaluate': 0.06147575378417969}
 ```

When employing an inductive conformal predictor, the predicted
p-values (and consequently the errors made) for a test set are not
independent. Semi-online conformal predictors can however make them
independent by updating the calibration set immediately after each
prediction (assuming that the true label is then available). We can
turn the conformal classifiers into semi-online conformal classifiers
by enabling online calibration, i.e., setting `online=True` when calling
the above methods, while also providing the true labels, e.g.,

```python
rf_classcond.predict_p(X_test, y_test, online=True)
```

```numpy
array([[8.13837566e-05, 8.86436603e-01],
       [6.60518590e-02, 4.02350293e-01],
       [4.28646783e-01, 4.29930890e-02],
       ...,
       [7.05118942e-01, 9.45056960e-03],
       [7.27003479e-01, 1.27347189e-02],
       [1.76403756e-01, 1.21434924e-01]])
```

Similarly, we can evaluate the conformal classifier while using online
calibration:

```python
rf_classcond.evaluate(X_test, y_test, confidence=0.99, online=True)
```

```python
{'error': 0.007575757575757569,
 'avg_c': 1.6117424242424243,
 'one_c': 0.38825757575757575,
 'empty': 0.0,
 'ks_test': 0.14097384777782784,
 'time_fit': 1.9073486328125e-06,
 'time_evaluate': 0.05298352241516113}
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
array([[1938866.06, 3146372.54],
       [ 225335.1 , 1432841.58],
       [-403305.49,  804200.99],
       ...,
       [ 443742.33, 1651248.81],
       [-343684.48,  863822.  ],
       [-153629.93, 1053876.55]])
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
array([[2302049.84, 2783188.76],
       [ 588518.88, 1069657.8 ],
       [      0.  ,  441017.21],
       ...,
       [ 806926.11, 1288065.03],
       [  19499.3 ,  500638.22],
       [ 209553.85,  690692.77]])
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
array([[1769594.36212355, 3315644.23787645],
       [ 693827.99796647,  964348.68203353],
       [ 124886.97469338,  276008.52530662],
       ...,
       [ 661373.45043166, 1433617.68956833],
       [ 178769.2939384 ,  341368.2260616 ],
       [ 222837.12801117,  677409.49198883]])
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
array([[1152528.9 , 3932709.7 ],
       [ 692366.75,  965809.93],
       [ 124254.81,  276640.69],
       ...,
       [ 622939.57, 1472051.57],
       [ 155346.82,  364790.7 ],
       [ 239474.31,  660772.31]])
```

Similarly to semi-online conformal classifiers, we may enable online calibration
also for conformal regressors; this is again done by setting `online=True` when
calling any of the applicable methods, while also providing the true labels, e.g.,

```python
rf.predict_p(X_test, y_test, online=True)
```

```numpy
array([0.09369225, 0.52548032, 0.49992477, ..., 0.72979714, 0.87495964,
       0.58352253])
```

We can easily switch from conformal regressors to conformal
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
through several different methods, e.g., `predict_percentiles`:

```python
rf.predict_percentiles(X_test, higher_percentiles=[90, 95, 99])
```

```numpy
array([[3120432.14791764, 3403976.16608241, 3952384.13595105],
       [ 930191.36994287,  979804.59585495, 1075762.49571536],
       [ 236278.82580469,  253387.66592079,  329933.49293406],
       ...,
       [1336110.21956702, 1477739.04927264, 1751666.10820498],
       [ 298621.13482031,  317029.35735016,  399388.68783836],
       [ 564574.75363948,  615226.06727944,  762212.9912238 ]])
```

Similarly to semi-online conformal classifiers and regressors, we can enable
online calibration also for conformal predictive systems; here we generate
prediction intervals at the default (95%) confidence level:

```python
rf.predict_int(X_test, y_test, y_min=0, online=True)
```

```numpy
array([[1719676.80439219, 3707173.76806116],
       [ 684289.27240227, 1032856.71531186],
       [ 127189.61835749,  274385.59426486],
       ...,
       [ 630347.70469164, 1594876.58130005],
       [ 167399.51044545,  337513.60197203],
       [ 232815.51352497,  641580.14787679]])
```

We may also obtain the full conformal predictive distribution for each test
instance, as defined by the threshold values:

```python
rf.predict_cpds(X_test)
```

For a Mondrian conformal predictive system (or any semi-online conformal
predictive system), the output is a vector containing one CPD per test instance,
while for a standard or normalized conformal predictive system (for which online
calibration is not enabled), the output is a 2-dimensional array.

The resulting vector of vectors is not displayed here, but we instead provide a plot
for the CPD of a random test instance:

![cpd](https://user-images.githubusercontent.com/7838741/235081969-328d7a23-26c9-4799-a246-8c35fd7ac88e.png)

## Examples

For additional examples of how to use the package and module, see [the documentation](https://crepes.readthedocs.io/en/latest/), [this Jupyter notebook using WrapClassifier and WrapRegressor](https://github.com/henrikbostrom/crepes/blob/main/docs/crepes_nb_wrap.ipynb), and [this Jupyter notebook using ConformalClassifier, ConformalRegressor, and ConformalPredictiveSystem](https://github.com/henrikbostrom/crepes/blob/main/docs/crepes_nb.ipynb).

You may also take a look at the [slides from my tutorial at COPA 2024](<https://github.com/henrikbostrom/crepes/blob/main/docs/COPA Tutorial 2024.pdf>) and the accompanying [Jupyter notebook](<https://github.com/henrikbostrom/crepes/blob/main/docs/COPA Tutorial 2024.ipynb>).

## Citing crepes

You are welcome to cite the following paper:

Boström, H. 2024. Conformal Prediction in Python with crepes. Proceedings of the 13th Symposium on Conformal and Probabilistic Prediction with Applications, PMLR 230:236-249 [Link](https://raw.githubusercontent.com/mlresearch/v230/main/assets/bostrom24a/bostrom24a.pdf)

Bibtex entry:

```bibtex
@inproceedings{bostrom2024,
  title={Conformal Prediction in Python with crepes},
  author={Bostr{\"o}m, Henrik},
  booktitle={Proc. of the 13th Symposium on Conformal and Probabilistic Prediction with Applications},
  pages={236--249},
  year={2024},
  organization={PMLR}
}
```

An early version of the package was described in:

Boström, H., 2022. crepes: a Python Package for Generating Conformal Regressors and Predictive Systems. Proceedings of the 11th Symposium on Conformal and Probabilistic Prediction with Applications, PMLR 179:24-41 [Link](https://proceedings.mlr.press/v179/bostrom22a/bostrom22a.pdf)

Bibtex entry:

```bibtex
@inproceedings{bostrom2022,
  title={crepes: a Python Package for Generating Conformal Regressors and Predictive Systems},
  author={Bostr{\"o}m, Henrik},
  booktitle={Proc. of the 11th Symposium on Conformal and Probabilistic Prediction with Applications},
  pages={24--41},
  year={2022},
  organization={PMLR}
}
```

## References

<a id="1">[1]</a> Vovk, V., Gammerman, A. and Shafer, G., 2022. Algorithmic learning in a random world. 2nd edition. Springer [Link](https://link.springer.com/book/10.1007/978-3-031-06649-8)

<a id="2">[2]</a> Papadopoulos, H., Proedrou, K., Vovk, V. and Gammerman, A., 2002. Inductive confidence machines for regression. European Conference on Machine Learning, pp. 345-356. [Link](https://link.springer.com/chapter/10.1007/3-540-36755-1_29)

<a id="3">[3]</a> Johansson, U., Boström, H., Löfström, T. and Linusson, H., 2014. Regression conformal prediction with random forests. Machine learning, 97(1-2), pp. 155-176. [Link](https://link.springer.com/article/10.1007/s10994-014-5453-0)

<a id="4">[4]</a> Boström, H., Linusson, H., Löfström, T. and Johansson, U., 2017. Accelerating difficulty estimation for conformal regression forests. Annals of Mathematics and Artificial Intelligence, 81(1-2), pp.125-144. [Link](https://link.springer.com/article/10.1007/s10472-017-9539-9)

<a id="5">[5]</a> Boström, H. and Johansson, U., 2020. Mondrian conformal regressors. In Conformal and Probabilistic Prediction and Applications. PMLR, 128, pp. 114-133. [Link](https://proceedings.mlr.press/v128/bostrom20a.html)

<a id="6">[6]</a> Vovk, V., Petej, I., Nouretdinov, I., Manokhin, V. and Gammerman, A., 2020. Computationally efficient versions of conformal predictive distributions. Neurocomputing, 397, pp.292-308. [Link](https://www.aminer.org/pub/5e09aac9df1a9c0c416c9b70/computationally-efficient-versions-of-conformal-predictive-distributions)

<a id="7">[7]</a> Boström, H., Johansson, U. and Löfström, T., 2021. Mondrian conformal predictive distributions. In Conformal and Probabilistic Prediction and Applications. PMLR, 152, pp. 24-38. [Link](https://proceedings.mlr.press/v152/bostrom21a.html)

<a id="8">[8]</a> Vovk, V., 2022. Universal predictive systems. Pattern Recognition. 126: pp. 108536 [Link](https://dl.acm.org/doi/abs/10.1016/j.patcog.2022.108536)


- - -

Author: Henrik Boström (bostromh@kth.se)
Copyright 2025 Henrik Boström
License: BSD 3 clause
