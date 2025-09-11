import numpy as np
import pandas as pd

from crepes.extras import hinge, margin, binning, MondrianCategorizer, DifficultyEstimator, get_oob


def test_hinge_and_margin_basic():
    probs = np.array([[0.1, 0.9], [0.6, 0.4]])
    classes = np.array([0, 1])
    y = np.array([1, 0])
    h = hinge(probs, classes=classes, y=y)
    assert np.allclose(h, np.array([0.1, 0.4]))
    h2 = hinge(probs)
    assert h2.shape == probs.shape
    m = margin(probs, classes=classes, y=y)
    assert m.shape == (2,)
    m2 = margin(probs)
    assert m2.shape == probs.shape


def test_binning_and_mondrian_repr_and_get_oob():
    rng = np.random.RandomState(0)
    values = rng.rand(50)
    assigned, thresholds = binning(values, bins=5)
    assert len(assigned) == 50
    assert thresholds[0] == -np.inf and thresholds[-1] == np.inf
    mc = MondrianCategorizer()
    r = repr(mc)
    assert 'MondrianCategorizer' in r

    # get_oob helper: for simple DataFrame provide n_samples and assert return
    # call get_oob with an integer seed and check boolean mask length
    mask = get_oob(0, 2)
    assert hasattr(mask, '__len__') and len(mask) == 2


def test_difficulty_estimator_function_and_knn():
    X = np.vstack([np.linspace(0, 1, 10), np.linspace(1, 2, 10)]).T
    def f(X_in):
        return X_in[:, 0]
    de = DifficultyEstimator()
    de.fit(X=X, f=f, scaler=True)
    sig = de.apply(X[:3])
    assert sig.shape[0] == 3
    de2 = DifficultyEstimator()
    de2.fit(X=X, k=3, scaler=False)
    sig2 = de2.apply(X[:2])
    assert sig2.shape[0] == 2


# ----- copied from legacy tests/test_extras.py (small utilities) -----
def test_hinge_margin_and_binning_from_legacy():
    probs = np.array([[0.1, 0.9], [0.6, 0.4]])
    classes = np.array([0, 1])
    y = np.array([1, 0])
    h = hinge(probs, classes=classes, y=y)
    assert np.allclose(h, np.array([0.1, 0.4]))
    h2 = hinge(probs)
    assert h2.shape == probs.shape
    m = margin(probs, classes=classes, y=y)
    assert m.shape == (2,)
    m2 = margin(probs)
    assert m2.shape == probs.shape

    rng = np.random.RandomState(0)
    values = rng.rand(50)
    assigned, thresholds = binning(values, bins=5)
    assert len(assigned) == 50
    assert thresholds[0] == -np.inf and thresholds[-1] == np.inf


def test_get_oob_and_variance_oob_path():
    from crepes.extras import get_oob, DifficultyEstimator
    # get_oob returns boolean mask of length n
    mask = get_oob(0, 5)
    assert hasattr(mask, '__len__') and len(mask) == 5

    # test variance estimator with oob path by mocking a learner with estimators_
    class MockEstimator:
        def __init__(self, pred):
            self.pred = pred
            self.random_state = 0
        def predict(self, X):
            return np.array(self.pred)

    class MockLearner:
        def __init__(self):
            self.estimators_ = [MockEstimator([0.0, 1.0]), MockEstimator([0.0, 1.0])]

    de = DifficultyEstimator()
    # monkey-patch a learner and test variance branch (avoid function-path)
    de.estimator_type = "variance"
    de.learner = MockLearner()
    de.oob = True
    de.scaler = False
    de.beta = 0.001
    # small X
    X = np.zeros((2, 1))
    sigmas = de.apply(X)
    assert hasattr(sigmas, '__len__') and len(sigmas) == 2


def test_hinge_margin_and_binning_additional_branches():
    import pandas as pd
    from crepes.extras import hinge, margin, binning
    # hinge with pandas Series y
    X_prob = np.array([[0.8, 0.2], [0.4, 0.6]])
    classes = np.array([0, 1])
    y = pd.Series([0, 1])
    h = hinge(X_prob, classes=classes, y=y)
    assert h.shape == (2,)

    # margin with pandas Series y
    m = margin(X_prob, classes=classes, y=y)
    assert m.shape == (2,)

    # binning with explicit thresholds array
    vals = np.array([-1.0, 0.5, 2.0])
    bins = [-np.inf, 0.0, np.inf]
    assigned = binning(vals, bins=bins)
    assert len(assigned) == 3


def test_mondrian_categorizer_fit_apply_and_repr_branches():
    from crepes.extras import MondrianCategorizer, DifficultyEstimator

    # f-function branch with bin thresholds produced
    def f_vals(X):
        return np.array([0, 1])

    mc = MondrianCategorizer()
    mc.fit(X=np.zeros((2, 1)), f=f_vals, no_bins=2)
    r = repr(mc)
    assert 'f=' in r or 'f=' in r
    # apply should return assigned bins (length matches)
    bins = mc.apply(np.zeros((2, 1)))
    assert len(bins) == 2

    # f present but bin_thresholds None -> apply returns f(X)
    mc2 = MondrianCategorizer()
    mc2.f = f_vals
    mc2.bin_thresholds = None
    out = mc2.apply(np.zeros((2, 1)))
    assert np.array_equal(out, f_vals(np.zeros((2, 1))))

    # de branch with oob
    de = DifficultyEstimator()
    de.oob = True
    de.sigmas = np.array([0.1, 0.9])
    de.beta = 0.001
    mc3 = MondrianCategorizer()
    mc3.fit(de=de, no_bins=2, oob=True)
    assert mc3.fitted

    # learner branch with oob prediction
    class Learner:
        def __init__(self):
            self.oob_prediction_ = np.array([0.2, 0.8])

    learner = Learner()
    mc4 = MondrianCategorizer()
    mc4.fit(learner=learner, no_bins=2, oob=True)
    assert mc4.fitted

    # learner apply with oob estimators_
    class Est:
        def __init__(self, rs, pred):
            self.random_state = rs
            self._pred = pred
        def predict(self, X):
            return np.array(self._pred)

    class Lear2:
        def __init__(self):
            self.estimators_ = [Est(0, [0.1, 0.2]), Est(1, [0.3, 0.4])]

    l2 = Lear2()
    mc5 = MondrianCategorizer()
    # set bin_thresholds so apply uses binning
    mc5.learner = l2
    mc5.oob = True
    mc5.bin_thresholds = np.array([-np.inf, 0.25, np.inf])
    bins_out = mc5.apply(np.zeros((2, 1)))
    assert len(bins_out) == 2


def test_difficulty_estimator_fit_and_apply_errors_and_function_scaler():
    from crepes.extras import DifficultyEstimator

    # function estimator with scaler True requires X
    def f_fun(X):
        return np.array([1.0, 2.0])

    de_fun = DifficultyEstimator()
    de_fun.fit(X=np.zeros((2, 1)), f=f_fun, scaler=True)
    assert de_fun.estimator_type == 'function'
    assert hasattr(de_fun, 'sigma_scaler')

    # calling fit with f and scaler True but X=None should raise
    de_err = DifficultyEstimator()
    try:
        de_err.fit(f=f_fun, scaler=True)
        raised = False
    except ValueError:
        raised = True
    assert raised

    # learner without estimators_ should raise
    class BadLear:
        pass

    de_bad = DifficultyEstimator()
    try:
        de_bad.fit(X=np.zeros((2, 1)), learner=BadLear())
        raised2 = False
    except ValueError:
        raised2 = True
    assert raised2

    # learner with estimators_ but missing random_state when oob=True
    class ENoRS:
        def predict(self, X):
            return np.array([0.1, 0.2])

    class LearNoRS:
        def __init__(self):
            self.estimators_ = [ENoRS(), ENoRS()]

    de_no_rs = DifficultyEstimator()
    try:
        de_no_rs.fit(X=np.zeros((2, 1)), learner=LearNoRS(), oob=True)
        raised3 = False
    except ValueError:
        raised3 = True
    assert raised3

    # apply without oob and X=None should raise
    de_apply = DifficultyEstimator()
    de_apply.oob = False
    try:
        de_apply.apply()
        raised4 = False
    except ValueError:
        raised4 = True
    assert raised4


def test_difficulty_estimator_knn_branches_and_mondrian_non_oob():
    from crepes.extras import DifficultyEstimator, MondrianCategorizer, binning
    # knn with labels (y provided)
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])
    de_knn = DifficultyEstimator()
    de_knn.fit(X=X, y=y, k=2, scaler=True, oob=True)
    assert de_knn.estimator_type == 'knn'
    # apply on same X
    sig = de_knn.apply(X)
    assert len(sig) == 3

    # knn with residuals
    residuals = np.array([0.1, 0.2, 0.3])
    de_knn2 = DifficultyEstimator()
    de_knn2.fit(X=X, residuals=residuals, k=2, scaler=False, oob=True)
    out = de_knn2.apply(X)
    assert len(out) == 3

    # MondrianCategorizer with learner non-oob predict path
    class LearP:
        def __init__(self):
            pass
        def predict(self, X):
            return np.array([0.1, 0.9, 0.5])

    learner = LearP()
    mc = MondrianCategorizer()
    mc.fit(X=X, learner=learner, no_bins=2, oob=False)
    bins = mc.apply(X)
    assert len(bins) == 3

