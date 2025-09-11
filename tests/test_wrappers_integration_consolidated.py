import numpy as np
import warnings

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from crepes.base import WrapClassifier, WrapRegressor
from crepes.extras import DifficultyEstimator, MondrianCategorizer


def test_wrapclassifier_flow_and_mondrian():
    X_prop, y_prop, X_cal, y_cal, X_test, y_test = (
        np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8], [0.15, 0.25], [0.85, 0.75]])[:4],
        np.array(["a", "a", "b", "b"])[:4],
        np.array([[0.15, 0.25]]),
        np.array(["a"]),
        np.array([[0.85, 0.75]]),
        np.array(["b"]),
    )
    learner = DecisionTreeClassifier(random_state=0)
    w = WrapClassifier(learner)
    w.fit(X_prop, y_prop)
    w.calibrate(X_cal, y_cal, seed=42)
    pvals = w.predict_p(X_test, smoothing=False, seed=42)
    assert pvals.shape[0] == X_test.shape[0]
    assert np.all((pvals >= 0) & (pvals <= 1))
    sets = w.predict_set(X_test, confidence=0.9, smoothing=False, seed=42)
    assert sets.shape == pvals.shape
    results = w.evaluate(X_test, y_test, confidence=0.9, seed=42)
    assert "error" in results and "avg_c" in results

    def mc_fn(X):
        return np.array([0 if x[0] < 0.5 else 1 for x in X])
    w2 = WrapClassifier(DecisionTreeClassifier(random_state=1))
    w2.fit(X_prop, y_prop)
    X_cal2 = np.vstack([X_cal, X_test])
    y_cal2 = np.concatenate([y_cal, y_test])
    w2.calibrate(X_cal2, y_cal2, mc=mc_fn, seed=1)
    pv2 = w2.predict_p(X_test, smoothing=False, seed=1)
    assert pv2.shape == (X_test.shape[0], len(w2.learner.classes_))

    mc = MondrianCategorizer()
    mc.fit(X_cal, f=lambda X_in: X_in[:, 0], no_bins=2)
    w3 = WrapClassifier(DecisionTreeClassifier(random_state=2))
    w3.fit(X_prop, y_prop)
    w3.calibrate(X_cal, y_cal, mc=mc, seed=2)
    assert w3.calibrated


def test_wrapregressor_flow_and_cps():
    X_prop = np.linspace(0, 1, 8)[:5, None]
    y_prop = (X_prop.ravel() * 2.0) + 0.1
    X_cal = np.linspace(0, 1, 8)[5:6, None]
    y_cal = ((X_cal.ravel()) * 2.0) + 0.1
    X_test = np.linspace(0, 1, 8)[6:8, None]
    y_test = ((X_test.ravel()) * 2.0) + 0.1
    learner = DecisionTreeRegressor(random_state=0)
    w = WrapRegressor(learner)
    w.fit(X_prop, y_prop)
    w.calibrate(X_cal, y_cal, seed=5)
    ints = w.predict_int(X_test, confidence=0.9, seed=5)
    assert ints.shape == (X_test.shape[0], 2)
    res = w.evaluate(X_test, y_test)
    assert "error" in res and "eff_mean" in res
    w2 = WrapRegressor(DecisionTreeRegressor(random_state=1))
    w2.fit(X_prop, y_prop)
    w2.calibrate(X_cal, y_cal, cps=True, seed=7)
    out = w2.predict_cps(X_test, lower_percentiles=2.5, higher_percentiles=97.5, seed=7)
    assert out is not None


def test_difficulty_estimator_and_knn():
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


def test_oob_paths_for_wrappers():
    class DummyLearner:
        def __init__(self):
            self.oob_decision_function_ = np.array([[0.4, 0.6]])
            self.classes_ = np.array([0, 1])

        def predict_proba(self, X):
            return np.array([[0.5, 0.5] for _ in range(len(X))])

    clf = DummyLearner()
    w = WrapClassifier(clf)
    X = np.zeros((1, 2))
    y = np.array([1])
    w.calibrate(X, y, oob=True)
    p = w.predict_p(X, smoothing=False)
    assert p.shape == (1, 2)

    class DummyReg:
        def __init__(self):
            self.oob_prediction_ = np.array([0.0])

        def predict(self, X):
            return np.zeros(len(X))

        def fit(self, X, y, **kwargs):
            return None

    wr = WrapRegressor(DummyReg())
    Xr = np.zeros((1, 1))
    yr = np.array([0.0])
    wr.fit(Xr, yr)
    wr.calibrate(Xr, yr, oob=True)
    intervals = wr.predict_int(Xr)
    assert intervals.shape[0] == 1


# ----- copied from legacy wrappers tests -----
def test_wrapclassifier_calibrate_class_cond_and_mc_function_from_legacy():
    class DummyLearner:
        def __init__(self):
            self.classes_ = np.array([0, 1])

        def predict_proba(self, X):
            n = X.shape[0]
            return np.tile(np.array([[0.7, 0.3]]), (n, 1))

        def fit(self, X, y, **kwargs):
            return self

    X = np.zeros((5, 2))
    y = np.array([0, 1, 0, 1, 0])
    learner = DummyLearner()
    w = WrapClassifier(learner)
    w.fit(X, y)
    def mc_func(X_in):
        return np.array([0 if i % 2 == 0 else 1 for i in range(X_in.shape[0])])
    w.calibrate(X, y, mc=mc_func)
    assert w.calibrated

    w2 = WrapClassifier(learner)
    w2.fit(X, y)
    w2.calibrate(X, y, class_cond=True)
    assert w2.calibrated


def test_conformal_classifier_alpha_index_all_labels_and_warning():
    from crepes.base import ConformalClassifier
    cc = ConformalClassifier()
    alphas_cal = np.array([0.1])
    cc.fit(alphas_cal)
    alphas_test = np.array([[0.2, 0.3]])
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ps = cc.predict_set(alphas_test, confidence=0.999, smoothing=False)
        assert ps.shape == (1, 2)
        assert ps[0].tolist() == [1, 1]
        assert any('too small' in str(x.message) for x in w)


def test_wrapregressor_predict_cps_requires_cps_flag():
    from crepes.base import WrapRegressor
    class DummyReg:
        def fit(self, X, y, **kwargs):
            return None
        def predict(self, X):
            return np.zeros(len(X))
    wr = WrapRegressor(DummyReg())
    X = np.zeros((3, 1))
    y = np.zeros(3)
    wr.fit(X, y)
    wr.calibrate(X, y, cps=False)
    try:
        wr.predict_cps(X)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_mondrian_categorizer_fit_repr():
    mc = MondrianCategorizer()
    # f returns the first column as category
    X = np.array([[0.1], [0.6], [0.4]])
    mc.fit(X, f=lambda X_in: X_in[:, 0], no_bins=2)
    r = repr(mc)
    assert 'MondrianCategorizer' in r

