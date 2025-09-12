import sys
import types
import pytest


def pytest_sessionstart(session):
    # Provide light stubs for heavy optional deps so import works without install
    if 'matplotlib' not in sys.modules:
        matplotlib = types.ModuleType('matplotlib')
        pyplot = types.ModuleType('matplotlib.pyplot')
        def _noop(*args, **kwargs):
            return None
        # Minimal API used in code
        class _Style:
            def use(self, *_args, **_kwargs):
                return None
        pyplot.style = _Style()
        pyplot.figure = _noop
        pyplot.subplots = lambda *a, **k: ((None, None), (None, None)) if a and a[0] == 2 else (None, None, None)
        pyplot.plot = _noop
        pyplot.axhline = _noop
        pyplot.tight_layout = _noop
        pyplot.savefig = _noop
        pyplot.show = _noop
        matplotlib.pyplot = pyplot
        sys.modules['matplotlib'] = matplotlib
        sys.modules['matplotlib.pyplot'] = pyplot

    if 'seaborn' not in sys.modules:
        seaborn = types.ModuleType('seaborn')
        def set_palette(*args, **kwargs):
            return None
        seaborn.set_palette = set_palette
        sys.modules['seaborn'] = seaborn

    if 'scipy' not in sys.modules:
        scipy = types.ModuleType('scipy')
        ndimage = types.ModuleType('scipy.ndimage')
        def gaussian_filter1d(x, sigma):
            return x
        ndimage.gaussian_filter1d = gaussian_filter1d
        scipy.ndimage = ndimage
        sys.modules['scipy'] = scipy
        sys.modules['scipy.ndimage'] = ndimage

    # Minimal sklearn stubs
    if 'sklearn' not in sys.modules:
        sklearn = types.ModuleType('sklearn')
        sys.modules['sklearn'] = sklearn

        # preprocessing.StandardScaler
        preprocessing = types.ModuleType('sklearn.preprocessing')
        class _StdScaler:
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                return X
            def fit_transform(self, X, y=None):
                return X
        preprocessing.StandardScaler = _StdScaler
        sys.modules['sklearn.preprocessing'] = preprocessing

        # model_selection APIs
        model_selection = types.ModuleType('sklearn.model_selection')
        def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
            n = len(X)
            split = int(n * (1 - test_size))
            return X[:split], X[split:], y[:split], y[split:]
        def cross_val_score(*args, **kwargs):
            return [0.0]
        class GridSearchCV:
            def __init__(self, estimator, param_grid, cv=3, scoring=None, n_jobs=None):
                self.best_estimator_ = estimator
                self.best_params_ = {}
            def fit(self, X, y):
                return self
        model_selection.train_test_split = train_test_split
        model_selection.cross_val_score = cross_val_score
        model_selection.GridSearchCV = GridSearchCV
        sys.modules['sklearn.model_selection'] = model_selection

        # ensemble classifiers
        ensemble = types.ModuleType('sklearn.ensemble')
        class RandomForestClassifier:
            def __init__(self, *args, **kwargs):
                pass
            def fit(self, X, y):
                return self
            def predict(self, X):
                import numpy as _np
                return _np.zeros(len(X))
            def predict_proba(self, X):
                import numpy as _np
                return _np.c_[1 - _np.zeros(len(X)), _np.zeros(len(X))]
            @property
            def feature_importances_(self):
                return []
        class GradientBoostingClassifier(RandomForestClassifier):
            pass
        ensemble.RandomForestClassifier = RandomForestClassifier
        ensemble.GradientBoostingClassifier = GradientBoostingClassifier
        sys.modules['sklearn.ensemble'] = ensemble

        # linear_model.LogisticRegression
        linear_model = types.ModuleType('sklearn.linear_model')
        class LogisticRegression(RandomForestClassifier):
            pass
        linear_model.LogisticRegression = LogisticRegression
        sys.modules['sklearn.linear_model'] = linear_model

        # metrics.classification_report
        metrics = types.ModuleType('sklearn.metrics')
        def classification_report(y_true, y_pred, output_dict=False):
            return {"dummy": 1.0} if output_dict else ""
        metrics.classification_report = classification_report
        sys.modules['sklearn.metrics'] = metrics

    # joblib stub
    if 'joblib' not in sys.modules:
        joblib = types.ModuleType('joblib')
        def dump(obj, path):
            return None
        def load(path):
            return {}
        joblib.dump = dump
        joblib.load = load
        sys.modules['joblib'] = joblib

