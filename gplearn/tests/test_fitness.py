"""Testing the Genetic Programming fitness module."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import pickle

import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.metrics import mean_absolute_error
from sklearn.utils._testing import assert_raises
from sklearn.utils.validation import check_random_state

from gplearn.genetic import SymbolicRegressor, SymbolicClassifier
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import make_fitness, _mean_square_error

# load the breast cancer dataset and randomly permute it
cancer = load_breast_cancer()
perm = check_random_state(0).permutation(cancer.target.size)
cancer.data = cancer.data[perm]
cancer.target = cancer.target[perm]

# load the diabetes dataset and randomly permute it
diabetes = load_diabetes()
perm = check_random_state(0).permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]


def test_validate_fitness():
    """Check that valid fitness measures are accepted & invalid raise error"""

    # Check arg count checks
    _ = make_fitness(function=_mean_square_error, greater_is_better=True)
    # non-bool greater_is_better
    assert_raises(ValueError,
                  make_fitness,
                  function=_mean_square_error,
                  greater_is_better='Sure')
    assert_raises(ValueError,
                  make_fitness,
                  function=_mean_square_error,
                  greater_is_better=1)
    # non-bool wrap
    assert_raises(ValueError,
                  make_fitness,
                  function=_mean_square_error,
                  greater_is_better=True, wrap='f')

    # Check arg count tests
    def bad_fun1(x1, x2):
        return 1.0
    assert_raises(ValueError,
                  make_fitness,
                  function=bad_fun1,
                  greater_is_better=True)

    # Check return type tests
    def bad_fun2(x1, x2, w):
        return 'ni'
    assert_raises(ValueError,
                  make_fitness,
                  function=bad_fun2,
                  greater_is_better=True)

    def _custom_metric(y, y_pred, w):
        """Calculate the root mean square error."""
        return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))

    custom_metric = make_fitness(function=_custom_metric,
                                 greater_is_better=True)

    for Symbolic in (SymbolicRegressor, SymbolicTransformer):
        # These should be fine
        est = Symbolic(generations=2, random_state=0, metric=custom_metric)
        est.fit(diabetes.data, diabetes.target)


def test_custom_regressor_metrics():
    """Check whether greater_is_better works for SymbolicRegressor."""

    x_data = check_random_state(0).uniform(-1, 1, 100).reshape(50, 2)
    y_true = x_data[:, 0] ** 2 + x_data[:, 1] ** 2

    est_gp = SymbolicRegressor(metric='mean absolute error',
                               stopping_criteria=0.000001, random_state=415,
                               parsimony_coefficient=0.001, init_method='full',
                               init_depth=(2, 4))
    est_gp.fit(x_data, y_true)
    formula = est_gp.__str__()
    assert('add(mul(X0, X0), mul(X1, X1))' == formula)

    def neg_mean_absolute_error(y, y_pred, sample_weight):
        return -1 * mean_absolute_error(y, y_pred,
                                        sample_weight=sample_weight)

    customized_fitness = make_fitness(function=neg_mean_absolute_error,
                                      greater_is_better=True)

    c_est_gp = SymbolicRegressor(metric=customized_fitness,
                                 stopping_criteria=-0.000001, random_state=415,
                                 parsimony_coefficient=0.001, verbose=0,
                                 init_method='full', init_depth=(2, 4))
    c_est_gp.fit(x_data, y_true)
    c_formula = c_est_gp.__str__()
    assert('add(mul(X0, X0), mul(X1, X1))' == c_formula)


def test_custom_transformer_metrics():
    """Check whether greater_is_better works for SymbolicTransformer."""

    est_gp = SymbolicTransformer(generations=2, population_size=100,
                                 hall_of_fame=10, n_components=1,
                                 metric='pearson', random_state=415)
    est_gp.fit(diabetes.data, diabetes.target)
    for program in est_gp:
        formula = program.__str__()
    expected_formula = 'mul(-0.111, add(add(X9, sub(X2, 0.606)), X3))'
    assert(expected_formula == formula)

    def _neg_weighted_pearson(y, y_pred, w):
        """Calculate the weighted Pearson correlation coefficient."""
        with np.errstate(divide='ignore', invalid='ignore'):
            y_pred_demean = y_pred - np.average(y_pred, weights=w)
            y_demean = y - np.average(y, weights=w)
            corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                    np.sqrt((np.sum(w * y_pred_demean ** 2) *
                             np.sum(w * y_demean ** 2)) /
                            (np.sum(w) ** 2)))
        if np.isfinite(corr):
            return -1 * np.abs(corr)
        return 0.

    neg_weighted_pearson = make_fitness(function=_neg_weighted_pearson,
                                        greater_is_better=False)

    c_est_gp = SymbolicTransformer(generations=2, population_size=100,
                                   hall_of_fame=10, n_components=1,
                                   stopping_criteria=-1,
                                   metric=neg_weighted_pearson,
                                   random_state=415)
    c_est_gp.fit(diabetes.data, diabetes.target)
    for program in c_est_gp:
        c_formula = program.__str__()
    assert(expected_formula == c_formula)


def test_custom_classifier_metrics():
    """Check whether greater_is_better works for SymbolicClassifier."""

    x_data = check_random_state(0).uniform(-1, 1, 100).reshape(50, 2)
    y_true = x_data[:, 0] ** 2 + x_data[:, 1] ** 2
    y_true = (y_true < y_true.mean()).astype(int)

    est_gp = SymbolicClassifier(metric='log loss',
                                stopping_criteria=0.000001,
                                random_state=415,
                                parsimony_coefficient=0.01,
                                init_method='full',
                                init_depth=(2, 4))
    est_gp.fit(x_data, y_true)
    formula = est_gp.__str__()
    expected_formula = 'sub(0.364, mul(add(X0, X0), add(X0, X0)))'
    assert(expected_formula == formula)

    def negative_log_loss(y, y_pred, w):
        """Calculate the log loss."""
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        score = y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        return np.average(score, weights=w)

    customized_fitness = make_fitness(function=negative_log_loss,
                                      greater_is_better=True)

    c_est_gp = SymbolicClassifier(metric=customized_fitness,
                                  stopping_criteria=0.000001,
                                  random_state=415,
                                  parsimony_coefficient=0.01,
                                  init_method='full',
                                  init_depth=(2, 4))
    c_est_gp.fit(x_data, y_true)
    c_formula = c_est_gp.__str__()
    assert(expected_formula == c_formula)


def test_parallel_custom_metric():
    """Regression test for running parallel training with custom transformer"""

    def _custom_metric(y, y_pred, w):
        """Calculate the root mean square error."""
        return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))

    custom_metric = make_fitness(function=_custom_metric,
                                 greater_is_better=True)
    est = SymbolicRegressor(generations=2,
                            metric=custom_metric,
                            random_state=0,
                            n_jobs=2)
    est.fit(diabetes.data, diabetes.target)
    _ = pickle.dumps(est)

    # Unwrapped functions should fail
    custom_metric = make_fitness(function=_custom_metric,
                                 greater_is_better=True,
                                 wrap=False)
    est = SymbolicRegressor(generations=2,
                            metric=custom_metric,
                            random_state=0,
                            n_jobs=2)
    est.fit(diabetes.data, diabetes.target)
    assert_raises(AttributeError, pickle.dumps, est)

    # Single threaded will also fail in non-interactive sessions
    est = SymbolicRegressor(generations=2,
                            metric=custom_metric,
                            random_state=0)
    est.fit(diabetes.data, diabetes.target)
    assert_raises(AttributeError, pickle.dumps, est)
