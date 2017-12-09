"""Testing the Genetic Programming fitness module."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error
from sklearn.utils.testing import assert_equal, assert_raises
from sklearn.utils.validation import check_random_state

from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from gplearn.fitness import make_fitness, _mean_square_error

# load the boston dataset and randomly permute it
rng = check_random_state(0)
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_validate_fitness():
    """Check that valid fitness measures are accepted & invalid raise error"""

    # Check arg count checks
    fun = make_fitness(function=_mean_square_error, greater_is_better=True)
    # non-bool greater_is_better
    assert_raises(ValueError, make_fitness, _mean_square_error, 'Sure')
    assert_raises(ValueError, make_fitness, _mean_square_error, 1)

    # Check arg count tests
    def bad_fun1(x1, x2):
        return 1.0
    assert_raises(ValueError, make_fitness, bad_fun1, True)

    # Check return type tests
    def bad_fun2(x1, x2, w):
        return 'ni'
    assert_raises(ValueError, make_fitness, bad_fun2, True)


def test_validate_fitness():
    """Check that custom fitness functions are accepted"""

    def _custom_metric(y, y_pred, w):
        """Calculate the root mean square error."""
        return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))

    custom_metric = make_fitness(function=_custom_metric,
                                 greater_is_better=True)

    for Symbolic in (SymbolicRegressor, SymbolicTransformer):
        # These should be fine
        est = Symbolic(generations=2, random_state=0, metric=custom_metric)
        est.fit(boston.data, boston.target)


def test_customized_regressor_metrics():
    """Check whether greater_is_better works for SymbolicRegressor."""

    x_data = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_true = x_data[:, 0] ** 2 + x_data[:, 1] ** 2

    est_gp = SymbolicRegressor(metric='mean absolute error',
                               stopping_criteria=0.000001, random_state=415,
                               parsimony_coefficient=0.001, init_method='full',
                               init_depth=(2, 4))
    est_gp.fit(x_data, y_true)
    formula = est_gp.__str__()
    assert_equal('add(mul(X1, X1), mul(X0, X0))', formula, True)

    def neg_mean_absolute_error(y, y_pred, sample_weight):
        return -1 * mean_absolute_error(y, y_pred, sample_weight)

    customized_fitness = make_fitness(neg_mean_absolute_error,
                                      greater_is_better=True)

    c_est_gp = SymbolicRegressor(metric=customized_fitness,
                                 stopping_criteria=-0.000001, random_state=415,
                                 parsimony_coefficient=0.001, verbose=0,
                                 init_method='full', init_depth=(2, 4))
    c_est_gp.fit(x_data, y_true)
    c_formula = c_est_gp.__str__()
    assert_equal('add(mul(X1, X1), mul(X0, X0))', c_formula, True)


def test_customized_transformer_metrics():
    """Check whether greater_is_better works for SymbolicTransformer."""

    est_gp = SymbolicTransformer(generations=2, population_size=100,
                                 hall_of_fame=10, n_components=1,
                                 metric='pearson', random_state=415)
    est_gp.fit(boston.data, boston.target)
    for program in est_gp:
        formula = program.__str__()
    expected_formula = ('sub(div(mul(X4, X12), div(X9, X9)), '
                        'sub(div(X11, X12), add(X12, X0)))')
    assert_equal(expected_formula, formula, True)

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
    c_est_gp.fit(boston.data, boston.target)
    for program in c_est_gp:
        c_formula = program.__str__()
    assert_equal(expected_formula, c_formula, True)
