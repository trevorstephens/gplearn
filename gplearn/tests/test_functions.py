"""Testing the Genetic Programming functions module."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import pickle

import numpy as np
from numpy import maximum
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.utils._testing import assert_raises
from sklearn.utils.validation import check_random_state

from gplearn.functions import _protected_sqrt, make_function
from gplearn.genetic import SymbolicRegressor, SymbolicTransformer
from gplearn.genetic import SymbolicClassifier

# load the diabetes dataset and randomly permute it
rng = check_random_state(0)
diabetes = load_diabetes()
perm = rng.permutation(diabetes.target.size)
diabetes.data = diabetes.data[perm]
diabetes.target = diabetes.target[perm]

# load the breast cancer dataset and randomly permute it
cancer = load_breast_cancer()
perm = check_random_state(0).permutation(cancer.target.size)
cancer.data = cancer.data[perm]
cancer.target = cancer.target[perm]


def test_validate_function():
    """Check that valid functions are accepted & invalid ones raise error"""

    # Check arity tests
    _ = make_function(function=_protected_sqrt, name='sqrt', arity=1)
    # non-integer arity
    assert_raises(ValueError,
                  make_function,
                  function=_protected_sqrt,
                  name='sqrt',
                  arity='1')
    assert_raises(ValueError,
                  make_function,
                  function=_protected_sqrt,
                  name='sqrt',
                  arity=1.0)
    # non-bool wrap
    assert_raises(ValueError,
                  make_function,
                  function=_protected_sqrt,
                  name='sqrt',
                  arity=1,
                  wrap='f')
    # non-matching arity
    assert_raises(ValueError,
                  make_function,
                  function=_protected_sqrt,
                  name='sqrt',
                  arity=2)
    assert_raises(ValueError,
                  make_function,
                  function=maximum,
                  name='max',
                  arity=1)

    # Check name test
    assert_raises(ValueError,
                  make_function,
                  function=_protected_sqrt,
                  name=2,
                  arity=1)

    # Check return type tests
    def bad_fun1(x1, x2):
        return 'ni'
    assert_raises(ValueError,
                  make_function,
                  function=bad_fun1,
                  name='ni',
                  arity=2)

    # Check return shape tests
    def bad_fun2(x1):
        return np.ones((2, 1))
    assert_raises(ValueError,
                  make_function,
                  function=bad_fun2,
                  name='ni',
                  arity=1)

    # Check closure for negatives test
    def _unprotected_sqrt(x1):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sqrt(x1)
    assert_raises(ValueError,
                  make_function,
                  function=_unprotected_sqrt,
                  name='sqrt',
                  arity=1)

    # Check closure for zeros test
    def _unprotected_div(x1, x2):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.divide(x1, x2)
    assert_raises(ValueError,
                  make_function,
                  function=_unprotected_div,
                  name='div',
                  arity=2)


def test_function_in_program():
    """Check that using a custom function in a program works"""

    def logic(x1, x2, x3, x4):
        return np.where(x1 > x2, x3, x4)

    logical = make_function(function=logic,
                            name='logical',
                            arity=4)
    function_set = ['add', 'sub', 'mul', 'div', logical]
    est = SymbolicTransformer(generations=2, population_size=2000,
                              hall_of_fame=100, n_components=10,
                              function_set=function_set,
                              parsimony_coefficient=0.0005,
                              max_samples=0.9, random_state=0)
    est.fit(diabetes.data[:300, :], diabetes.target[:300])

    formula = est._programs[0][3].__str__()
    expected_formula = ('add(X3, logical(div(X5, sub(X5, X5)), '
                        'add(X9, -0.621), X8, X4))')
    assert(expected_formula == formula)


def test_parallel_custom_function():
    """Regression test for running parallel training with custom functions"""

    def _logical(x1, x2, x3, x4):
        return np.where(x1 > x2, x3, x4)

    logical = make_function(function=_logical,
                            name='logical',
                            arity=4)
    est = SymbolicRegressor(generations=2,
                            function_set=['add', 'sub', 'mul', 'div', logical],
                            random_state=0,
                            n_jobs=2)
    est.fit(diabetes.data, diabetes.target)
    _ = pickle.dumps(est)

    # Unwrapped functions should fail
    logical = make_function(function=_logical,
                            name='logical',
                            arity=4,
                            wrap=False)
    est = SymbolicRegressor(generations=2,
                            function_set=['add', 'sub', 'mul', 'div', logical],
                            random_state=0,
                            n_jobs=2)
    est.fit(diabetes.data, diabetes.target)
    assert_raises(AttributeError, pickle.dumps, est)

    # Single threaded will also fail in non-interactive sessions
    est = SymbolicRegressor(generations=2,
                            function_set=['add', 'sub', 'mul', 'div', logical],
                            random_state=0)
    est.fit(diabetes.data, diabetes.target)
    assert_raises(AttributeError, pickle.dumps, est)


def test_parallel_custom_transformer():
    """Regression test for running parallel training with custom transformer"""

    def _sigmoid(x1):
        with np.errstate(over='ignore', under='ignore'):
            return 1 / (1 + np.exp(-x1))

    sigmoid = make_function(function=_sigmoid,
                            name='sig',
                            arity=1)
    est = SymbolicClassifier(generations=2,
                             transformer=sigmoid,
                             random_state=0,
                             n_jobs=2)
    est.fit(cancer.data, cancer.target)
    _ = pickle.dumps(est)

    # Unwrapped functions should fail
    sigmoid = make_function(function=_sigmoid,
                            name='sig',
                            arity=1,
                            wrap=False)
    est = SymbolicClassifier(generations=2,
                             transformer=sigmoid,
                             random_state=0,
                             n_jobs=2)
    est.fit(cancer.data, cancer.target)
    assert_raises(AttributeError, pickle.dumps, est)

    # Single threaded will also fail in non-interactive sessions
    est = SymbolicClassifier(generations=2,
                             transformer=sigmoid,
                             random_state=0)
    est.fit(cancer.data, cancer.target)
    assert_raises(AttributeError, pickle.dumps, est)
