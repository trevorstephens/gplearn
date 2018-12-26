"""Testing the Genetic Programming functions module."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from numpy import maximum
from sklearn.datasets import load_boston
from sklearn.utils.testing import assert_equal, assert_raises
from sklearn.utils.validation import check_random_state

from gplearn.functions import _protected_sqrt, make_function
from gplearn.genetic import SymbolicTransformer

# load the boston dataset and randomly permute it
rng = check_random_state(0)
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_validate_function():
    """Check that valid functions are accepted & invalid ones raise error"""

    # Check arity tests
    fun = make_function(function=_protected_sqrt, name='sqrt', arity=1)
    # non-integer arity
    assert_raises(ValueError, make_function, _protected_sqrt, 'sqrt', '1')
    assert_raises(ValueError, make_function, _protected_sqrt, 'sqrt', 1.0)
    # non-matching arity
    assert_raises(ValueError, make_function, _protected_sqrt, 'sqrt', 2)
    assert_raises(ValueError, make_function, maximum, 'max', 1)

    # Check name test
    assert_raises(ValueError, make_function, _protected_sqrt, 2, 1)

    # Check return type tests
    def bad_fun1(x1, x2):
        return 'ni'
    assert_raises(ValueError, make_function, bad_fun1, 'ni', 2)

    # Check return shape tests
    def bad_fun2(x1):
        return np.ones((2, 1))
    assert_raises(ValueError, make_function, bad_fun2, 'ni', 1)

    # Check closure for negatives test
    def _unprotected_sqrt(x1):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.sqrt(x1)
    assert_raises(ValueError, make_function, _unprotected_sqrt, 'sqrt', 1)

    # Check closure for zeros test
    def _unprotected_div(x1, x2):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.divide(x1, x2)
    assert_raises(ValueError, make_function, _unprotected_div, 'div', 2)


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
    est.fit(boston.data[:300, :], boston.target[:300])

    formula = est._programs[0][906].__str__()
    expected_formula = 'sub(logical(X6, add(X11, 0.898), X10, X2), X5)'
    assert_equal(expected_formula, formula, True)
