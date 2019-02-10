"""Testing the Genetic Programming functions module."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from numpy import maximum
from sklearn.datasets import load_boston
from sklearn.utils.testing import assert_equal, assert_raises, assert_true
from sklearn.utils.validation import check_random_state

from gplearn.functions import make_function, _protected_sqrt, _protected_log
from gplearn.functions import _protected_division, _protected_inverse
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

def test_protected_functions():
    """Check that protected functions return expected values"""

    x = np.array([1e-5, -1e-4, 1e-1, -1, 10, -100])
    assert_true(np.allclose(_protected_division(x, x), 1.0))

    # Zero division by zero == 0 / eps -> 0
    assert_true(_protected_division(0, 0) == 0)

    x = np.array([1e-5, 1e-4, 1e-1, 1, 10, 100])
    ex = np.exp(x)
    assert_true(np.allclose(_protected_log(ex), x))

    # Protected log takes logarithm of absolute value of args
    assert_true(np.allclose(_protected_log(-x), _protected_log(x)))

    x = np.array([1, -2, -3, 10, -100, 1000])
    y = [1, -1/2, -1/3, 1e-1, -1e-2, 1e-3]
    assert_true(np.allclose(_protected_inverse(x), y))

    # Protected sqrt takes sqrt of absolute value of args
    x = np.array([0, -0, 1, -4, 9, -16, 25])
    y = [0, 0, 1, 2, 3, 4, 5]
    assert_true(np.allclose(_protected_sqrt(x), y))

if __name__ == "__main__":
    import nose
    nose.runmodule()
