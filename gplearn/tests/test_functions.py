"""Testing the Genetic Programming functions module."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np

from numpy import maximum

from gplearn.functions import _protected_sqrt, make_function
from gplearn.skutils.testing import assert_raises


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
