"""Testing the Genetic Programming fitness module."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from gplearn.fitness import make_fitness, _mean_square_error
from gplearn.skutils.testing import assert_raises


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
