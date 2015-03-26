import warnings

import numpy as np

from sklearn.utils.testing import (assert_equal, assert_raises, assert_true,
                                   assert_almost_equal, assert_array_equal,
                                   SkipTest)

from sklearn.utils import check_random_state
from sklearn.utils import deprecated


def test_make_rng():
    """Check the check_random_state utility function behavior"""
    assert_true(check_random_state(None) is np.random.mtrand._rand)
    assert_true(check_random_state(np.random) is np.random.mtrand._rand)

    rng_42 = np.random.RandomState(42)
    assert_true(check_random_state(42).randint(100) == rng_42.randint(100))

    rng_42 = np.random.RandomState(42)
    assert_true(check_random_state(rng_42) is rng_42)

    rng_42 = np.random.RandomState(42)
    assert_true(check_random_state(43).randint(100) != rng_42.randint(100))

    assert_raises(ValueError, check_random_state, "some invalid seed")

