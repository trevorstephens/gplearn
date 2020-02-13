"""Testing the utils module."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.utils._testing import assert_raises

from gplearn.utils import _get_n_jobs, check_random_state, cpu_count


def test_check_random_state():
    """Check the check_random_state utility function behavior"""

    assert(check_random_state(None) is np.random.mtrand._rand)
    assert(check_random_state(np.random) is np.random.mtrand._rand)

    rng_42 = np.random.RandomState(42)
    assert(check_random_state(42).randint(100) == rng_42.randint(100))

    rng_42 = np.random.RandomState(42)
    assert(check_random_state(rng_42) is rng_42)

    rng_42 = np.random.RandomState(42)
    assert(check_random_state(43).randint(100) != rng_42.randint(100))

    assert_raises(ValueError, check_random_state, "some invalid seed")


def test_get_n_jobs():
    """Check that _get_n_jobs returns expected values"""

    jobs = _get_n_jobs(4)
    assert(jobs == 4)

    jobs = -2
    expected = cpu_count() + 1 + jobs
    jobs = _get_n_jobs(jobs)
    assert(jobs == expected)

    assert_raises(ValueError, _get_n_jobs, 0)
