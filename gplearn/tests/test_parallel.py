"""Testing the Genetic Programming module's underlying datastructure
(gplearn.genetic._Program) as well as the classes that use it,
gplearn.genetic.SymbolicRegressor and gplearn.genetic.SymbolicTransformer."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import pickle

import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.validation import check_random_state

from gplearn.genetic import SymbolicClassifier, SymbolicRegressor
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function

# load the boston dataset and randomly permute it
rng = check_random_state(0)
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]


def test_parallel_train():
    """Check predictions are the same for different n_jobs"""

    # Check the regressor
    ests = [
        SymbolicRegressor(population_size=100, generations=4, n_jobs=n_jobs,
                          random_state=0).fit(boston.data[:100, :],
                                              boston.target[:100])
        for n_jobs in [1, 2, 3, 8, 16]
    ]

    preds = [e.predict(boston.data[500:, :]) for e in ests]
    for pred1, pred2 in zip(preds, preds[1:]):
        assert_array_almost_equal(pred1, pred2)
    lengths = np.array([[gp.length_ for gp in e._programs[-1]] for e in ests])
    for len1, len2 in zip(lengths, lengths[1:]):
        assert_array_almost_equal(len1, len2)

    # Check the transformer
    ests = [
        SymbolicTransformer(population_size=100, hall_of_fame=50,
                            generations=4, n_jobs=n_jobs,
                            random_state=0).fit(boston.data[:100, :],
                                                boston.target[:100])
        for n_jobs in [1, 2, 3, 8, 16]
    ]

    preds = [e.transform(boston.data[500:, :]) for e in ests]
    for pred1, pred2 in zip(preds, preds[1:]):
        assert_array_almost_equal(pred1, pred2)
    lengths = np.array([[gp.length_ for gp in e._programs[-1]] for e in ests])
    for len1, len2 in zip(lengths, lengths[1:]):
        assert_array_almost_equal(len1, len2)

    # Check the classifier
    ests = [
        SymbolicClassifier(population_size=100, generations=4, n_jobs=n_jobs,
                           random_state=0).fit(cancer.data[:100, :],
                                               cancer.target[:100])
        for n_jobs in [1, 2, 3, 8, 16]
    ]

    preds = [e.predict(cancer.data[500:, :]) for e in ests]
    for pred1, pred2 in zip(preds, preds[1:]):
        assert_array_almost_equal(pred1, pred2)
    lengths = np.array([[gp.length_ for gp in e._programs[-1]] for e in ests])
    for len1, len2 in zip(lengths, lengths[1:]):
        assert_array_almost_equal(len1, len2)


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
    est.fit(boston.data, boston.target)
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
    est.fit(boston.data, boston.target)
    assert_raises(AttributeError, pickle.dumps, est)
