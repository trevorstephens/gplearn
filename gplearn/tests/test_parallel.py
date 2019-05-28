"""Testing the Genetic Programming module's underlying datastructure
(gplearn.genetic._Program) as well as the classes that use it,
gplearn.genetic.SymbolicRegressor and gplearn.genetic.SymbolicTransformer."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import pickle
import sys
from io import StringIO

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_equal, assert_almost_equal
from sklearn.utils.testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils.testing import assert_raises, assert_warns
from sklearn.utils.validation import check_random_state

from gplearn.genetic import SymbolicClassifier, SymbolicRegressor
from gplearn.genetic import SymbolicTransformer
from gplearn.fitness import weighted_pearson, weighted_spearman
from gplearn._program import _Program
from gplearn.fitness import _fitness_map
from gplearn.functions import (add2, sub2, mul2, div2, sqrt1, log1, abs1, max2,
                               min2)
from gplearn.functions import _Function, make_function
from gplearn.tests.check_estimator import custom_check_estimator
from gplearn.tests.check_estimator import rewritten_check_estimator

# load the boston dataset and randomly permute it
rng = check_random_state(0)
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

# load the breast cancer dataset and randomly permute it
rng = check_random_state(0)
cancer = load_breast_cancer()
perm = rng.permutation(cancer.target.size)
cancer.data = cancer.data[perm]
cancer.target = cancer.target[perm]


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


def test_parallel_custom_fun():
    """Regression test for running parallel training with custom functions"""

    def _logical(x1, x2, x3, x4):
        return np.where(x1 > x2, x3, x4)

    logical = make_function(function=_logical,
                            name='logical',
                            arity=4)

    est = SymbolicRegressor(generations=5,
                            function_set=('add', 'sub', 'mul', 'div', logical),
                            n_jobs=2)
    est.fit(boston.data, boston.target)
    pickle_object = pickle.dumps(est)
