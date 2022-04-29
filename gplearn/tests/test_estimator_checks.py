"""Testing the Genetic Programming module's underlying datastructure
(gplearn.genetic._Program) as well as the classes that use it,
gplearn.genetic.SymbolicRegressor and gplearn.genetic.SymbolicTransformer."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from sklearn.utils.estimator_checks import check_estimator

from gplearn.genetic import SymbolicClassifier, SymbolicRegressor
from gplearn.genetic import SymbolicTransformer


def test_sklearn_regressor_checks():
    """Run the sklearn estimator validation checks on SymbolicRegressor"""

    check_estimator(SymbolicRegressor(population_size=1000,
                                      generations=5))


def test_sklearn_classifier_checks():
    """Run the sklearn estimator validation checks on SymbolicClassifier"""

    check_estimator(SymbolicClassifier(population_size=50,
                                       generations=5))


def test_sklearn_transformer_checks():
    """Run the sklearn estimator validation checks on SymbolicTransformer"""

    check_estimator(SymbolicTransformer(population_size=50,
                                        hall_of_fame=10,
                                        generations=5))
