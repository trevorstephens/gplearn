"""Testing the Genetic Programming module's underlying datastructure
(gplearn.genetic._Program) as well as the classes that use it,
gplearn.genetic.SymbolicRegressor and gplearn.genetic.SymbolicTransformer."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from sklearn.utils.estimator_checks import check_estimator

from gplearn.genetic import SymbolicClassifier


def test_sklearn_estimator_checks_classifier():
    """Run the sklearn estimator validation checks on SymbolicClassifier"""

    check_estimator(SymbolicClassifier)
