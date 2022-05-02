"""Testing the examples from the documentation."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils.validation import check_random_state

from gplearn.genetic import SymbolicClassifier, SymbolicRegressor
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function


def test_symbolic_regressor():
    """Check that SymbolicRegressor example works"""

    rng = check_random_state(0)
    X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1
    X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_test = X_test[:, 0] ** 2 - X_test[:, 1] ** 2 + X_test[:, 1] - 1

    est_gp = SymbolicRegressor(population_size=5000, generations=20,
                               stopping_criteria=0.01, p_crossover=0.7,
                               p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                               p_point_mutation=0.1, max_samples=0.9,
                               parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_train, y_train)

    assert(len(est_gp._programs) == 7)
    expected = 'sub(add(-0.999, X1), mul(sub(X1, X0), add(X0, X1)))'
    assert(est_gp.__str__() == expected)
    assert_almost_equal(est_gp.score(X_test, y_test), 0.99999, decimal=5)
    dot_data = est_gp._program.export_graphviz()
    expected = ('digraph program {\nnode [style=filled]\n0 [label="sub", '
                'fillcolor="#136ed4"] ;\n1 [label="add", fillcolor="#136ed4"] '
                ';\n2 [label="-0.999", fillcolor="#60a6f6"] ;\n3 [label="X1", '
                'fillcolor="#60a6f6"] ;\n1 -> 3 ;\n1 -> 2 ;\n4 [label="mul", '
                'fillcolor="#136ed4"] ;\n5 [label="sub", fillcolor="#136ed4"] '
                ';\n6 [label="X1", fillcolor="#60a6f6"] ;\n7 [label="X0", '
                'fillcolor="#60a6f6"] ;\n5 -> 7 ;\n5 -> 6 ;\n8 [label="add", '
                'fillcolor="#136ed4"] ;\n9 [label="X0", fillcolor="#60a6f6"] '
                ';\n10 [label="X1", fillcolor="#60a6f6"] ;\n8 -> 10 ;\n8 -> 9 '
                ';\n4 -> 8 ;\n4 -> 5 ;\n0 -> 4 ;\n0 -> 1 ;\n}')
    assert(dot_data == expected)
    assert(est_gp._program.parents == {'method': 'Crossover',
                                       'parent_idx': 1555,
                                       'parent_nodes': range(1, 4),
                                       'donor_idx': 78,
                                       'donor_nodes': []})
    idx = est_gp._program.parents['donor_idx']
    fade_nodes = est_gp._program.parents['donor_nodes']
    assert(est_gp._programs[-2][idx].__str__() == 'add(-0.999, X1)')
    assert_almost_equal(est_gp._programs[-2][idx].fitness_, 0.351803319075)
    dot_data = est_gp._programs[-2][idx].export_graphviz(fade_nodes=fade_nodes)
    expected = ('digraph program {\nnode [style=filled]\n0 [label="add", '
                'fillcolor="#136ed4"] ;\n1 [label="-0.999", '
                'fillcolor="#60a6f6"] ;\n2 [label="X1", fillcolor="#60a6f6"] '
                ';\n0 -> 2 ;\n0 -> 1 ;\n}')
    assert(dot_data == expected)
    idx = est_gp._program.parents['parent_idx']
    fade_nodes = est_gp._program.parents['parent_nodes']
    expected = 'sub(sub(X1, 0.939), mul(sub(X1, X0), add(X0, X1)))'
    assert(est_gp._programs[-2][idx].__str__() == expected)
    assert_almost_equal(est_gp._programs[-2][idx].fitness_, 0.17080204042)
    dot_data = est_gp._programs[-2][idx].export_graphviz(fade_nodes=fade_nodes)
    expected = ('digraph program {\nnode [style=filled]\n0 [label="sub", '
                'fillcolor="#136ed4"] ;\n1 [label="sub", fillcolor="#cecece"] '
                ';\n2 [label="X1", fillcolor="#cecece"] ;\n3 [label="0.939", '
                'fillcolor="#cecece"] ;\n1 -> 3 ;\n1 -> 2 ;\n4 [label="mul", '
                'fillcolor="#136ed4"] ;\n5 [label="sub", fillcolor="#136ed4"] '
                ';\n6 [label="X1", fillcolor="#60a6f6"] ;\n7 [label="X0", '
                'fillcolor="#60a6f6"] ;\n5 -> 7 ;\n5 -> 6 ;\n8 [label="add", '
                'fillcolor="#136ed4"] ;\n9 [label="X0", fillcolor="#60a6f6"] '
                ';\n10 [label="X1", fillcolor="#60a6f6"] ;\n8 -> 10 ;\n8 -> 9 '
                ';\n4 -> 8 ;\n4 -> 5 ;\n0 -> 4 ;\n0 -> 1 ;\n}')
    assert(dot_data == expected)


def test_symbolic_transformer():
    """Check that SymbolicTransformer example works"""

    rng = check_random_state(0)
    diabetes = load_diabetes()
    perm = rng.permutation(diabetes.target.size)
    diabetes.data = diabetes.data[perm]
    diabetes.target = diabetes.target[perm]

    est = Ridge()
    est.fit(diabetes.data[:300, :], diabetes.target[:300])
    assert_almost_equal(est.score(diabetes.data[300:, :],
                                  diabetes.target[300:]),
                        desired=0.43406, decimal=5)

    function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                    'abs', 'neg', 'inv', 'max', 'min']
    gp = SymbolicTransformer(generations=20, population_size=2000,
                             hall_of_fame=100, n_components=10,
                             function_set=function_set,
                             parsimony_coefficient=0.0005,
                             max_samples=0.9,
                             random_state=0)
    gp.fit(diabetes.data[:300, :], diabetes.target[:300])

    gp_features = gp.transform(diabetes.data)
    new_diabetes = np.hstack((diabetes.data, gp_features))

    est = Ridge()
    est.fit(new_diabetes[:300, :], diabetes.target[:300])
    assert_almost_equal(est.score(new_diabetes[300:, :],
                                  diabetes.target[300:]),
                        desired=0.53368, decimal=5)


def test_custom_functions():
    """Test the custom programs example works"""

    rng = check_random_state(0)
    diabetes = load_diabetes()
    perm = rng.permutation(diabetes.target.size)
    diabetes.data = diabetes.data[perm]
    diabetes.target = diabetes.target[perm]

    def logic(x1, x2, x3, x4):
        return np.where(x1 > x2, x3, x4)

    logical = make_function(function=logic,
                            name='logical',
                            arity=4)

    function_set = ['add', 'sub', 'mul', 'div', logical]
    gp = SymbolicTransformer(generations=2, population_size=2000,
                             hall_of_fame=100, n_components=10,
                             function_set=function_set,
                             parsimony_coefficient=0.0005,
                             max_samples=0.9, random_state=0)

    gp.fit(diabetes.data[:300, :], diabetes.target[:300])

    expected = ('add(X3, logical(div(X5, sub(X5, X5)), '
                'add(X9, -0.621), X8, X4))')
    assert(gp._programs[0][3].__str__() == expected)

    dot_data = gp._programs[0][3].export_graphviz()
    expected = ('digraph program {\nnode [style=filled]\n0 [label="add", '
                'fillcolor="#136ed4"] ;\n1 [label="X3", fillcolor="#60a6f6"] ;'
                '\n2 [label="logical", fillcolor="#136ed4"] ;\n3 [label="div",'
                ' fillcolor="#136ed4"] ;\n4 [label="X5", fillcolor="#60a6f6"] '
                ';\n5 [label="sub", fillcolor="#136ed4"] ;\n6 [label="X5", '
                'fillcolor="#60a6f6"] ;\n7 [label="X5", fillcolor="#60a6f6"] '
                ';\n5 -> 7 ;\n5 -> 6 ;\n3 -> 5 ;\n3 -> 4 ;\n8 [label="add", '
                'fillcolor="#136ed4"] ;\n9 [label="X9", fillcolor="#60a6f6"] '
                ';\n10 [label="-0.621", fillcolor="#60a6f6"] ;\n8 -> 10 ;\n8 '
                '-> 9 ;\n11 [label="X8", fillcolor="#60a6f6"] ;\n12 '
                '[label="X4", fillcolor="#60a6f6"] ;\n2 -> 12 ;\n2 -> 11 ;\n2 '
                '-> 8 ;\n2 -> 3 ;\n0 -> 2 ;\n0 -> 1 ;\n}')
    assert(dot_data == expected)


def test_classifier_comparison():
    """Test the classifier comparison example works"""

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)
    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable]
    scores = []
    for ds in datasets:
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
        clf = SymbolicClassifier(random_state=0)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(('%.2f' % score).lstrip('0'))

    assert(scores == ['.95', '.93', '.95'])


def test_symbolic_classifier():
    """Check that SymbolicClassifier example works"""

    rng = check_random_state(0)
    cancer = load_breast_cancer()
    perm = rng.permutation(cancer.target.size)
    cancer.data = cancer.data[perm]
    cancer.target = cancer.target[perm]

    est = SymbolicClassifier(parsimony_coefficient=.01,
                             feature_names=cancer.feature_names,
                             random_state=1)
    est.fit(cancer.data[:400], cancer.target[:400])

    y_true = cancer.target[400:]
    y_score = est.predict_proba(cancer.data[400:])[:, 1]
    assert_almost_equal(roc_auc_score(y_true, y_score), 0.96937869822485212)

    dot_data = est._program.export_graphviz()
    expected = ('digraph program {\nnode [style=filled]\n0 [label="sub", '
                'fillcolor="#136ed4"] ;\n1 [label="div", fillcolor="#136ed4"] '
                ';\n2 [label="worst fractal dimension", fillcolor="#60a6f6"] '
                ';\n3 [label="mean concave points", fillcolor="#60a6f6"] '
                ';\n1 -> 3 ;\n1 -> 2 ;\n4 [label="mul", fillcolor="#136ed4"] '
                ';\n5 [label="mean concave points", fillcolor="#60a6f6"] ;\n6 '
                '[label="area error", fillcolor="#60a6f6"] ;\n4 -> 6 ;\n4 -> '
                '5 ;\n0 -> 4 ;\n0 -> 1 ;\n}')
    assert(dot_data == expected)
