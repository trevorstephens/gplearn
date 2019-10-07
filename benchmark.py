from gplearn.genetic import SymbolicClassifier
from gplearn.functions import _function_map
from sklearn.utils.random import check_random_state
from sklearn.datasets import load_breast_cancer

from time import time

# Training samples
rng = check_random_state(0)
cancer = load_breast_cancer()
perm = rng.permutation(cancer.target.size)
cancer.data = cancer.data[perm]
cancer.target = cancer.target[perm]

t0 = time()

est = SymbolicClassifier(
    population_size=5000, generations=20,
    function_set=_function_map.keys(),
    parsimony_coefficient=0.001,
    feature_names=cancer.feature_names,
    random_state=1,
    verbose=True
)
est.fit(cancer.data[:400], cancer.target[:400])


print(f"GP Result in {time()-t0} seconds:", est._program)
