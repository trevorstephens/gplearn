import sklearn
from sklearn.datasets import *
from sklearn.linear_model import *
from gplearn.genetic import *
import numpy as np

fitness = [52, 23, 56, 14]
fitness = np.array(fitness)
print(fitness.argsort())
# ascending
exit()
rng = sklearn.utils.check_random_state(0)
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

est = Ridge()
est.fit(boston.data[:300, :], boston.target[:300])
print(est.score(boston.data[300:, :], boston.target[300:]))

function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']
gp = SymbolicTransformer(generations=10, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=1.0, verbose=1,
                         random_state=0, n_jobs=3)
gp.fit(boston.data[:10, :], boston.target[:10])

print(boston.data[:10, :])
gp_features = gp.transform(boston.data[:10, :])
print(gp_features)

# new_boston = np.hstack((boston.data, gp_features))
# print(gp)
# est = Ridge()
# est.fit(new_boston[:300, :], boston.target[:300])
# print(est.score(new_boston[300:, :], boston.target[300:]))
