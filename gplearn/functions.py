"""The functions used to create programs.

The :mod:`gplearn.functions` module contains all of the functions used by
gplearn programs. It also contains helper methods for a user to define their
own custom functions.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.externals import six
import operator
import math

__all__ = ['make_function']
from sympy import *
import sympy.mpmath as gg
#m, n = 2, 5.9
def func1(a,b):
    return abs(a) * abs(b) #* (cos(a)+cos(b))
    #lambda x: legendre(m,x)*legendre(n,x)
def zegax(m,n):
    return quak(m,n)

def quak(m,n):
    q = []
    for x in np.arange(-1,1,.0001):
        q.append(func1(m,x)*func1(n,x))
    return sum(q)/10000

def remainder(x1,x2):
    return np.remainder(x1,x2)
def heaviside(x1,x2):
    return np.heaviside(x1,x2)
def hypot(x1,x2):
    return np.hypot(x1,x2)
def cbrt(x):
    return np.cbrt(x)
def ceil(x):
    return np.ceil(x)
def fabs(x):
    return np.fabs(x)
def factorial(x):
    return np.factorial(x)
def floor(x):
    return np.floor(x)
def frexp(x):
    return np.frexp(x)
def trunc(x):
    return np.trunc(x)
def exp(x):
    return np.exp(x)
def expm1(x):
    return np.expm1(x)
def arccos(x):
    return np.arccos(x)
def arcsin(x):
    return np.arcsin(x)
def arctan(x):
    return np.arctan(x)
def degrees(x):
    return np.degrees(x)
def radians(x):
    return np.radians(x)
def cosh(x):
    return np.cosh(x)
def sinh(x):
    return np.sinh(x)
def tanh(x):
    return np.tanh(x)
def gamma(x):
    return np.gamma(x)
def lgamma(x):
    return np.lgamma(x)

def nnnfunc0(x1, x2):
    return np.maximum(np.maximum(x1, x1), (x1/x1)), np.maximum(np.maximum(x2, x2), (x2/x2))

def nnnfunc(x1):
    try:
        return np.maximum(abs(np.negative(x1)), math.sqrt(math.cos(math.log(x1))))
    except:
        return (x1-x1)+1



def andB(a,b):
    return a and b 
def orB(a,b):
    return a or b
def xorB(a,b):
    return operator.xor(a,b)


"""
import GP as IAMANHPI
proggenetic = IAMANHPI(500,100,0.1)
proggenetic.load("nonnulpass.model")
"""
def modulox(x1, x2):
    return operator.mod(nnnfunc(x1),nnnfunc(x2))
    #abs(nonzero(nonnegativ(x1)) % nonzero(nonnegativ(x2)))

from scipy.stats import logistic
def sigmoid(x):
  return logistic.cdf(x)

class _Function(object):

    """A representation of a npematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a npematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(function, name, arity):
    """Make a function node, a representation of a npematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a npematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if six.get_function_code(function).co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity,
                                six.get_function_code(function).co_argcount))
    if not isinstance(name, six.string_types):
        raise ValueError('name must be a string, got %s' % type(name))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    return _Function(function, name, arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)

add2 = make_function(function=np.add, name='add', arity=2)
sub2 = make_function(function=np.subtract, name='sub', arity=2)
mul2 = make_function(function=np.multiply, name='mul', arity=2)
div2 = make_function(function=_protected_division, name='div', arity=2)
sqrt1 = make_function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = make_function(function=_protected_log, name='log', arity=1)
neg1 = make_function(function=np.negative, name='neg', arity=1)
inv1 = make_function(function=_protected_inverse, name='inv', arity=1)
abs1 = make_function(function=np.abs, name='abs', arity=1)
max2 = make_function(function=np.maximum, name='max', arity=2)
min2 = make_function(function=np.minimum, name='min', arity=2)
sin1 = make_function(function=np.sin, name='sin', arity=1)
cos1 = make_function(function=np.cos, name='cos', arity=1)
tan1 = make_function(function=np.tan, name='tan', arity=1)

sigmoid1 = make_function(function=sigmoid, name='sigmoid', arity=1)
ceil1 = make_function(function=ceil, name='ceil', arity=1)
fabs1 = make_function(function=fabs, name='fabs', arity=1)
floor1 = make_function(function=floor, name='floor', arity=1)
trunc1 = make_function(function=trunc, name='trunc', arity=1)
exp1 = make_function(function=exp, name='exp', arity=1)
expm11 = make_function(function=expm1, name='expm1', arity=1)
arccos1 = make_function(function=arccos, name='arccos', arity=1)
arcsin1 = make_function(function=arcsin, name='arcsin', arity=1)
arctan1 = make_function(function=arctan, name='arctan', arity=1)
cosh1 = make_function(function=cosh, name='cosh', arity=1)
sinh1 = make_function(function=sinh, name='sinh', arity=1)
tanh1 = make_function(function=tanh, name='tanh', arity=1)
cbrt1 = make_function(function=cbrt, name='cbrt', arity=1)
zegax1 = make_function(function=zegax, name='zegax', arity=2)
hypot1 = make_function(function=hypot, name='hypot', arity=2)
modulo1 = make_function(function=modulox, name='modulox', arity=2)
heaviside1 = make_function(function=heaviside, name='heaviside', arity=2)

and1 = make_function(function=andB, name='and', arity=2)
or1 = make_function(function=orB, name='or', arity=2)
xor1 = make_function(function=xorB, name='xor', arity=2)
_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1,
                 'ceil': ceil1,
                 'fabs': fabs1,
                 'floor': floor1,
                 'trunc': trunc1,
                 'cbrt': cbrt1,
                 'hypot': hypot1,
                 'heaviside': heaviside1,
                 'zegax': zegax1,
                 'modulox': modulo1,
                 'sigmoid': sigmoid1,
                 'and': and1,
                 'or': or1,
                 'xor': xor1 }#'modulo': modulo1,
