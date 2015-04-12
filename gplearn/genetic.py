"""Genetic Programming in Python, with a scikit-learn inspired API

The :mod:`sklearn.genetic` module implements Genetic Programming. These
are supervised learning methods based on applying evolutionary operations on
computer programs.
"""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
import itertools

from copy import deepcopy
from time import time

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.random import sample_without_replacement

from .skutils import _partition_estimators
from .skutils.fixes import bincount
from .skutils.validation import check_random_state, NotFittedError
from .skutils.validation import check_X_y, check_array, column_or_1d

__all__ = ['SymbolicRegressor']

MAX_INT = np.iinfo(np.int32).max


def protected_devision(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def protected_log(x1):
    """Closure of log for zero arguments."""
    return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def protected_inverse(x1):
    """Closure of log for zero arguments."""
    return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


# Format is '<name><arity>': function
FUNCTIONS = {'add2': np.add,
             'sub2': np.subtract,
             'mul2': np.multiply,
             'div2': protected_devision,
             'sqrt1': protected_sqrt,
             'log1': protected_log,
             'neg1': np.negative,
             'inv1': protected_inverse,
             'abs1': np.abs,
             'max2': np.maximum,
             'min2': np.minimum,
             'sin1': np.sin,
             'cos1': np.cos,
             'tan1': np.tan}


def _parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
    """Private function used to build a batch of programs within a job."""
    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['metric']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    bootstrap = params['bootstrap']
    max_samples = params['max_samples']

    max_samples = int(max_samples * n_samples)

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           random_state=random_state,
                           program=program)

        program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()

        if bootstrap:
            indices = random_state.randint(0, n_samples, max_samples)
            sample_counts = bincount(indices, minlength=n_samples)
            curr_sample_weight *= sample_counts
        else:
            not_indices = sample_without_replacement(
                n_samples,
                n_samples - max_samples,
                random_state=random_state)
            curr_sample_weight[not_indices] = 0

        program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)

        programs.append(program)

    return programs


class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in this
    module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program, must match keys from
        the FUNCTIONS dict global variable.

    arities : dict
        A dictionary of the form `{arity: [function names]}`. The arity is the
        number of arguments that the function takes, the function names must
        match those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : str
        The name of the raw fitness metric. Available options include
        'mean absolute error', 'mse', 'rmse' and 'rmsle'.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.
    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None

    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        if self.init_method == 'half and half':
            method = ['grow', 'full'][random_state.randint(2)]
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        program = [function]
        terminal_stack = [int(function[-1])]

        while len(terminal_stack) != 0:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(int(function[-1]))
            else:
                # We need a terminal, add a variable or constant
                terminal = random_state.randint(self.n_features + 1)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if len(terminal_stack) == 0:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, six.string_types):
                terminals.append(int(node[-1]))
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, six.string_types):
                terminals.append(int(node[-1]))
                output += node[:-1] + '('
            else:
                if isinstance(node, int):
                    output += 'X%s' % node
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self):
        """Returns a string, Graphviz script for visualizing the program."""
        terminals = []
        output = "digraph program {\nnode [style=filled]"
        for i, node in enumerate(self.program):
            if isinstance(node, six.string_types):
                terminals.append([int(node[-1]), i])
                output += ('%d [label="%s", fillcolor="#3499cd"] ;\n'
                           % (i, node[:-1]))
            else:
                if isinstance(node, int):
                    output += ('%d [label="%s%s", fillcolor="#f89939"] ;\n'
                               % (i, 'X', node))
                else:
                    output += ('%d [label="%.3f", fillcolor="#f89939"] ;\n'
                               % (i, node))
                if i == 0:
                    # A degenerative program of only one node
                    return output + "}"
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if len(terminals) == 0:
                            return output + "}"
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, six.string_types):
                terminals.append(int(node[-1]))
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.
        """
        # Stop warnings being raised for protected division, etc
        old_settings = np.seterr(divide='ignore', invalid='ignore')

        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, six.string_types):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == int(apply_stack[-1][0][-1]) + 1:
                # Apply functions that have sufficient arguments
                function = FUNCTIONS[apply_stack[-1][0]]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    np.seterr(**old_settings)
                    # Protect for rmsle:
                    if self.metric == 'rmsle':
                        intermediate_result[intermediate_result <= 1e-16] = 0
                    return intermediate_result

        # We should never get here
        np.seterr(**old_settings)
        return None

    def raw_fitness(self, X, y, sample_weight=None):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.
        """
        y_pred = self.execute(X)

        if self.metric == 'mean absolute error':
            raw_fitness = np.average(np.abs(y_pred - y),
                                     weights=sample_weight)

        elif self.metric == 'mse':
            raw_fitness = np.average(((y_pred - y) ** 2),
                                     weights=sample_weight)

        elif self.metric == 'rmse':
            raw_fitness = np.sqrt(np.average(((y_pred - y) ** 2),
                                             weights=sample_weight))

        elif self.metric == 'rmsle':
            raw_fitness = np.sqrt(np.average((np.log(y_pred + 1) -
                                              np.log(y + 1)) ** 2,
                                             weights=sample_weight))

        else:
            raise ValueError('Unsupported metric: %s' % self.metric)

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.
        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        return self.raw_fitness_ + (parsimony_coefficient * len(self.program))

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.
        """
        if program is None:
            program = self.program
        probs = np.array([0.9 if isinstance(node, six.string_types) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, six.string_types):
                stack += int(node[-1])
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return deepcopy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.
        """
        program = deepcopy(self.program)

        # Get the nodes to modify
        mutate = np.where([True if (random_state.uniform() <
                                    self.p_point_replace)
                           else False
                           for _ in range(len(program))])[0]

        for node in mutate:
            if isinstance(program[node], six.string_types):
                arity = int(program[node][-1])
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                terminal = random_state.randint(self.n_features + 1)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)


class SymbolicRegressor(BaseEstimator, RegressorMixin):

    """A Genetic Programming symbolic regressor.

    A symbolic regressor is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional (default=500)
        The number of programs in each generation.

    generations : integer, optional (default=10)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    const_range : tuple of two floats, optional (default=(-1., 1.))
        The range of constants to include in the formulas.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    transformer : bool, optional (default=True)
        Whether to include protected square root, protected log, absolute
        value, negative, and inverse functions in the function set.

    comparison : bool, optional (default=True)
        Whether to include maximum and minimum functions in the function set.

    trigonometric : bool, optional (default=False)
        Whether to include sin, cos and tan functions in the function set. Note
        that these functions work on radian angles, if your data is presented
        as degrees, you may wish to covert using, for example, `np.radians`.

    metric : str, optional (default='mean absolute error')
        The name of the raw fitness metric. Available options include:
        - 'mean absolute error',
        - 'mse' for mean squared error,
        - 'rmse' for root mean squared error, and
        - 'rmsle' for root mean squared logarithmic error.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    program_ : _Program object
        The fittest individual in the final generation.

    fitness_ : float
        The fitness of the fittest individual in the final generation.

    See also
    --------
    DecisionTreeRegressor

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.
    """

    def __init__(self,
                 population_size=500,
                 generations=10,
                 tournament_size=20,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 transformer=True,
                 comparison=True,
                 trigonometric=False,
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 bootstrap=False,
                 max_samples=1.0,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):

        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.transformer = transformer
        self.comparison = comparison
        self.trigonometric = trigonometric
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit Symbolic Regressor according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        # Check arrays
        X, y = check_X_y(X, y, y_numeric=True)

        self._function_set = ['add2', 'sub2', 'mul2', 'div2']
        if self.transformer:
            self._function_set.extend(['sqrt1', 'log1', 'abs1', 'neg1',
                                       'inv1'])
        if self.comparison:
            self._function_set.extend(['max2', 'min2'])
        if self.trigonometric:
            self._function_set.extend(['sin1', 'cos1', 'tan1'])

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = int(function[-1])
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if (not isinstance(self.const_range, tuple) or
                len(self.const_range) != 2):
            raise ValueError('const_range should be a tuple with length two.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        params = self.get_params()
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        self._programs = []

        if self.verbose:
            # header fields and line format str
            header_fields = ['Gen', 'AveFit', 'BestFit', 'AveLen', 'BestLen',
                             'OOBFit', 'TimeLeft']
            print(('%10s ' + '%16s ' * (len(header_fields) - 1)) %
                  tuple(header_fields))
            start_time = time()

        for gen in range(self.generations):

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            if self.verbose > 1:
                verbose = 1
            else:
                verbose = 0

            population = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                fitness = [program.raw_fitness_ for program in population]
                length = [program.length_ for program in population]
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)

            self._programs.append(population)

            if self.verbose:

                remaining_time = ((self.generations - gen - 1) *
                                  (time() - start_time) / float(gen + 1))
                if remaining_time > 60:
                    remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
                else:
                    remaining_time = '{0:.2f}s'.format(remaining_time)

                if self.parsimony_coefficient != 'auto':
                    fitness = [program.fitness_ for program in population]
                    length = [program.length_ for program in population]
                best_program = population[np.argmin(fitness)]

                print(('%10s ' + '%16s ' * (len(header_fields) - 1)) %
                      (gen,
                       np.mean(fitness),
                       best_program.fitness_,
                       np.mean(length),
                       best_program.length_,
                       1.0,
                       remaining_time))

        # Find the best individual in the final generation
        self.fitness_ = [program.fitness_ for program in self._programs[-1]]
        self.program_ = self._programs[-1][np.argmin(self.fitness_)]
        self.fitness_ = self.program_.fitness_

    def predict(self, X):
        """
        Perform classification on test vectors X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted target values for X.
        """
        if not hasattr(self, "program_"):
            raise NotFittedError("SymbolicRegressor not fitted.")

        X = check_array(X)

        y = self.program_.execute(X)

        return y
