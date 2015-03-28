"""
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

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.externals import six
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error

from .skutils import _partition_estimators
from .skutils.validation import check_random_state, check_X_y, check_array
from .skutils.validation import NotFittedError

__all__ = ['SymbolicRegressor']

MAX_INT = np.iinfo(np.int32).max


def protected_devision(x1, x2):
    """Closure of division for zero denominator"""
    return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def protected_sqrt(x1):
    """Closure of square root for negative arguments"""
    return np.sqrt(np.abs(x1))


def protected_log(x1):
    """Closure of log for zero arguments"""
    return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


# Format is '<name><arity>': function
FUNCTIONS = {'add2': np.add,
             'sub2': np.subtract,
             'mul2': np.multiply,
             'div2': protected_devision,
             'sqrt1': protected_sqrt,
             'log1': protected_log,
             'abs1': np.abs,
             'max2': np.maximum,
             'min2': np.minimum}


def _parallel_evolve(n_programs, parents, X, y, seeds, params):
    """Private function used to build a batch of programs within a job."""

    def _tournament():
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
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']

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
                program = parent.crossover(donor.program, random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'donor_idx': donor_index}
            elif method < method_probs[1]:
                # subtree_mutation
                program = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index}
            elif method < method_probs[2]:
                # hoist_mutation
                program = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index}
            elif method < method_probs[3]:
                # point_mutation
                program = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           random_state=random_state,
                           program=program)

        # TODO: parent mutated nodes need to be serialized
        program.parents = genome
        program.fitness_ = program.fitness(mean_absolute_error, X, y)

        programs.append(program)

    return programs


class _Program(object):
    """A program-like representation of the evolved program"""

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
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
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.fitness_ = None
        self.parents = None

    def build_program(self, random_state):
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
        return len(self.program)

    def execute(self, X):

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
                    return intermediate_result

        # We should never get here
        np.seterr(**old_settings)
        return None

    def fitness(self, metric, X, y):
        return (metric(y, self.execute(X)) +
                (self.parsimony_coefficient * len(self.program)))

    def get_subtree(self, random_state, program=None):

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
        return deepcopy(self.program)

    def crossover(self, donor, random_state):
        """donor is a program-like list"""
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:])

    def subtree_mutation(self, random_state):
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):

        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]

        return self.program[:start] + hoist + self.program[end:]

    def point_mutation(self, random_state):

        program = deepcopy(self.program)

        # Get the nodes to modify
        mutate = np.where([True if (random_state.uniform() <
                                    self.p_point_replace)
                           else False
                           for _ in xrange(len(program))])[0]

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

        return program

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
        Whether to include protected square root, protected log, and absolute
        value functions in the function set.

    comparison : bool, optional (default=True)
        Whether to include maximum and minimum functions in the function set.

    parsimony_coefficient : float, optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

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
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
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
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        """Fit Symbolic Regressor according to X, y

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

        X, y = check_X_y(X, y)
        y = np.ascontiguousarray(y, dtype=np.float64)

        random_state = check_random_state(self.random_state)

        self._function_set = ['add2', 'sub2', 'mul2', 'div2']
        if self.transformer:
            self._function_set.extend(['sqrt1', 'log1', 'abs1'])
        if self.comparison:
            self._function_set.extend(['max2', 'min2'])

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

        params = self.get_params()
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        self._programs = []

        for gen in range(self.generations):

            if self.verbose:
                print('Evolving generation %d of %d.' % (gen + 1,
                                                         self.generations))

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))
            self._programs.append(population)

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
