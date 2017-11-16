.. currentmodule:: gplearn
.. _changelog:

Release History
===============

Version 0.3.0
-------------

- Fixed two bugs in :class:`genetic.SymbolicTransformer` where the final
  solution selection logic was incorrect and suboptimal. This fix will change
  the solutions from all previous versions of `gplearn`. Thanks to
  `iblasi <https://github.com/iblasi>`_ for diagnosing the problem and helping
  craft the solution.
- Fixed bug in :class:`genetic.SymbolicRegressor` where a custom fitness
  measure was defined in :func:`fitness.make_fitness()` with the parameter
  `greater_is_better=True`. This was ignored during final solution selection.
  This change will alter the results from previous releases where
  `greater_is_better=True` was set in a custom fitness measure. By
  `sun ao <https://github.com/eggachecat>`_.
- Increase minimum required version of ``scikit-learn`` to 0.18.1. This allows
  streamlining the test suite and removal of many utilities to reduce future
  technical debt. **Please note that due to this change, previous versions
  may have different results** due to a change in random sampling noted
  `here <http://scikit-learn.org/stable/whats_new.html#version-0-18-1>`_.
- Drop support for Python 2.6 and add support for Python 3.5 and 3.6 in order
  to support the latest release of ``scikit-learn`` 0.19 and avoid future test
  failures. By `hugovk <https://github.com/hugovk>`_.

Version 0.2.0
-------------

- Allow more generations to be evolved on top of those already trained using a
  previous call to fit. The :class:`genetic.SymbolicRegressor` and
  :class:`genetic.SymbolicTransformer` classes now support the ``warm_start``
  parameter which, when set to ``True``, reuse the solution of the previous
  call to fit and add more generations to the evolution.
- Allow users to define their own fitness measures. Supported by the
  :func:`fitness.make_fitness()` factory function. Using this a user may define
  any metric by which to measure the fitness of a program to optimize any
  problem. This also required modifying the API slightly with the deprecation
  of the ``'rmsle'`` error measure for the :class:`genetic.SymbolicRegressor`.
- Allow users to define their own functions for use in genetic programs.
  Supported by the :func:`functions.make_function()` factory function. Using
  this a user may define any mathematical relationship with any number of
  arguments and grow totally customized programs. This also required modifying
  the API with the deprecation of the ``'comparison'``, ``'transformer'`` and
  ``'trigonometric'`` arguments to the :class:`genetic.SymbolicRegressor` and
  :class:`genetic.SymbolicTransformer` classes in favor of the new
  ``function_set`` where any combination of preset and user-defined functions
  can be supplied. To restore previous behavior initialize the estimator with
  ``function_set=['add2', 'sub2', 'mul2', 'div2', 'sqrt1', 'log1', 'abs1',
  'neg1', 'inv1', 'max2', 'min2']``.
- Reduce memory consumption for large datasets, large populations or many
  generations. Indices for in-sample/out-of-sample fitness calculations are now
  generated on demand rather than being stored in the program objects which
  reduces the size significantly for large datasets. Additionally "irrelevant"
  programs from earlier generations are removed if they did not contribute to
  the current population through genetic operations. This reduces the number of
  programs stored in the estimator which helps for large populations, high
  number of generations, as well as for runs with significant bloat.

Version 0.1.0
-------------

Initial public release supporting symbolic regression tasks through the
:class:`genetic.SymbolicRegressor` class for regression problems and the
:class:`genetic.SymbolicTransformer` class for automated feature engineering.
